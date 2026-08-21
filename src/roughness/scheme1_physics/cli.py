import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from .config import (
    EXPECTED_MODELS,
    load_scheme1_physics_config,
    write_physics_protocol_checkpoint,
)
from .data import prepare_physics_fold
from .evaluation import (
    assess_physics_comparisons,
    physics_metric_rows,
    select_final_candidate,
    validate_complete_physics_oof,
)
from .physics import exact_ra_um, word_ra_um
from .reporting import (
    write_formula_difference_report,
    write_gate_reports,
    write_method_notes,
    write_physics_error_breakdowns,
    write_physics_prediction_figures,
)
from .training import (
    completed_physics_run_matches,
    physics_run_fingerprint,
    train_gated_physics_fold,
    train_physics_fold,
    write_physics_baseline_oof,
)


NEURAL_MODELS = ("PW1", "PE1", "PW2", "PE2")


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ordered_neural_models(requested: list[str]) -> list[str]:
    unknown = set(requested) - set(NEURAL_MODELS)
    if unknown:
        raise ValueError(
            f"Only neural models {NEURAL_MODELS} are allowed; got {sorted(unknown)}"
        )
    expanded = set(requested)
    if "PW2" in expanded:
        expanded.add("PW1")
    if "PE2" in expanded:
        expanded.add("PE1")
    return [model for model in NEURAL_MODELS if model in expanded]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scheme 1 physical-formula residual experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True)

    baselines = subparsers.add_parser("baselines")
    baselines.add_argument("--config", required=True)
    baselines.add_argument("--folds", default="0,1,2,3,4")
    baselines.add_argument("--seeds", default="20260723,20260724,20260725")
    baselines.add_argument("--force", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--models", default="PW1,PW2,PE1,PE2")
    train.add_argument("--folds", default="0,1,2,3,4")
    train.add_argument("--seeds", default="20260723,20260724,20260725")
    train.add_argument("--max-epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--resume", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--max-epochs", type=int, default=None)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--resume", action="store_true")
    return parser


def _parent_checkpoint(config, model: str, fold: int, seed: int) -> Path:
    parent = "PW1" if model == "PW2" else "PE1"
    return (
        config.output_dir
        / "neural"
        / parent
        / f"fold_{fold}"
        / f"seed_{seed}"
        / "best.pt"
    )


def _run_dir(config, model: str, fold: int, seed: int) -> Path:
    return (
        config.output_dir
        / "neural"
        / model
        / f"fold_{fold}"
        / f"seed_{seed}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def train_requested(
    config,
    models: list[str],
    folds: list[int],
    seeds: list[int],
    max_epochs: int | None,
    batch_size: int,
    resume: bool,
) -> None:
    actual_epochs = int(max_epochs or config.source.max_epochs)
    for model in ordered_neural_models(models):
        formula = "word" if model.startswith("PW") else "exact"
        for fold in folds:
            for seed in seeds:
                parent = (
                    _parent_checkpoint(config, model, fold, seed)
                    if model.endswith("2")
                    else None
                )
                if parent is not None and not parent.is_file():
                    raise ValueError(
                        f"Required parent checkpoint does not exist: {parent}"
                    )
                prepared = prepare_physics_fold(
                    config, formula=formula, outer_fold=fold, seed=seed
                )
                fingerprint = physics_run_fingerprint(
                    config,
                    model,
                    fold,
                    seed,
                    prepared.selection.radius_mm,
                    actual_epochs,
                    batch_size,
                    config.validation_fraction,
                    parent_checkpoint=parent,
                )
                parent_fingerprint = _sha256(parent) if parent else None
                if resume and completed_physics_run_matches(
                    _run_dir(config, model, fold, seed),
                    fingerprint,
                    parent_fingerprint,
                ):
                    print(f"resume: {model} fold={fold} seed={seed}")
                    continue
                if model.endswith("1"):
                    train_physics_fold(
                        model,
                        fold,
                        seed,
                        config,
                        max_epochs=actual_epochs,
                        batch_size=batch_size,
                    )
                else:
                    train_gated_physics_fold(
                        model,
                        fold,
                        seed,
                        config,
                        parent_checkpoint=parent,
                        max_epochs=actual_epochs,
                        batch_size=batch_size,
                    )


def aggregate_oof(
    config,
    folds: list[int],
    seeds: list[int],
) -> tuple[pd.DataFrame, Path]:
    full = set(folds) == set(range(5)) and set(seeds) == set(
        config.source.seeds
    )
    suffix = "" if full else "_partial"
    baseline_path = (
        config.output_dir / "physics" / f"oof_predictions{suffix}.csv"
    )
    if not baseline_path.is_file():
        raise ValueError(f"Missing physical baseline OOF: {baseline_path}")
    parts = [pd.read_csv(baseline_path)]
    for model in NEURAL_MODELS:
        for fold in folds:
            for seed in seeds:
                path = (
                    _run_dir(config, model, fold, seed)
                    / "oof_predictions.csv"
                )
                if not path.is_file():
                    raise ValueError(f"Missing neural OOF: {path}")
                parts.append(pd.read_csv(path))
    combined = pd.concat(parts, ignore_index=True)
    output_path = config.output_dir / f"oof_predictions{suffix}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    metrics = physics_metric_rows(combined)
    metrics.to_csv(config.output_dir / f"metrics{suffix}.csv", index=False)
    return combined, output_path


def evaluate_complete(config) -> dict:
    predictions = pd.read_csv(config.output_dir / "oof_predictions.csv")
    manifest = pd.read_csv(config.source.manifest_path)
    validate_complete_physics_oof(
        predictions,
        set(manifest["sample_id"].astype(str)),
        EXPECTED_MODELS,
        config.source.seeds,
    )
    metrics = physics_metric_rows(predictions)
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    classic = pd.read_csv(
        config.source.output_dir / "classic" / "oof_predictions.csv"
    )
    m0 = classic[classic["model"].eq("M0")].copy()
    if m0.empty:
        raise ValueError("Source Scheme 1 contains no M0 OOF predictions")
    assessed = assess_physics_comparisons(metrics, predictions, m0, config)
    acceptance_by_model = assessed["m0_acceptance"]
    selected = select_final_candidate(acceptance_by_model, metrics)
    conclusion = (
        selected.get("classification")
        if selected.get("model")
        else "no_stable_increment"
    )
    comparisons = assessed["comparisons"]
    acceptance = {
        "formula_comparisons": assessed["formula_comparisons"],
        "residual_gates": {
            key: value
            for key, value in comparisons.items()
            if key in {"PW1_vs_PW0", "PE1_vs_PE0"}
        },
        "gating_gates": {
            key: value
            for key, value in comparisons.items()
            if key in {"PW2_vs_PW1", "PE2_vs_PE1"}
        },
        "m0_comparisons": acceptance_by_model,
        "bootstrap": {
            model: result["bootstrap"]
            for model, result in acceptance_by_model.items()
        },
        "selected_model": selected,
        "scheme1_physics_conclusion": conclusion,
    }
    evaluation_dir = config.output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "comparisons.json").write_text(
        json.dumps(assessed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (evaluation_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"model": model, **result["bootstrap"]}
            for model, result in acceptance_by_model.items()
        ]
    ).to_csv(evaluation_dir / "paired_bootstrap.csv", index=False)

    formula_rows = []
    for radius in config.source.re_candidates_mm:
        for row in manifest[["sample_id", "fz_mm_per_tooth"]].itertuples(
            index=False
        ):
            formula_rows.append(
                {
                    "sample_id": str(row.sample_id),
                    "radius_mm": radius,
                    "fz_mm_per_tooth": row.fz_mm_per_tooth,
                    "exact_um": exact_ra_um(row.fz_mm_per_tooth, radius),
                    "word_um": word_ra_um(row.fz_mm_per_tooth, radius),
                }
            )
    write_formula_difference_report(
        pd.DataFrame.from_records(formula_rows), evaluation_dir
    )
    baseline = predictions[predictions["model"].isin(["PW0", "PE0"])]
    selected_difference = baseline.pivot(
        index=["sample_id", "group_id", "fold", "seed"],
        columns="model",
        values=["y_pred", "radius_mm"],
    )
    selected_difference.columns = [
        f"{'exact' if model == 'PE0' else 'word'}_{'prediction_um' if value == 'y_pred' else 'radius_mm'}"
        for value, model in selected_difference.columns
    ]
    selected_difference.reset_index().to_csv(
        evaluation_dir / "formula_selected_prediction_difference.csv",
        index=False,
    )
    report_predictions = pd.concat([predictions, m0], ignore_index=True)
    write_physics_prediction_figures(
        report_predictions, evaluation_dir / "figures"
    )
    gated = predictions[predictions["model"].isin(["PW2", "PE2"])].merge(
        manifest[
            ["sample_id", "n_rpm", "fz_mm_per_tooth", "ap_mm"]
        ],
        on="sample_id",
        validate="many_to_one",
    )
    write_gate_reports(gated, evaluation_dir)
    write_physics_error_breakdowns(
        report_predictions, manifest, evaluation_dir / "breakdowns"
    )
    write_method_notes(evaluation_dir)
    return acceptance


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_scheme1_physics_config(args.config)
    if args.command == "prepare":
        write_physics_protocol_checkpoint(config)
        return 0
    if args.command == "baselines":
        write_physics_baseline_oof(
            config,
            force=args.force,
            folds=_csv_ints(args.folds),
            seeds=_csv_ints(args.seeds),
        )
        return 0
    if args.command == "train":
        folds = _csv_ints(args.folds)
        seeds = _csv_ints(args.seeds)
        train_requested(
            config,
            _csv_strings(args.models),
            folds,
            seeds,
            args.max_epochs,
            args.batch_size,
            args.resume,
        )
        aggregate_oof(config, folds, seeds)
        return 0
    if args.command == "evaluate":
        evaluate_complete(config)
        return 0
    if args.command == "run":
        write_physics_protocol_checkpoint(config)
        write_physics_baseline_oof(config, force=not args.resume)
        train_requested(
            config,
            list(NEURAL_MODELS),
            list(range(5)),
            list(config.source.seeds),
            args.max_epochs,
            args.batch_size,
            args.resume,
        )
        aggregate_oof(
            config, list(range(5)), list(config.source.seeds)
        )
        evaluate_complete(config)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
