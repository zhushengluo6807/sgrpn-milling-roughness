import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from roughness.metrics import regression_metrics

from .classic import run_classic_outer_cv
from .config import load_scheme1_config, write_protocol_checkpoint
from .evaluation import assess_model_gate, paired_group_bootstrap
from .features import build_segment_feature_table, write_feature_artifacts
from .folds import validate_outer_folds
from .signals import write_signal_artifacts
from .training import scheme1_run_fingerprint, train_one_fold
from .windows import build_window_index


def _csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def completed_run_matches(
    run_dir: str | Path,
    requested_max_epochs: int,
    expected_fingerprint: str | None = None,
) -> bool:
    directory = Path(run_dir)
    metadata_path = directory / "run_metadata.json"
    oof_path = directory / "oof_predictions.csv"
    if not metadata_path.is_file() or not oof_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    matches = (
        metadata.get("status") == "complete"
        and int(metadata.get("requested_max_epochs", -1))
        == int(requested_max_epochs)
    )
    if expected_fingerprint is not None:
        matches = (
            matches
            and metadata.get("run_fingerprint") == expected_fingerprint
        )
    return matches


def validate_complete_neural_oof(
    frame: pd.DataFrame,
    expected_sample_ids: set[str],
    models: list[str],
    seeds: list[int],
) -> None:
    required = {"sample_id", "model", "seed", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Neural OOF missing columns: {missing}")
    if frame.duplicated(["sample_id", "model", "seed"]).any():
        raise ValueError("Neural OOF contains duplicate predictions")
    for model in models:
        for seed in seeds:
            observed = set(
                frame.loc[
                    (frame["model"] == model)
                    & (frame["seed"].astype(int) == int(seed)),
                    "sample_id",
                ].astype(str)
            )
            if observed != expected_sample_ids:
                missing_ids = sorted(expected_sample_ids - observed)
                extra_ids = sorted(observed - expected_sample_ids)
                raise ValueError(
                    f"Incomplete OOF for {model}/seed={seed}; "
                    f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
                )
    if not np.isfinite(frame["y_pred"].to_numpy(dtype=float)).all():
        raise ValueError("Neural OOF contains non-finite predictions")


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, fold), group in predictions.groupby(
        ["model", "seed", "fold"], sort=True
    ):
        plain = regression_metrics(group["y_true"], group["y_pred"])
        weighted = regression_metrics(
            group["y_true"], group["y_pred"], group["sample_weight"]
        )
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "fold": int(fold),
                **plain,
                **{f"weighted_{key}": value for key, value in weighted.items()},
            }
        )
    return pd.DataFrame.from_records(rows)


def _handle_audit(config, force: bool = False) -> None:
    manifest = pd.read_csv(config.manifest_path)
    folds = pd.read_csv(config.folds_path)
    fold_audit = validate_outer_folds(manifest, folds)
    write_protocol_checkpoint(config, fold_audit)
    if force or not (config.output_dir / "audit.json").is_file():
        write_signal_artifacts(
            manifest,
            folds,
            config.output_dir,
            config.sample_rate_hz,
        )
    print(json.dumps(fold_audit, ensure_ascii=False))


def _handle_features(config, force: bool = False) -> None:
    output = config.output_dir / "features" / "segment_features.csv"
    if output.is_file() and not force:
        print(f"reuse {output}")
        return
    manifest = pd.read_csv(config.manifest_path)
    windows = build_window_index(
        manifest, config.window_samples, config.stride_samples
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(config.output_dir / "window_index.csv", index=False)
    table = build_segment_feature_table(
        manifest,
        windows,
        config.re_candidates_mm,
        sample_rate_hz=config.sample_rate_hz,
        welch_nperseg=config.welch_nperseg,
        welch_noverlap=config.welch_noverlap,
        nominal_band_min_halfwidth_hz=config.nominal_band_min_halfwidth_hz,
        nominal_band_relative_halfwidth=config.nominal_band_relative_halfwidth,
        hf_band_hz=config.hf_band_hz,
    )
    print(write_feature_artifacts(table, config.output_dir / "features"))


def _handle_classic(config, force: bool = False) -> None:
    destination = config.output_dir / "classic"
    output = destination / "oof_predictions.csv"
    if output.is_file() and not force:
        print(f"reuse {output}")
        return
    table = pd.read_csv(config.output_dir / "features" / "segment_features.csv")
    folds = pd.read_csv(config.folds_path)
    runs = [
        run_classic_outer_cv(
            table,
            folds,
            seed,
            config.inner_splits,
        )
        for seed in config.seeds
    ]
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat([run[0] for run in runs], ignore_index=True).to_csv(
        destination / "oof_predictions.csv", index=False
    )
    pd.concat(
        [
            run[1].assign(seed=seed)
            for run, seed in zip(runs, config.seeds, strict=True)
        ],
        ignore_index=True,
    ).to_csv(destination / "metrics.csv", index=False)
    pd.concat([run[2] for run in runs], ignore_index=True).to_csv(
        destination / "selection.csv", index=False
    )
    print(output)


def _handle_train(
    config,
    models: list[str],
    folds: list[int],
    seeds: list[int],
    max_epochs: int,
    batch_size: int,
    resume: bool,
    fusion_encoder: str,
) -> None:
    collected = []
    for model in models:
        for fold in folds:
            for seed in seeds:
                run_dir = (
                    config.output_dir
                    / "neural"
                    / model
                    / f"fold_{fold}"
                    / f"seed_{seed}"
                )
                fingerprint = scheme1_run_fingerprint(
                    config,
                    model,
                    fold,
                    seed,
                    max_epochs,
                    batch_size,
                    0.2,
                    fusion_encoder,
                )
                if resume and completed_run_matches(
                    run_dir, max_epochs, fingerprint
                ):
                    print(f"resume-skip {model} fold={fold} seed={seed}", flush=True)
                else:
                    print(f"train {model} fold={fold} seed={seed}", flush=True)
                    train_one_fold(
                        model,
                        fold,
                        seed,
                        config,
                        batch_size=batch_size,
                        max_epochs=max_epochs,
                        fusion_encoder=fusion_encoder,
                    )
                collected.append(pd.read_csv(run_dir / "oof_predictions.csv"))
    predictions = pd.concat(collected, ignore_index=True)
    destination = config.output_dir / "neural"
    all_folds = sorted(folds) == [0, 1, 2, 3, 4]
    if all_folds:
        expected = set(
            pd.read_csv(config.manifest_path)["sample_id"].astype(str)
        )
        validate_complete_neural_oof(predictions, expected, models, seeds)
        predictions.to_csv(destination / "oof_predictions.csv", index=False)
        _metric_rows(predictions).to_csv(
            destination / "metrics.csv", index=False
        )
    else:
        predictions.to_csv(
            destination / "oof_predictions_partial.csv", index=False
        )


def _handle_evaluate(config) -> None:
    from .reporting import write_error_breakdowns, write_prediction_figures

    neural = pd.read_csv(config.output_dir / "neural" / "oof_predictions.csv")
    manifest = pd.read_csv(config.manifest_path)
    models = ["N1", "N2", "N3", "N4"]
    validate_complete_neural_oof(
        neural,
        set(manifest["sample_id"].astype(str)),
        models,
        list(config.seeds),
    )
    neural_metrics = _metric_rows(neural)
    classic_metrics = pd.read_csv(config.output_dir / "classic" / "metrics.csv")
    m0_metrics = classic_metrics[classic_metrics["model"] == "M0"]
    metrics = pd.concat([m0_metrics, neural_metrics], ignore_index=True)
    evaluation_dir = config.output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(evaluation_dir / "summary_metrics.csv", index=False)

    gates = {
        "N2_vs_N1": assess_model_gate(
            metrics, "N2", "N1", 0.02, 3, False
        ),
        "N3_vs_M0": assess_model_gate(
            metrics, "N3", "M0", 0.03, 3, False
        ),
        "N4_vs_M0": assess_model_gate(
            metrics, "N4", "M0", 0.05, 4, True
        ),
        "N5": {"passed": False, "skipped_reason": "P4_no_stable_increment"},
        "N6": {"passed": False, "skipped_reason": "N5_not_enabled"},
    }
    classic_predictions = pd.read_csv(
        config.output_dir / "classic" / "oof_predictions.csv"
    )
    m0 = classic_predictions[
        (classic_predictions["model"] == "M0")
        & (classic_predictions["seed"].astype(int) == config.seeds[0])
    ][["sample_id", "group_id", "y_true", "sample_weight", "y_pred"]].rename(
        columns={"y_pred": "reference"}
    )
    n4 = (
        neural[neural["model"] == "N4"]
        .groupby("sample_id", as_index=False)["y_pred"]
        .mean()
        .rename(columns={"y_pred": "candidate"})
    )
    paired = m0.merge(n4, on="sample_id", validate="one_to_one")
    bootstrap = paired_group_bootstrap(
        paired,
        "reference",
        "candidate",
        repetitions=config.bootstrap_repetitions,
        seed=config.seeds[0],
    )
    pd.DataFrame([bootstrap.to_dict()]).to_csv(
        evaluation_dir / "paired_bootstrap.csv", index=False
    )
    n4_gate = gates["N4_vs_M0"]
    if n4_gate["passed"] and bootstrap.ci_low > 0:
        conclusion = "stable_effective"
    elif (
        n4_gate["relative_improvement"] >= 0.03
        and n4_gate["fold_wins"] >= 3
    ):
        conclusion = "exploratory_increment"
    else:
        conclusion = "no_stable_increment"
    acceptance = {
        "gates": gates,
        "bootstrap": bootstrap.to_dict(),
        "scheme1_conclusion": conclusion,
    }
    (evaluation_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_frame = pd.concat(
        [
            m0.assign(model="M0").rename(columns={"reference": "y_pred"}),
            paired.assign(model="N4").rename(columns={"candidate": "y_pred"}),
        ],
        ignore_index=True,
    )
    write_prediction_figures(
        figure_frame[["model", "y_true", "y_pred"]],
        evaluation_dir / "figures",
    )
    n4_breakdown = paired[
        ["sample_id", "y_true", "candidate", "sample_weight"]
    ].rename(columns={"candidate": "y_pred"})
    write_error_breakdowns(
        n4_breakdown,
        manifest,
        evaluation_dir / "breakdowns",
    )
    classic_selection = pd.read_csv(
        config.output_dir / "classic" / "selection.csv"
    )
    (
        classic_selection.groupby(
            ["stage", "model_name", "xy_mode", "re_mm"],
            dropna=False,
        )
        .size()
        .rename("selection_count")
        .reset_index()
        .to_csv(evaluation_dir / "classic_selection_summary.csv", index=False)
    )
    notes = {
        "horizontal_direction": (
            "A/B are nested predictive hypotheses, not measurements of "
            "sensor installation direction."
        ),
        "effective_radius": (
            "re values are sensitivity proxies, not measured tool radii; "
            "P4 showed no stable increment."
        ),
        "displacement_proxy": (
            "Band-limited displacement features remain in relative units "
            "because sensor calibration is unavailable."
        ),
        "network_fallback": (
            "N2 failed its gate, so N3 and N4 used the CNN encoder and do "
            "not claim an independent TCN contribution."
        ),
        "stopped_models": {
            "N5": "skipped because P4 had no stable increment",
            "N6": "skipped because N5 was not enabled",
        },
    }
    (evaluation_dir / "method_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roughness-scheme1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "features", "classic", "evaluate"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", required=True)
        if name in {"audit", "features", "classic"}:
            subparser.add_argument("--force", action="store_true")
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--models", default="N1,N2,N3,N4")
    train.add_argument("--folds", default="0,1,2,3,4")
    train.add_argument("--seeds", default="20260723,20260724,20260725")
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--fusion-encoder", choices=["cnn", "cnn_tcn"], default="cnn"
    )
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--max-epochs", type=int)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--fusion-encoder", choices=["cnn", "cnn_tcn"], default="cnn"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_scheme1_config(args.config)
    if args.command == "audit":
        _handle_audit(config, args.force)
    elif args.command == "features":
        _handle_features(config, args.force)
    elif args.command == "classic":
        _handle_classic(config, args.force)
    elif args.command == "train":
        _handle_train(
            config,
            _csv_list(args.models),
            _csv_list(args.folds, int),
            _csv_list(args.seeds, int),
            args.max_epochs or config.max_epochs,
            args.batch_size,
            args.resume,
            args.fusion_encoder,
        )
    elif args.command == "evaluate":
        _handle_evaluate(config)
    elif args.command == "run":
        _handle_audit(config)
        _handle_features(config)
        _handle_classic(config)
        _handle_train(
            config,
            ["N1", "N2", "N3", "N4"],
            [0, 1, 2, 3, 4],
            list(config.seeds),
            args.max_epochs or config.max_epochs,
            args.batch_size,
            args.resume,
            args.fusion_encoder,
        )
        _handle_evaluate(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
