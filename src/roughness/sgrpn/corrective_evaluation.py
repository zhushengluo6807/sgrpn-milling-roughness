"""One-shot evaluation for the frozen group split-conformal correction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
import uuid

import pandas as pd
from pandas.testing import assert_frame_equal

from . import phase_b_training
from .config import PhaseBConfig
from .corrective import (
    CORRECTIVE_PROTOCOL,
    CorrectiveConfig,
    corrective_fingerprint,
    load_validated_corrective_fold,
)
from .data import DataBundle
from .evaluation import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    paired_probability_bootstrap,
    phase_b_mean_bootstrap_table,
    phase_b_mean_metric_table,
    phase_b_negative_transfer_table,
    probability_metric_table,
    seed_uncertainty_summary,
)
from .order_spectrum import OrderSpectrumCache


_ANALYSIS_STATUS = "post_audit_corrective_reanalysis"
_METHOD_NOTES = {
    "analysis_status": _ANALYSIS_STATUS,
    "calibration": "group_split_conformal_with_locked_predictor",
    "calibration_split_seed": 20260723,
    "calibration_inner_block": 0,
    "calibration_group_count_per_outer_fold": 43,
    "quantile_rule": "ceil((m+1)*(1-alpha))_order_statistic_without_interpolation",
    "outer_results": "corrective_empirical_evidence_on_previously_analyzed_data",
    "seed_dependence": "three_seeds_share_the_same_212_groups",
    "coverage_scope": "exchangeable_new_groups_of_existing_type",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_tree_inventory(root: str | Path) -> tuple[str, int, int]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise ValueError("corrective training root is missing")
    files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and "evaluation" not in path.relative_to(base).parts
        and path.name != "training_seal.json"
    )
    if not files:
        raise ValueError("corrective training root contains no files")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(base).as_posix()
        record = f"{relative}\0{path.stat().st_size}\0{_sha256_file(path)}\n"
        digest.update(record.encode("utf-8"))
        total_bytes += path.stat().st_size
    return digest.hexdigest(), len(files), total_bytes


def corrective_training_tree_digest(root: str | Path) -> str:
    """Hash only frozen training artifacts, excluding seal/evaluation files."""
    return _training_tree_inventory(root)[0]


def _load_training_seal(root: Path) -> dict[str, object]:
    path = root / "training_seal.json"
    try:
        seal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("corrective training seal is missing or invalid") from error
    required = {
        "protocol",
        "status",
        "training_tree_sha256",
        "file_count",
        "total_bytes",
        "completed_units",
        "phase_a_config_file_sha256",
        "phase_b_config_file_sha256",
        "phase_b_run_manifest_sha256",
        "phase_b_immutable_after_sha256",
        "source_revision",
        "source_files",
        "unit_completion_sha256",
        "run_manifest_sha256",
    }
    if (
        not isinstance(seal, dict)
        or set(seal) != required
        or seal.get("protocol") != CORRECTIVE_PROTOCOL
        or seal.get("status") != "sealed"
        or seal.get("completed_units") != 15
        or not isinstance(seal.get("source_files"), dict)
        or not isinstance(seal.get("unit_completion_sha256"), dict)
        or len(seal["unit_completion_sha256"]) != 15
    ):
        raise ValueError("corrective training seal is incompatible")
    digest = str(seal["training_tree_sha256"])
    if len(digest) != 64 or corrective_training_tree_digest(root) != digest:
        raise ValueError("corrective training tree hash does not match its seal")
    return seal


def create_corrective_training_seal(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    *,
    source_revision: str,
    source_paths: Mapping[str, str | Path],
) -> Path:
    """Publish an exclusive root seal after the independently replayed 15/15 gate."""
    root = Path(corrective_config.output_dir).resolve()
    seal_path = root / "training_seal.json"
    if seal_path.exists():
        raise ValueError("corrective training seal already exists")
    if (root / "evaluation").exists():
        raise ValueError("corrective training cannot be sealed after evaluation starts")
    revision = str(source_revision)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("corrective source revision must be a 40-character Git commit")
    if not isinstance(source_paths, Mapping) or not source_paths:
        raise ValueError("corrective source files must be provided")
    actual_phase_a_hash = _sha256_file(Path(phase_b_config.phase_a_config_path))
    if actual_phase_a_hash != phase_b_config.phase_a_config_file_sha256:
        raise ValueError("Phase A config hash does not match the frozen registration")
    expected_markers = {
        f"folds/fold_{fold}/seed_{seed}/complete.json": (
            root / "folds" / f"fold_{fold}" / f"seed_{seed}" / "complete.json"
        )
        for fold in range(5)
        for seed in phase_b_training.PHASE_B_SEEDS
    }
    if not all(path.is_file() for path in expected_markers.values()):
        raise ValueError("corrective training seal requires exactly 15 completed units")
    actual_marker_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("complete.json")
        if path.is_file()
    }
    if actual_marker_paths != set(expected_markers):
        raise ValueError("corrective training seal completion marker set is incompatible")
    run_manifest_path = root / "run_manifest.json"
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("corrective training run manifest is missing or invalid") from error
    if (
        run_manifest.get("training_status") != "complete"
        or run_manifest.get("completed_units") != 15
    ):
        raise ValueError("corrective training run manifest is incomplete")
    source_hashes = {}
    for name, source in source_paths.items():
        label = str(name)
        path = Path(source).resolve()
        if not label or label in source_hashes or not path.is_file():
            raise ValueError("corrective source file set is incompatible")
        source_hashes[label] = _sha256_file(path)
    tree_hash, file_count, total_bytes = _training_tree_inventory(root)
    return _exclusive_json(
        seal_path,
        {
            "protocol": CORRECTIVE_PROTOCOL,
            "status": "sealed",
            "training_tree_sha256": tree_hash,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "completed_units": 15,
            "phase_a_config_file_sha256": actual_phase_a_hash,
            "phase_b_config_file_sha256": corrective_config.phase_b_config_file_sha256,
            "phase_b_run_manifest_sha256": corrective_config.phase_b_run_manifest_sha256,
            "phase_b_immutable_after_sha256": corrective_config.phase_b_immutable_after_sha256,
            "source_revision": revision,
            "source_files": source_hashes,
            "unit_completion_sha256": {
                relative: _sha256_file(path)
                for relative, path in expected_markers.items()
            },
            "run_manifest_sha256": _sha256_file(run_manifest_path),
        },
        exists_message="corrective training seal already exists",
    )


def _load_training_predictions(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(corrective_config.output_dir).resolve()
    units = []
    for fold in range(5):
        for seed in phase_b_training.PHASE_B_SEEDS:
            fingerprint = corrective_fingerprint(
                corrective_config,
                phase_b_config,
                bundle,
                cache,
                fold=fold,
                seed=seed,
            )
            units.append(
                load_validated_corrective_fold(
                    root / "folds" / f"fold_{fold}" / f"seed_{seed}",
                    fingerprint=fingerprint,
                    fold=fold,
                    seed=seed,
                )
            )
    probability = pd.concat(
        [unit.predictions for unit in units], ignore_index=True
    ).loc[:, phase_b_training.PHASE_B_PREDICTION_COLUMNS]
    mean = pd.concat([unit.mean_predictions for unit in units], ignore_index=True).loc[
        :, phase_b_training.PHASE_B_MEAN_COLUMNS
    ]
    saved_probability = pd.read_csv(
        root / "predictions" / "oof_probability_predictions.csv",
        float_precision="round_trip",
    )
    saved_mean = pd.read_csv(
        root / "predictions" / "oof_mean_predictions.csv",
        float_precision="round_trip",
    )
    assert_frame_equal(saved_probability, probability, check_dtype=False)
    assert_frame_equal(saved_mean, mean, check_dtype=False)
    phase_b_training._validate_phase_b_oof_cartesian(
        saved_probability, saved_mean, bundle
    )
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != CORRECTIVE_PROTOCOL
        or manifest.get("training_status") != "complete"
        or manifest.get("completed_units") != 15
        or manifest.get("selected_device") not in {"cpu", "cuda"}
    ):
        raise ValueError("corrective training manifest is incomplete")
    artifacts = manifest.get("combined_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("corrective combined artifact hash closure is missing")
    for relative, expected in artifacts.items():
        path = root / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("corrective combined artifact hash closure changed")
    return saved_probability, saved_mean


def _paired_bootstrap_table(
    probability_predictions: pd.DataFrame, mean_predictions: pd.DataFrame
) -> pd.DataFrame:
    rows = phase_b_mean_bootstrap_table(mean_predictions).to_dict(orient="records")
    for metric, interval_type, coverage in (
        ("gaussian_nll", None, None),
        ("gaussian_crps", None, None),
        ("mean_interval_width", "conformal", 0.90),
        ("mean_interval_width", "conformal", 0.95),
        ("winkler_score", "conformal", 0.90),
        ("winkler_score", "conformal", 0.95),
    ):
        result = paired_probability_bootstrap(
            probability_predictions,
            metric=metric,
            interval_type=interval_type,
            nominal_coverage=coverage,
        )
        rows.append(
            {
                "baseline": "homoscedastic",
                "candidate": "heteroscedastic",
                "metric": metric,
                "interval_type": interval_type,
                "nominal_coverage": coverage,
                **result.to_dict(),
                "confidence_level": 0.95,
            }
        )
    return pd.DataFrame(rows)


def _ablation_differences(probability_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = probability_metrics.loc[
        probability_metrics["aggregation"].isin(("seed", "all_seed"))
    ].copy()
    identifiers = [
        "aggregation",
        "seed",
        "fold",
        "interval_type",
        "nominal_coverage",
        "metric",
    ]
    heteroscedastic = selected.loc[
        selected["scale_model"] == "heteroscedastic", identifiers + ["value"]
    ].rename(columns={"value": "heteroscedastic_value"})
    homoscedastic = selected.loc[
        selected["scale_model"] == "homoscedastic", identifiers + ["value"]
    ].rename(columns={"value": "homoscedastic_value"})
    merged = heteroscedastic.merge(
        homoscedastic, on=identifiers, validate="one_to_one"
    )
    merged["heteroscedastic_minus_homoscedastic"] = (
        merged["heteroscedastic_value"] - merged["homoscedastic_value"]
    )
    return merged


def _build_non_claim_artifacts(
    probability_predictions: pd.DataFrame, mean_predictions: pd.DataFrame
) -> dict[str, object]:
    probability_metrics = probability_metric_table(probability_predictions)
    mean_metrics = phase_b_mean_metric_table(mean_predictions)
    fold_metrics = pd.concat(
        [
            mean_metrics.loc[mean_metrics["aggregation"] == "fold"].assign(
                metric_family="mean"
            ),
            probability_metrics.loc[
                probability_metrics["aggregation"] == "fold"
            ].assign(metric_family="probability"),
        ],
        ignore_index=True,
        sort=False,
    )
    seed_metrics = pd.concat(
        [
            mean_metrics.loc[
                mean_metrics["aggregation"].isin(("seed", "all_seed"))
            ].assign(metric_family="mean"),
            probability_metrics.loc[
                probability_metrics["aggregation"].isin(("seed", "all_seed"))
            ].assign(metric_family="probability"),
        ],
        ignore_index=True,
        sort=False,
    )
    return {
        "probability_metrics": probability_metrics,
        "mean_metrics": mean_metrics,
        "fold_metrics": fold_metrics,
        "seed_metrics": seed_metrics,
        "ablation_differences": _ablation_differences(probability_metrics),
        "paired_bootstrap": _paired_bootstrap_table(
            probability_predictions, mean_predictions
        ),
        "negative_transfer": phase_b_negative_transfer_table(mean_predictions),
        "seed_uncertainty_summary": seed_uncertainty_summary(
            probability_predictions
        ),
        "method_notes": dict(_METHOD_NOTES),
    }


def _build_claim_decision(probability_metrics: pd.DataFrame) -> dict[str, object]:
    selected = probability_metrics.loc[
        (probability_metrics["aggregation"] == "all_seed")
        & (probability_metrics["interval_type"] == "conformal")
        & probability_metrics["metric"].isin(
            ("simultaneous_group_coverage", "winkler_score")
        )
        & probability_metrics["scale_model"].isin(phase_b_training.SCALE_MODELS),
        ["scale_model", "nominal_coverage", "metric", "value"],
    ].copy()
    selected["nominal_coverage"] = pd.to_numeric(
        selected["nominal_coverage"], errors="raise"
    )
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    expected = {
        (model, coverage, metric)
        for model in phase_b_training.SCALE_MODELS
        for coverage in (0.90, 0.95)
        for metric in ("simultaneous_group_coverage", "winkler_score")
    }
    actual = set(
        zip(
            selected["scale_model"],
            selected["nominal_coverage"],
            selected["metric"],
            strict=True,
        )
    )
    if len(selected) != len(expected) or actual != expected:
        raise ValueError("corrective claim rule requires the exact registered metric rows")
    values = selected.set_index(["scale_model", "nominal_coverage", "metric"])[
        "value"
    ]
    coverage_met = {
        model: all(
            float(
                values.loc[(model, coverage, "simultaneous_group_coverage")]
            )
            >= coverage
            for coverage in (0.90, 0.95)
        )
        for model in phase_b_training.SCALE_MODELS
    }
    heteroscedastic_winkler_better = all(
        float(values.loc[("heteroscedastic", coverage, "winkler_score")])
        < float(values.loc[("homoscedastic", coverage, "winkler_score")])
        for coverage in (0.90, 0.95)
    )
    emphasize = bool(
        coverage_met["heteroscedastic"] and heteroscedastic_winkler_better
    )
    return {
        "analysis_status": _ANALYSIS_STATUS,
        "rule_status": "frozen_before_corrective_evaluation",
        "emphasize_heteroscedasticity": emphasize,
        "empirical_coverage_targets_met": coverage_met,
        "heteroscedastic_winkler_better_at_90_and_95": bool(
            heteroscedastic_winkler_better
        ),
        "retain_both_scale_variants": True,
        "retain_group_split_conformal_method": True,
        "practical_width_threshold_registered": False,
        "coverage_claim_scope": "exchangeable_new_groups_of_existing_type",
        "evidence_status": "corrective_empirical_evidence_on_previously_analyzed_data",
        "reasons": [
            (
                "heteroscedastic_corrective_rule_met"
                if emphasize
                else "heteroscedastic_corrective_rule_not_met"
            ),
            *[
                f"{model}_empirical_group_coverage_"
                + ("met" if met else "not_met")
                for model, met in coverage_met.items()
            ],
            "both_scale_variants_retained",
            "no_numeric_practical_width_threshold_registered",
        ],
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _exclusive_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    exists_message: str = "corrective evaluation has already been invoked",
) -> Path:
    """Acquire a durable one-shot marker with O_EXCL before any formal work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(exists_message) from error
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)
    return path


def evaluate_corrective_once(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
) -> Mapping[str, Path]:
    """Consume the single evaluation invocation and publish marker-last outputs."""
    root = Path(corrective_config.output_dir).resolve()
    evaluation = root / "evaluation"
    invocation = evaluation / "evaluation_invocation.json"
    _exclusive_json(
        invocation,
        {
            "protocol": CORRECTIVE_PROTOCOL,
            "analysis_status": _ANALYSIS_STATUS,
            "evaluation_invocations": 1,
            "training_seal": "../training_seal.json",
            "status": "started",
        },
    )
    if any(path != invocation for path in evaluation.iterdir()):
        raise ValueError("corrective evaluation directory is contaminated")
    seal = _load_training_seal(root)
    actual = str(seal["training_tree_sha256"])
    probability, mean = _load_training_predictions(
        corrective_config, phase_b_config, bundle, cache
    )
    artifacts = _build_non_claim_artifacts(probability, mean)
    artifacts["claim_decision"] = _build_claim_decision(
        artifacts["probability_metrics"]
    )
    written: dict[str, Path] = {"evaluation_invocation": invocation}
    for name, artifact in artifacts.items():
        if isinstance(artifact, pd.DataFrame):
            written[name] = _atomic_csv(evaluation / f"{name}.csv", artifact)
        elif isinstance(artifact, dict):
            written[name] = _atomic_json(evaluation / f"{name}.json", artifact)
        else:
            raise TypeError(f"unsupported corrective evaluation artifact: {name}")
    artifact_hashes = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in written.values()
    }
    manifest = _atomic_json(
        evaluation / "evaluation_manifest.json",
        {
            "protocol": CORRECTIVE_PROTOCOL,
            "analysis_status": _ANALYSIS_STATUS,
            "status": "complete",
            "evaluation_invocations": 1,
            "training_tree_sha256": actual,
            "training_seal_sha256": _sha256_file(root / "training_seal.json"),
            "bootstrap": {
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": BOOTSTRAP_SEED,
                "resampling_unit": "group_id",
            },
            "artifacts": artifact_hashes,
        },
    )
    written["evaluation_manifest"] = manifest
    written["evaluation_complete"] = _atomic_json(
        evaluation / "evaluation_complete.json",
        {
            "protocol": CORRECTIVE_PROTOCOL,
            "status": "complete",
            "evaluation_invocations": 1,
            "training_tree_sha256": actual,
            "evaluation_manifest_sha256": _sha256_file(manifest),
        },
    )
    return written


def validate_corrective_evaluation_outputs(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
) -> None:
    """Recompute evaluation artifacts without opening the sealed claim payload."""
    root = Path(corrective_config.output_dir).resolve()
    evaluation = root / "evaluation"
    invocation_path = evaluation / "evaluation_invocation.json"
    manifest_path = evaluation / "evaluation_manifest.json"
    complete_path = evaluation / "evaluation_complete.json"
    if not all(path.is_file() for path in (invocation_path, manifest_path, complete_path)):
        raise ValueError("corrective evaluation completion evidence is missing")
    seal = _load_training_seal(root)
    expected_training_tree_sha256 = str(seal["training_tree_sha256"])
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        invocation.get("protocol") != CORRECTIVE_PROTOCOL
        or invocation.get("evaluation_invocations") != 1
        or invocation.get("training_seal") != "../training_seal.json"
        or invocation.get("status") != "started"
    ):
        raise ValueError("corrective evaluation invocation identity is incompatible")
    for payload in (manifest, complete):
        if (
            payload.get("protocol") != CORRECTIVE_PROTOCOL
            or payload.get("evaluation_invocations") != 1
            or payload.get("training_tree_sha256") != expected_training_tree_sha256
        ):
            raise ValueError("corrective evaluation identity is incompatible")
    if (
        manifest.get("status") != "complete"
        or complete.get("status") != "complete"
        or manifest.get("training_seal_sha256")
        != _sha256_file(root / "training_seal.json")
        or complete.get("evaluation_manifest_sha256") != _sha256_file(manifest_path)
    ):
        raise ValueError("corrective evaluation completion evidence is incompatible")
    artifact_hashes = manifest.get("artifacts")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("corrective evaluation artifact hash closure is missing")
    for relative, expected in artifact_hashes.items():
        path = root / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("corrective evaluation artifact hash closure changed")

    probability, mean = _load_training_predictions(
        corrective_config, phase_b_config, bundle, cache
    )
    recomputed = _build_non_claim_artifacts(probability, mean)
    expected_relative = {
        "evaluation/evaluation_invocation.json",
        "evaluation/claim_decision.json",
        *{
            f"evaluation/{name}.csv" if isinstance(artifact, pd.DataFrame) else f"evaluation/{name}.json"
            for name, artifact in recomputed.items()
        },
    }
    if set(map(str, artifact_hashes)) != expected_relative:
        raise ValueError("corrective evaluation artifact set is incompatible")
    for name, expected_artifact in recomputed.items():
        if isinstance(expected_artifact, pd.DataFrame):
            actual_artifact = pd.read_csv(
                evaluation / f"{name}.csv", float_precision="round_trip"
            )
            try:
                assert_frame_equal(
                    actual_artifact, expected_artifact, check_dtype=False
                )
            except AssertionError as error:
                raise ValueError(
                    f"corrective evaluation artifact does not recompute: {name}"
                ) from error
        else:
            actual_artifact = json.loads(
                (evaluation / f"{name}.json").read_text(encoding="utf-8")
            )
            if actual_artifact != expected_artifact:
                raise ValueError(
                    f"corrective evaluation artifact does not recompute: {name}"
                )


__all__ = [
    "corrective_training_tree_digest",
    "create_corrective_training_seal",
    "evaluate_corrective_once",
    "validate_corrective_evaluation_outputs",
]
