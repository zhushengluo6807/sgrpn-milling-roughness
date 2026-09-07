"""Checkpoint-independent, atomic Phase A report generation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import io
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping
import uuid

_matplotlib_cache = Path.cwd() / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    MATERIAL_MARGIN_UM,
    PHASE_A_MODELS,
    assess_phase_a,
    build_acceptance_inputs,
    negative_transfer,
    paired_group_bootstrap,
    validate_prediction_cartesian,
    weighted_metrics,
    assess_phase_b_claims,
    paired_probability_bootstrap,
    phase_b_gate_statistics,
    phase_b_mean_bootstrap_table,
    phase_b_mean_metric_table,
    phase_b_negative_transfer_table,
    probability_metric_table,
    seed_uncertainty_summary,
    validate_phase_b_prediction_cartesian,
)
from .config import (
    PhaseAHandoff,
    PhaseBConfig,
    load_sgrpn_config,
    validate_phase_b_handoff,
    validate_phase_b_output_root,
)
from .data import DataBundle, load_data_bundle
from .phase_b_training import (
    CALIBRATION_SCORE_COLUMNS,
    MEAN_MODELS,
    PHASE_B_MEAN_COLUMNS,
    PHASE_B_PROTOCOL,
    PHASE_B_PREDICTION_COLUMNS,
    SCALE_MODELS,
    _load_completed_phase_b_fold,
    _phase_b_fingerprint,
)
from .training import validate_phase_a_output_root


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    return _atomic_bytes(
        path,
        json.dumps(
            _jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8"),
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    return _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_figure(path: Path, figure: plt.Figure) -> Path:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return _atomic_bytes(path, buffer.getvalue())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration_frames_equal(path: Path, canonical: pd.DataFrame) -> bool:
    try:
        persisted = pd.read_csv(
            path,
            dtype={"sample_id": str},
            float_precision="round_trip",
        )
        pd.testing.assert_frame_equal(
            persisted,
            canonical.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except (OSError, ValueError, AssertionError):
        return False
    return True


def _validate_manifest_binding(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "group_id",
        "version",
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"report manifest missing columns: {missing}")
    values = manifest.copy()
    if values["sample_id"].isna().any() or values["sample_id"].astype(str).duplicated().any():
        raise ValueError("report manifest sample IDs must be non-missing and unique")
    for column in ("sample_id", "group_id", "version"):
        values[column] = values[column].astype(str)
    expected_ids = tuple(values["sample_id"])
    validated = validate_prediction_cartesian(
        predictions, expected_sample_ids=expected_ids, models=PHASE_A_MODELS
    )
    source = values.set_index("sample_id")
    first = validated.drop_duplicates("sample_id").set_index("sample_id")
    if (
        not first.index.equals(source.reindex(first.index).index)
        or not (first["group_id"] == source.reindex(first.index)["group_id"]).all()
        or not (first["version"] == source.reindex(first.index)["version"]).all()
    ):
        raise ValueError("OOF IDs/group/version do not match the report manifest")
    process = values.loc[:, ["n_rpm", "fz_mm_per_tooth", "ap_mm"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(process).all():
        raise ValueError("report manifest process values must be finite")
    return validated


def _group_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    values = predictions.copy()
    values["weighted_target"] = values["target"] * values["sample_weight"]
    values["weighted_prediction"] = values["prediction"] * values["sample_weight"]
    grouped = values.groupby(
        ["group_id", "version", "fold", "seed", "model"],
        sort=True,
        observed=True,
        as_index=False,
    ).agg(
        weighted_target_sum=("weighted_target", "sum"),
        weighted_prediction_sum=("weighted_prediction", "sum"),
        sample_weight=("sample_weight", "sum"),
        segment_count=("sample_id", "size"),
    )
    grouped["target"] = grouped["weighted_target_sum"] / grouped["sample_weight"]
    grouped["prediction"] = (
        grouped["weighted_prediction_sum"] / grouped["sample_weight"]
    )
    return grouped.loc[
        :,
        [
            "group_id",
            "version",
            "fold",
            "seed",
            "model",
            "target",
            "prediction",
            "sample_weight",
            "segment_count",
        ],
    ]


def _metric_table(frame: pd.DataFrame, aggregation_unit: str) -> pd.DataFrame:
    rows = []
    for model in PHASE_A_MODELS:
        values = frame.loc[frame["model"] == model]
        rows.append(
            {
                "aggregation_unit": aggregation_unit,
                "model": model,
                "seed": BOOTSTRAP_SEED,
                "sample_count": int(len(values)),
                "weight_sum": float(values["sample_weight"].sum()),
                **weighted_metrics(
                    values["target"], values["prediction"], values["sample_weight"]
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, fold), values in predictions.groupby(
        ["model", "seed", "fold"], sort=True, observed=True
    ):
        rows.append(
            {
                "aggregation_unit": "segment",
                "model": str(model),
                "seed": int(seed),
                "fold": int(fold),
                "sample_count": int(len(values)),
                "weight_sum": float(values["sample_weight"].sum()),
                **weighted_metrics(
                    values["target"], values["prediction"], values["sample_weight"]
                ),
            }
        )
    result = pd.DataFrame.from_records(rows)
    expected = {
        (model, BOOTSTRAP_SEED, fold)
        for model in PHASE_A_MODELS
        for fold in range(5)
    }
    actual = set(zip(result["model"], result["seed"], result["fold"], strict=True))
    if actual != expected or len(result) != len(expected):
        raise ValueError("formal fold metrics require every model on folds 0..4")
    return result


def _paired_errors(predictions: pd.DataFrame, candidate: str) -> pd.DataFrame:
    columns = ["sample_id", "group_id", "target", "prediction", "sample_weight"]
    p1 = predictions.loc[predictions["model"] == "P1", columns].rename(
        columns={"prediction": "p1_prediction"}
    )
    other = predictions.loc[predictions["model"] == candidate, columns].rename(
        columns={
            "group_id": "candidate_group_id",
            "target": "candidate_target",
            "prediction": "candidate_prediction",
            "sample_weight": "candidate_weight",
        }
    )
    paired = p1.merge(other, on="sample_id", validate="one_to_one")
    if len(paired) != len(p1) or (
        not (paired["group_id"] == paired["candidate_group_id"]).all()
        or not np.allclose(paired["target"], paired["candidate_target"], rtol=0, atol=1e-12)
        or not np.allclose(paired["sample_weight"], paired["candidate_weight"], rtol=0, atol=1e-12)
    ):
        raise ValueError(f"{candidate}/P1 OOF pairing is incompatible")
    return pd.DataFrame(
        {
            "sample_id": paired["sample_id"].astype(str),
            "group_id": paired["group_id"].astype(str),
            "p1_abs_error": np.abs(paired["target"] - paired["p1_prediction"]),
            "candidate_abs_error": np.abs(
                paired["target"] - paired["candidate_prediction"]
            ),
            "sample_weight": paired["sample_weight"],
        }
    )


def _negative_transfer_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("V1", "F1", "R1", "G1"):
        result = negative_transfer(
            _paired_errors(predictions, model), material_margin_um=MATERIAL_MARGIN_UM
        )
        rows.append(
            {
                "baseline": "P1",
                "model": model,
                "raw_rate": result.raw_rate,
                "material_rate": result.material_rate,
                "group_count": result.group_count,
                "material_margin_um": MATERIAL_MARGIN_UM,
                "aggregation": "sample_weighted_group_mean_absolute_error",
            }
        )
    return pd.DataFrame.from_records(rows)


def _bootstrap_table(
    predictions: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    comparisons = (("M0", "P1"), ("P1", "V1"), ("P1", "F1"), ("P1", "R1"), ("P1", "G1"))
    rows = []
    for baseline, candidate in comparisons:
        result = paired_group_bootstrap(
            predictions,
            baseline=baseline,
            candidate=candidate,
            repetitions=repetitions,
            seed=seed,
        )
        rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "metric": "weighted_mae_improvement_um",
                **result.to_dict(),
                "confidence_level": 0.95,
            }
        )
    return pd.DataFrame.from_records(rows)


def _gate_statistics(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    quality_features: pd.DataFrame | None,
) -> pd.DataFrame:
    gate = predictions.loc[
        predictions["model"] == "G1", ["sample_id", "gate", "sample_weight"]
    ].merge(
        manifest.loc[
            :, ["sample_id", "version", "n_rpm", "fz_mm_per_tooth", "ap_mm"]
        ].assign(sample_id=lambda frame: frame["sample_id"].astype(str)),
        on="sample_id",
        validate="one_to_one",
    )

    def record(dimension: str, level: str, values: pd.DataFrame) -> dict[str, Any]:
        gates = values["gate"].to_numpy(dtype=np.float64)
        weights = values["sample_weight"].to_numpy(dtype=np.float64)
        return {
            "dimension": dimension,
            "level": level,
            "segment_count": len(values),
            "weight_sum": float(weights.sum()),
            "gate_mean": float(gates.mean()),
            "gate_weighted_mean": float(np.dot(gates, weights) / weights.sum()),
            "gate_p05": float(np.quantile(gates, 0.05)),
            "gate_median": float(np.median(gates)),
            "gate_p95": float(np.quantile(gates, 0.95)),
        }

    rows = [record("overall", "all", gate)]
    for dimension in ("n_rpm", "fz_mm_per_tooth", "ap_mm", "version"):
        for level, values in gate.groupby(dimension, sort=True, dropna=False, observed=True):
            rows.append(record(dimension, str(level), values))
    if quality_features is not None:
        required = {"sample_id", *(f"quality_{index}" for index in range(7))}
        missing = sorted(required - set(quality_features.columns))
        if missing:
            raise ValueError(f"gate quality diagnostics missing columns: {missing}")
        quality = quality_features.loc[:, sorted(required)].copy()
        if quality["sample_id"].isna().any() or quality["sample_id"].astype(str).duplicated().any():
            raise ValueError("gate quality diagnostic sample IDs must be unique")
        quality["sample_id"] = quality["sample_id"].astype(str)
        names = [f"quality_{index}" for index in range(7)]
        numeric = quality.loc[:, names].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or set(quality["sample_id"]) != set(gate["sample_id"]):
            raise ValueError("gate quality diagnostics must be finite with exact OOF coverage")
        gate_quality = gate.merge(quality, on="sample_id", validate="one_to_one")
        for dimension in names:
            values = gate_quality[dimension]
            if values.nunique() > 8:
                levels = pd.qcut(values, q=4, duplicates="drop").astype(str)
            else:
                levels = values.astype(str)
            for level, subset in gate_quality.assign(_level=levels).groupby(
                "_level", sort=True, observed=True
            ):
                rows.append(record(dimension, str(level), subset))
    return pd.DataFrame.from_records(rows)


def _error_breakdowns(predictions: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    diagnostic = predictions.merge(
        manifest.loc[
            :,
            [
                "sample_id",
                "n_rpm",
                "fz_mm_per_tooth",
                "ap_mm",
            ],
        ].assign(sample_id=lambda frame: frame["sample_id"].astype(str)),
        on="sample_id",
        validate="many_to_one",
    )
    diagnostic["weighted_absolute_error"] = (
        np.abs(diagnostic["target"] - diagnostic["prediction"])
        * diagnostic["sample_weight"]
    )
    tables: dict[str, pd.DataFrame] = {}
    for dimension in ("n_rpm", "fz_mm_per_tooth", "ap_mm", "version"):
        grouped = diagnostic.groupby(
            ["model", dimension], sort=True, dropna=False, observed=True, as_index=False
        ).agg(
            weighted_error_sum=("weighted_absolute_error", "sum"),
            weight_sum=("sample_weight", "sum"),
            segment_count=("sample_id", "size"),
        )
        grouped["weighted_mae"] = grouped["weighted_error_sum"] / grouped["weight_sum"]
        tables[f"error_by_{dimension}"] = grouped
    return tables


def _write_figures(
    predictions: pd.DataFrame, manifest: pd.DataFrame, output: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    figure, axis = plt.subplots(figsize=(7, 6))
    for model, values in predictions.groupby("model", sort=True, observed=True):
        axis.scatter(values["target"], values["prediction"], s=13, alpha=0.5, label=model)
    low = float(min(predictions["target"].min(), predictions["prediction"].min()))
    high = float(max(predictions["target"].max(), predictions["prediction"].max()))
    axis.plot([low, high], [low, high], "k--", linewidth=1)
    axis.set_xlabel("Measured Ra (μm)")
    axis.set_ylabel("Predicted Ra (μm)")
    axis.legend(ncol=2)
    paths["prediction_scatter"] = _atomic_figure(output / "prediction_scatter.png", figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for model, values in predictions.groupby("model", sort=True, observed=True):
        axis.scatter(
            values["prediction"], values["target"] - values["prediction"],
            s=13, alpha=0.5, label=model,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Predicted Ra (μm)")
    axis.set_ylabel("Residual: measured - predicted (μm)")
    axis.legend(ncol=2)
    paths["residual_plot"] = _atomic_figure(output / "residual_plot.png", figure)

    gate = predictions.loc[predictions["model"] == "G1", ["sample_id", "gate"]].merge(
        manifest.loc[:, ["sample_id", "n_rpm", "fz_mm_per_tooth"]].assign(
            sample_id=lambda frame: frame["sample_id"].astype(str)
        ),
        on="sample_id",
        validate="one_to_one",
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(gate["gate"], bins=np.linspace(0.0, 1.0, 21), color="#4472C4", edgecolor="white")
    axis.set_xlabel("G1 trust gate")
    axis.set_ylabel("Segment count")
    paths["gate_distribution"] = _atomic_figure(output / "gate_distribution.png", figure)

    heat = gate.pivot_table(
        index="n_rpm", columns="fz_mm_per_tooth", values="gate", aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(heat.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(heat.columns)), [f"{value:g}" for value in heat.columns])
    axis.set_yticks(range(len(heat.index)), [f"{value:g}" for value in heat.index])
    axis.set_xlabel("fz (mm/tooth)")
    axis.set_ylabel("n (rpm)")
    figure.colorbar(image, ax=axis, label="Mean G1 trust gate")
    paths["gate_condition_heatmap"] = _atomic_figure(
        output / "gate_condition_heatmap.png", figure
    )

    fold = (
        predictions.assign(
            weighted_error=lambda frame: np.abs(frame["target"] - frame["prediction"])
            * frame["sample_weight"]
        )
        .groupby(["model", "fold"], sort=True, observed=True)
        .agg(error=("weighted_error", "sum"), weight=("sample_weight", "sum"))
        .assign(weighted_mae=lambda frame: frame["error"] / frame["weight"])
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for model, values in fold.groupby("model", sort=True, observed=True):
        axis.plot(values["fold"], values["weighted_mae"], marker="o", label=model)
    axis.set_xlabel("Outer fold")
    axis.set_ylabel("Weighted MAE (μm)")
    axis.legend(ncol=2)
    paths["fold_stability"] = _atomic_figure(output / "fold_stability.png", figure)
    return paths


def write_phase_a_report(
    output_dir: str | Path,
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    output_root: str | Path | None = None,
    duration_audit: pd.DataFrame | None = None,
    quality_features: pd.DataFrame | None = None,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Create every Phase A evaluation artifact from persisted tabular inputs."""
    output = validate_phase_a_output_root(output_dir, output_root=output_root)
    output.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output / "run_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.is_file():
        try:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing Phase A run manifest is invalid") from error
        if not isinstance(existing_manifest, dict):
            raise ValueError("existing Phase A run manifest must be an object")
    manifest_payload = dict(existing_manifest)
    manifest_payload.update(run_manifest or {})
    values = _validate_manifest_binding(predictions, manifest)
    if int(bootstrap_seed) != BOOTSTRAP_SEED:
        raise ValueError("Phase A bootstrap seed must be exactly 20260723")
    if isinstance(bootstrap_repetitions, bool) or int(bootstrap_repetitions) < 1:
        raise ValueError("bootstrap repetitions must be a positive integer")
    written: dict[str, Path] = {}
    written["oof_predictions"] = _atomic_csv(
        output / "predictions" / "oof_predictions.csv", values
    )
    duration_path = output / "audit" / "duration_audit.csv"
    if duration_audit is None:
        raise ValueError("current canonical duration audit is required")
    duration_audit = duration_audit.reset_index(drop=True).copy()
    if "sample_id" not in duration_audit or duration_audit["sample_id"].isna().any():
        raise ValueError("duration audit requires non-missing sample_id values")
    duration_ids = tuple(duration_audit["sample_id"].astype(str))
    expected_ids = tuple(manifest["sample_id"].astype(str))
    if len(set(duration_ids)) != len(duration_ids) or set(duration_ids) != set(expected_ids):
        raise ValueError("duration audit must exactly cover current manifest sample IDs")
    prior_duration = existing_manifest.get("duration_audit")
    registered_existing = bool(
        isinstance(prior_duration, dict)
        and type(prior_duration.get("row_count")) is int
        and prior_duration["row_count"] == len(duration_audit)
        and duration_path.is_file()
        and prior_duration.get("sha256") == _sha256(duration_path)
        and _duration_frames_equal(duration_path, duration_audit)
    )
    if registered_existing:
        duration_provenance = "existing_registered"
        written["duration_audit"] = duration_path
    else:
        duration_provenance = "canonical_current_data"
        written["duration_audit"] = _atomic_csv(
            duration_path, duration_audit
        )
    groups = _group_predictions(values)
    summary = pd.concat(
        [_metric_table(values, "segment"), _metric_table(groups, "group")],
        ignore_index=True,
    )
    fold_metrics = _fold_metrics(values)
    transfers = _negative_transfer_table(values)
    bootstrap = _bootstrap_table(values, int(bootstrap_repetitions), int(bootstrap_seed))
    gate_statistics = _gate_statistics(values, manifest, quality_features)
    evaluation = output / "evaluation"
    for name, frame in (
        ("summary_metrics", summary),
        ("fold_metrics", fold_metrics),
        ("paired_bootstrap", bootstrap),
        ("negative_transfer", transfers),
        ("gate_statistics", gate_statistics),
    ):
        written[name] = _atomic_csv(evaluation / f"{name}.csv", frame)
    written["group_predictions"] = _atomic_csv(
        evaluation / "breakdowns" / "group_oof_predictions.csv", groups
    )
    for name, frame in _error_breakdowns(values, manifest).items():
        written[name] = _atomic_csv(evaluation / "breakdowns" / f"{name}.csv", frame)
    inputs = build_acceptance_inputs(
        summary,
        fold_metrics,
        transfers,
        values.loc[values["model"] == "G1", "gate"].to_numpy(dtype=np.float64),
    )
    decision = assess_phase_a(inputs)
    g1_bootstrap = bootstrap.loc[
        (bootstrap["baseline"] == "P1") & (bootstrap["candidate"] == "G1")
    ].iloc[0]
    acceptance = {
        "schema_version": "sgrpn-phase-a-acceptance-v1",
        "proceed_to_phase_b": decision.proceed_to_phase_b,
        "decision": decision.to_dict(),
        "inputs": asdict(inputs),
        "thresholds": {
            "p1_mae_ratio_max": 1.05,
            "p1_r2_drop_max": 0.02,
            "g1_mae_ratio_noninferiority_max": 1.01,
            "g1_rmse_ratio_noninferiority_max": 1.03,
            "g1_r2_drop_max": 0.0,
            "g1_mean_path_mae_ratio_max": 0.99,
            "g1_mean_path_fold_wins_min": 3,
            "transfer_reduction_min": 0.10,
            "material_negative_transfer_margin_um": MATERIAL_MARGIN_UM,
            "collapsed_gate_median_below": 0.05,
            "collapsed_gate_p95_below": 0.20,
        },
        "bootstrap": {
            "metric": "P1_minus_G1_weighted_mae_um",
            "point_estimate": float(g1_bootstrap["point_estimate"]),
            "lower": float(g1_bootstrap["lower"]),
            "upper": float(g1_bootstrap["upper"]),
            "confidence_level": 0.95,
            "repetitions": int(g1_bootstrap["repetitions"]),
            "seed": int(g1_bootstrap["seed"]),
            "resampling_unit": str(g1_bootstrap["resampling_unit"]),
        },
        "subjective_override_allowed": False,
        "phase_b_executed": False,
    }
    written["acceptance"] = _atomic_json(evaluation / "acceptance.json", acceptance)
    method_notes = {
        "version_input": "excluded",
        "v3_v4_stress_test": "v3/v4 is a composite-domain stress test; differences are confounded by the speed grid and other domain changes.",
        "ch9_ch10_orientation": "Ch9/Ch10 physical orientation is uncertain; inference averages original and swapped channels.",
        "phase_a_seed_count": 1,
        "phase_a_seed": BOOTSTRAP_SEED,
        "scientific_units": {
            "roughness": "μm",
            "speed": "rpm",
            "feed_per_tooth": "mm/tooth",
            "depth_of_cut": "mm",
        },
        "group_error_aggregation": "sum(sample_weight * absolute_error) / sum(sample_weight)",
        "gate_quality_diagnostics": [f"quality_{index}" for index in range(7)],
        "phase_b_status": "not_executed",
    }
    written["method_notes"] = _atomic_json(evaluation / "method_notes.json", method_notes)
    written.update(_write_figures(values, manifest, evaluation / "figures"))
    manifest_payload.pop("proceed_to_phase_b", None)
    try:
        installed_version = package_version("roughness-training")
    except PackageNotFoundError:
        installed_version = "not-installed"
    selected_device = manifest_payload.get(
        "selected_device",
        manifest_payload.get(
            "device", manifest_payload.get("selected_devices", "not-recorded")
        ),
    )
    manifest_payload.update(
        {
            "schema_version": "sgrpn-phase-a-run-v1",
            "seed": BOOTSTRAP_SEED,
            "seed_count": 1,
            "bootstrap": {
                "seed": int(bootstrap_seed),
                "repetitions": int(bootstrap_repetitions),
                "resampling_unit": "group_id",
            },
            "model_matrix": list(PHASE_A_MODELS),
            "phase_b_executed": False,
            "duration_audit": {
                "provenance": duration_provenance,
                "row_count": int(len(duration_audit)),
                "sha256": _sha256(duration_path),
            },
            "environment": {
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "package_version": installed_version,
                "device": selected_device,
            },
            "artifacts": {
                str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
                for path in written.values()
            },
        }
    )
    written["run_manifest"] = _atomic_json(output / "run_manifest.json", manifest_payload)
    return written


_PHASE_B_QUANTILE_COLUMNS = (
    "fold",
    "seed",
    "scale_model",
    "alpha",
    "group_count",
    "order_index",
    "quantile",
)

_PHASE_B_METHOD_NOTES = {
    "three_readings_one_region": True,
    "probability_loss_preserves_region_weight": True,
    "sigma_interpretation": "sigma combines repeat dispersion and unmodeled error",
    "variance_limitations": "no Gauge R&R or variance-source decomposition",
    "seed_spread": "seed spread is descriptive instability only",
    "version_input": "version is excluded from inputs",
    "composite_domain_stress_test": "v3/v4 is a composite-domain stress test confounded with speed",
    "orientation": "Ch9/Ch10 orientation is unresolved",
    "coverage_scope": "coverage applies only to exchangeable new groups of existing type",
    "practical_width_threshold_registered": False,
    "outer_result_design_change": "no outer result changed the design",
}


def _phase_b_frame(path: Path, *, string_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Phase B report artifact is missing: {path}")
    try:
        return pd.read_csv(path, dtype={column: str for column in string_columns}, float_precision="round_trip")
    except (OSError, ValueError) as error:
        raise ValueError(f"Phase B report artifact is unreadable: {path}") from error


def _phase_b_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Phase B {name} is missing or invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Phase B {name} must be a JSON object")
    return payload


def _phase_b_config_hash(config: PhaseBConfig) -> str:
    payload = json.dumps(_jsonable(asdict(config)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _phase_b_fold_key(fold: int, seed: int) -> str:
    return f"fold_{fold}/seed_{seed}"


def _phase_b_expected_fold_keys() -> tuple[str, ...]:
    return tuple(
        _phase_b_fold_key(fold, seed)
        for fold in range(5)
        for seed in (20260723, 20260724, 20260725)
    )


def _phase_b_fingerprint_map(
    config: PhaseBConfig, handoff: PhaseAHandoff, bundle: DataBundle
) -> dict[str, str]:
    """Recompute every registered fold/seed fingerprint from immutable inputs."""
    from .order_spectrum import load_order_cache

    phase_a = load_sgrpn_config(config.phase_a_config_path)
    cache = load_order_cache(bundle, phase_a)
    return {
        _phase_b_fold_key(fold, seed): _phase_b_fingerprint(
            config, handoff, bundle, cache, fold=fold, seed=seed
        ).value
        for fold in range(5)
        for seed in (20260723, 20260724, 20260725)
    }


def _phase_b_device_provenance_hash(
    devices: Mapping[str, str],
    fingerprints: Mapping[str, str],
    completion_hashes: Mapping[str, str],
) -> str:
    payload = json.dumps(
        {
            "selected_device_by_fold_seed": dict(devices),
            "fold_fingerprints": dict(fingerprints),
            "fold_completion_sha256": dict(completion_hashes),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _phase_b_environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


_PHASE_B_MANIFEST_KEYS = {
    "schema_version",
    "phase_b_config",
    "phase_b_config_sha256",
    "phase_a_handoff",
    "handoff_file_sha256",
    "protocol",
    "status",
    "seeds",
    "alphas",
    "input_fingerprints",
    "cache_sha256",
    "training_fingerprint",
    "fold_fingerprints",
    "fold_completion_sha256",
    "selected_device_by_fold_seed",
    "device_provenance_sha256",
    "bootstrap",
    "environment",
    "artifacts",
}


def _phase_b_validate_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    config: PhaseBConfig,
    handoff: PhaseAHandoff,
    handoff_file_sha256: str,
    fingerprints: Mapping[str, str],
    completion_hashes: Mapping[str, str],
) -> None:
    if set(manifest) != _PHASE_B_MANIFEST_KEYS:
        raise ValueError("Phase B run manifest schema is incompatible")
    expected_keys = set(_phase_b_expected_fold_keys())
    devices = manifest["selected_device_by_fold_seed"]
    environment = manifest["environment"]
    if (
        manifest["schema_version"] != "sgrpn-phase-b-run-v1"
        or manifest["phase_b_config"] != _jsonable(asdict(config))
        or manifest["phase_b_config_sha256"] != _phase_b_config_hash(config)
    ):
        raise ValueError("Phase B run manifest config is incompatible")
    if (
        manifest["phase_a_handoff"] != asdict(handoff)
        or manifest["handoff_file_sha256"] != handoff_file_sha256
        or manifest["protocol"] != PHASE_B_PROTOCOL
        or manifest["status"] != "complete"
        or manifest["seeds"] != list(config.seeds)
        or manifest["alphas"] != list(config.alphas)
        or manifest["input_fingerprints"] != dict(handoff.input_sha256)
        or manifest["cache_sha256"] != handoff.cache_sha256
        or manifest["training_fingerprint"] != handoff.training_fingerprint
    ):
        raise ValueError("Phase B run manifest provenance is incompatible")
    if manifest["fold_fingerprints"] != dict(fingerprints):
        raise ValueError("Phase B fold fingerprint provenance is incompatible")
    if manifest["fold_completion_sha256"] != dict(completion_hashes):
        raise ValueError("Phase B completion provenance is incompatible")
    if (
        manifest["bootstrap"]
        != {"repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED, "resampling_unit": "group_id"}
        or not isinstance(environment, dict)
        or set(environment) != {"python_version", "python_executable", "platform"}
        or any(not isinstance(value, str) or not value for value in environment.values())
        or not isinstance(manifest["artifacts"], dict)
    ):
        raise ValueError("Phase B run manifest metadata is incompatible")
    if (
        not isinstance(devices, dict)
        or set(devices) != expected_keys
        or any(value not in {"cpu", "cuda"} for value in devices.values())
    ):
        raise ValueError("Phase B selected device records are incomplete or incompatible")
    if manifest["device_provenance_sha256"] != _phase_b_device_provenance_hash(
        devices, fingerprints, completion_hashes
    ):
        raise ValueError("Phase B device provenance is incompatible")


def _phase_b_hash_tree(root: Path) -> dict[str, str]:
    """Return a deterministic byte-level inventory without following links."""
    resolved = root.resolve()
    if not resolved.exists():
        return {}
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"immutable input root is not a regular directory: {resolved}")
    hashes: dict[str, str] = {}
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"immutable input tree contains a link: {path}")
        if path.is_file():
            hashes[path.relative_to(resolved).as_posix()] = _sha256(path)
    return hashes


def _phase_b_immutable_roots(config: PhaseBConfig) -> dict[str, Path]:
    phase_a_root = Path(config.phase_a_run_manifest_path).resolve().parent
    project_root = Path(config.phase_a_config_path).resolve().parents[1]
    return {
        "phase_a": phase_a_root,
        "scheme1": project_root / "outputs" / "scheme1",
        "scheme1_physics": project_root / "outputs" / "scheme1_physics",
    }


def _phase_b_immutable_hashes(config: PhaseBConfig) -> dict[str, dict[str, str]]:
    return {
        name: _phase_b_hash_tree(path)
        for name, path in _phase_b_immutable_roots(config).items()
    }


def _phase_b_quality_features(config: PhaseBConfig, bundle: DataBundle) -> pd.DataFrame:
    provided = bundle.manifest.attrs.get("quality_features")
    if isinstance(provided, pd.DataFrame):
        return provided.copy()
    from .order_spectrum import load_order_cache

    phase_a = load_sgrpn_config(config.phase_a_config_path)
    cache = load_order_cache(bundle, phase_a)
    return pd.DataFrame(
        {
            "sample_id": tuple(cache.segment_ids),
            **{
                f"quality_{index}": cache.quality[:, index]
                for index in range(cache.quality.shape[1])
            },
        }
    )


def _phase_b_group_predictions(mean_predictions: pd.DataFrame, bundle: DataBundle) -> pd.DataFrame:
    manifest = bundle.manifest.loc[
        :, ["sample_id", "group_id", "version", "n_rpm", "fz_mm_per_tooth", "ap_mm"]
    ].copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    g1 = mean_predictions.loc[
        mean_predictions["model"] == "G1",
        ["sample_id", "group_id", "version", "fold", "seed", "target_mean", "prediction", "sample_weight", "gate"],
    ].copy()
    g1["weighted_target"] = g1["target_mean"] * g1["sample_weight"]
    g1["weighted_prediction"] = g1["prediction"] * g1["sample_weight"]
    g1["weighted_gate"] = g1["gate"] * g1["sample_weight"]
    grouped = g1.groupby(
        ["group_id", "version", "fold", "seed"], sort=True, observed=True, as_index=False
    ).agg(
        weighted_target=("weighted_target", "sum"),
        weighted_prediction=("weighted_prediction", "sum"),
        weighted_gate=("weighted_gate", "sum"),
        sample_weight=("sample_weight", "sum"),
        region_count=("sample_id", "size"),
    )
    grouped["target_mean"] = grouped["weighted_target"] / grouped["sample_weight"]
    grouped["prediction"] = grouped["weighted_prediction"] / grouped["sample_weight"]
    grouped["gate"] = grouped["weighted_gate"] / grouped["sample_weight"]
    conditions = manifest.groupby(
        ["group_id", "version"], sort=True, observed=True, as_index=False
    ).agg(
        n_rpm=("n_rpm", "first"),
        fz_mm_per_tooth=("fz_mm_per_tooth", "first"),
        ap_mm=("ap_mm", "first"),
    )
    return grouped.merge(conditions, on=["group_id", "version"], validate="many_to_one").loc[
        :,
        [
            "group_id", "version", "fold", "seed", "target_mean", "prediction", "gate",
            "sample_weight", "region_count", "n_rpm", "fz_mm_per_tooth", "ap_mm",
        ],
    ]


def _phase_b_breakdowns(group_predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for column in ("version", "n_rpm", "fz_mm_per_tooth", "ap_mm"):
        values = group_predictions.copy()
        values["weighted_absolute_error"] = (
            np.abs(values["target_mean"] - values["prediction"]) * values["sample_weight"]
        )
        grouped = values.groupby([column, "seed"], sort=True, observed=True, as_index=False).agg(
            weighted_error=("weighted_absolute_error", "sum"),
            weight_sum=("sample_weight", "sum"),
            group_count=("group_id", "nunique"),
        )
        grouped["weighted_mae"] = grouped["weighted_error"] / grouped["weight_sum"]
        if column == "version":
            grouped["interpretation"] = "v3/v4 composite-domain stress test is confounded with speed"
        tables[column] = grouped
    return tables


def _phase_b_ablation_differences(probability_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = probability_metrics.loc[
        probability_metrics["aggregation"].isin(("seed", "all_seed"))
    ].copy()
    identifiers = ["aggregation", "seed", "fold", "interval_type", "nominal_coverage", "metric"]
    hetero = selected.loc[selected["scale_model"] == "heteroscedastic", identifiers + ["value"]].rename(
        columns={"value": "heteroscedastic_value"}
    )
    homo = selected.loc[selected["scale_model"] == "homoscedastic", identifiers + ["value"]].rename(
        columns={"value": "homoscedastic_value"}
    )
    merged = hetero.merge(homo, on=identifiers, validate="one_to_one")
    merged["heteroscedastic_minus_homoscedastic"] = (
        merged["heteroscedastic_value"] - merged["homoscedastic_value"]
    )
    return merged


def _phase_b_paired_bootstrap(
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


def _phase_b_plot_bars(
    frame: pd.DataFrame, *, title: str, ylabel: str
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 5))
    if frame.empty:
        plt.close(figure)
        raise ValueError(f"Phase B figure input is empty: {title}")
    labels = [
        " / ".join(str(value) for value in row)
        for row in frame.iloc[:, :-1].itertuples(index=False, name=None)
    ]
    axis.bar(np.arange(len(frame)), frame.iloc[:, -1].to_numpy(dtype=np.float64), color="#4472C4")
    axis.set_xticks(np.arange(len(frame)), labels, rotation=35, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    return figure


def _phase_b_coverage_width_figure(
    coverage: pd.DataFrame, width: pd.DataFrame
) -> plt.Figure:
    if coverage.empty or width.empty:
        raise ValueError("Phase B coverage/width figure input is empty")
    identifiers = ["scale_model", "nominal_coverage"]
    merged = coverage.merge(
        width,
        on=identifiers,
        how="inner",
        validate="one_to_one",
        suffixes=("_coverage", "_width"),
    ).sort_values(identifiers, kind="stable")
    if len(merged) != len(coverage) or len(merged) != len(width):
        raise ValueError("Phase B coverage/width figure inputs are incompatible")
    figure, coverage_axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(merged), dtype=np.float64)
    labels = [
        f"{scale_model} / {float(nominal):.2f}"
        for scale_model, nominal in merged.loc[:, identifiers].itertuples(index=False, name=None)
    ]
    coverage_axis.bar(
        positions - 0.2,
        merged["value_coverage"].to_numpy(dtype=np.float64),
        width=0.4,
        color="#4472C4",
        label="Coverage",
    )
    coverage_axis.set_ylabel("Coverage")
    coverage_axis.set_ylim(0.0, 1.05)
    width_axis = coverage_axis.twinx()
    width_axis.bar(
        positions + 0.2,
        merged["value_width"].to_numpy(dtype=np.float64),
        width=0.4,
        color="#ED7D31",
        label="Mean interval width",
    )
    width_axis.set_ylabel("Mean interval width (μm)")
    coverage_axis.set_xticks(positions, labels, rotation=35, ha="right")
    coverage_axis.set_title("Conformal simultaneous coverage and interval width")
    handles, labels = coverage_axis.get_legend_handles_labels()
    width_handles, width_labels = width_axis.get_legend_handles_labels()
    coverage_axis.legend(handles + width_handles, labels + width_labels, loc="best")
    return figure


def _phase_b_figure_bytes(figure: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(figure)


def _render_phase_b_figures_from_tables(output: Path) -> dict[str, bytes]:
    """Render deterministic Phase B PNG bytes from persisted report tables only."""
    probability_metrics = _phase_b_frame(
        output / "evaluation" / "probability_metrics.csv",
        string_columns=("aggregation", "scale_model", "interval_type", "metric"),
    )
    mean_metrics = _phase_b_frame(
        output / "evaluation" / "mean_metrics.csv",
        string_columns=("aggregation", "model", "metric"),
    )
    mean = _phase_b_frame(
        output / "predictions" / "oof_mean_predictions.csv",
        string_columns=("sample_id", "group_id", "version", "model", "fold", "seed"),
    )
    groups = _phase_b_frame(
        output / "predictions" / "group_oof_predictions.csv",
        string_columns=("group_id", "version", "fold", "seed"),
    )
    rendered: dict[str, bytes] = {}

    def render(name: str, figure: plt.Figure) -> None:
        rendered[name] = _phase_b_figure_bytes(figure)

    conformal = probability_metrics.loc[
        (probability_metrics["aggregation"] == "all_seed")
        & (probability_metrics["interval_type"] == "conformal")
    ]
    coverage = conformal.loc[
        conformal["metric"].eq("simultaneous_group_coverage"),
        ["scale_model", "nominal_coverage", "value"],
    ]
    width = conformal.loc[
        conformal["metric"].eq("mean_interval_width"),
        ["scale_model", "nominal_coverage", "value"],
    ]
    render("coverage_width", _phase_b_coverage_width_figure(coverage, width))
    interval = conformal.loc[
        conformal["metric"].eq("winkler_score"),
        ["scale_model", "nominal_coverage", "value"],
    ]
    render(
        "interval_score",
        _phase_b_plot_bars(
            interval, title="Conformal interval score", ylabel="Winkler score"
        ),
    )
    nll_crps = probability_metrics.loc[
        (probability_metrics["aggregation"] == "seed")
        & probability_metrics["metric"].isin(("gaussian_nll", "gaussian_crps")),
        ["seed", "scale_model", "metric", "value"],
    ]
    render(
        "nll_crps_by_seed",
        _phase_b_plot_bars(
            nll_crps, title="NLL and CRPS by seed", ylabel="Metric value"
        ),
    )
    render(
        "group_simultaneous_coverage",
        _phase_b_plot_bars(
            coverage,
            title="Group simultaneous coverage",
            ylabel="Coverage",
        ),
    )

    g1 = mean.loc[mean["model"] == "G1"].copy()
    figure, axis = plt.subplots(figsize=(7, 6))
    for seed, values in g1.groupby("seed", sort=True, observed=True):
        axis.scatter(
            values["target_mean"], values["prediction"], s=8, alpha=0.45, label=str(seed)
        )
    low = float(min(g1["target_mean"].min(), g1["prediction"].min()))
    high = float(max(g1["target_mean"].max(), g1["prediction"].max()))
    axis.plot([low, high], [low, high], "k--", linewidth=1)
    axis.set_title("G1 OOF predictions; v3/v4 stress test is confounded with speed")
    axis.set_xlabel("Measured mean Ra (μm)")
    axis.set_ylabel("G1 prediction (μm)")
    axis.legend()
    render("prediction_scatter", figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for seed, values in g1.groupby("seed", sort=True, observed=True):
        axis.scatter(
            values["prediction"],
            values["target_mean"] - values["prediction"],
            s=8,
            alpha=0.45,
            label=str(seed),
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_title("G1 residuals; v3/v4 stress test is confounded with speed")
    axis.set_xlabel("G1 prediction (μm)")
    axis.set_ylabel("Mean residual (μm)")
    axis.legend()
    render("residual_plot", figure)

    fold = mean_metrics.loc[
        (mean_metrics["aggregation"] == "fold")
        & mean_metrics["metric"].eq("mean_mae")
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    for model, values in fold.groupby("model", sort=True, observed=True):
        aggregate = values.groupby("fold", sort=True)["value"].mean()
        axis.plot(aggregate.index, aggregate.values, marker="o", label=model)
    axis.set_xlabel("Outer fold")
    axis.set_ylabel("Weighted MAE (μm)")
    axis.legend()
    render("fold_stability", figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(
        groups["gate"], bins=np.linspace(0.0, 1.0, 21), color="#4472C4", edgecolor="white"
    )
    axis.set_xlabel("G1 trust gate")
    axis.set_ylabel("Group-seed count")
    render("gate_distribution", figure)

    heat = (
        groups.pivot_table(
            index="n_rpm", columns="fz_mm_per_tooth", values="gate", aggfunc="mean"
        )
        .sort_index()
        .sort_index(axis=1)
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(
        heat.to_numpy(dtype=np.float64),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axis.set_xticks(range(len(heat.columns)), [f"{value:g}" for value in heat.columns])
    axis.set_yticks(range(len(heat.index)), [f"{value:g}" for value in heat.index])
    axis.set_xlabel("fz (mm/tooth)")
    axis.set_ylabel("n (rpm)")
    figure.colorbar(image, ax=axis, label="Mean G1 trust gate")
    render("gate_condition_heatmap", figure)
    return rendered


def _write_phase_b_figures_from_tables(output: Path) -> dict[str, Path]:
    """Render figures solely from report CSVs already saved under ``output``."""
    figures = output / "evaluation" / "figures"
    return {
        f"figure_{name}": _atomic_bytes(figures / f"{name}.png", payload)
        for name, payload in _render_phase_b_figures_from_tables(output).items()
    }


def _phase_b_completion_metadata(output: Path) -> tuple[dict[str, str], dict[str, str]]:
    hashes: dict[str, str] = {}
    devices: dict[str, str] = {}
    for fold in range(5):
        for seed in (20260723, 20260724, 20260725):
            key = f"fold_{fold}/seed_{seed}"
            marker = output / "folds" / f"fold_{fold}" / f"seed_{seed}" / "complete.json"
            if marker.is_file():
                hashes[key] = _sha256(marker)
    return hashes, devices


def _phase_b_existing_inner_folds(output: Path) -> pd.DataFrame:
    paths = [
        output / "folds" / f"fold_{fold}" / f"seed_{seed}" / "calibration" / "inner_folds.csv"
        for fold in range(5)
        for seed in (20260723, 20260724, 20260725)
    ]
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return pd.DataFrame()
    if len(existing) != len(paths):
        raise ValueError("Phase B report requires every completed fold/seed inner-fold table")
    return pd.concat(
        [_phase_b_frame(path, string_columns=()) for path in existing], ignore_index=True
    )


def write_phase_b_report(
    config: PhaseBConfig,
    handoff: PhaseAHandoff,
    bundle: DataBundle,
    mean_predictions: pd.DataFrame,
    probability_predictions: pd.DataFrame,
    calibration_scores: pd.DataFrame,
    quantiles: pd.DataFrame,
) -> Mapping[str, Path]:
    """Publish every Phase B report artifact from OOF CSV/JSON-derived inputs."""
    if not isinstance(config, PhaseBConfig) or not isinstance(handoff, PhaseAHandoff):
        raise ValueError("Phase B report requires validated configuration and handoff")
    if not isinstance(bundle, DataBundle):
        raise ValueError("Phase B report requires the canonical data bundle")
    output = validate_phase_b_output_root(config.output_dir)
    immutable_before = _phase_b_immutable_hashes(config)
    output.mkdir(parents=True, exist_ok=True)
    existing_run_manifest = (
        _phase_b_json(output / "run_manifest.json", "run manifest")
        if (output / "run_manifest.json").is_file()
        else {}
    )
    probability = validate_phase_b_prediction_cartesian(
        probability_predictions,
        manifest=bundle.manifest,
        folds=bundle.folds,
        mean_predictions=mean_predictions,
        calibration_scores=calibration_scores,
        calibration_quantiles=quantiles,
    )
    mean = mean_predictions.loc[:, PHASE_B_MEAN_COLUMNS].copy()
    scores = calibration_scores.loc[:, CALIBRATION_SCORE_COLUMNS].copy()
    quantile_table = quantiles.loc[:, _PHASE_B_QUANTILE_COLUMNS].copy()
    probability_metrics = probability_metric_table(probability)
    mean_metrics = phase_b_mean_metric_table(mean)
    seed_summary = seed_uncertainty_summary(probability)
    paired_bootstrap = _phase_b_paired_bootstrap(probability, mean)
    negative_transfer = phase_b_negative_transfer_table(mean)
    quality = _phase_b_quality_features(config, bundle)
    gate_statistics = phase_b_gate_statistics(mean, bundle.manifest, quality)
    groups = _phase_b_group_predictions(mean, bundle)
    breakdowns = _phase_b_breakdowns(groups)
    decision = assess_phase_b_claims(probability_metrics)
    fold_metrics = pd.concat(
        [
            mean_metrics.loc[mean_metrics["aggregation"] == "fold"].assign(metric_family="mean"),
            probability_metrics.loc[probability_metrics["aggregation"] == "fold"].assign(metric_family="probability"),
        ],
        ignore_index=True,
        sort=False,
    )
    seed_metrics = pd.concat(
        [
            mean_metrics.loc[mean_metrics["aggregation"].isin(("seed", "all_seed"))].assign(metric_family="mean"),
            probability_metrics.loc[probability_metrics["aggregation"].isin(("seed", "all_seed"))].assign(metric_family="probability"),
        ],
        ignore_index=True,
        sort=False,
    )
    written: dict[str, Path] = {}
    written["handoff"] = _atomic_json(output / "handoff.json", asdict(handoff))
    written["immutable_hashes_before"] = _atomic_json(output / "immutable_hashes_before.json", immutable_before)
    written["outer_folds"] = _atomic_csv(output / "folds" / "outer_folds.csv", bundle.folds)
    written["inner_folds"] = _atomic_csv(
        output / "folds" / "inner_folds.csv", _phase_b_existing_inner_folds(output)
    )
    written["oof_probability_predictions"] = _atomic_csv(output / "predictions" / "oof_probability_predictions.csv", probability)
    written["oof_mean_predictions"] = _atomic_csv(output / "predictions" / "oof_mean_predictions.csv", mean)
    written["group_oof_predictions"] = _atomic_csv(output / "predictions" / "group_oof_predictions.csv", groups)
    written["seed_uncertainty_summary"] = _atomic_csv(output / "predictions" / "seed_uncertainty_summary.csv", seed_summary)
    written["group_scores"] = _atomic_csv(output / "calibration" / "group_scores.csv", scores)
    written["quantiles"] = _atomic_csv(output / "calibration" / "quantiles.csv", quantile_table)
    written["mean_metrics"] = _atomic_csv(output / "evaluation" / "mean_metrics.csv", mean_metrics)
    written["probability_metrics"] = _atomic_csv(output / "evaluation" / "probability_metrics.csv", probability_metrics)
    written["fold_metrics"] = _atomic_csv(output / "evaluation" / "fold_metrics.csv", fold_metrics)
    written["seed_metrics"] = _atomic_csv(output / "evaluation" / "seed_metrics.csv", seed_metrics)
    written["ablation_differences"] = _atomic_csv(output / "evaluation" / "ablation_differences.csv", _phase_b_ablation_differences(probability_metrics))
    written["paired_bootstrap"] = _atomic_csv(output / "evaluation" / "paired_bootstrap.csv", paired_bootstrap)
    written["negative_transfer"] = _atomic_csv(output / "evaluation" / "negative_transfer.csv", negative_transfer)
    written["gate_statistics"] = _atomic_csv(output / "evaluation" / "gate_statistics.csv", gate_statistics)
    written["claim_decision"] = _atomic_json(output / "evaluation" / "claim_decision.json", decision.to_dict())
    written["method_notes"] = _atomic_json(output / "evaluation" / "method_notes.json", _PHASE_B_METHOD_NOTES)
    for name, table in breakdowns.items():
        written[f"breakdown_{name}"] = _atomic_csv(output / "evaluation" / "breakdowns" / f"by_{name}.csv", table)
    written.update(_write_phase_b_figures_from_tables(output))
    immutable_after = _phase_b_immutable_hashes(config)
    if immutable_after != immutable_before:
        raise RuntimeError("immutable Phase A or legacy input changed during Phase B reporting")
    written["immutable_hashes_after"] = _atomic_json(output / "immutable_hashes_after.json", immutable_after)
    completion_hashes, _ = _phase_b_completion_metadata(output)
    fingerprints = _phase_b_fingerprint_map(config, handoff, bundle)
    recorded_devices = existing_run_manifest.get("selected_device_by_fold_seed", {})
    devices = (
        {str(key): str(value) for key, value in recorded_devices.items()}
        if isinstance(recorded_devices, dict)
        else {}
    )
    if any(device not in {"cpu", "cuda"} for device in devices.values()):
        raise ValueError("Phase B recorded device is incompatible")
    if completion_hashes and set(devices) != set(completion_hashes):
        raise ValueError("Phase B completion/device provenance is incomplete")
    artifact_hashes = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(written.values())
    }
    manifest_payload = {
        "schema_version": "sgrpn-phase-b-run-v1",
        "phase_b_config": _jsonable(asdict(config)),
        "phase_b_config_sha256": _phase_b_config_hash(config),
        "phase_a_handoff": asdict(handoff),
        "handoff_file_sha256": _sha256(written["handoff"]),
        "protocol": PHASE_B_PROTOCOL,
        "status": "complete",
        "seeds": list(config.seeds),
        "alphas": list(config.alphas),
        "input_fingerprints": dict(handoff.input_sha256),
        "cache_sha256": handoff.cache_sha256,
        "training_fingerprint": handoff.training_fingerprint,
        "fold_fingerprints": fingerprints,
        "fold_completion_sha256": completion_hashes,
        "selected_device_by_fold_seed": devices,
        "device_provenance_sha256": _phase_b_device_provenance_hash(
            devices, fingerprints, completion_hashes
        ),
        "bootstrap": {"repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED, "resampling_unit": "group_id"},
        "environment": _phase_b_environment(),
        "artifacts": artifact_hashes,
    }
    written["run_manifest"] = _atomic_json(output / "run_manifest.json", manifest_payload)
    return written


def _phase_b_flatten_quantiles(output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        for seed in (20260723, 20260724, 20260725):
            path = output / "folds" / f"fold_{fold}" / f"seed_{seed}" / "calibration" / "quantiles.json"
            payload = _phase_b_json(path, "calibration quantiles")
            if payload.get("fold") != fold or payload.get("seed") != seed or not isinstance(payload.get("quantiles"), dict):
                raise ValueError("Phase B calibration quantile identity is incompatible")
            for scale_model in SCALE_MODELS:
                model_quantiles = payload["quantiles"].get(scale_model)
                if not isinstance(model_quantiles, dict):
                    raise ValueError("Phase B calibration quantile model is incompatible")
                for alpha in (0.10, 0.05):
                    value = model_quantiles.get(f"{alpha:.2f}")
                    if not isinstance(value, dict):
                        raise ValueError("Phase B calibration quantile alpha is incompatible")
                    rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "scale_model": scale_model,
                            "alpha": alpha,
                            "group_count": value.get("group_count"),
                            "order_index": value.get("order_index"),
                            "quantile": value.get("quantile"),
                        }
                    )
    return pd.DataFrame(rows, columns=_PHASE_B_QUANTILE_COLUMNS)


def _phase_b_saved_fold_inputs(
    config: PhaseBConfig, handoff: PhaseAHandoff, bundle: DataBundle
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .order_spectrum import load_order_cache

    phase_a = load_sgrpn_config(config.phase_a_config_path)
    cache = load_order_cache(bundle, phase_a)
    output = validate_phase_b_output_root(config.output_dir)
    probability_rows: list[pd.DataFrame] = []
    mean_rows: list[pd.DataFrame] = []
    score_rows: list[pd.DataFrame] = []
    inner_rows: list[pd.DataFrame] = []
    for fold in range(5):
        for seed in (20260723, 20260724, 20260725):
            fingerprint = _phase_b_fingerprint(config, handoff, bundle, cache, fold=fold, seed=seed)
            fold_dir = output / "folds" / f"fold_{fold}" / f"seed_{seed}"
            artifact = _load_completed_phase_b_fold(fold_dir, fingerprint, fold, seed)
            block = artifact.predictions.copy()
            block.attrs = {}
            probability_rows.append(block)
            mean_rows.append(artifact.predictions.attrs["mean_predictions"].copy())
            score_rows.extend(item.group_scores.copy() for item in artifact.calibration.values())
            inner_rows.append(_phase_b_frame(fold_dir / "calibration" / "inner_folds.csv", string_columns=()))
    return (
        pd.concat(probability_rows, ignore_index=True),
        pd.concat(mean_rows, ignore_index=True),
        pd.concat(score_rows, ignore_index=True),
        pd.concat(inner_rows, ignore_index=True),
    )


def _phase_b_assert_same(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False, check_exact=True
        )
    except AssertionError as error:
        raise ValueError(f"Phase B persisted {label} does not match the reproducible source") from error


def validate_phase_b_outputs(config: PhaseBConfig) -> None:
    """Deeply validate report, folds, metrics, calibration, and immutable inputs without writing."""
    handoff = validate_phase_b_handoff(config)
    phase_a = load_sgrpn_config(config.phase_a_config_path)
    bundle = load_data_bundle(phase_a)
    output = validate_phase_b_output_root(config.output_dir)
    if not output.is_dir():
        raise ValueError("Phase B output root is missing")
    before = _phase_b_json(output / "immutable_hashes_before.json", "immutable hashes before")
    after = _phase_b_json(output / "immutable_hashes_after.json", "immutable hashes after")
    current = _phase_b_immutable_hashes(config)
    if before != current or after != current:
        raise ValueError("Phase B immutable input hashes changed or are incompatible")
    handoff_path = output / "handoff.json"
    if _phase_b_json(handoff_path, "handoff") != asdict(handoff):
        raise ValueError("Phase B serialized handoff is incompatible")
    completion_hashes, _ = _phase_b_completion_metadata(output)
    if len(completion_hashes) != 15:
        raise ValueError("Phase B requires all fifteen completed fold/seed artifacts")
    run_manifest = _phase_b_json(output / "run_manifest.json", "run manifest")
    _phase_b_validate_manifest_contract(
        run_manifest,
        config=config,
        handoff=handoff,
        handoff_file_sha256=_sha256(handoff_path),
        fingerprints=_phase_b_fingerprint_map(config, handoff, bundle),
        completion_hashes=completion_hashes,
    )
    probability, mean, scores, inner = _phase_b_saved_fold_inputs(config, handoff, bundle)
    quantiles = _phase_b_flatten_quantiles(output)
    persisted_probability = _phase_b_frame(output / "predictions" / "oof_probability_predictions.csv", string_columns=("sample_id", "group_id", "version", "scale_model"))
    persisted_mean = _phase_b_frame(output / "predictions" / "oof_mean_predictions.csv", string_columns=("sample_id", "group_id", "version", "model"))
    persisted_scores = _phase_b_frame(output / "calibration" / "group_scores.csv", string_columns=("group_id", "scale_model"))
    persisted_quantiles = _phase_b_frame(output / "calibration" / "quantiles.csv", string_columns=("scale_model",))
    _phase_b_assert_same(persisted_probability, probability, "probability OOF rows")
    _phase_b_assert_same(persisted_mean, mean, "mean OOF rows")
    _phase_b_assert_same(persisted_scores, scores, "calibration scores")
    _phase_b_assert_same(persisted_quantiles, quantiles, "calibration quantiles")
    _phase_b_assert_same(
        _phase_b_frame(output / "folds" / "outer_folds.csv", string_columns=("sample_id", "group_id")),
        bundle.folds,
        "outer folds",
    )
    _phase_b_assert_same(
        _phase_b_frame(output / "folds" / "inner_folds.csv", string_columns=()), inner,
        "inner folds",
    )
    validated = validate_phase_b_prediction_cartesian(
        persisted_probability,
        manifest=bundle.manifest,
        folds=bundle.folds,
        mean_predictions=persisted_mean,
        calibration_scores=persisted_scores,
        calibration_quantiles=persisted_quantiles,
    )
    expected_probability_metrics = probability_metric_table(validated)
    expected_mean_metrics = phase_b_mean_metric_table(persisted_mean)
    _phase_b_assert_same(
        _phase_b_frame(output / "evaluation" / "probability_metrics.csv", string_columns=("aggregation", "scale_model", "interval_type", "metric")),
        expected_probability_metrics,
        "probability metrics",
    )
    _phase_b_assert_same(
        _phase_b_frame(output / "evaluation" / "mean_metrics.csv", string_columns=("aggregation", "model", "metric")),
        expected_mean_metrics,
        "mean metrics",
    )
    expected_groups = _phase_b_group_predictions(persisted_mean, bundle)
    _phase_b_assert_same(
        _phase_b_frame(
            output / "predictions" / "group_oof_predictions.csv",
            string_columns=("group_id", "version"),
        ),
        expected_groups,
        "group OOF rows",
    )
    expected_breakdowns = _phase_b_breakdowns(expected_groups)
    for name, expected_breakdown in expected_breakdowns.items():
        _phase_b_assert_same(
            _phase_b_frame(
                output / "evaluation" / "breakdowns" / f"by_{name}.csv",
                string_columns=("interpretation",) if name == "version" else (),
            ),
            expected_breakdown,
            f"{name} breakdown",
        )
    quality = _phase_b_quality_features(config, bundle)
    _phase_b_assert_same(
        _phase_b_frame(output / "evaluation" / "gate_statistics.csv", string_columns=("dimension", "level")),
        phase_b_gate_statistics(persisted_mean, bundle.manifest, quality),
        "gate statistics",
    )
    _phase_b_assert_same(
        _phase_b_frame(output / "evaluation" / "negative_transfer.csv", string_columns=("aggregation", "baseline", "candidate")),
        phase_b_negative_transfer_table(persisted_mean),
        "negative-transfer table",
    )
    _phase_b_assert_same(
        _phase_b_frame(
            output / "evaluation" / "ablation_differences.csv",
            string_columns=("aggregation", "interval_type", "metric"),
        ),
        _phase_b_ablation_differences(expected_probability_metrics),
        "ablation differences",
    )
    expected_fold_metrics = pd.concat(
        [
            expected_mean_metrics.loc[expected_mean_metrics["aggregation"] == "fold"].assign(metric_family="mean"),
            expected_probability_metrics.loc[expected_probability_metrics["aggregation"] == "fold"].assign(metric_family="probability"),
        ],
        ignore_index=True,
        sort=False,
    )
    expected_seed_metrics = pd.concat(
        [
            expected_mean_metrics.loc[expected_mean_metrics["aggregation"].isin(("seed", "all_seed"))].assign(metric_family="mean"),
            expected_probability_metrics.loc[expected_probability_metrics["aggregation"].isin(("seed", "all_seed"))].assign(metric_family="probability"),
        ],
        ignore_index=True,
        sort=False,
    )
    _phase_b_assert_same(
        _phase_b_frame(output / "evaluation" / "fold_metrics.csv", string_columns=("aggregation", "model", "scale_model", "interval_type", "metric", "metric_family")),
        expected_fold_metrics,
        "fold metrics",
    )
    _phase_b_assert_same(
        _phase_b_frame(output / "evaluation" / "seed_metrics.csv", string_columns=("aggregation", "model", "scale_model", "interval_type", "metric", "metric_family")),
        expected_seed_metrics,
        "seed metrics",
    )
    _phase_b_assert_same(
        _phase_b_frame(output / "predictions" / "seed_uncertainty_summary.csv", string_columns=("scale_model", "sample_id", "group_id", "version")),
        seed_uncertainty_summary(validated),
        "seed uncertainty summary",
    )
    _phase_b_assert_same(
        _phase_b_frame(output / "evaluation" / "paired_bootstrap.csv", string_columns=("baseline", "candidate", "metric", "interval_type", "resampling_unit")),
        _phase_b_paired_bootstrap(validated, persisted_mean),
        "paired bootstrap",
    )
    if _phase_b_json(output / "evaluation" / "claim_decision.json", "claim decision") != _jsonable(
        assess_phase_b_claims(expected_probability_metrics).to_dict()
    ):
        raise ValueError("Phase B claim decision is incompatible")
    if _phase_b_json(output / "evaluation" / "method_notes.json", "method notes") != _PHASE_B_METHOD_NOTES:
        raise ValueError("Phase B method notes are incompatible")
    artifacts = run_manifest.get("artifacts")
    required_artifacts = {
        "handoff.json",
        "immutable_hashes_before.json",
        "immutable_hashes_after.json",
        "folds/outer_folds.csv",
        "folds/inner_folds.csv",
        "predictions/oof_probability_predictions.csv",
        "predictions/oof_mean_predictions.csv",
        "predictions/group_oof_predictions.csv",
        "predictions/seed_uncertainty_summary.csv",
        "calibration/group_scores.csv",
        "calibration/quantiles.csv",
        "evaluation/mean_metrics.csv",
        "evaluation/probability_metrics.csv",
        "evaluation/fold_metrics.csv",
        "evaluation/seed_metrics.csv",
        "evaluation/ablation_differences.csv",
        "evaluation/paired_bootstrap.csv",
        "evaluation/negative_transfer.csv",
        "evaluation/gate_statistics.csv",
        "evaluation/claim_decision.json",
        "evaluation/method_notes.json",
        "evaluation/breakdowns/by_version.csv",
        "evaluation/breakdowns/by_n_rpm.csv",
        "evaluation/breakdowns/by_fz_mm_per_tooth.csv",
        "evaluation/breakdowns/by_ap_mm.csv",
        *(f"evaluation/figures/{name}.png" for name in (
            "coverage_width", "interval_score", "nll_crps_by_seed", "group_simultaneous_coverage",
            "prediction_scatter", "residual_plot", "fold_stability", "gate_distribution", "gate_condition_heatmap",
        )),
    }
    expected_figures = _render_phase_b_figures_from_tables(output)
    if any(
        not (output / "evaluation" / "figures" / f"{name}.png").is_file()
        or (output / "evaluation" / "figures" / f"{name}.png").read_bytes() != payload
        for name, payload in expected_figures.items()
    ):
        raise ValueError("Phase B persisted figure does not match deterministic reconstruction")
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts or any(
        not (output / relative).is_file() or _sha256(output / relative) != digest
        for relative, digest in artifacts.items()
    ):
        raise ValueError("Phase B report artifact hashes are incompatible")


__all__ = ["validate_phase_b_outputs", "write_phase_a_report", "write_phase_b_report"]
