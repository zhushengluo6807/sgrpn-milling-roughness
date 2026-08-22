"""Checkpoint-independent, atomic Phase A report generation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any
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
)


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
    duration_audit: pd.DataFrame | None = None,
    quality_features: pd.DataFrame | None = None,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Create every Phase A evaluation artifact from persisted tabular inputs."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = _validate_manifest_binding(predictions, manifest)
    if int(bootstrap_seed) != BOOTSTRAP_SEED:
        raise ValueError("Phase A bootstrap seed must be exactly 20260723")
    if isinstance(bootstrap_repetitions, bool) or int(bootstrap_repetitions) < 1:
        raise ValueError("bootstrap repetitions must be a positive integer")
    written: dict[str, Path] = {}
    written["oof_predictions"] = _atomic_csv(
        output / "predictions" / "oof_predictions.csv", values
    )
    if duration_audit is not None:
        written["duration_audit"] = _atomic_csv(
            output / "audit" / "duration_audit.csv", duration_audit
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
    manifest_payload = dict(run_manifest or {})
    manifest_payload.pop("proceed_to_phase_b", None)
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
            "artifacts": {
                str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
                for path in written.values()
            },
        }
    )
    written["run_manifest"] = _atomic_json(output / "run_manifest.json", manifest_payload)
    return written


__all__ = ["write_phase_a_report"]
