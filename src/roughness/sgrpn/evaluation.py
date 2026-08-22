"""Registered Phase A metrics, group comparisons, and acceptance gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Sequence

import numpy as np
import pandas as pd


PHASE_A_MODELS = ("M0", "P1", "V1", "F1", "R1", "G1")
PREDICTION_COLUMNS = (
    "sample_id",
    "group_id",
    "version",
    "fold",
    "seed",
    "model",
    "target",
    "prediction",
    "sample_weight",
    "process_mean",
    "residual",
    "gate",
)
BOOTSTRAP_SEED = 20260723
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_UNIT = "group_id"
MATERIAL_MARGIN_UM = 0.01


@dataclass(frozen=True)
class NegativeTransferResult:
    raw_rate: float
    material_rate: float
    group_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    repetitions: int
    resampling_unit: str
    seed: int = BOOTSTRAP_SEED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceInputs:
    p1_mae_ratio_to_m0: float
    p1_r2_drop_from_m0: float
    g1_mae_ratio_to_p1: float
    g1_rmse_ratio_to_p1: float
    g1_r2_drop_from_p1: float
    g1_fold_wins: int
    transfer_reduction_vs_f1: float
    transfer_reduction_vs_r1: float
    gate_median: float
    gate_p95: float


@dataclass(frozen=True)
class PhaseADecision:
    process_expert_credible: bool
    g1_noninferior: bool
    mean_improvement_path: bool
    transfer_safety_path: bool
    gate_collapsed_without_benefit: bool
    proceed_to_phase_b: bool
    path: str
    reasons: Sequence[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_vectors(*values: Any) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    length: int | None = None
    for value in values:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("metric inputs must be finite one-dimensional arrays")
        if length is None:
            length = len(array)
        elif len(array) != length:
            raise ValueError("metric inputs must have equal length")
        arrays.append(array)
    if not arrays or len(arrays[0]) == 0:
        raise ValueError("metric inputs must not be empty")
    return arrays


def weighted_metrics(
    y_true: Any, y_pred: Any, sample_weight: Any
) -> dict[str, float]:
    """Return exact sum-of-weight normalized MAE, RMSE, and weighted R2.

    Weighted R2 uses the weighted target mean. For a constant target it is
    defined as 1 for a perfect prediction and 0 otherwise, keeping formal
    artifacts finite while matching the standard force-finite convention.
    """
    target, prediction, weight = _finite_vectors(y_true, y_pred, sample_weight)
    if np.any(weight <= 0.0):
        raise ValueError("sample_weight must be finite and strictly positive")
    weight_sum = float(weight.sum())
    error = target - prediction
    absolute_sum = float(np.dot(weight, np.abs(error)))
    squared_sum = float(np.dot(weight, error * error))
    target_mean = float(np.dot(weight, target) / weight_sum)
    total_sum = float(np.dot(weight, (target - target_mean) ** 2))
    if total_sum == 0.0:
        r2 = 1.0 if squared_sum == 0.0 else 0.0
    else:
        r2 = 1.0 - squared_sum / total_sum
    return {
        "weighted_mae": absolute_sum / weight_sum,
        "weighted_rmse": float(np.sqrt(squared_sum / weight_sum)),
        "weighted_r2": float(r2),
    }


def negative_transfer(
    paired: pd.DataFrame, *, material_margin_um: float = MATERIAL_MARGIN_UM
) -> NegativeTransferResult:
    """Calculate rates after sample-weighted error aggregation within groups."""
    required = {"group_id", "p1_abs_error", "candidate_abs_error"}
    missing = sorted(required - set(paired.columns))
    if missing:
        raise ValueError(f"negative-transfer frame missing columns: {missing}")
    if not np.isfinite(float(material_margin_um)) or material_margin_um < 0.0:
        raise ValueError("material margin must be finite and non-negative")
    values = paired.copy()
    if values["group_id"].isna().any():
        raise ValueError("negative-transfer group_id must not be missing")
    values["group_id"] = values["group_id"].astype(str)
    if "sample_weight" not in values:
        values["sample_weight"] = 1.0
    numeric = values.loc[
        :, ["p1_abs_error", "candidate_abs_error", "sample_weight"]
    ].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric[:, :2] < 0.0)
        or np.any(numeric[:, 2] <= 0.0)
    ):
        raise ValueError("negative-transfer errors/weights are incompatible")
    values["p1_weighted_error"] = values["p1_abs_error"] * values["sample_weight"]
    values["candidate_weighted_error"] = (
        values["candidate_abs_error"] * values["sample_weight"]
    )
    grouped = values.groupby("group_id", sort=True, observed=True).agg(
        p1_error_sum=("p1_weighted_error", "sum"),
        candidate_error_sum=("candidate_weighted_error", "sum"),
        weight_sum=("sample_weight", "sum"),
    )
    if grouped.empty:
        raise ValueError("negative-transfer frame must contain at least one group")
    excess = (
        grouped["candidate_error_sum"] - grouped["p1_error_sum"]
    ) / grouped["weight_sum"]
    return NegativeTransferResult(
        raw_rate=float((excess > 0.0).mean()),
        material_rate=float((excess > float(material_margin_um)).mean()),
        group_count=int(len(grouped)),
    )


def _paired_model_rows(
    predictions: pd.DataFrame, baseline: str, candidate: str
) -> pd.DataFrame:
    required = {"group_id", "model", "target", "prediction", "sample_weight"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"bootstrap predictions missing columns: {missing}")
    values = predictions.loc[predictions["model"].isin((baseline, candidate))].copy()
    if values.empty or values["group_id"].isna().any():
        raise ValueError("bootstrap requires non-missing paired model rows")
    values["group_id"] = values["group_id"].astype(str)
    numeric = values.loc[:, ["target", "prediction", "sample_weight"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all() or np.any(numeric[:, 2] <= 0.0):
        raise ValueError("bootstrap values must be finite with positive sample weights")
    if "sample_id" in values:
        if values["sample_id"].isna().any():
            raise ValueError("bootstrap sample_id must not be missing")
        values["_pair_id"] = values["sample_id"].astype(str)
    else:
        values["_pair_id"] = values.groupby(
            ["model", "group_id"], sort=False, observed=True
        ).cumcount().astype(str)
    if values.duplicated(["_pair_id", "model"]).any():
        raise ValueError("bootstrap contains duplicate sample/model rows")
    index_columns = ["_pair_id"]
    baseline_rows = values.loc[
        values["model"] == baseline,
        index_columns + ["group_id", "target", "prediction", "sample_weight"],
    ].rename(columns={"prediction": "baseline_prediction"})
    candidate_rows = values.loc[
        values["model"] == candidate,
        index_columns + ["group_id", "target", "prediction", "sample_weight"],
    ].rename(
        columns={
            "group_id": "candidate_group_id",
            "target": "candidate_target",
            "prediction": "candidate_prediction",
            "sample_weight": "candidate_weight",
        }
    )
    paired = baseline_rows.merge(
        candidate_rows, on=index_columns, how="outer", validate="one_to_one", indicator=True
    )
    if (paired["_merge"] != "both").any() or paired.empty:
        raise ValueError("bootstrap baseline/candidate Cartesian pairing is incomplete")
    if (
        not (paired["group_id"] == paired["candidate_group_id"]).all()
        or not np.allclose(paired["target"], paired["candidate_target"], rtol=0.0, atol=1e-12)
        or not np.allclose(
            paired["sample_weight"], paired["candidate_weight"], rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError("bootstrap paired target/group/weight mapping is incompatible")
    return paired


def paired_group_bootstrap(
    predictions: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """Bootstrap the weighted-MAE improvement, sampling whole groups."""
    if isinstance(repetitions, (bool, np.bool_)) or not isinstance(
        repetitions, (int, np.integer)
    ) or int(repetitions) < 1:
        raise ValueError("bootstrap repetitions must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("bootstrap seed must be an integer")
    paired = _paired_model_rows(predictions, str(baseline), str(candidate))
    weight = paired["sample_weight"].to_numpy(dtype=np.float64)
    paired["baseline_weighted_error"] = (
        np.abs(paired["target"] - paired["baseline_prediction"]) * weight
    )
    paired["candidate_weighted_error"] = (
        np.abs(paired["target"] - paired["candidate_prediction"]) * weight
    )
    grouped = paired.groupby("group_id", sort=True, observed=True).agg(
        baseline_error=("baseline_weighted_error", "sum"),
        candidate_error=("candidate_weighted_error", "sum"),
        weight=("sample_weight", "sum"),
    )
    baseline_errors = grouped["baseline_error"].to_numpy(dtype=np.float64)
    candidate_errors = grouped["candidate_error"].to_numpy(dtype=np.float64)
    weights = grouped["weight"].to_numpy(dtype=np.float64)
    point = float((baseline_errors.sum() - candidate_errors.sum()) / weights.sum())
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=np.float64)
    group_count = len(grouped)
    for index in range(int(repetitions)):
        sampled = rng.integers(0, group_count, size=group_count)
        draws[index] = float(
            (baseline_errors[sampled].sum() - candidate_errors[sampled].sum())
            / weights[sampled].sum()
        )
    return BootstrapResult(
        point_estimate=point,
        lower=float(np.quantile(draws, 0.025)),
        upper=float(np.quantile(draws, 0.975)),
        repetitions=int(repetitions),
        resampling_unit=BOOTSTRAP_UNIT,
        seed=int(seed),
    )


def validate_prediction_cartesian(
    predictions: pd.DataFrame,
    *,
    expected_sample_ids: Sequence[str],
    models: Sequence[str] = PHASE_A_MODELS,
) -> pd.DataFrame:
    """Validate the formal segment x model OOF table without coercing IDs to numbers."""
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("OOF prediction schema is incompatible")
    values = predictions.copy()
    for column in ("sample_id", "group_id", "version", "model"):
        if values[column].isna().any():
            raise ValueError(f"OOF {column} must not be missing")
        values[column] = values[column].astype(str)
    expected_ids = tuple(str(value) for value in expected_sample_ids)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected sample IDs must be unique")
    registered_models = tuple(str(value) for value in models)
    expected_pairs = {
        (sample_id, model) for sample_id in expected_ids for model in registered_models
    }
    actual_pairs = set(zip(values["sample_id"], values["model"], strict=True))
    if (
        len(values) != len(expected_pairs)
        or values.duplicated(["sample_id", "model"]).any()
        or actual_pairs != expected_pairs
    ):
        raise ValueError("OOF predictions must equal the exact sample/model Cartesian product")
    from .training import _canonical_integer_text

    def exact_integer_column(column: str) -> np.ndarray:
        parsed: list[int] = []
        for value in values[column]:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"OOF {column} must use canonical integral values")
            if isinstance(value, (int, np.integer)):
                parsed.append(int(value))
            elif isinstance(value, str) and _canonical_integer_text(value, signed=True):
                parsed.append(int(value))
            else:
                raise ValueError(f"OOF {column} must use canonical integral values")
        return np.asarray(parsed, dtype=np.int64)

    numeric_columns = ("target", "prediction", "sample_weight")
    folds = exact_integer_column("fold")
    seeds = exact_integer_column("seed")
    try:
        numeric = values.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("OOF numeric values are incompatible") from error
    if (
        not np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
        or np.any(numeric["sample_weight"].to_numpy(dtype=np.float64) <= 0.0)
        or set(seeds) != {BOOTSTRAP_SEED}
    ):
        raise ValueError("OOF fold/seed/target/prediction/weight values are incompatible")
    values.loc[:, "fold"] = folds
    values.loc[:, "seed"] = seeds
    values.loc[:, numeric_columns] = numeric
    invariants = values.groupby("sample_id", sort=False, observed=True).agg(
        groups=("group_id", "nunique"),
        versions=("version", "nunique"),
        folds=("fold", "nunique"),
        targets=("target", "nunique"),
        weights=("sample_weight", "nunique"),
    )
    if (invariants != 1).any().any():
        raise ValueError("OOF sample metadata differs across models")
    semantics = {
        "M0": ((), ("process_mean", "residual", "gate")),
        "P1": (("process_mean",), ("residual", "gate")),
        "V1": ((), ("process_mean", "residual", "gate")),
        "F1": ((), ("process_mean", "residual", "gate")),
        "R1": (("process_mean", "residual"), ("gate",)),
        "G1": (("process_mean", "residual", "gate"), ()),
    }
    for model in registered_models:
        rows = values.loc[values["model"] == model]
        finite_columns, missing_columns = semantics.get(model, ((), ()))
        for column in finite_columns:
            if not np.isfinite(pd.to_numeric(rows[column], errors="coerce")).all():
                raise ValueError(f"OOF {model} populated components must be finite")
        for column in missing_columns:
            if not rows[column].isna().all():
                raise ValueError(f"OOF {model} absent components must be NaN")
    if "G1" in registered_models:
        gates = pd.to_numeric(
            values.loc[values["model"] == "G1", "gate"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(gates).all() or np.any((gates < 0.0) | (gates > 1.0)):
            raise ValueError("OOF G1 gates must lie in [0, 1]")
    return values


def _single_metric_row(summary_metrics: pd.DataFrame, model: str) -> pd.Series:
    required = {"model", "weighted_mae", "weighted_rmse", "weighted_r2"}
    missing = sorted(required - set(summary_metrics.columns))
    if missing:
        raise ValueError(f"summary metrics missing columns: {missing}")
    values = summary_metrics
    if "aggregation_unit" in values:
        values = values.loc[values["aggregation_unit"] == "segment"]
    rows = values.loc[values["model"] == model]
    if len(rows) != 1:
        raise ValueError(f"summary metrics require exactly one segment row for {model}")
    return rows.iloc[0]


def build_acceptance_inputs(
    summary_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    negative_transfer_table: pd.DataFrame,
    gate_values: Any,
) -> AcceptanceInputs:
    """Derive threshold inputs from saved tables; no gate logic lives here."""
    rows = {model: _single_metric_row(summary_metrics, model) for model in ("M0", "P1", "G1")}
    metric_values = np.asarray(
        [rows[m][c] for m in rows for c in ("weighted_mae", "weighted_rmse", "weighted_r2")],
        dtype=np.float64,
    )
    if not np.isfinite(metric_values).all() or rows["M0"]["weighted_mae"] <= 0 or rows["P1"]["weighted_mae"] <= 0 or rows["P1"]["weighted_rmse"] <= 0:
        raise ValueError("acceptance summary metrics must be finite with positive denominators")
    fold_required = {"model", "fold", "seed", "weighted_mae"}
    missing = sorted(fold_required - set(fold_metrics.columns))
    if missing:
        raise ValueError(f"fold metrics missing columns: {missing}")
    fold_subset = fold_metrics.loc[fold_metrics["model"].isin(("P1", "G1"))].copy()
    if fold_subset.duplicated(["model", "fold", "seed"]).any():
        raise ValueError("fold metrics contain duplicate model/fold/seed rows")
    pivot = fold_subset.pivot(index=["seed", "fold"], columns="model", values="weighted_mae")
    if (
        set(pivot.columns) != {"P1", "G1"}
        or pivot.isna().any().any()
        or set(pivot.index.get_level_values("seed")) != {BOOTSTRAP_SEED}
        or set(pivot.index.get_level_values("fold")) != {0, 1, 2, 3, 4}
        or len(pivot) != 5
    ):
        raise ValueError("fold metrics require exact P1/G1 coverage for folds 0..4 and seed 20260723")
    transfer_required = {"model", "material_rate"}
    missing = sorted(transfer_required - set(negative_transfer_table.columns))
    if missing:
        raise ValueError(f"negative-transfer table missing columns: {missing}")
    transfers = negative_transfer_table.loc[
        negative_transfer_table["model"].isin(("F1", "R1", "G1"))
    ]
    if transfers["model"].duplicated().any() or set(transfers["model"]) != {"F1", "R1", "G1"}:
        raise ValueError("negative-transfer table requires exactly F1/R1/G1")
    rates = transfers.set_index("model")["material_rate"].astype(float)
    if not np.isfinite(rates).all() or ((rates < 0.0) | (rates > 1.0)).any():
        raise ValueError("negative-transfer material rates must lie in [0, 1]")
    gates = np.asarray(gate_values, dtype=np.float64)
    if gates.ndim != 1 or gates.size == 0 or not np.isfinite(gates).all() or np.any((gates < 0.0) | (gates > 1.0)):
        raise ValueError("gate values must be a non-empty finite vector in [0, 1]")
    def exact_decimal_reduction(reference: float, candidate: float) -> float:
        return float(Decimal(str(reference)) - Decimal(str(candidate)))

    return AcceptanceInputs(
        p1_mae_ratio_to_m0=float(rows["P1"]["weighted_mae"] / rows["M0"]["weighted_mae"]),
        p1_r2_drop_from_m0=float(rows["M0"]["weighted_r2"] - rows["P1"]["weighted_r2"]),
        g1_mae_ratio_to_p1=float(rows["G1"]["weighted_mae"] / rows["P1"]["weighted_mae"]),
        g1_rmse_ratio_to_p1=float(rows["G1"]["weighted_rmse"] / rows["P1"]["weighted_rmse"]),
        g1_r2_drop_from_p1=float(rows["P1"]["weighted_r2"] - rows["G1"]["weighted_r2"]),
        g1_fold_wins=int((pivot["G1"] < pivot["P1"]).sum()),
        transfer_reduction_vs_f1=exact_decimal_reduction(rates["F1"], rates["G1"]),
        transfer_reduction_vs_r1=exact_decimal_reduction(rates["R1"], rates["G1"]),
        gate_median=float(np.median(gates)),
        gate_p95=float(np.quantile(gates, 0.95)),
    )


def assess_phase_a(inputs: AcceptanceInputs) -> PhaseADecision:
    """Apply the fixed machine gate. There is deliberately no override input."""
    values = np.asarray([getattr(inputs, field) for field in inputs.__dataclass_fields__], dtype=np.float64)
    if not np.isfinite(values).all() or isinstance(inputs.g1_fold_wins, bool) or not isinstance(inputs.g1_fold_wins, (int, np.integer)):
        raise ValueError("acceptance inputs must be finite and fold wins integral")
    credible = bool(inputs.p1_mae_ratio_to_m0 <= 1.05 and inputs.p1_r2_drop_from_m0 <= 0.02)
    noninferior = bool(
        inputs.g1_mae_ratio_to_p1 <= 1.01
        and inputs.g1_rmse_ratio_to_p1 <= 1.03
        and inputs.g1_r2_drop_from_p1 <= 0.0
    )
    mean_path = bool(
        noninferior
        and inputs.g1_mae_ratio_to_p1 <= 0.99
        and inputs.g1_fold_wins >= 3
    )
    safety_path = bool(
        noninferior
        and Decimal(str(inputs.transfer_reduction_vs_f1)) >= Decimal("0.10")
        and Decimal(str(inputs.transfer_reduction_vs_r1)) >= Decimal("0.10")
    )
    collapsed = bool(
        inputs.gate_median < 0.05
        and inputs.gate_p95 < 0.20
        and not mean_path
        and not safety_path
    )
    proceed = bool(credible and noninferior and (mean_path or safety_path) and not collapsed)
    reasons: list[str] = []
    if not credible:
        reasons.append("process_expert_not_credible")
    if not noninferior:
        reasons.append("g1_not_noninferior")
    if not mean_path:
        reasons.append("mean_improvement_path_not_met")
    if not safety_path:
        reasons.append("transfer_safety_path_not_met")
    if collapsed:
        reasons.append("collapsed")
    if mean_path:
        reasons.append("mean_improvement_path_met")
    if safety_path:
        reasons.append("transfer_safety_path_met")
    path = (
        "mean_improvement_and_transfer_safety"
        if mean_path and safety_path
        else "mean_improvement"
        if mean_path
        else "transfer_safety"
        if safety_path
        else "stop"
    )
    return PhaseADecision(
        process_expert_credible=credible,
        g1_noninferior=noninferior,
        mean_improvement_path=mean_path,
        transfer_safety_path=safety_path,
        gate_collapsed_without_benefit=collapsed,
        proceed_to_phase_b=proceed,
        path=path,
        reasons=tuple(reasons),
    )


__all__ = [
    "AcceptanceInputs",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_UNIT",
    "BootstrapResult",
    "MATERIAL_MARGIN_UM",
    "NegativeTransferResult",
    "PHASE_A_MODELS",
    "PREDICTION_COLUMNS",
    "PhaseADecision",
    "assess_phase_a",
    "build_acceptance_inputs",
    "negative_transfer",
    "paired_group_bootstrap",
    "validate_prediction_cartesian",
    "weighted_metrics",
]
