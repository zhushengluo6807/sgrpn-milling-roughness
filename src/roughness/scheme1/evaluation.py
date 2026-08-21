from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    reference_mae: float
    candidate_mae: float
    delta_mae: float
    relative_improvement: float
    ci_low: float
    ci_high: float
    repetitions: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def paired_group_bootstrap(
    frame: pd.DataFrame,
    reference_col: str,
    candidate_col: str,
    repetitions: int = 10_000,
    seed: int = 20260723,
) -> BootstrapResult:
    required = {
        "group_id",
        "y_true",
        "sample_weight",
        reference_col,
        candidate_col,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Bootstrap frame missing columns: {missing}")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    values = frame.copy()
    numeric = values[
        ["y_true", "sample_weight", reference_col, candidate_col]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(values["sample_weight"] <= 0):
        raise ValueError("Bootstrap inputs must be finite with positive weights")

    values["reference_weighted_error"] = (
        np.abs(values["y_true"] - values[reference_col])
        * values["sample_weight"]
    )
    values["candidate_weighted_error"] = (
        np.abs(values["y_true"] - values[candidate_col])
        * values["sample_weight"]
    )
    grouped = values.groupby("group_id", sort=True).agg(
        reference_error=("reference_weighted_error", "sum"),
        candidate_error=("candidate_weighted_error", "sum"),
        weight=("sample_weight", "sum"),
    )
    reference_mae = float(
        grouped["reference_error"].sum() / grouped["weight"].sum()
    )
    candidate_mae = float(
        grouped["candidate_error"].sum() / grouped["weight"].sum()
    )
    rng = np.random.default_rng(seed)
    group_count = len(grouped)
    deltas = np.empty(repetitions, dtype=np.float64)
    reference_errors = grouped["reference_error"].to_numpy()
    candidate_errors = grouped["candidate_error"].to_numpy()
    weights = grouped["weight"].to_numpy()
    for index in range(repetitions):
        sampled = rng.integers(0, group_count, size=group_count)
        denominator = weights[sampled].sum()
        reference = reference_errors[sampled].sum() / denominator
        candidate = candidate_errors[sampled].sum() / denominator
        deltas[index] = reference - candidate
    delta = reference_mae - candidate_mae
    return BootstrapResult(
        reference_mae=reference_mae,
        candidate_mae=candidate_mae,
        delta_mae=delta,
        relative_improvement=delta / reference_mae,
        ci_low=float(np.quantile(deltas, 0.025)),
        ci_high=float(np.quantile(deltas, 0.975)),
        repetitions=int(repetitions),
        seed=int(seed),
    )


def assess_model_gate(
    metrics: pd.DataFrame,
    candidate: str,
    reference: str,
    minimum_relative_improvement: float,
    minimum_fold_wins: int,
    require_all_seeds: bool,
    maximum_fold_std_relative_increase: float | None = None,
) -> dict:
    required = {"model", "seed", "fold", "weighted_mae"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Metrics missing columns: {missing}")
    subset = metrics[metrics["model"].isin([candidate, reference])].copy()
    if subset.duplicated(["model", "seed", "fold"]).any():
        raise ValueError("Duplicate model/seed/fold metric rows")
    pivot = subset.pivot(
        index=["seed", "fold"], columns="model", values="weighted_mae"
    )
    if candidate not in pivot or reference not in pivot or pivot.isna().any().any():
        raise ValueError("Candidate and reference metrics are incomplete")
    reference_mean = float(pivot[reference].mean())
    candidate_mean = float(pivot[candidate].mean())
    relative_improvement = (reference_mean - candidate_mean) / reference_mean
    fold_means = pivot.groupby(level="fold").mean()
    fold_wins = int((fold_means[candidate] < fold_means[reference]).sum())
    seed_means = pivot.groupby(level="seed").mean()
    all_seeds_improve = bool(
        (seed_means[candidate] < seed_means[reference]).all()
    )
    failed_conditions = []
    if relative_improvement < minimum_relative_improvement:
        failed_conditions.append("relative_improvement")
    if fold_wins < minimum_fold_wins:
        failed_conditions.append("fold_consistency")
    if require_all_seeds and not all_seeds_improve:
        failed_conditions.append("seed_consistency")
    fold_std_relative_increase = None
    if maximum_fold_std_relative_increase is not None:
        reference_std = float(fold_means[reference].std(ddof=0))
        candidate_std = float(fold_means[candidate].std(ddof=0))
        fold_std_relative_increase = (
            (candidate_std - reference_std) / reference_std
            if reference_std > 0
            else (0.0 if candidate_std == 0 else float("inf"))
        )
        if fold_std_relative_increase > maximum_fold_std_relative_increase:
            failed_conditions.append("fold_variance")
    return {
        "candidate": candidate,
        "reference": reference,
        "reference_weighted_mae": reference_mean,
        "candidate_weighted_mae": candidate_mean,
        "relative_improvement": relative_improvement,
        "fold_wins": fold_wins,
        "all_seeds_improve": all_seeds_improve,
        "fold_std_relative_increase": fold_std_relative_increase,
        "passed": not failed_conditions,
        "failed_conditions": failed_conditions,
    }
