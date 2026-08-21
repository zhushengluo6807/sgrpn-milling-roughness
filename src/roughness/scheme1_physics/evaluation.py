from copy import deepcopy
import warnings

import numpy as np
import pandas as pd

from roughness.metrics import regression_metrics
from roughness.scheme1.evaluation import (
    assess_model_gate,
    paired_group_bootstrap,
)


COMPARISONS = {
    "PW1_vs_PW0": ("PW1", "PW0", 0.05, 3, True, None),
    "PE1_vs_PE0": ("PE1", "PE0", 0.05, 3, True, None),
    "PW2_vs_PW1": ("PW2", "PW1", 0.01, 3, True, 0.20),
    "PE2_vs_PE1": ("PE2", "PE1", 0.01, 3, True, 0.20),
    "PE0_vs_PW0": ("PE0", "PW0", 0.01, 3, True, None),
    "PE1_vs_PW1": ("PE1", "PW1", 0.01, 3, True, None),
    "PE2_vs_PW2": ("PE2", "PW2", 0.01, 3, True, None),
}


def validate_complete_physics_oof(
    frame: pd.DataFrame,
    expected_sample_ids: set[str],
    models,
    seeds,
) -> None:
    required = {
        "sample_id",
        "group_id",
        "model",
        "formula",
        "radius_mm",
        "fold",
        "seed",
        "y_true",
        "y_pred",
        "base_ra",
        "sample_weight",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF missing columns: {missing}")
    data = frame.copy()
    data["sample_id"] = data["sample_id"].astype(str)
    if data.duplicated(["sample_id", "model", "seed"]).any():
        raise ValueError("OOF contains duplicate sample/model/seed rows")
    expected_ids = set(map(str, expected_sample_ids))
    for model in models:
        for seed in seeds:
            observed = set(
                data[
                    (data["model"] == model)
                    & (data["seed"].astype(int) == int(seed))
                ]["sample_id"]
            )
            if observed != expected_ids:
                raise ValueError(
                    f"Incomplete OOF for model={model}, seed={seed}"
                )
    numeric = data[
        ["y_true", "y_pred", "sample_weight", "base_ra", "radius_mm"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("OOF numeric values must be finite")
    if (data["sample_weight"].astype(float) <= 0).any():
        raise ValueError("OOF sample_weight must be positive")
    word_bad = data["model"].astype(str).str.startswith("PW") & data[
        "formula"
    ].ne("word")
    exact_bad = data["model"].astype(str).str.startswith("PE") & data[
        "formula"
    ].ne("exact")
    if word_bad.any() or exact_bad.any():
        raise ValueError("OOF formula does not match model family")
    family = np.where(
        data["model"].astype(str).str.startswith("PW"), "word", "exact"
    )
    radius_counts = (
        data.assign(_family=family)
        .groupby(["sample_id", "_family", "seed"])["radius_mm"]
        .nunique()
    )
    if (radius_counts > 1).any():
        raise ValueError("Models in one formula family use different radii")


def physics_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "seed",
        "fold",
        "y_true",
        "y_pred",
        "sample_weight",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")
    rows = []
    for keys, group in predictions.groupby(
        ["model", "seed", "fold"], sort=True
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unweighted = regression_metrics(
                group["y_true"], group["y_pred"]
            )
            weighted = regression_metrics(
                group["y_true"],
                group["y_pred"],
                sample_weight=group["sample_weight"],
            )
        rows.append(
            {
                "model": keys[0],
                "seed": int(keys[1]),
                "fold": int(keys[2]),
                **unweighted,
                **{f"weighted_{key}": value for key, value in weighted.items()},
                "n_samples": int(len(group)),
            }
        )
    return pd.DataFrame.from_records(rows)


def assess_pair(
    metrics: pd.DataFrame,
    candidate: str,
    reference: str,
    minimum_relative_improvement: float,
    minimum_fold_wins: int,
    require_all_seed_means: bool,
    maximum_fold_std_relative_increase: float | None = None,
) -> dict:
    return assess_model_gate(
        metrics,
        candidate=candidate,
        reference=reference,
        minimum_relative_improvement=minimum_relative_improvement,
        minimum_fold_wins=minimum_fold_wins,
        require_all_seeds=require_all_seed_means,
        maximum_fold_std_relative_increase=(
            maximum_fold_std_relative_increase
        ),
    )


def apply_secondary_metric_downgrade(
    result: dict,
    reference_metrics: dict,
    candidate_metrics: dict,
) -> dict:
    updated = deepcopy(result)
    reasons = list(updated.get("downgrade_reasons", []))
    reference_rmse = float(reference_metrics["weighted_rmse"])
    candidate_rmse = float(candidate_metrics["weighted_rmse"])
    if candidate_rmse > reference_rmse * 1.03:
        reasons.append("rmse_conflict")
    if float(candidate_metrics["weighted_r2"]) < float(
        reference_metrics["weighted_r2"]
    ):
        reasons.append("r2_conflict")
    if reasons:
        levels = {
            "stable_effective": "exploratory_increment",
            "exploratory_increment": "no_stable_increment",
            "no_stable_increment": "no_stable_increment",
        }
        updated["classification"] = levels.get(
            updated.get("classification"), updated.get("classification")
        )
    updated["downgrade_reasons"] = reasons
    return updated


def _bootstrap_model_pair(
    predictions: pd.DataFrame,
    candidate: str,
    reference: str,
    repetitions: int,
    seed: int,
) -> dict:
    subset = predictions[predictions["model"].isin([candidate, reference])]
    averaged = (
        subset.groupby(["sample_id", "group_id", "model"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            sample_weight=("sample_weight", "first"),
            y_pred=("y_pred", "mean"),
        )
    )
    reference_rows = averaged[averaged["model"] == reference].rename(
        columns={"y_pred": "reference_pred"}
    )
    candidate_rows = averaged[averaged["model"] == candidate][
        ["sample_id", "y_pred"]
    ].rename(columns={"y_pred": "candidate_pred"})
    paired = reference_rows[
        ["sample_id", "group_id", "y_true", "sample_weight", "reference_pred"]
    ].merge(candidate_rows, on="sample_id", validate="one_to_one")
    return paired_group_bootstrap(
        paired,
        reference_col="reference_pred",
        candidate_col="candidate_pred",
        repetitions=repetitions,
        seed=seed,
    ).to_dict()


def assess_formula_pair(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    level: int,
    repetitions: int,
    seed: int,
) -> dict:
    if level not in {0, 1, 2}:
        raise ValueError("formula level must be 0, 1 or 2")
    exact = f"PE{level}"
    word = f"PW{level}"
    exact_gate = assess_pair(
        metrics, exact, word, 0.01, 3, True
    )
    word_gate = assess_pair(
        metrics, word, exact, 0.01, 3, True
    )
    exact_bootstrap = _bootstrap_model_pair(
        predictions, exact, word, repetitions, seed
    )
    word_bootstrap = _bootstrap_model_pair(
        predictions, word, exact, repetitions, seed
    )
    exact_passed = exact_gate["passed"] and exact_bootstrap["ci_low"] > 0
    word_passed = word_gate["passed"] and word_bootstrap["ci_low"] > 0
    direction = (
        "exact_better"
        if exact_passed
        else "word_better"
        if word_passed
        else "inconclusive"
    )
    return {
        "level": int(level),
        "direction": direction,
        "passed": bool(exact_passed or word_passed),
        "exact_vs_word": {
            "gate": exact_gate,
            "bootstrap": exact_bootstrap,
            "passed": bool(exact_passed),
        },
        "word_vs_exact": {
            "gate": word_gate,
            "bootstrap": word_bootstrap,
            "passed": bool(word_passed),
        },
    }


def _aggregate_model_metrics(metrics: pd.DataFrame, model: str) -> dict:
    subset = metrics[metrics["model"] == model]
    return {
        column: float(subset[column].mean())
        for column in ("weighted_mae", "weighted_rmse", "weighted_r2")
    }


def _bootstrap_candidate_vs_m0(
    predictions: pd.DataFrame,
    m0_predictions: pd.DataFrame,
    candidate: str,
    repetitions: int,
    seed: int,
) -> dict:
    candidate_rows = predictions[predictions["model"] == candidate]
    averaged = (
        candidate_rows.groupby(["sample_id", "group_id"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            sample_weight=("sample_weight", "first"),
            candidate_pred=("y_pred", "mean"),
        )
    )
    m0 = m0_predictions.copy()
    if "model" in m0.columns:
        m0 = m0[m0["model"].eq("M0")].copy()
    if m0.empty:
        raise ValueError("M0 predictions are missing")
    prediction_column = (
        "y_pred"
        if "y_pred" in m0
        else "prediction"
        if "prediction" in m0
        else "base_ra"
    )
    m0_by_sample = (
        m0.groupby("sample_id", as_index=False)[prediction_column].mean()
    )
    merged = averaged.merge(
        m0_by_sample.rename(columns={prediction_column: "m0_pred"}),
        on="sample_id",
        validate="one_to_one",
    )
    return paired_group_bootstrap(
        merged,
        reference_col="m0_pred",
        candidate_col="candidate_pred",
        repetitions=repetitions,
        seed=seed,
    ).to_dict()


def assess_physics_comparisons(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    m0_predictions: pd.DataFrame,
    config,
) -> dict:
    comparisons = {}
    for name, (
        candidate,
        reference,
        minimum,
        fold_wins,
        all_seeds,
        max_std,
    ) in COMPARISONS.items():
        comparisons[name] = assess_pair(
            metrics,
            candidate,
            reference,
            minimum,
            fold_wins,
            all_seeds,
            max_std,
        )
    formula_comparisons = {
        f"level_{level}": assess_formula_pair(
            metrics,
            predictions,
            level=level,
            repetitions=config.source.bootstrap_repetitions,
            seed=config.source.seeds[0],
        )
        for level in (0, 1, 2)
    }
    m0_metrics = physics_metric_rows(
        m0_predictions.assign(model="M0")
        if "model" not in m0_predictions
        else m0_predictions
    )
    combined_metrics = pd.concat([metrics, m0_metrics], ignore_index=True)
    m0_acceptance = {}
    for candidate in ("PW1", "PW2", "PE1", "PE2"):
        exploratory = assess_pair(
            combined_metrics,
            candidate,
            "M0",
            0.03,
            3,
            True,
        )
        stable = assess_pair(
            combined_metrics,
            candidate,
            "M0",
            0.05,
            4,
            True,
        )
        bootstrap = _bootstrap_candidate_vs_m0(
            predictions,
            m0_predictions,
            candidate,
            repetitions=config.source.bootstrap_repetitions,
            seed=config.source.seeds[0],
        )
        classification = "no_stable_increment"
        if exploratory["passed"]:
            classification = "exploratory_increment"
        if stable["passed"] and bootstrap["ci_low"] > 0:
            classification = "stable_effective"
        result = {
            "classification": classification,
            "stage_passed": True,
            "exploratory_gate": exploratory,
            "stable_gate": stable,
            "bootstrap": bootstrap,
        }
        result = apply_secondary_metric_downgrade(
            result,
            _aggregate_model_metrics(combined_metrics, "M0"),
            _aggregate_model_metrics(combined_metrics, candidate),
        )
        m0_acceptance[candidate] = result
    for gated, key in (("PW2", "PW2_vs_PW1"), ("PE2", "PE2_vs_PE1")):
        m0_acceptance[gated]["stage_passed"] = comparisons[key]["passed"]
    return {
        "comparisons": comparisons,
        "formula_comparisons": formula_comparisons,
        "m0_acceptance": m0_acceptance,
    }


def select_final_candidate(acceptance: dict, metrics: pd.DataFrame) -> dict:
    ranks = {
        "no_stable_increment": 0,
        "exploratory_increment": 1,
        "stable_effective": 2,
    }
    fixed_order = {"PW1": 0, "PE1": 1, "PW2": 2, "PE2": 3}
    candidates = []
    for model in ("PW1", "PE1", "PW2", "PE2"):
        result = acceptance.get(model)
        if not result or not result.get("stage_passed", False):
            continue
        model_metrics = metrics[metrics["model"] == model]
        if model_metrics.empty:
            continue
        candidates.append(
            (
                -ranks.get(result.get("classification"), 0),
                float(model_metrics["weighted_mae"].mean()),
                fixed_order[model],
                model,
            )
        )
    if not candidates:
        return {"model": None, "reason": "no eligible physical candidate"}
    candidates.sort()
    _, mae, _, model = candidates[0]
    return {
        "model": model,
        "classification": acceptance[model]["classification"],
        "weighted_mae": mae,
    }
