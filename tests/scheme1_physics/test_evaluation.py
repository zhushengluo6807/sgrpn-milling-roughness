import pandas as pd
import pytest

from roughness.scheme1_physics.evaluation import (
    _bootstrap_candidate_vs_m0,
    apply_secondary_metric_downgrade,
    assess_formula_pair,
    assess_pair,
    physics_metric_rows,
    select_final_candidate,
    validate_complete_physics_oof,
)


MODELS = ("PW0", "PW1", "PW2", "PE0", "PE1", "PE2")


def _complete_oof():
    rows = []
    for index, sample_id in enumerate(("s0", "s1")):
        for model in MODELS:
            rows.append(
                {
                    "sample_id": sample_id,
                    "group_id": f"g{index}",
                    "model": model,
                    "formula": "word" if model.startswith("PW") else "exact",
                    "radius_mm": 0.2,
                    "fold": index,
                    "seed": 20260723,
                    "y_true": 0.2 + 0.1 * index,
                    "y_pred": 0.21 + 0.1 * index,
                    "base_ra": 0.05,
                    "sample_weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_complete_oof_rejects_duplicate_and_missing_rows():
    complete = _complete_oof()
    duplicate = pd.concat([complete, complete.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_physics_oof(
            duplicate, set(complete["sample_id"]), MODELS, [20260723]
        )
    with pytest.raises(ValueError, match="Incomplete OOF"):
        validate_complete_physics_oof(
            complete[complete["model"] != "PE2"],
            set(complete["sample_id"]),
            MODELS,
            [20260723],
        )


def test_metrics_include_weighted_and_unweighted_regression_scores():
    metrics = physics_metric_rows(_complete_oof())
    assert {
        "mae",
        "rmse",
        "r2",
        "weighted_mae",
        "weighted_rmse",
        "weighted_r2",
    } <= set(metrics)


def test_gate_requires_one_percent_three_folds_and_stable_variance():
    rows = []
    for seed in (20260723, 20260724, 20260725):
        for fold in range(5):
            rows.extend(
                [
                    {
                        "model": "PW1",
                        "seed": seed,
                        "fold": fold,
                        "weighted_mae": 1.0 + 0.01 * fold,
                    },
                    {
                        "model": "PW2",
                        "seed": seed,
                        "fold": fold,
                        "weighted_mae": 0.98 + 0.005 * fold,
                    },
                ]
            )
    result = assess_pair(
        pd.DataFrame(rows),
        candidate="PW2",
        reference="PW1",
        minimum_relative_improvement=0.01,
        minimum_fold_wins=3,
        require_all_seed_means=True,
        maximum_fold_std_relative_increase=0.20,
    )
    assert result["passed"]


def test_secondary_metric_conflict_downgrades_one_level():
    result = {"classification": "stable_effective"}
    downgraded = apply_secondary_metric_downgrade(
        result,
        reference_metrics={"weighted_rmse": 0.10, "weighted_r2": 0.80},
        candidate_metrics={"weighted_rmse": 0.104, "weighted_r2": 0.81},
    )
    assert downgraded["classification"] == "exploratory_increment"
    assert "rmse_conflict" in downgraded["downgrade_reasons"]


def test_final_candidate_excludes_failed_gate_and_is_deterministic():
    acceptance = {
        "PW1": {"classification": "exploratory_increment", "stage_passed": True},
        "PW2": {"classification": "stable_effective", "stage_passed": False},
        "PE1": {"classification": "exploratory_increment", "stage_passed": True},
        "PE2": {"classification": "no_stable_increment", "stage_passed": True},
    }
    metrics = pd.DataFrame(
        [
            {"model": "PW1", "weighted_mae": 0.11},
            {"model": "PE1", "weighted_mae": 0.10},
            {"model": "PE2", "weighted_mae": 0.09},
        ]
    )
    assert select_final_candidate(acceptance, metrics)["model"] == "PE1"


def test_m0_bootstrap_excludes_other_classic_models():
    candidate = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "group_id": ["g0", "g1"],
            "model": ["PW1", "PW1"],
            "seed": [1, 1],
            "y_true": [0.2, 0.3],
            "y_pred": [0.25, 0.35],
            "sample_weight": [1.0, 1.0],
        }
    )
    classic = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s0", "s1"],
            "model": ["M0", "M0", "P1", "P1"],
            "y_pred": [0.21, 0.31, 2.0, 2.0],
        }
    )
    result = _bootstrap_candidate_vs_m0(
        candidate, classic, "PW1", repetitions=10, seed=1
    )
    assert result["reference_mae"] == pytest.approx(0.01)


def test_formula_comparison_requires_bootstrap_interval_excluding_zero(
    monkeypatch,
):
    from types import SimpleNamespace
    import roughness.scheme1_physics.evaluation as evaluation

    rows = []
    for seed in (1, 2, 3):
        for fold in range(5):
            rows.extend(
                [
                    {
                        "model": "PW1",
                        "seed": seed,
                        "fold": fold,
                        "weighted_mae": 1.0,
                    },
                    {
                        "model": "PE1",
                        "seed": seed,
                        "fold": fold,
                        "weighted_mae": 0.98,
                    },
                ]
            )
    predictions = pd.DataFrame(
        {
            "sample_id": ["s0", "s0"],
            "group_id": ["g0", "g0"],
            "model": ["PW1", "PE1"],
            "y_true": [1.0, 1.0],
            "y_pred": [1.1, 1.0],
            "sample_weight": [1.0, 1.0],
        }
    )
    monkeypatch.setattr(
        evaluation,
        "paired_group_bootstrap",
        lambda *args, **kwargs: SimpleNamespace(
            to_dict=lambda: {"ci_low": -0.01, "ci_high": 0.01}
        ),
    )
    result = assess_formula_pair(
        pd.DataFrame(rows),
        predictions,
        level=1,
        repetitions=10,
        seed=1,
    )
    assert not result["passed"]
    assert result["direction"] == "inconclusive"
