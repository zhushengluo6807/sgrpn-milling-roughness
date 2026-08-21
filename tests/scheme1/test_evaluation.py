import numpy as np
import pandas as pd
import pytest

from roughness.scheme1.evaluation import (
    assess_model_gate,
    paired_group_bootstrap,
)


def test_paired_group_bootstrap_detects_consistently_better_candidate():
    rows = []
    for group_index in range(20):
        for segment_index in range(2):
            truth = 1.0 + group_index * 0.05
            rows.append(
                {
                    "group_id": f"g{group_index}",
                    "sample_id": f"g{group_index}_s{segment_index}",
                    "y_true": truth,
                    "sample_weight": 0.5,
                    "reference": truth + 0.20,
                    "candidate": truth + 0.05,
                }
            )
    frame = pd.DataFrame(rows)

    result = paired_group_bootstrap(
        frame,
        reference_col="reference",
        candidate_col="candidate",
        repetitions=1000,
        seed=20260723,
    )

    assert result.delta_mae == pytest.approx(0.15)
    assert result.ci_low > 0
    assert result.relative_improvement == pytest.approx(0.75)


def test_model_gate_requires_fold_and_seed_consistency():
    rows = []
    for seed in (1, 2, 3):
        for fold in range(5):
            rows.append(
                {
                    "model": "M0",
                    "seed": seed,
                    "fold": fold,
                    "weighted_mae": 0.10,
                }
            )
            rows.append(
                {
                    "model": "N4",
                    "seed": seed,
                    "fold": fold,
                    "weighted_mae": 0.093 if fold < 4 else 0.101,
                }
            )
    metrics = pd.DataFrame(rows)

    result = assess_model_gate(
        metrics,
        candidate="N4",
        reference="M0",
        minimum_relative_improvement=0.05,
        minimum_fold_wins=4,
        require_all_seeds=True,
    )

    assert result["passed"] is True
    assert result["fold_wins"] == 4
    assert result["all_seeds_improve"] is True


def test_model_gate_fails_when_one_seed_degrades():
    rows = []
    for seed in (1, 2, 3):
        for fold in range(5):
            rows.extend(
                [
                    {
                        "model": "M0",
                        "seed": seed,
                        "fold": fold,
                        "weighted_mae": 0.10,
                    },
                    {
                        "model": "N4",
                        "seed": seed,
                        "fold": fold,
                        "weighted_mae": 0.09 if seed != 3 else 0.11,
                    },
                ]
            )

    result = assess_model_gate(
        pd.DataFrame(rows),
        candidate="N4",
        reference="M0",
        minimum_relative_improvement=0.03,
        minimum_fold_wins=3,
        require_all_seeds=True,
    )

    assert result["passed"] is False
    assert result["all_seeds_improve"] is False
    assert "seed_consistency" in result["failed_conditions"]
