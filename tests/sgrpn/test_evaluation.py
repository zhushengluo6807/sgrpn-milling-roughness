import numpy as np
import pandas as pd
import pytest
import inspect
import math
import torch
from torch import nn

import roughness.sgrpn.evaluation as evaluation

from roughness.sgrpn.evaluation import (
    AcceptanceInputs,
    BOOTSTRAP_REPETITIONS,
    PhaseBPaperDecision,
    ProbabilityMetrics,
    assess_phase_a,
    assess_phase_b_claims,
    build_acceptance_inputs,
    negative_transfer,
    paired_group_bootstrap,
    paired_probability_bootstrap,
    probability_metric_table,
    seed_uncertainty_summary,
    validate_phase_b_prediction_cartesian,
    validate_prediction_cartesian,
    weighted_metrics,
)
from roughness.sgrpn.dataset import swap_horizontal
from roughness.sgrpn.models import (
    ModelOutput,
    SelectiveGatedModel,
    average_swap_predictions,
)
from roughness.sgrpn.phase_b_training import (
    MEAN_MODELS,
    PHASE_B_MEAN_COLUMNS,
    PHASE_B_PREDICTION_COLUMNS,
    PHASE_B_SEEDS,
    SCALE_MODELS,
)


def test_weighted_metrics_use_registered_normalizations():
    result = weighted_metrics(
        y_true=np.array([0.0, 2.0, 4.0]),
        y_pred=np.array([1.0, 2.0, 2.0]),
        sample_weight=np.array([1.0, 2.0, 1.0]),
    )

    assert result == pytest.approx(
        {
            "weighted_mae": 0.75,
            "weighted_rmse": np.sqrt(1.25),
            "weighted_r2": 0.375,
        }
    )


def test_material_negative_transfer_uses_weighted_group_error_and_point_zero_one_um():
    paired = pd.DataFrame(
        {
            "group_id": ["01", "01", "02"],
            "p1_abs_error": [0.0, 0.20, 0.10],
            "candidate_abs_error": [0.20, 0.0, 0.121],
            "sample_weight": [0.9, 0.1, 1.0],
        }
    )

    result = negative_transfer(paired, material_margin_um=0.01)

    assert result.raw_rate == 1.0
    assert result.material_rate == 1.0
    assert result.group_count == 2


def test_material_negative_transfer_supports_registered_unweighted_fixture():
    paired = pd.DataFrame(
        {
            "group_id": ["a", "b"],
            "p1_abs_error": [0.10, 0.10],
            "candidate_abs_error": [0.105, 0.121],
        }
    )
    result = negative_transfer(paired, material_margin_um=0.01)
    assert result.raw_rate == 1.0
    assert result.material_rate == 0.5


def test_bootstrap_resamples_groups_not_segments_and_is_reproducible():
    rows = []
    for group, target in (("01", 0.5), ("02", 1.0), ("03", 1.5)):
        for segment in ("01", "02"):
            rows.extend(
                [
                    {
                        "sample_id": f"{group}-{segment}",
                        "group_id": group,
                        "model": "P1",
                        "target": target,
                        "prediction": target + 0.1,
                        "sample_weight": 0.5,
                    },
                    {
                        "sample_id": f"{group}-{segment}",
                        "group_id": group,
                        "model": "G1",
                        "target": target,
                        "prediction": target + (0.01 if group == "01" else 0.05),
                        "sample_weight": 0.5,
                    },
                ]
            )
    predictions = pd.DataFrame(rows)

    first = paired_group_bootstrap(
        predictions, baseline="P1", candidate="G1", repetitions=50, seed=20260723
    )
    second = paired_group_bootstrap(
        predictions, baseline="P1", candidate="G1", repetitions=50, seed=20260723
    )

    assert first == second
    assert first.resampling_unit == "group_id"
    assert first.repetitions == 50
    assert first.seed == 20260723
    assert first.point_estimate == pytest.approx(0.06333333333333334)
    assert first.lower <= first.point_estimate <= first.upper


def test_prediction_validation_requires_exact_six_model_cartesian_and_preserves_ids():
    rows = [
        {
            "sample_id": sample,
            "group_id": f"g-{sample}",
            "version": "v3",
            "fold": fold,
            "seed": 20260723,
            "model": model,
            "target": 1.0,
            "prediction": 1.0,
            "sample_weight": 1.0,
            "process_mean": 1.0 if model in {"P1", "R1", "G1"} else np.nan,
            "residual": 0.0 if model in {"R1", "G1"} else np.nan,
            "gate": 0.5 if model == "G1" else np.nan,
        }
        for sample, fold in (("001", 0), ("002", 1))
        for model in ("M0", "P1", "V1", "F1", "R1", "G1")
    ]
    frame = pd.DataFrame(rows)

    validated = validate_prediction_cartesian(frame, expected_sample_ids=("001", "002"))
    assert validated["sample_id"].tolist()[0] == "001"

    with pytest.raises(ValueError, match="Cartesian"):
        validate_prediction_cartesian(frame.iloc[:-1], expected_sample_ids=("001", "002"))


def test_phase_b_proceeds_on_mean_gain_path():
    decision = assess_phase_a(
        AcceptanceInputs(
            p1_mae_ratio_to_m0=1.04,
            p1_r2_drop_from_m0=0.01,
            g1_mae_ratio_to_p1=0.98,
            g1_rmse_ratio_to_p1=0.99,
            g1_r2_drop_from_p1=-0.01,
            g1_fold_wins=3,
            transfer_reduction_vs_f1=0.0,
            transfer_reduction_vs_r1=0.0,
            gate_median=0.3,
            gate_p95=0.8,
        )
    )
    assert decision.proceed_to_phase_b is True
    assert decision.path == "mean_improvement"


def test_phase_b_proceeds_on_transfer_safety_path_at_exact_thresholds():
    decision = assess_phase_a(
        AcceptanceInputs(
            p1_mae_ratio_to_m0=1.05,
            p1_r2_drop_from_m0=0.02,
            g1_mae_ratio_to_p1=1.01,
            g1_rmse_ratio_to_p1=1.03,
            g1_r2_drop_from_p1=0.0,
            g1_fold_wins=0,
            transfer_reduction_vs_f1=0.10,
            transfer_reduction_vs_r1=0.10,
            gate_median=0.05,
            gate_p95=0.20,
        )
    )
    assert decision.proceed_to_phase_b is True
    assert decision.path == "transfer_safety"


def test_phase_b_stops_when_gate_collapses_without_benefit():
    decision = assess_phase_a(
        AcceptanceInputs(
            p1_mae_ratio_to_m0=1.00,
            p1_r2_drop_from_m0=0.0,
            g1_mae_ratio_to_p1=1.00,
            g1_rmse_ratio_to_p1=1.00,
            g1_r2_drop_from_p1=0.0,
            g1_fold_wins=2,
            transfer_reduction_vs_f1=0.02,
            transfer_reduction_vs_r1=0.02,
            gate_median=0.01,
            gate_p95=0.10,
        )
    )
    assert decision.proceed_to_phase_b is False
    assert "collapsed" in decision.reasons


def test_acceptance_builder_uses_segment_summary_exact_five_folds_and_material_rates():
    summary = pd.DataFrame(
        [
            {"aggregation_unit": "segment", "model": "M0", "weighted_mae": 1.0, "weighted_rmse": 1.0, "weighted_r2": 0.50},
            {"aggregation_unit": "segment", "model": "P1", "weighted_mae": 1.04, "weighted_rmse": 1.0, "weighted_r2": 0.49},
            {"aggregation_unit": "segment", "model": "G1", "weighted_mae": 1.0192, "weighted_rmse": 0.99, "weighted_r2": 0.51},
        ]
    )
    fold_rows = []
    for fold in range(5):
        fold_rows.extend(
            [
                {"model": "P1", "fold": fold, "seed": 20260723, "weighted_mae": 1.0},
                {"model": "G1", "fold": fold, "seed": 20260723, "weighted_mae": 0.9 if fold < 3 else 1.1},
            ]
        )
    transfers = pd.DataFrame(
        {"model": ["F1", "R1", "G1"], "material_rate": [0.30, 0.25, 0.10]}
    )

    result = build_acceptance_inputs(summary, pd.DataFrame(fold_rows), transfers, [0.1, 0.2, 0.9])

    assert result.p1_mae_ratio_to_m0 == pytest.approx(1.04)
    assert result.g1_mae_ratio_to_p1 == pytest.approx(0.98)
    assert result.g1_fold_wins == 3
    assert result.transfer_reduction_vs_f1 == pytest.approx(0.20)
    assert result.transfer_reduction_vs_r1 == pytest.approx(0.15)
    assert result.gate_p95 == pytest.approx(0.83)


def _one_sample_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "001",
                "group_id": "01",
                "version": "v3",
                "fold": 0,
                "seed": 20260723,
                "model": model,
                "target": 1.0,
                "prediction": 1.0,
                "sample_weight": 1.0,
                "process_mean": 1.0 if model in {"P1", "R1", "G1"} else np.nan,
                "residual": 0.0 if model in {"R1", "G1"} else np.nan,
                "gate": 0.5 if model == "G1" else np.nan,
            }
            for model in ("M0", "P1", "V1", "F1", "R1", "G1")
        ]
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("fold", "0.0"),
        ("fold", "+0"),
        ("fold", "00"),
        ("fold", "0e0"),
        ("fold", "true"),
        ("seed", "20260723.0"),
        ("seed", "+20260723"),
        ("seed", "020260723"),
        ("seed", "2.0260723e7"),
        ("seed", "false"),
    ],
)
def test_prediction_validation_rejects_noncanonical_integer_text(column, value):
    frame = _one_sample_prediction_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="canonical integral"):
        validate_prediction_cartesian(frame, expected_sample_ids=("001",))


def test_material_negative_transfer_is_strict_at_exact_point_zero_one_um():
    paired = pd.DataFrame(
        {
            "group_id": ["a", "b"],
            "p1_abs_error": [0.10, 0.10],
            "candidate_abs_error": [0.11, 0.1100001],
            "sample_weight": [1.0, 1.0],
        }
    )
    result = negative_transfer(paired, material_margin_um=0.01)
    assert result.raw_rate == 1.0
    assert result.material_rate == 0.5


def test_safety_equality_survives_builder_float_subtraction():
    summary = pd.DataFrame(
        [
            {"aggregation_unit": "segment", "model": "M0", "weighted_mae": 1.0, "weighted_rmse": 1.0, "weighted_r2": 0.50},
            {"aggregation_unit": "segment", "model": "P1", "weighted_mae": 1.0, "weighted_rmse": 1.0, "weighted_r2": 0.50},
            {"aggregation_unit": "segment", "model": "G1", "weighted_mae": 1.0, "weighted_rmse": 1.0, "weighted_r2": 0.50},
        ]
    )
    folds = pd.DataFrame(
        [
            {"model": model, "fold": fold, "seed": 20260723, "weighted_mae": 1.0}
            for fold in range(5)
            for model in ("P1", "G1")
        ]
    )
    transfers = pd.DataFrame(
        {"model": ["F1", "R1", "G1"], "material_rate": [0.15, 0.15, 0.05]}
    )
    inputs = build_acceptance_inputs(summary, folds, transfers, [0.2, 0.4, 0.8])
    decision = assess_phase_a(inputs)
    assert inputs.transfer_reduction_vs_f1 == 0.10
    assert inputs.transfer_reduction_vs_r1 == 0.10
    assert decision.transfer_safety_path is True
    assert decision.proceed_to_phase_b is True


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"p1_mae_ratio_to_m0": 1.0500001}, False),
        ({"p1_r2_drop_from_m0": 0.0200001}, False),
        ({"g1_mae_ratio_to_p1": 1.0100001}, False),
        ({"g1_rmse_ratio_to_p1": 1.0300001}, False),
        ({"g1_r2_drop_from_p1": 0.0000001}, False),
    ],
)
def test_noninferiority_equalities_and_just_outside(changes, expected):
    payload = dict(
        p1_mae_ratio_to_m0=1.05,
        p1_r2_drop_from_m0=0.02,
        g1_mae_ratio_to_p1=0.99,
        g1_rmse_ratio_to_p1=1.03,
        g1_r2_drop_from_p1=0.0,
        g1_fold_wins=3,
        transfer_reduction_vs_f1=0.0,
        transfer_reduction_vs_r1=0.0,
        gate_median=0.2,
        gate_p95=0.8,
    )
    payload.update(changes)
    assert assess_phase_a(AcceptanceInputs(**payload)).proceed_to_phase_b is expected


def test_mean_path_accepts_exact_point_nine_nine_ratio():
    decision = assess_phase_a(
        AcceptanceInputs(
            p1_mae_ratio_to_m0=1.0,
            p1_r2_drop_from_m0=0.0,
            g1_mae_ratio_to_p1=0.99,
            g1_rmse_ratio_to_p1=1.0,
            g1_r2_drop_from_p1=0.0,
            g1_fold_wins=3,
            transfer_reduction_vs_f1=0.0,
            transfer_reduction_vs_r1=0.0,
            gate_median=0.2,
            gate_p95=0.8,
        )
    )
    assert decision.mean_improvement_path is True
    assert decision.proceed_to_phase_b is True


@pytest.mark.parametrize(
    ("median", "p95", "collapsed"),
    [
        (0.05, 0.20, False),
        (0.049999, 0.199999, True),
        (0.050001, 0.199999, False),
        (0.049999, 0.200001, False),
    ],
)
def test_collapse_boundaries_require_both_strictly_below(median, p95, collapsed):
    decision = assess_phase_a(
        AcceptanceInputs(
            p1_mae_ratio_to_m0=1.0,
            p1_r2_drop_from_m0=0.0,
            g1_mae_ratio_to_p1=1.0,
            g1_rmse_ratio_to_p1=1.0,
            g1_r2_drop_from_p1=0.0,
            g1_fold_wins=2,
            transfer_reduction_vs_f1=0.0,
            transfer_reduction_vs_r1=0.0,
            gate_median=median,
            gate_p95=p95,
        )
    )
    assert decision.gate_collapsed_without_benefit is collapsed


def test_machine_gate_has_no_subjective_override_parameter_or_field():
    assert tuple(inspect.signature(assess_phase_a).parameters) == ("inputs",)
    assert "subjective_override" not in AcceptanceInputs.__dataclass_fields__


def _acceptance_payload(**changes):
    payload = {
        "p1_mae_ratio_to_m0": 1.0,
        "p1_r2_drop_from_m0": 0.0,
        "g1_mae_ratio_to_p1": 0.99,
        "g1_rmse_ratio_to_p1": 1.0,
        "g1_r2_drop_from_p1": 0.0,
        "g1_fold_wins": 3,
        "transfer_reduction_vs_f1": 0.0,
        "transfer_reduction_vs_r1": 0.0,
        "gate_median": 0.2,
        "gate_p95": 0.8,
    }
    payload.update(changes)
    return AcceptanceInputs(**payload)


@pytest.mark.parametrize(
    ("field", "threshold"),
    [
        ("p1_mae_ratio_to_m0", 1.05),
        ("p1_r2_drop_from_m0", 0.02),
    ],
)
def test_process_credibility_accepts_equality_and_rejects_nearest_outside(field, threshold):
    at_limit = assess_phase_a(_acceptance_payload(**{field: threshold}))
    outside = assess_phase_a(
        _acceptance_payload(**{field: np.nextafter(threshold, np.inf)})
    )
    assert at_limit.process_expert_credible is True
    assert at_limit.proceed_to_phase_b is True
    assert outside.process_expert_credible is False
    assert outside.proceed_to_phase_b is False


@pytest.mark.parametrize(
    ("field", "threshold"),
    [
        ("g1_mae_ratio_to_p1", 1.01),
        ("g1_rmse_ratio_to_p1", 1.03),
        ("g1_r2_drop_from_p1", 0.0),
    ],
)
def test_g1_noninferiority_accepts_equality_and_rejects_nearest_outside(field, threshold):
    safety = {
        "g1_mae_ratio_to_p1": 1.0,
        "g1_fold_wins": 2,
        "transfer_reduction_vs_f1": 0.10,
        "transfer_reduction_vs_r1": 0.10,
    }
    at_limit = assess_phase_a(
        _acceptance_payload(**{**safety, field: threshold})
    )
    outside = assess_phase_a(
        _acceptance_payload(
            **{**safety, field: np.nextafter(threshold, np.inf)}
        )
    )
    assert at_limit.g1_noninferior is True
    assert at_limit.proceed_to_phase_b is True
    assert outside.g1_noninferior is False
    assert outside.proceed_to_phase_b is False


def test_mean_path_ratio_and_fold_win_boundaries_one_field_at_a_time():
    at_ratio = assess_phase_a(_acceptance_payload(g1_mae_ratio_to_p1=0.99))
    outside_ratio = assess_phase_a(
        _acceptance_payload(g1_mae_ratio_to_p1=np.nextafter(0.99, np.inf))
    )
    at_wins = assess_phase_a(_acceptance_payload(g1_fold_wins=3))
    outside_wins = assess_phase_a(_acceptance_payload(g1_fold_wins=2))
    assert (at_ratio.mean_improvement_path, at_ratio.proceed_to_phase_b) == (True, True)
    assert (outside_ratio.mean_improvement_path, outside_ratio.proceed_to_phase_b) == (False, False)
    assert (at_wins.mean_improvement_path, at_wins.proceed_to_phase_b) == (True, True)
    assert (outside_wins.mean_improvement_path, outside_wins.proceed_to_phase_b) == (False, False)


@pytest.mark.parametrize(
    "field", ("transfer_reduction_vs_f1", "transfer_reduction_vs_r1")
)
def test_each_safety_reduction_requires_exact_decimal_point_ten(field):
    safety = {
        "g1_mae_ratio_to_p1": 1.0,
        "g1_fold_wins": 2,
        "transfer_reduction_vs_f1": 0.10,
        "transfer_reduction_vs_r1": 0.10,
    }
    at_limit = assess_phase_a(_acceptance_payload(**safety))
    safety[field] = 0.0999999999995
    just_below = assess_phase_a(_acceptance_payload(**safety))
    assert (at_limit.transfer_safety_path, at_limit.proceed_to_phase_b) == (True, True)
    assert (just_below.transfer_safety_path, just_below.proceed_to_phase_b) == (False, False)


@pytest.mark.parametrize(
    ("field", "at_limit", "inside"),
    [
        ("gate_median", 0.05, np.nextafter(0.05, -np.inf)),
        ("gate_p95", 0.20, np.nextafter(0.20, -np.inf)),
    ],
)
def test_each_collapse_threshold_is_strict(field, at_limit, inside):
    no_benefit = {
        "g1_mae_ratio_to_p1": 1.0,
        "g1_fold_wins": 2,
        "gate_median": np.nextafter(0.05, -np.inf),
        "gate_p95": np.nextafter(0.20, -np.inf),
    }
    boundary = assess_phase_a(_acceptance_payload(**{**no_benefit, field: at_limit}))
    collapsed = assess_phase_a(_acceptance_payload(**{**no_benefit, field: inside}))
    assert boundary.gate_collapsed_without_benefit is False
    assert collapsed.gate_collapsed_without_benefit is True


@pytest.mark.parametrize(
    ("changes", "mean_path", "safety_path", "collapsed"),
    [
        ({"g1_mae_ratio_to_p1": 0.99, "g1_fold_wins": 3}, True, False, False),
        (
            {
                "g1_mae_ratio_to_p1": 1.0,
                "g1_fold_wins": 2,
                "transfer_reduction_vs_f1": 0.10,
                "transfer_reduction_vs_r1": 0.10,
            },
            False,
            True,
            False,
        ),
        ({"g1_mae_ratio_to_p1": 1.0, "g1_fold_wins": 2}, False, False, True),
    ],
)
def test_collapse_requires_no_mean_or_safety_benefit(changes, mean_path, safety_path, collapsed):
    decision = assess_phase_a(
        _acceptance_payload(gate_median=0.01, gate_p95=0.10, **changes)
    )
    assert decision.mean_improvement_path is mean_path
    assert decision.transfer_safety_path is safety_path
    assert decision.gate_collapsed_without_benefit is collapsed


@pytest.mark.parametrize(
    ("excess", "material_rate"),
    [(0.01, 0.0), (np.nextafter(0.01, np.inf), 1.0)],
)
def test_material_negative_transfer_margin_equality_and_nearest_above(excess, material_rate):
    result = negative_transfer(
        pd.DataFrame(
            {
                "group_id": ["g"],
                "p1_abs_error": [0.0],
                "candidate_abs_error": [excess],
                "sample_weight": [1.0],
            }
        ),
        material_margin_um=0.01,
    )
    assert result.material_rate == material_rate


def _phase_b_fixture(sample_count=586):
    sample_ids = [f"s{index:03d}" for index in range(sample_count)]
    manifest_rows = []
    fold_rows = []
    probability_rows = []
    mean_rows = []
    z90 = 1.6448536269514722
    z95 = 1.959963984540054
    for index, sample_id in enumerate(sample_ids):
        group_id = f"g{index // 2:03d}"
        fold = (index // 2) % 5
        target = 0.5 + index / 1000.0
        readings = (target - 0.1, target, target + 0.1)
        split_count = 2 if index % 2 == 0 else 1
        weight = 1.0 / split_count
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "version": "v3" if index % 2 == 0 else "v4",
                "ra_1": readings[0],
                "ra_2": readings[1],
                "ra_3": readings[2],
                "ra_mean": target,
                "sample_weight": weight,
                "split_count": split_count,
                "n_rpm": 4000.0 + 1000.0 * fold,
                "fz_mm_per_tooth": 0.03 + 0.01 * fold,
                "ap_mm": 0.5 + 0.25 * fold,
            }
        )
        fold_rows.append({"sample_id": sample_id, "group_id": group_id, "fold": fold})
        for seed_offset, seed in enumerate(PHASE_B_SEEDS):
            mu = target + 0.005 * seed_offset
            process_mean = mu - 0.02
            residual = 0.04
            gate = 0.5
            for scale_model in SCALE_MODELS:
                sigma = 0.2 if scale_model == "heteroscedastic" else 0.25
                probability_rows.append(
                    {
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "version": manifest_rows[-1]["version"],
                        "fold": fold,
                        "seed": seed,
                        "scale_model": scale_model,
                        "target_mean": target,
                        "ra_1": readings[0],
                        "ra_2": readings[1],
                        "ra_3": readings[2],
                        "sample_weight": weight,
                        "mu": mu,
                        "sigma": sigma,
                        "gate": gate,
                        "correction": mu - process_mean,
                        "raw_lower_90": mu - z90 * sigma,
                        "raw_upper_90": mu + z90 * sigma,
                        "raw_lower_95": mu - z95 * sigma,
                        "raw_upper_95": mu + z95 * sigma,
                        "conformal_q_90": 2.0,
                        "conformal_lower_90": mu - 2.0 * sigma,
                        "conformal_upper_90": mu + 2.0 * sigma,
                        "conformal_q_95": 2.5,
                        "conformal_lower_95": mu - 2.5 * sigma,
                        "conformal_upper_95": mu + 2.5 * sigma,
                    }
                )
            for model in MEAN_MODELS:
                if model == "P1":
                    prediction, model_residual, model_gate, correction = (
                        process_mean,
                        residual,
                        0.0,
                        0.0,
                    )
                elif model == "R1":
                    prediction, model_residual, model_gate, correction = (
                        process_mean + residual,
                        residual,
                        1.0,
                        residual,
                    )
                else:
                    prediction, model_residual, model_gate, correction = (
                        mu,
                        residual,
                        gate,
                        mu - process_mean,
                    )
                mean_rows.append(
                    {
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "version": manifest_rows[-1]["version"],
                        "fold": fold,
                        "seed": seed,
                        "model": model,
                        "target_mean": target,
                        "prediction": prediction,
                        "sample_weight": weight,
                        "process_mean": process_mean,
                        "residual": model_residual,
                        "gate": model_gate,
                        "correction": correction,
                    }
                )
    return (
        pd.DataFrame(manifest_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(probability_rows, columns=PHASE_B_PREDICTION_COLUMNS),
        pd.DataFrame(mean_rows, columns=PHASE_B_MEAN_COLUMNS),
    )


def _phase_b_calibration_fixture(manifest, folds):
    sample_folds = folds.set_index("sample_id")["fold"]
    group_metadata = (
        manifest.assign(fold=manifest["sample_id"].map(sample_folds))
        .groupby("group_id", sort=True, observed=True)
        .agg(fold=("fold", "first"), region_count=("sample_id", "size"))
    )
    assert (
        manifest.assign(fold=manifest["sample_id"].map(sample_folds))
        .groupby("group_id", observed=True)["fold"]
        .nunique()
        .eq(1)
        .all()
    )
    score_rows = []
    quantile_rows = []
    for fold in range(5):
        outer_groups = group_metadata.loc[group_metadata["fold"] != fold]
        group_count = len(outer_groups)
        order_90 = math.ceil((group_count + 1) * 0.90)
        order_95 = math.ceil((group_count + 1) * 0.95)
        scores = np.full(group_count, 2.5, dtype=np.float64)
        scores[: order_90 - 1] = 1.0
        scores[order_90 - 1 : order_95 - 1] = 2.0
        for seed in PHASE_B_SEEDS:
            for scale_model in SCALE_MODELS:
                for index, ((group_id, metadata), score) in enumerate(
                    zip(outer_groups.iterrows(), scores, strict=True)
                ):
                    score_rows.append(
                        {
                            "group_id": group_id,
                            "outer_fold": fold,
                            "inner_fold": index % 4,
                            "seed": seed,
                            "scale_model": scale_model,
                            "score": score,
                            "region_count": int(metadata["region_count"]),
                            "reading_count": 3 * int(metadata["region_count"]),
                        }
                    )
                quantile_rows.extend(
                    [
                        {
                            "fold": fold,
                            "seed": seed,
                            "scale_model": scale_model,
                            "alpha": 0.10,
                            "group_count": group_count,
                            "order_index": order_90,
                            "quantile": 2.0,
                        },
                        {
                            "fold": fold,
                            "seed": seed,
                            "scale_model": scale_model,
                            "alpha": 0.05,
                            "group_count": group_count,
                            "order_index": order_95,
                            "quantile": 2.5,
                        },
                    ]
                )
    return pd.DataFrame(score_rows), pd.DataFrame(quantile_rows)


def test_phase_b_cartesian_validation_enforces_registered_probability_and_mean_rows():
    manifest, folds, probability, mean = _phase_b_fixture()
    scores, quantiles = _phase_b_calibration_fixture(manifest, folds)

    validated = validate_phase_b_prediction_cartesian(
        probability,
        manifest=manifest,
        folds=folds,
        mean_predictions=mean,
        calibration_scores=scores,
        calibration_quantiles=quantiles,
    )
    assert len(validated) == 3516
    assert validated["sample_id"].iloc[0] == "s000"

    bad_frames = []
    bad_frames.append(probability.iloc[:-1].copy())
    duplicate = probability.copy()
    duplicate.iloc[-1] = duplicate.iloc[0]
    bad_frames.append(duplicate)
    wrong_group = probability.copy()
    wrong_group.loc[0, "group_id"] = "swapped"
    bad_frames.append(wrong_group)
    wrong_reading = probability.copy()
    wrong_reading.loc[0, "ra_1"] += 0.01
    bad_frames.append(wrong_reading)
    wrong_seed = probability.copy()
    wrong_seed["seed"] = wrong_seed["seed"].astype(object)
    wrong_seed.loc[0, "seed"] = "20260723.0"
    bad_frames.append(wrong_seed)
    zero_sigma = probability.copy()
    zero_sigma.loc[0, "sigma"] = 0.0
    bad_frames.append(zero_sigma)
    quantile_drift = probability.copy()
    quantile_drift.loc[0, "conformal_q_90"] = 2.01
    quantile_drift.loc[0, "conformal_lower_90"] = (
        quantile_drift.loc[0, "mu"] - 2.01 * quantile_drift.loc[0, "sigma"]
    )
    quantile_drift.loc[0, "conformal_upper_90"] = (
        quantile_drift.loc[0, "mu"] + 2.01 * quantile_drift.loc[0, "sigma"]
    )
    bad_frames.append(quantile_drift)

    for bad in bad_frames:
        with pytest.raises(ValueError):
            validate_phase_b_prediction_cartesian(
                bad,
                manifest=manifest,
                folds=folds,
                mean_predictions=mean,
                calibration_scores=scores,
                calibration_quantiles=quantiles,
            )

    bad_mean = mean.copy()
    bad_mean.loc[bad_mean["model"] == "G1", "gate"] = 1.5
    with pytest.raises(ValueError, match="mean"):
        validate_phase_b_prediction_cartesian(
            probability,
            manifest=manifest,
            folds=folds,
            mean_predictions=bad_mean,
            calibration_scores=scores,
            calibration_quantiles=quantiles,
        )


def test_phase_b_validator_accepts_real_noninvariant_swap_correction_with_float32_rounding():
    class ConstantProcess(nn.Module):
        def forward(self, process):
            return torch.full_like(process[:, 0], 0.1)

    class OrientationResidual(nn.Module):
        def forward(self, spectrum, window_mask):
            del window_mask
            orientation = spectrum[:, :, 0].mean(dim=(1, 2))
            residual = 4.0 * orientation - 2.0
            embedding = torch.zeros(
                orientation.shape[0], 64, dtype=orientation.dtype
            )
            embedding[:, 0] = orientation
            return ModelOutput(
                prediction=residual,
                residual=residual,
                embedding=embedding,
            )

    class OrientationGate(nn.Module):
        def forward(self, gate_input):
            return gate_input[:, 9:10]

    model = SelectiveGatedModel()
    model.process_expert = ConstantProcess()
    model.residual_expert = OrientationResidual()
    model.gate = OrientationGate()
    model.eval()
    batch = {
        "spectrum": torch.zeros(1, 1, 3, 361, dtype=torch.float32),
        "window_mask": torch.ones(1, 1, dtype=torch.bool),
        "process": torch.zeros(1, 9, dtype=torch.float32),
        "quality": torch.zeros(1, 7, dtype=torch.float32),
        "target": torch.zeros(1, dtype=torch.float32),
        "sample_weight": torch.ones(1, dtype=torch.float32),
        "sample_id": ["s000"],
        "group_id": ["g000"],
    }
    batch["spectrum"][:, :, 0] = 1.0

    original = model(
        batch["spectrum"], batch["window_mask"], batch["process"], batch["quality"]
    )
    swapped_batch = swap_horizontal(batch)
    swapped = model(
        swapped_batch["spectrum"],
        swapped_batch["window_mask"],
        swapped_batch["process"],
        swapped_batch["quality"],
    )
    averaged = average_swap_predictions(model, batch)
    assert averaged.process_mean is not None
    assert averaged.residual is not None
    assert averaged.gate is not None
    actual_correction = (
        original.gate * original.residual + swapped.gate * swapped.residual
    ) / 2.0
    descriptive_product = averaged.gate * averaged.residual
    torch.testing.assert_close(actual_correction, torch.tensor([1.0]))
    torch.testing.assert_close(descriptive_product, torch.tensor([0.0]))
    assert float(averaged.prediction) != float(averaged.process_mean) + float(
        actual_correction
    )

    manifest, folds, probability, mean = _phase_b_fixture()
    probability_mask = (probability["sample_id"] == "s000") & (
        probability["seed"] == PHASE_B_SEEDS[0]
    )
    mu = float(averaged.prediction)
    process_mean = float(averaged.process_mean)
    gate = float(averaged.gate)
    residual = float(averaged.residual)
    correction = float(actual_correction)
    probability.loc[
        probability_mask, ["mu", "gate", "correction"]
    ] = [mu, gate, correction]
    for suffix, multiplier in (("90", 1.6448536269514722), ("95", 1.959963984540054)):
        sigma = probability.loc[probability_mask, "sigma"]
        probability.loc[probability_mask, f"raw_lower_{suffix}"] = mu - multiplier * sigma
        probability.loc[probability_mask, f"raw_upper_{suffix}"] = mu + multiplier * sigma
        quantile = {"90": 2.0, "95": 2.5}[suffix]
        probability.loc[probability_mask, f"conformal_lower_{suffix}"] = mu - quantile * sigma
        probability.loc[probability_mask, f"conformal_upper_{suffix}"] = mu + quantile * sigma

    mean_mask = (mean["sample_id"] == "s000") & (
        mean["seed"] == PHASE_B_SEEDS[0]
    )
    mean.loc[mean_mask, ["process_mean", "residual"]] = [process_mean, residual]
    for model_name, prediction, model_gate, model_correction in (
        ("P1", process_mean, 0.0, 0.0),
        ("R1", process_mean + residual, 1.0, residual),
        ("G1", mu, gate, correction),
    ):
        row_mask = mean_mask & (mean["model"] == model_name)
        mean.loc[row_mask, ["prediction", "gate"]] = [prediction, model_gate]
        if "correction" in mean.columns:
            mean.loc[row_mask, "correction"] = model_correction

    scores, quantiles = _phase_b_calibration_fixture(manifest, folds)
    validate_phase_b_prediction_cartesian(
        probability,
        manifest=manifest,
        folds=folds,
        mean_predictions=mean,
        calibration_scores=scores,
        calibration_quantiles=quantiles,
    )


def test_phase_b_validator_consumes_scores_and_recomputes_persisted_quantiles():
    manifest, folds, probability, mean = _phase_b_fixture()
    scores, quantiles = _phase_b_calibration_fixture(manifest, folds)

    validated = validate_phase_b_prediction_cartesian(
        probability,
        manifest=manifest,
        folds=folds,
        mean_predictions=mean,
        calibration_scores=scores,
        calibration_quantiles=quantiles,
    )
    assert len(validated) == 3516

    stale_scores = scores.copy()
    stale_scores.loc[
        (stale_scores["outer_fold"] == 0)
        & (stale_scores["seed"] == PHASE_B_SEEDS[0])
        & (stale_scores["scale_model"] == SCALE_MODELS[0]),
        "score",
    ] += 1.0
    with pytest.raises(ValueError, match="quantile"):
        validate_phase_b_prediction_cartesian(
            probability,
            manifest=manifest,
            folds=folds,
            mean_predictions=mean,
            calibration_scores=stale_scores,
            calibration_quantiles=quantiles,
        )

    forged_probability = probability.copy()
    forged_quantiles = quantiles.copy()
    quantile_mask = (
        (forged_quantiles["fold"] == 0)
        & (forged_quantiles["seed"] == PHASE_B_SEEDS[0])
        & (forged_quantiles["scale_model"] == SCALE_MODELS[0])
        & (forged_quantiles["alpha"] == 0.10)
    )
    forged_quantiles.loc[quantile_mask, "quantile"] = 2.25
    prediction_mask = (
        (forged_probability["fold"] == 0)
        & (forged_probability["seed"] == PHASE_B_SEEDS[0])
        & (forged_probability["scale_model"] == SCALE_MODELS[0])
    )
    forged_probability.loc[prediction_mask, "conformal_q_90"] = 2.25
    forged_probability.loc[prediction_mask, "conformal_lower_90"] = (
        forged_probability.loc[prediction_mask, "mu"]
        - 2.25 * forged_probability.loc[prediction_mask, "sigma"]
    )
    forged_probability.loc[prediction_mask, "conformal_upper_90"] = (
        forged_probability.loc[prediction_mask, "mu"]
        + 2.25 * forged_probability.loc[prediction_mask, "sigma"]
    )
    with pytest.raises(ValueError, match="quantile"):
        validate_phase_b_prediction_cartesian(
            forged_probability,
            manifest=manifest,
            folds=folds,
            mean_predictions=mean,
            calibration_scores=scores,
            calibration_quantiles=forged_quantiles,
        )


@pytest.mark.parametrize(
    ("model", "column", "delta", "message"),
    [
        ("P1", "prediction", 0.1, "P1"),
        ("R1", "prediction", 0.1, "R1"),
        ("G1", "prediction", 0.1, "G1"),
        ("P1", "process_mean", 0.1, "shared process"),
        ("P1", "residual", 0.1, "shared residual"),
        ("P1", "gate", 0.1, "gate semantics"),
        ("R1", "gate", -0.1, "gate semantics"),
        ("P1", "correction", 0.1, "P1/R1 correction"),
        ("R1", "correction", 0.1, "P1/R1 correction"),
        ("G1", "correction", 0.1, "G1"),
    ],
)
def test_phase_b_mean_arithmetic_rejects_targeted_corruption(
    model, column, delta, message
):
    manifest, folds, probability, mean = _phase_b_fixture()
    scores, quantiles = _phase_b_calibration_fixture(manifest, folds)
    corrupted = mean.copy()
    row = corrupted.index[
        (corrupted["sample_id"] == "s000")
        & (corrupted["seed"] == PHASE_B_SEEDS[0])
        & (corrupted["model"] == model)
    ][0]
    corrupted.loc[row, column] += delta

    with pytest.raises(ValueError, match=message):
        validate_phase_b_prediction_cartesian(
            probability,
            manifest=manifest,
            folds=folds,
            mean_predictions=corrupted,
            calibration_scores=scores,
            calibration_quantiles=quantiles,
        )


def test_phase_b_probability_all_seed_rejects_incomplete_cartesian_first():
    _, _, probability, _ = _phase_b_fixture()

    for incomplete in (
        probability.loc[probability["seed"] != PHASE_B_SEEDS[-1]],
        probability.iloc[:-1],
    ):
        with pytest.raises(ValueError, match="3516|Cartesian|three registered seeds"):
            probability_metric_table(incomplete)


def test_phase_b_mean_evaluation_chain_covers_registered_outputs():
    manifest, _, _, mean = _phase_b_fixture()
    quality = pd.DataFrame(
        {
            "sample_id": manifest["sample_id"],
            **{
                f"quality_{index}": (
                    np.arange(len(manifest), dtype=np.float64) % (index + 4)
                )
                for index in range(7)
            },
        }
    )

    metrics = evaluation.phase_b_mean_metric_table(mean)
    seed_zero = metrics.loc[
        (metrics["aggregation"] == "seed")
        & (metrics["seed"] == PHASE_B_SEEDS[0])
    ].set_index(["model", "metric"])["value"]
    assert set(metrics["metric"]) == {"mean_mae", "mean_rmse", "mean_r2"}
    assert set(metrics["model"]) == set(MEAN_MODELS)
    assert set(metrics["aggregation"]) == {"fold", "seed", "all_seed"}
    assert seed_zero.loc[("P1", "mean_mae")] == pytest.approx(0.02)
    assert seed_zero.loc[("P1", "mean_rmse")] == pytest.approx(0.02)
    assert seed_zero.loc[("G1", "mean_r2")] == pytest.approx(1.0)

    transfers = evaluation.phase_b_negative_transfer_table(mean)
    assert set(transfers["candidate"]) == {"R1", "G1"}
    assert set(transfers["aggregation"]) == {"seed", "all_seed"}
    g1_seed_zero = transfers.loc[
        (transfers["aggregation"] == "seed")
        & (transfers["seed"] == PHASE_B_SEEDS[0])
        & (transfers["candidate"] == "G1")
    ].iloc[0]
    assert g1_seed_zero["raw_rate"] == 0.0
    assert g1_seed_zero["material_rate"] == 0.0

    bootstraps = evaluation.phase_b_mean_bootstrap_table(mean)
    assert set(bootstraps["candidate"]) == {"R1", "G1"}
    assert set(bootstraps["repetitions"]) == {10_000}
    assert set(bootstraps["seed"]) == {20260723}
    assert set(bootstraps["resampling_unit"]) == {"group_id"}
    assert bootstraps.loc[
        bootstraps["candidate"] == "G1", "point_estimate"
    ].item() > 0.0
    with pytest.raises(ValueError, match="exactly 10000"):
        evaluation.paired_phase_b_mean_bootstrap(
            mean, candidate="G1", repetitions=9999
        )

    gates = evaluation.phase_b_gate_statistics(mean, manifest, quality)
    assert set(gates["dimension"]) == {
        "overall",
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
        *(f"quality_{index}" for index in range(7)),
    }
    assert {
        "prediction_count",
        "sample_count",
        "weight_sum",
        "gate_mean",
        "gate_weighted_mean",
        "gate_p05",
        "gate_median",
        "gate_p95",
    }.issubset(gates.columns)


def test_probability_table_contains_every_registered_measure():
    _, _, probability, _ = _phase_b_fixture()
    table = probability_metric_table(probability)
    assert set(table["metric"]) == {
        "mean_mae",
        "mean_rmse",
        "mean_r2",
        "gaussian_nll",
        "gaussian_crps",
        "single_reading_coverage",
        "simultaneous_group_coverage",
        "mean_interval_width",
        "winkler_score",
    }
    assert set(table["interval_type"].dropna()) == {"raw", "conformal"}
    assert set(table["nominal_coverage"].dropna()) == {0.90, 0.95}
    assert set(table["aggregation"]) == {"fold", "seed", "all_seed"}


def test_probability_all_seed_metrics_average_seed_replicates():
    _, _, probability, _ = _phase_b_fixture()
    table = probability_metric_table(probability)
    seed_rows = table.loc[
        (table["aggregation"] == "seed")
        & (table["scale_model"] == "heteroscedastic")
        & (table["metric"] == "gaussian_nll")
    ]
    all_seed = table.loc[
        (table["aggregation"] == "all_seed")
        & (table["scale_model"] == "heteroscedastic")
        & (table["metric"] == "gaussian_nll"),
        "value",
    ].item()
    assert all_seed == pytest.approx(seed_rows["value"].mean())


def test_seed_uncertainty_is_descriptive_and_never_builds_ensemble_intervals():
    _, _, probability, _ = _phase_b_fixture(sample_count=2)
    summary = seed_uncertainty_summary(probability)

    row = summary.loc[
        (summary["sample_id"] == "s000")
        & (summary["scale_model"] == "heteroscedastic")
    ].iloc[0]
    assert row["mu_mean"] == pytest.approx(0.505)
    assert row["aleatoric_variance_mean"] == pytest.approx(0.04)
    assert row["seed_prediction_variance"] == pytest.approx(1.0 / 60000.0)
    assert row["seed_spread_interpretation"] == "descriptive_seed_instability"
    assert not any("total_variance" in column for column in summary.columns)
    assert not any("ensemble" in column for column in summary.columns)

    missing_seed = probability.loc[probability["seed"] != PHASE_B_SEEDS[-1]]
    with pytest.raises(ValueError, match="three registered seeds"):
        seed_uncertainty_summary(missing_seed)


@pytest.mark.parametrize(
    ("metric", "interval_type", "nominal_coverage"),
    [
        ("gaussian_nll", None, None),
        ("gaussian_crps", None, None),
        ("mean_interval_width", "raw", 0.90),
        ("mean_interval_width", "conformal", 0.95),
        ("winkler_score", "raw", 0.90),
        ("winkler_score", "conformal", 0.95),
    ],
)
def test_probability_bootstrap_is_exact_group_paired_and_reproducible(
    metric, interval_type, nominal_coverage
):
    _, _, probability, _ = _phase_b_fixture(sample_count=10)
    kwargs = {
        "metric": metric,
        "interval_type": interval_type,
        "nominal_coverage": nominal_coverage,
    }
    first = paired_probability_bootstrap(probability, **kwargs)
    second = paired_probability_bootstrap(probability, **kwargs)
    expected_point = {
        ("gaussian_nll", None, None): -0.19295605131420973,
        ("gaussian_crps", None, None): -0.009142278913736972,
        ("mean_interval_width", "raw", 0.90): -0.16448536269514713,
        ("mean_interval_width", "conformal", 0.95): -0.25,
        ("winkler_score", "raw", 0.90): -0.16448536269514713,
        ("winkler_score", "conformal", 0.95): -0.25,
    }[(metric, interval_type, nominal_coverage)]

    assert first == second
    assert first.point_estimate == pytest.approx(expected_point)
    assert first.repetitions == BOOTSTRAP_REPETITIONS == 10_000
    assert first.seed == 20260723
    assert first.resampling_unit == "group_id"
    assert np.isfinite([first.point_estimate, first.lower, first.upper]).all()

    with pytest.raises(ValueError, match="exactly 10000"):
        paired_probability_bootstrap(
            probability, metric="gaussian_nll", repetitions=9999
        )
    with pytest.raises(ValueError, match="seed 20260723"):
        paired_probability_bootstrap(
            probability, metric="gaussian_nll", seed=20260724
        )


def test_probability_bootstrap_rejects_incomplete_or_ambiguous_cartesian_rows():
    _, _, probability, _ = _phase_b_fixture(sample_count=4)
    duplicate = pd.concat([probability, probability.iloc[[0]]], ignore_index=True)
    wrong_group = probability.copy()
    wrong_group.loc[0, "group_id"] = "mismatched"
    split_seed_group = probability.copy()
    split_seed_group.loc[
        (split_seed_group["sample_id"] == "s000")
        & (split_seed_group["seed"] == PHASE_B_SEEDS[0]),
        "group_id",
    ] = "mismatched"

    for malformed in (
        probability.iloc[:-1],
        duplicate,
        wrong_group,
        split_seed_group,
    ):
        with pytest.raises(ValueError, match="registered|paired"):
            paired_probability_bootstrap(malformed, metric="gaussian_nll")

    with pytest.raises(ValueError, match="interval type and nominal coverage"):
        paired_probability_bootstrap(probability, metric="winkler_score")
    with pytest.raises(ValueError, match="registered probability bootstrap metric"):
        paired_probability_bootstrap(probability, metric="mean_mae")


def test_phase_b_probability_dataclasses_expose_only_registered_claim_fields():
    assert tuple(ProbabilityMetrics.__dataclass_fields__) == (
        "mean_mae",
        "mean_rmse",
        "mean_r2",
        "gaussian_nll",
        "gaussian_crps",
        "single_reading_coverage",
        "simultaneous_group_coverage",
        "mean_interval_width",
        "winkler_score",
    )
    assert tuple(PhaseBPaperDecision.__dataclass_fields__) == (
        "emphasize_heteroscedasticity",
        "retain_group_conformal",
        "practical_width_threshold_registered",
        "coverage_claim_scope",
        "sigma_interpretation",
        "seed_spread_interpretation",
        "reasons",
    )


def _phase_b_claim_metrics(hetero_90, hetero_95, homo_90=1.0, homo_95=1.0):
    return pd.DataFrame(
        [
            {
                "aggregation": "all_seed",
                "scale_model": scale_model,
                "interval_type": "conformal",
                "nominal_coverage": coverage,
                "metric": "winkler_score",
                "value": value,
            }
            for scale_model, coverage, value in (
                ("heteroscedastic", 0.90, hetero_90),
                ("heteroscedastic", 0.95, hetero_95),
                ("homoscedastic", 0.90, homo_90),
                ("homoscedastic", 0.95, homo_95),
            )
        ]
    )


def test_no_winkler_gain_disables_heteroscedastic_emphasis():
    decision = assess_phase_b_claims(_phase_b_claim_metrics(1.01, 1.01))
    assert decision.emphasize_heteroscedasticity is False
    assert decision.retain_group_conformal is True
    assert decision.practical_width_threshold_registered is False
    assert decision.coverage_claim_scope == "exchangeable_new_groups_of_existing_type"
    assert decision.sigma_interpretation == (
        "conditional_predictive_dispersion_combining_repeat_variation_"
        "and_unmodeled_error"
    )
    assert decision.seed_spread_interpretation == (
        "descriptive_model_instability_not_posterior_epistemic_variance"
    )
    assert decision.reasons == (
        "heteroscedastic_conformal_winkler_rule_not_met",
        "group_conformal_calibration_retained",
        "no_numeric_practical_width_threshold_preregistered",
    )


def test_heteroscedastic_emphasis_requires_strict_gain_at_both_coverages():
    emphasized = assess_phase_b_claims(_phase_b_claim_metrics(0.99, 0.99))
    assert emphasized.emphasize_heteroscedasticity is True
    assert emphasized.reasons == (
        "heteroscedastic_conformal_winkler_rule_met_at_90_and_95",
        "group_conformal_calibration_retained",
        "no_numeric_practical_width_threshold_preregistered",
    )
    assert assess_phase_b_claims(
        _phase_b_claim_metrics(0.99, 1.0)
    ).emphasize_heteroscedasticity is False

    valid = _phase_b_claim_metrics(0.99, 0.99)
    malformed_frames = [
        valid.iloc[:-1],
        pd.concat([valid, valid.iloc[[0]]], ignore_index=True),
        valid.assign(value=[0.99, np.nan, 1.0, 1.0]),
    ]
    for malformed in malformed_frames:
        with pytest.raises(ValueError, match="exactly one|finite"):
            assess_phase_b_claims(malformed)
