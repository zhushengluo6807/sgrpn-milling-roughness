import numpy as np
import pandas as pd
import pytest
import inspect

from roughness.sgrpn.evaluation import (
    AcceptanceInputs,
    assess_phase_a,
    build_acceptance_inputs,
    negative_transfer,
    paired_group_bootstrap,
    validate_prediction_cartesian,
    weighted_metrics,
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
    assert inputs.transfer_reduction_vs_f1 == pytest.approx(0.10)
    assert inputs.transfer_reduction_vs_r1 == pytest.approx(0.10)
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
