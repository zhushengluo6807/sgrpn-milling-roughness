import numpy as np
import pandas as pd

from roughness.scheme1.classic import (
    ClassicSelection,
    build_classic_feature_sets,
    run_classic_outer_cv,
    select_ridge_candidate,
)


def test_classic_feature_sets_keep_models_and_geometry_candidates_separate():
    columns = [
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
        "Z_rms_mean",
        "H_sym_mean_rms_mean",
        "A_X_rms_mean",
        "A_Y_rms_mean",
        "B_X_rms_mean",
        "B_Y_rms_mean",
        "ra_geo_bottom_re_0p10_um",
        "ra_geo_bottom_re_0p20_um",
    ]
    frame = pd.DataFrame(columns=columns)

    sets = build_classic_feature_sets(frame)

    assert set(sets) == {
        "P1",
        "P2",
        "P3A",
        "P3B",
        "P4_none",
        "P4_re_0p10",
        "P4_re_0p20",
    }
    assert "ra_geo_bottom_re_0p10_um" not in sets["P2"]
    assert sets["P4_re_0p10"][-1] == "ra_geo_bottom_re_0p10_um"


def test_ridge_candidate_selection_uses_grouped_inner_predictions():
    rows = []
    for index in range(20):
        rows.append(
            {
                "sample_id": f"s{index}",
                "group_id": f"g{index}",
                "signal": float(index),
                "noise": float((index * 7) % 11),
                "ra_mean": 1.0 + 2.0 * index,
                "sample_weight": 1.0,
            }
        )
    frame = pd.DataFrame(rows)

    selection = select_ridge_candidate(
        frame,
        candidates={"informative": ["signal"], "noise": ["noise"]},
        outer_fold=2,
        seed=20260723,
        inner_splits=4,
        alphas=(0.01, 1.0),
    )

    assert isinstance(selection, ClassicSelection)
    assert selection.outer_fold == 2
    assert selection.model_name == "informative"
    assert selection.inner_weighted_mae < 0.1


def test_classic_outer_cv_produces_one_oof_prediction_per_model_and_segment():
    rows = []
    fold_rows = []
    for index in range(12):
        fz = 0.03 + index * 0.005
        rows.append(
            {
                "sample_id": f"s{index}",
                "group_id": f"g{index}",
                "n_rpm": 4000.0 + index * 100,
                "fz_mm_per_tooth": fz,
                "ap_mm": 0.5 + (index % 3) * 0.5,
                "ra_mean": 0.4 + 3.0 * fz + 0.00001 * (4000 + index * 100),
                "sample_weight": 1.0,
                "Z_rms_mean": float(index),
                "H_sym_mean_rms_mean": float(index) / 2,
                "A_X_rms_mean": float(index),
                "A_Y_rms_mean": float(11 - index),
                "B_X_rms_mean": float(11 - index),
                "B_Y_rms_mean": float(index),
                "ra_geo_bottom_re_0p20_um": 0.1 + fz,
            }
        )
        fold_rows.append(
            {
                "sample_id": f"s{index}",
                "group_id": f"g{index}",
                "fold": index % 3,
            }
        )

    predictions, metrics, selections = run_classic_outer_cv(
        pd.DataFrame(rows),
        pd.DataFrame(fold_rows),
        seed=20260723,
        inner_splits=3,
        alphas=(1.0,),
    )

    assert set(predictions["model"]) == {"M0", "P1", "P2", "P3", "P4"}
    assert len(predictions) == 12 * 5
    assert not predictions.duplicated(["sample_id", "model"]).any()
    assert len(metrics) == 3 * 5
    assert set(selections["stage"]) == {"P1", "P2", "P3", "P4"}
    assert np.isfinite(predictions["y_pred"]).all()
