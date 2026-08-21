import numpy as np
import pandas as pd
import warnings

from roughness.baselines import cross_validated_baselines, model_factories
from roughness.splits import make_group_folds


def test_expected_baselines_exist():
    names = set(model_factories(20260723))

    assert names == {
        "dummy_mean",
        "linear_process",
        "quadratic_process",
        "hist_process",
        "calibrated_geometry",
        "process_plus_geometry",
    }


def test_cross_validated_baselines_produce_one_oof_prediction_per_model():
    rows = []
    for group_index in range(10):
        for segment_index in range(2):
            fz = 0.03 + 0.01 * group_index
            rows.append(
                {
                    "sample_id": f"g{group_index}_s{segment_index}",
                    "group_id": f"g{group_index}",
                    "n_rpm": 4000 + 250 * group_index,
                    "fz_mm_per_tooth": fz,
                    "ap_mm": 0.5 + 0.1 * (group_index % 4),
                    "ra_geo_um": fz**2 / (32 * 5) * 1000,
                    "ra_mean": 0.5 + 4 * fz + 0.01 * segment_index,
                    "sample_weight": 0.5,
                }
            )
    manifest = pd.DataFrame(rows)
    folds = make_group_folds(manifest, n_splits=5, seed=20260723)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predictions, fold_metrics = cross_validated_baselines(
            manifest, folds, seed=20260723
        )

    assert len(predictions) == len(manifest) * len(model_factories(20260723))
    assert not predictions.duplicated(["sample_id", "model"]).any()
    assert set(fold_metrics.columns) >= {
        "mae",
        "weighted_mae",
        "fit_seconds",
        "predict_seconds",
    }
    assert np.isfinite(fold_metrics.select_dtypes(include="number")).all().all()
