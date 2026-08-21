import os
import time
from collections.abc import Callable

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .metrics import regression_metrics


FEATURES = {
    "dummy_mean": ["n_rpm"],
    "linear_process": ["n_rpm", "fz_mm_per_tooth", "ap_mm"],
    "quadratic_process": ["n_rpm", "fz_mm_per_tooth", "ap_mm"],
    "hist_process": ["n_rpm", "fz_mm_per_tooth", "ap_mm"],
    "calibrated_geometry": ["ra_geo_um"],
    "process_plus_geometry": [
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
        "ra_geo_um",
    ],
}


def model_factories(seed: int) -> dict[str, Callable]:
    return {
        "dummy_mean": lambda: DummyRegressor(strategy="mean"),
        "linear_process": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=1.0)
        ),
        "quadratic_process": lambda: make_pipeline(
            PolynomialFeatures(2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "hist_process": lambda: HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "calibrated_geometry": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=1.0)
        ),
        "process_plus_geometry": lambda: HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def _fit_with_weight(model, features, target, sample_weight) -> None:
    if hasattr(model, "steps"):
        final_name = model.steps[-1][0]
        model.fit(
            features,
            target,
            **{f"{final_name}__sample_weight": sample_weight},
        )
    else:
        model.fit(features, target, sample_weight=sample_weight)


def cross_validated_baselines(
    manifest: pd.DataFrame, folds: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = manifest.merge(
        folds[["sample_id", "fold"]],
        on="sample_id",
        validate="one_to_one",
    )
    predictions = []
    fold_metrics = []

    for model_name, factory in model_factories(seed).items():
        feature_names = FEATURES[model_name]
        for fold in sorted(data["fold"].unique()):
            train = data[data["fold"] != fold]
            valid = data[data["fold"] == fold]
            model = factory()

            fit_start = time.perf_counter()
            _fit_with_weight(
                model,
                train[feature_names],
                train["ra_mean"],
                train["sample_weight"],
            )
            fit_seconds = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            predicted = model.predict(valid[feature_names])
            predict_seconds = time.perf_counter() - predict_start

            segment_metrics = regression_metrics(valid["ra_mean"], predicted)
            weighted_metrics = {
                f"weighted_{key}": value
                for key, value in regression_metrics(
                    valid["ra_mean"], predicted, valid["sample_weight"]
                ).items()
            }
            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    **segment_metrics,
                    **weighted_metrics,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
            )
            predictions.extend(
                {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "model": model_name,
                    "fold": int(fold),
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "sample_weight": float(weight),
                }
                for sample_id, group_id, y_true, y_pred, weight in zip(
                    valid["sample_id"],
                    valid["group_id"],
                    valid["ra_mean"],
                    predicted,
                    valid["sample_weight"],
                    strict=True,
                )
            )

    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(fold_metrics)
    expected = len(data) * len(FEATURES)
    if (
        len(prediction_frame) != expected
        or prediction_frame.duplicated(["sample_id", "model"]).any()
    ):
        raise AssertionError("Incomplete or duplicate out-of-fold predictions")
    return prediction_frame, metric_frame
