from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from roughness.baselines import model_factories
from roughness.metrics import regression_metrics

from .crossfit import (
    cross_fitted_predictions,
    fit_with_sample_weight,
    make_group_inner_splits,
)


PROCESS_FEATURES = ["n_rpm", "fz_mm_per_tooth", "ap_mm"]


@dataclass(frozen=True)
class ClassicSelection:
    outer_fold: int
    xy_mode: Literal["A", "B", "symmetric"]
    re_mm: float | None
    model_name: str
    alpha: float
    inner_weighted_mae: float


def build_classic_feature_sets(
    frame: pd.DataFrame,
) -> dict[str, list[str]]:
    missing_process = sorted(set(PROCESS_FEATURES) - set(frame.columns))
    if missing_process:
        raise ValueError(f"Missing process features: {missing_process}")
    z_features = sorted(
        column for column in frame.columns if column.startswith("Z_")
    )
    symmetric = sorted(
        column for column in frame.columns if column.startswith("H_sym_")
    )
    a_features = sorted(
        column for column in frame.columns if column.startswith("A_")
    )
    b_features = sorted(
        column for column in frame.columns if column.startswith("B_")
    )
    geometry = sorted(
        column
        for column in frame.columns
        if column.startswith("ra_geo_bottom_re_")
    )
    p1 = [*PROCESS_FEATURES, *z_features]
    p2 = [*p1, *symmetric]
    result = {
        "P1": p1,
        "P2": p2,
        "P3A": [*p2, *a_features],
        "P3B": [*p2, *b_features],
        "P4_none": p2,
    }
    for column in geometry:
        radius_label = column.removeprefix("ra_geo_bottom_re_").removesuffix(
            "_um"
        )
        result[f"P4_re_{radius_label}"] = [*p2, column]
    return result


def _selection_metadata(name: str) -> tuple[str, float | None]:
    if name == "P3A":
        return "A", None
    if name == "P3B":
        return "B", None
    if name.startswith("P4_re_"):
        radius = float(name.removeprefix("P4_re_").replace("p", "."))
        return "symmetric", radius
    return "symmetric", None


def select_ridge_candidate(
    train_frame: pd.DataFrame,
    candidates: dict[str, list[str]],
    outer_fold: int,
    seed: int,
    inner_splits: int,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> ClassicSelection:
    required = {"group_id", "ra_mean", "sample_weight"}
    for columns in candidates.values():
        required.update(columns)
    missing = sorted(required - set(train_frame.columns))
    if missing:
        raise ValueError(f"Training frame missing columns: {missing}")
    splits = make_group_inner_splits(train_frame, inner_splits, seed)
    y = train_frame["ra_mean"].to_numpy(dtype=np.float64)
    weights = train_frame["sample_weight"].to_numpy(dtype=np.float64)
    best: ClassicSelection | None = None
    for name, feature_names in candidates.items():
        x = train_frame.loc[:, feature_names].to_numpy(dtype=np.float64)
        for alpha in alphas:
            predictions = cross_fitted_predictions(
                lambda alpha=alpha: make_pipeline(
                    StandardScaler(), Ridge(alpha=alpha)
                ),
                x,
                y,
                sample_weight=weights,
                splits=splits,
            )
            mae = float(np.average(np.abs(y - predictions), weights=weights))
            xy_mode, radius = _selection_metadata(name)
            selection = ClassicSelection(
                outer_fold=int(outer_fold),
                xy_mode=xy_mode,
                re_mm=radius,
                model_name=name,
                alpha=float(alpha),
                inner_weighted_mae=mae,
            )
            if best is None or (mae, name, alpha) < (
                best.inner_weighted_mae,
                best.model_name,
                best.alpha,
            ):
                best = selection
    if best is None:
        raise ValueError("No classic candidates were provided")
    return best


def run_classic_outer_cv(
    feature_table: pd.DataFrame,
    folds: pd.DataFrame,
    seed: int,
    inner_splits: int,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = feature_table.merge(
        folds[["sample_id", "fold"]],
        on="sample_id",
        validate="one_to_one",
    )
    feature_sets = build_classic_feature_sets(data)
    stage_candidates = {
        "P1": {"P1": feature_sets["P1"]},
        "P2": {"P2": feature_sets["P2"]},
        "P3": {
            "P3A": feature_sets["P3A"],
            "P3B": feature_sets["P3B"],
        },
        "P4": {
            name: columns
            for name, columns in feature_sets.items()
            if name.startswith("P4_")
        },
    }
    prediction_rows = []
    metric_rows = []
    selection_rows = []

    def record(
        model_name: str,
        fold: int,
        valid: pd.DataFrame,
        predicted: np.ndarray,
    ) -> None:
        unweighted = regression_metrics(valid["ra_mean"], predicted)
        weighted = regression_metrics(
            valid["ra_mean"], predicted, valid["sample_weight"]
        )
        metric_rows.append(
            {
                "model": model_name,
                "fold": int(fold),
                **unweighted,
                **{f"weighted_{key}": value for key, value in weighted.items()},
            }
        )
        prediction_rows.extend(
            {
                "sample_id": str(row.sample_id),
                "group_id": str(row.group_id),
                "model": model_name,
                "fold": int(fold),
                "seed": int(seed),
                "y_true": float(row.ra_mean),
                "y_pred": float(y_pred),
                "sample_weight": float(row.sample_weight),
            }
            for row, y_pred in zip(
                valid.itertuples(index=False), predicted, strict=True
            )
        )

    for fold in sorted(pd.to_numeric(data["fold"]).astype(int).unique()):
        train = data[data["fold"].astype(int) != fold].reset_index(drop=True)
        valid = data[data["fold"].astype(int) == fold].reset_index(drop=True)

        m0 = model_factories(seed)["quadratic_process"]()
        fit_with_sample_weight(
            m0,
            train[PROCESS_FEATURES],
            train["ra_mean"],
            train["sample_weight"],
        )
        record(
            "M0",
            fold,
            valid,
            m0.predict(valid[PROCESS_FEATURES]),
        )

        for stage, candidates in stage_candidates.items():
            selection = select_ridge_candidate(
                train,
                candidates=candidates,
                outer_fold=int(fold),
                seed=seed,
                inner_splits=inner_splits,
                alphas=alphas,
            )
            feature_names = candidates[selection.model_name]
            estimator = make_pipeline(
                StandardScaler(), Ridge(alpha=selection.alpha)
            )
            fit_with_sample_weight(
                estimator,
                train[feature_names],
                train["ra_mean"],
                train["sample_weight"],
            )
            record(
                stage,
                fold,
                valid,
                estimator.predict(valid[feature_names]),
            )
            selection_rows.append(
                {
                    "stage": stage,
                    "seed": int(seed),
                    **asdict(selection),
                    "feature_count": len(feature_names),
                }
            )

    predictions = pd.DataFrame.from_records(prediction_rows)
    metrics = pd.DataFrame.from_records(metric_rows)
    selections = pd.DataFrame.from_records(selection_rows)
    expected = len(data) * 5
    if len(predictions) != expected or predictions.duplicated(
        ["sample_id", "model"]
    ).any():
        raise AssertionError("Incomplete or duplicate classic OOF predictions")
    return predictions, metrics, selections


def build_outer_baseline_predictions(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    estimator_factory,
    feature_names: list[str],
    inner_splits: int,
    seed: int,
) -> pd.DataFrame:
    data = frame.merge(
        folds[["sample_id", "fold"]],
        on="sample_id",
        validate="one_to_one",
    )
    rows = []
    for outer_fold in sorted(data["fold"].astype(int).unique()):
        train = data[data["fold"].astype(int) != outer_fold].reset_index(
            drop=True
        )
        test = data[data["fold"].astype(int) == outer_fold].reset_index(
            drop=True
        )
        splits = make_group_inner_splits(train, inner_splits, seed)
        train_predictions = cross_fitted_predictions(
            estimator_factory,
            train[feature_names],
            train["ra_mean"],
            train["sample_weight"],
            splits,
        )
        estimator = estimator_factory()
        fit_with_sample_weight(
            estimator,
            train[feature_names],
            train["ra_mean"],
            train["sample_weight"],
        )
        test_predictions = estimator.predict(test[feature_names])
        rows.extend(
            {
                "outer_fold": int(outer_fold),
                "sample_id": str(row.sample_id),
                "group_id": str(row.group_id),
                "role": "train",
                "base_ra": float(prediction),
                "provenance": "inner_oof",
            }
            for row, prediction in zip(
                train.itertuples(index=False),
                train_predictions,
                strict=True,
            )
        )
        rows.extend(
            {
                "outer_fold": int(outer_fold),
                "sample_id": str(row.sample_id),
                "group_id": str(row.group_id),
                "role": "test",
                "base_ra": float(prediction),
                "provenance": "outer_train_fit",
            }
            for row, prediction in zip(
                test.itertuples(index=False),
                test_predictions,
                strict=True,
            )
        )
    result = pd.DataFrame.from_records(rows)
    if result.duplicated(["outer_fold", "sample_id"]).any():
        raise AssertionError("Duplicate outer-fold baseline prediction")
    if not np.isfinite(result["base_ra"]).all():
        raise ValueError("Baseline predictions contain non-finite values")
    return result
