from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def make_group_inner_splits(
    frame: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if "group_id" not in frame:
        raise ValueError("frame must contain group_id")
    groups = np.asarray(sorted(frame["group_id"].astype(str).unique()))
    if n_splits < 2 or len(groups) < n_splits:
        raise ValueError("n_splits must be between 2 and the number of groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    splits = []
    group_values = frame["group_id"].astype(str).to_numpy()
    for validation_groups in np.array_split(groups, n_splits):
        validation_mask = np.isin(group_values, validation_groups)
        splits.append(
            (
                np.flatnonzero(~validation_mask),
                np.flatnonzero(validation_mask),
            )
        )
    return splits


def fit_with_sample_weight(estimator, x, y, sample_weight):
    if hasattr(estimator, "steps"):
        final_name = estimator.steps[-1][0]
        estimator.fit(
            x,
            y,
            **{f"{final_name}__sample_weight": sample_weight},
        )
    else:
        estimator.fit(x, y, sample_weight=sample_weight)
    return estimator


def cross_fitted_predictions(
    estimator_factory: Callable,
    x,
    y,
    sample_weight,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    x_array = np.asarray(x)
    y_array = np.asarray(y, dtype=np.float64)
    weight_array = np.asarray(sample_weight, dtype=np.float64)
    if not (len(x_array) == len(y_array) == len(weight_array)):
        raise ValueError("x, y and sample_weight must have equal length")
    predictions = np.full(len(y_array), np.nan, dtype=np.float64)
    assigned = np.zeros(len(y_array), dtype=np.int64)
    for train_index, validation_index in splits:
        if np.intersect1d(train_index, validation_index).size:
            raise ValueError("Train and validation indices overlap")
        estimator = estimator_factory()
        fit_with_sample_weight(
            estimator,
            x_array[train_index],
            y_array[train_index],
            weight_array[train_index],
        )
        predictions[validation_index] = estimator.predict(
            x_array[validation_index]
        )
        assigned[validation_index] += 1
    if not np.all(assigned == 1) or not np.isfinite(predictions).all():
        raise ValueError("Every row must receive one finite cross-fitted prediction")
    return predictions

