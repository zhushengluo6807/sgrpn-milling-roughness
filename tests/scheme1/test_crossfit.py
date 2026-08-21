import numpy as np
import pandas as pd

from roughness.scheme1.crossfit import (
    cross_fitted_predictions,
    make_group_inner_splits,
)


class WeightedMeanRegressor:
    def fit(self, x, y, sample_weight=None):
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


def test_group_inner_splits_keep_groups_disjoint_and_validate_every_row_once():
    frame = pd.DataFrame(
        {
            "group_id": np.repeat([f"g{i}" for i in range(8)], 2),
            "value": np.arange(16),
        }
    )

    splits = make_group_inner_splits(frame, n_splits=4, seed=20260723)

    validation_rows = []
    for train_index, valid_index in splits:
        train_groups = set(frame.iloc[train_index]["group_id"])
        valid_groups = set(frame.iloc[valid_index]["group_id"])
        assert train_groups.isdisjoint(valid_groups)
        validation_rows.extend(valid_index.tolist())
    assert sorted(validation_rows) == list(range(len(frame)))


def test_cross_fitted_predictions_never_use_the_validation_targets():
    x = np.zeros((4, 1))
    y = np.array([1.0, 3.0, 10.0, 14.0])
    weights = np.ones(4)
    splits = [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ]

    predicted = cross_fitted_predictions(
        WeightedMeanRegressor,
        x,
        y,
        sample_weight=weights,
        splits=splits,
    )

    np.testing.assert_allclose(predicted, [12.0, 12.0, 2.0, 2.0])
