import numpy as np
import pandas as pd
import torch

from roughness.scheme1.classic import build_outer_baseline_predictions
from roughness.scheme1.models import (
    CNNEncoder,
    SegmentFusionRegressor,
)


class WeightedMeanRegressor:
    def fit(self, x, y, sample_weight=None):
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


def _batch():
    return {
        "signal": torch.randn(2, 2, 3, 2048),
        "window_mask": torch.ones(2, 2, dtype=torch.bool),
        "process": torch.randn(2, 3),
        "physics": torch.randn(2, 2),
        "base_ra": torch.tensor([0.8, 1.2]),
    }


def test_process_fusion_outputs_one_direct_prediction_per_segment():
    batch = _batch()
    model = SegmentFusionRegressor(
        CNNEncoder(), process_dim=3, physics_dim=0, mode="direct"
    )

    output = model(
        batch["signal"],
        batch["window_mask"],
        process=batch["process"],
    )

    assert output.prediction.shape == (2,)
    assert output.residual is None
    assert output.gate is None


def test_residual_prediction_is_exactly_base_plus_learned_delta():
    batch = _batch()
    model = SegmentFusionRegressor(
        CNNEncoder(), process_dim=3, physics_dim=2, mode="residual"
    )

    output = model(
        batch["signal"],
        batch["window_mask"],
        process=batch["process"],
        physics=batch["physics"],
        base_ra=batch["base_ra"],
    )

    torch.testing.assert_close(
        output.prediction, batch["base_ra"] + output.residual
    )
    assert output.gate is None


def test_gated_prediction_uses_bounded_gate_times_residual():
    batch = _batch()
    model = SegmentFusionRegressor(
        CNNEncoder(), process_dim=3, physics_dim=2, mode="gated"
    )

    output = model(
        batch["signal"],
        batch["window_mask"],
        process=batch["process"],
        physics=batch["physics"],
        base_ra=batch["base_ra"],
    )

    assert torch.all((0.0 <= output.gate) & (output.gate <= 1.0))
    torch.testing.assert_close(
        output.prediction,
        batch["base_ra"] + output.gate * output.residual,
    )


def test_outer_training_baselines_are_inner_cross_fitted():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "group_id": [f"g{i}" for i in range(6)],
            "x": np.zeros(6),
            "ra_mean": [1.0, 3.0, 5.0, 10.0, 12.0, 14.0],
            "sample_weight": np.ones(6),
        }
    )
    folds = pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "group_id": frame["group_id"],
            "fold": [0, 0, 1, 1, 2, 2],
        }
    )

    predictions = build_outer_baseline_predictions(
        frame,
        folds,
        estimator_factory=WeightedMeanRegressor,
        feature_names=["x"],
        inner_splits=2,
        seed=20260723,
    )

    assert len(predictions) == 18
    train = predictions[predictions["role"] == "train"]
    test = predictions[predictions["role"] == "test"]
    assert set(train["provenance"]) == {"inner_oof"}
    assert set(test["provenance"]) == {"outer_train_fit"}
    assert not train["base_ra"].isna().any()
    assert not test["base_ra"].isna().any()
