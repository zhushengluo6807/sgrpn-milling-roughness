from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import roughness.sgrpn.crossfit as crossfit
from roughness.sgrpn.crossfit import (
    fit_process_inner_fold,
    fit_process_scaler,
    generate_process_oof as _generate_process_oof,
    median_best_epoch,
)
from roughness.sgrpn.data import build_process_features


def make_frame(groups: int, rows_per_group: int) -> pd.DataFrame:
    group_id = [f"g{group}" for group in range(groups) for _ in range(rows_per_group)]
    count = len(group_id)
    return pd.DataFrame(
        {
            "sample_id": [f"sample-{row}" for row in range(count)],
            "group_id": group_id,
            "n_rpm": np.linspace(4000.0, 8000.0, count),
            "fz_mm_per_tooth": np.linspace(0.03, 0.12, count),
            "ap_mm": np.linspace(0.5, 2.0, count),
            "ra_mean": np.linspace(0.2, 1.2, count),
            "sample_weight": np.ones(count),
            "split_count": np.ones(count, dtype=int),
        }
    )


def _mean_trainer(
    x_train,
    y_train,
    w_train,
    x_valid,
    train_groups,
    valid_groups,
):
    del x_train, w_train, train_groups, valid_groups
    return np.full(len(x_valid), y_train.mean()), 7


def generate_process_oof(frame, process_features, trainer, **kwargs):
    if "sample_id" in frame:
        feature_sample_ids = frame["sample_id"].tolist()
    else:
        feature_sample_ids = frame.index.tolist()
    return _generate_process_oof(
        frame,
        process_features,
        trainer,
        feature_sample_ids=feature_sample_ids,
        **kwargs,
    )


def test_rejects_independently_permuted_feature_rows_and_ordered_ids():
    frame = make_frame(groups=8, rows_per_group=1)
    features = build_process_features(frame)
    permutation = np.roll(np.arange(len(frame)), 1)
    calls = 0

    def trainer(*args):
        nonlocal calls
        calls += 1
        return np.zeros(len(args[3])), 1

    with pytest.raises(ValueError, match="feature_sample_ids|ordered.*ID|order"):
        _generate_process_oof(
            frame,
            features[permutation],
            trainer,
            feature_sample_ids=frame["sample_id"].to_numpy()[permutation],
        )
    assert calls == 0


def test_each_group_receives_one_prediction_from_unseen_groups():
    frame = make_frame(groups=8, rows_per_group=2)
    seen = []

    def trainer(x_train, y_train, w_train, x_valid, train_groups, valid_groups):
        del w_train
        assert x_train.shape[1] == x_valid.shape[1] == 9
        assert set(train_groups).isdisjoint(valid_groups)
        seen.extend(valid_groups)
        return np.full(len(x_valid), y_train.mean()), 7

    result = generate_process_oof(
        frame,
        build_process_features(frame),
        trainer,
        n_splits=4,
        seed=20260723,
    )

    assert np.isfinite(result.prediction).all()
    assert result.assignment_count.tolist() == [1] * len(frame)
    assert sorted(seen) == sorted(frame.group_id.tolist())
    assert sorted(np.unique(result.inner_fold).tolist()) == [0, 1, 2, 3]
    assert tuple(result.best_epochs) == (7, 7, 7, 7)


def test_scaler_fits_only_inner_train_rows_and_handles_constant_columns():
    x = np.array([[0.0] * 9, [10.0] * 9, [1000.0] * 9])

    scaler = fit_process_scaler(x, np.array([0, 1]))

    np.testing.assert_allclose(scaler.mean, np.full(9, 5.0))
    np.testing.assert_allclose(scaler.scale, np.full(9, 5.0))
    np.testing.assert_allclose(scaler.transform(x[[0, 1]]).mean(axis=0), np.zeros(9))

    constant = fit_process_scaler(np.ones((3, 9)), np.array([0, 1]))
    np.testing.assert_allclose(constant.scale, np.ones(9))


def test_fold_assignment_is_deterministic_and_stays_aligned_to_sample_ids():
    frame = make_frame(groups=8, rows_per_group=2).sample(frac=1.0, random_state=91)
    features = build_process_features(frame)

    first = generate_process_oof(frame, features, _mean_trainer)
    second = generate_process_oof(frame.copy(), features.copy(), _mean_trainer)

    np.testing.assert_array_equal(first.inner_fold, second.inner_fold)
    np.testing.assert_allclose(first.prediction, second.prediction)
    mapping = dict(zip(frame["sample_id"], first.residual, strict=True))
    expected = frame["ra_mean"].to_numpy() - first.prediction
    assert mapping == dict(zip(frame["sample_id"], expected, strict=True))


def test_fit_only_callback_does_not_train_on_its_held_out_group_labels():
    frame = make_frame(groups=8, rows_per_group=2)
    original = generate_process_oof(frame, build_process_features(frame), _mean_trainer)
    changed = frame.copy()
    changed_group = "g3"
    changed.loc[changed["group_id"] == changed_group, "ra_mean"] += 10_000.0

    permuted = generate_process_oof(changed, build_process_features(changed), _mean_trainer)

    held_out = frame["group_id"].eq(changed_group).to_numpy()
    np.testing.assert_allclose(original.prediction[held_out], permuted.prediction[held_out])
    assert not np.allclose(original.prediction[~held_out], permuted.prediction[~held_out])


def test_extended_trainer_receives_only_matching_inner_validation_labels():
    frame = make_frame(groups=8, rows_per_group=2)
    calls = 0

    def trainer(
        x_train,
        y_train,
        w_train,
        x_valid,
        train_groups,
        valid_groups,
        *,
        y_valid,
        w_valid,
    ):
        nonlocal calls
        del x_train, y_train, w_train
        calls += 1
        assert len(y_valid) == len(w_valid) == len(x_valid) == len(valid_groups)
        expected = frame.loc[
            frame["group_id"].isin(set(valid_groups)), "ra_mean"
        ].to_numpy()
        np.testing.assert_allclose(y_valid, expected)
        return np.zeros(len(x_valid)), 3

    generate_process_oof(
        frame,
        build_process_features(frame),
        crossfit.ValidationAwareTrainer(trainer),
    )

    assert calls == 4


def test_generic_kwargs_trainer_is_not_silently_given_validation_labels():
    frame = make_frame(groups=8, rows_per_group=1)
    received_kwargs = []

    def trainer(
        x_train,
        y_train,
        w_train,
        x_valid,
        train_groups,
        valid_groups,
        **kwargs,
    ):
        del x_train, y_train, w_train, train_groups, valid_groups
        received_kwargs.append(kwargs)
        return np.zeros(len(x_valid)), 1

    generate_process_oof(frame, build_process_features(frame), trainer)

    assert received_kwargs == [{}, {}, {}, {}]


def test_validation_aware_adapter_rejects_kwargs_only_label_capability():
    def generic_trainer(*args, **kwargs):
        return np.zeros(len(args[3])), 1

    with pytest.raises(ValueError, match="explicit.*y_valid.*w_valid|keyword"):
        crossfit.ValidationAwareTrainer(generic_trainer)


def test_validation_aware_adapter_rejects_positional_only_validation_labels():
    def positional_only_trainer(
        x_train,
        y_train,
        w_train,
        x_valid,
        train_groups,
        valid_groups,
        y_valid,
        w_valid,
        /,
    ):
        return np.zeros(len(x_valid)), 1

    with pytest.raises(ValueError, match="keyword-capable"):
        crossfit.ValidationAwareTrainer(positional_only_trainer)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.assign(sample_id="duplicate"), "Duplicate.*sample_id"),
        (
            lambda frame: frame.assign(
                sample_id=frame["sample_id"].mask(frame.index == frame.index[0])
            ),
            "missing.*sample_id|sample_id.*missing",
        ),
        (
            lambda frame: frame.assign(
                group_id=frame["group_id"].mask(frame.index == frame.index[0])
            ),
            "missing.*group_id|group_id.*missing",
        ),
    ],
)
def test_rejects_duplicate_or_missing_audit_identifiers(mutate, message):
    frame = mutate(make_frame(groups=8, rows_per_group=1))

    with pytest.raises(ValueError, match=message):
        generate_process_oof(frame, build_process_features(frame), _mean_trainer)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (np.zeros((8, 8)), "9"),
        (np.full((8, 9), np.nan), "finite"),
        (np.zeros((7, 9)), "rows|length"),
    ],
)
def test_rejects_wrong_width_non_finite_or_misaligned_features(features, message):
    frame = make_frame(groups=8, rows_per_group=1)

    with pytest.raises(ValueError, match=message):
        generate_process_oof(frame, features, _mean_trainer)


def test_rejects_non_finite_predictions_and_invalid_best_epoch():
    frame = make_frame(groups=8, rows_per_group=1)
    features = build_process_features(frame)

    def non_finite(*args):
        return np.full(len(args[3]), np.nan), 1

    with pytest.raises(ValueError, match="finite"):
        generate_process_oof(frame, features, non_finite)

    def bad_epoch(*args):
        return np.zeros(len(args[3])), 0

    with pytest.raises(ValueError, match="epoch.*positive|positive.*epoch"):
        generate_process_oof(frame, features, bad_epoch)


def test_rejects_protocol_changes_and_invalid_weights():
    frame = make_frame(groups=8, rows_per_group=1)
    features = build_process_features(frame)

    with pytest.raises(ValueError, match="four|4"):
        generate_process_oof(frame, features, _mean_trainer, n_splits=3)
    with pytest.raises(ValueError, match="20260723"):
        generate_process_oof(frame, features, _mean_trainer, seed=1)

    frame.loc[0, "sample_weight"] = 0.0
    with pytest.raises(ValueError, match="weight.*positive|positive.*weight"):
        generate_process_oof(frame, features, _mean_trainer)


def test_rejects_duplicate_missing_and_leaking_split_assignments(monkeypatch):
    frame = make_frame(groups=8, rows_per_group=1)
    features = build_process_features(frame)

    duplicate_and_missing = [
        (np.arange(2, 8), np.array([0, 1])),
        (np.array([0, 1, 4, 5, 6, 7]), np.array([2, 3])),
        (np.array([0, 1, 2, 3, 6, 7]), np.array([4, 5])),
        (np.arange(8), np.array([0])),
    ]
    monkeypatch.setattr(
        "roughness.sgrpn.crossfit.make_group_inner_splits",
        lambda *_args, **_kwargs: duplicate_and_missing,
    )
    with pytest.raises(ValueError, match="assigned|assignment|overlap"):
        generate_process_oof(frame, features, _mean_trainer)

    leaking = [
        (np.array([0, 1, 2, 3, 4, 5, 6]), np.array([0, 7])),
        (np.array([0, 1, 4, 5, 6, 7]), np.array([2, 3])),
        (np.array([0, 1, 2, 3, 6, 7]), np.array([4, 5])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
    ]
    monkeypatch.setattr(
        "roughness.sgrpn.crossfit.make_group_inner_splits",
        lambda *_args, **_kwargs: leaking,
    )
    with pytest.raises(ValueError, match="overlap|leakage"):
        generate_process_oof(frame, features, _mean_trainer)


def test_rejects_inner_fold_that_does_not_partition_all_outer_train_rows(monkeypatch):
    frame = make_frame(groups=8, rows_per_group=1)
    features = build_process_features(frame)
    incomplete_partition = [
        (np.array([3, 4, 5, 6, 7]), np.array([0, 1])),
        (np.array([0, 1, 4, 5, 6, 7]), np.array([2, 3])),
        (np.array([0, 1, 2, 3, 6, 7]), np.array([4, 5])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7])),
    ]
    monkeypatch.setattr(
        "roughness.sgrpn.crossfit.make_group_inner_splits",
        lambda *_args, **_kwargs: incomplete_partition,
    )

    with pytest.raises(ValueError, match="partition|cover|missing"):
        generate_process_oof(frame, features, _mean_trainer)


def test_preflights_every_fold_before_any_trainer_callback(monkeypatch):
    frame = make_frame(groups=8, rows_per_group=1)
    features = build_process_features(frame)
    malformed_last_fold = [
        (np.array([2, 3, 4, 5, 6, 7]), np.array([0, 1])),
        (np.array([0, 1, 4, 5, 6, 7]), np.array([2, 3])),
        (np.array([0, 1, 2, 3, 6, 7]), np.array([4, 5])),
        (np.array([0, 1, 2, 3, 4, 5, 6]), np.array([6, 7])),
    ]
    monkeypatch.setattr(
        "roughness.sgrpn.crossfit.make_group_inner_splits",
        lambda *_args, **_kwargs: malformed_last_fold,
    )
    calls = 0

    def trainer(*args):
        nonlocal calls
        calls += 1
        return np.zeros(len(args[3])), 1

    with pytest.raises(ValueError, match="overlap|leakage"):
        generate_process_oof(frame, features, trainer)
    assert calls == 0


@pytest.mark.parametrize(
    ("epochs", "expected"),
    [
        ([1, 2, 3, 100], 2),
        ([5, 5, 5, 5], 5),
    ],
)
def test_median_best_epoch_uses_four_positive_epochs(epochs, expected):
    assert median_best_epoch(epochs) == expected


@pytest.mark.parametrize("epochs", [[], [1, 2, 3], [1, 2, 3, 4, 5], [1, 0, 2, 3]])
def test_median_best_epoch_rejects_non_four_or_non_positive_epochs(epochs):
    with pytest.raises(ValueError, match="four|positive"):
        median_best_epoch(epochs)


def test_real_process_fold_trainer_returns_finite_predictions_and_selected_epoch():
    x_train = np.array(
        [
            [-1.0] * 9,
            [-0.5] * 9,
            [0.5] * 9,
            [1.0] * 9,
        ],
        dtype=np.float32,
    )
    y_train = np.array([0.2, 0.4, 0.8, 1.0])
    weights = np.ones(4)

    prediction, best_epoch = fit_process_inner_fold(
        x_train,
        y_train,
        weights,
        np.zeros((2, 9), dtype=np.float32),
        np.array(["g0", "g1", "g2", "g3"]),
        np.array(["g4", "g5"]),
        y_valid=np.array([0.6, 0.7]),
        w_valid=np.ones(2),
    )

    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert 1 <= best_epoch <= 200


def test_real_process_fold_trainer_requires_validation_targets_for_selection():
    with pytest.raises(ValueError, match="validation.*target|y_valid"):
        fit_process_inner_fold(
            np.zeros((2, 9)),
            np.zeros(2),
            np.ones(2),
            np.zeros((1, 9)),
            np.array(["g0", "g1"]),
            np.array(["g2"]),
        )


class _ScalarProcessMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, process: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(len(process))


def test_real_trainer_keeps_validation_labels_out_of_optimizer_loss(monkeypatch):
    optimizer_targets = []
    selection_targets = []
    registered_weighted_huber = crossfit.weighted_huber

    def recording_weighted_huber(prediction, target, weight, delta):
        destination = optimizer_targets if torch.is_grad_enabled() else selection_targets
        destination.append(target.detach().cpu().numpy().copy())
        return registered_weighted_huber(prediction, target, weight, delta)

    monkeypatch.setattr(crossfit, "ProcessMLP", _ScalarProcessMLP)
    monkeypatch.setattr(crossfit, "weighted_huber", recording_weighted_huber)
    y_train = np.array([0.25, 0.75])
    y_valid = np.array([100.0, 200.0])

    fit_process_inner_fold(
        np.zeros((2, 9)),
        y_train,
        np.ones(2),
        np.zeros((2, 9)),
        np.array(["train-0", "train-1"]),
        np.array(["valid-0", "valid-1"]),
        y_valid=y_valid,
        w_valid=np.ones(2),
    )

    assert optimizer_targets
    assert selection_targets
    for target in optimizer_targets:
        np.testing.assert_array_equal(target, y_train.astype(np.float32))
    for target in selection_targets:
        np.testing.assert_array_equal(target, y_valid.astype(np.float32))


def test_validation_labels_can_change_selected_best_epoch(monkeypatch):
    monkeypatch.setattr(crossfit, "ProcessMLP", _ScalarProcessMLP)
    common = (
        np.zeros((2, 9)),
        np.ones(2),
        np.ones(2),
        np.zeros((1, 9)),
        np.array(["train-0", "train-1"]),
        np.array(["valid-0"]),
    )

    prediction_near_zero, epoch_near_zero = fit_process_inner_fold(
        *common,
        y_valid=np.zeros(1),
        w_valid=np.ones(1),
    )
    prediction_near_target, epoch_near_target = fit_process_inner_fold(
        *common,
        y_valid=np.ones(1),
        w_valid=np.ones(1),
    )

    assert epoch_near_zero < epoch_near_target
    assert prediction_near_zero[0] < prediction_near_target[0]
