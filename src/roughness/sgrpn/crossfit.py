"""Leakage-safe cross-fitting for the Phase A process expert."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
import inspect
from typing import Any

import numpy as np
import pandas as pd
import torch

from roughness.scheme1.crossfit import make_group_inner_splits

from .models import ProcessMLP, weighted_huber


_PROCESS_WIDTH = 9
_PHASE_A_INNER_SPLITS = 4
_PHASE_A_SEED = 20260723
_PROCESS_LEARNING_RATE = 1e-3
_WEIGHT_DECAY = 1e-4
_HUBER_DELTA = 0.10
_MAX_EPOCHS = 200
_PATIENCE = 20


@dataclass(frozen=True)
class ProcessScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = _finite_process_matrix(values, name="values")
        if self.mean.shape != (_PROCESS_WIDTH,) or self.scale.shape != (_PROCESS_WIDTH,):
            raise ValueError("ProcessScaler mean and scale must each contain 9 values")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("ProcessScaler mean and scale must be finite")
        if np.any(self.scale <= 0):
            raise ValueError("ProcessScaler scale must be positive")
        return (array - self.mean) / self.scale


@dataclass(frozen=True)
class ProcessOOFResult:
    prediction: np.ndarray
    residual: np.ndarray
    assignment_count: np.ndarray
    best_epochs: Sequence[int]
    inner_fold: np.ndarray


ProcessTrainer = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, int],
]


def _finite_process_matrix(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite matrix with 9 columns") from error
    if array.ndim != 2 or array.shape[1] != _PROCESS_WIDTH:
        raise ValueError(f"{name} must have shape [N, 9]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _finite_vector(values: Any, *, name: str, length: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite vector") from error
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have length {length}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validated_indices(values: Any, *, name: str, row_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional index")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    result = array.astype(np.int64, copy=False)
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} contains duplicate indices")
    if np.any(result < 0) or np.any(result >= row_count):
        raise ValueError(f"{name} contains out-of-range indices")
    return result


def fit_process_scaler(values: np.ndarray, train_index: np.ndarray) -> ProcessScaler:
    """Fit the nine-feature scaler on the explicitly supplied training rows."""
    array = _finite_process_matrix(values, name="values")
    index = _validated_indices(train_index, name="train_index", row_count=len(array))
    selected = array[index]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Process scaler statistics must be finite")
    return ProcessScaler(mean=mean, scale=scale)


def _validate_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame")
    required = {"group_id", "ra_mean", "sample_weight"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"frame missing columns: {missing_columns}")

    if frame["group_id"].isna().any():
        raise ValueError("frame contains missing group_id values")
    groups = frame["group_id"].astype(str).to_numpy()
    if np.any(np.char.str_len(groups.astype(str)) == 0):
        raise ValueError("frame contains missing group_id values")

    if "sample_id" in frame:
        if frame["sample_id"].isna().any():
            raise ValueError("frame contains missing sample_id values")
        audit_ids = frame["sample_id"].astype(str)
        if (audit_ids.str.len() == 0).any():
            raise ValueError("frame contains missing sample_id values")
        if audit_ids.duplicated().any():
            raise ValueError("Duplicate frame sample_id values")
    else:
        if frame.index.hasnans:
            raise ValueError("frame index contains missing audit identifiers")
        if frame.index.duplicated().any():
            raise ValueError("Duplicate frame index audit identifiers")

    target = _finite_vector(frame["ra_mean"], name="ra_mean", length=len(frame))
    weight = _finite_vector(
        frame["sample_weight"], name="sample_weight", length=len(frame)
    )
    if np.any(weight <= 0):
        raise ValueError("sample_weight must be positive")

    if "split_count" in frame:
        split_count = _finite_vector(
            frame["split_count"], name="split_count", length=len(frame)
        )
        if np.any(split_count <= 0):
            raise ValueError("split_count must be positive")
        if not np.allclose(weight, 1.0 / split_count, rtol=0.0, atol=1e-12):
            raise ValueError("sample_weight must equal 1/split_count")
    return groups, target, weight


def _supports_validation_labels(trainer: ProcessTrainer) -> bool:
    try:
        parameters = inspect.signature(trainer).parameters
    except (TypeError, ValueError):
        return False
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return True
    has_target = "y_valid" in parameters
    has_weight = "w_valid" in parameters
    if has_target != has_weight:
        raise ValueError("trainer must accept both y_valid and w_valid or neither")
    return has_target and has_weight


def _positive_epoch(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("best epoch must be a positive integer")
    epoch = int(value)
    if epoch <= 0:
        raise ValueError("best epoch must be positive")
    return epoch


def _call_trainer(
    trainer: ProcessTrainer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    train_groups: np.ndarray,
    valid_groups: np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray,
) -> tuple[np.ndarray, int]:
    arguments = (x_train, y_train, w_train, x_valid, train_groups, valid_groups)
    if _supports_validation_labels(trainer):
        result = trainer(*arguments, y_valid=y_valid, w_valid=w_valid)
    else:
        result = trainer(*arguments)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("trainer must return (prediction, best_epoch)")
    prediction = _finite_vector(result[0], name="trainer prediction", length=len(x_valid))
    return prediction, _positive_epoch(result[1])


def generate_process_oof(
    frame: pd.DataFrame,
    process_features: np.ndarray,
    trainer: ProcessTrainer,
    n_splits: int = _PHASE_A_INNER_SPLITS,
    seed: int = _PHASE_A_SEED,
) -> ProcessOOFResult:
    """Generate one P1 prediction per outer-training row without group leakage."""
    if isinstance(n_splits, bool) or n_splits != _PHASE_A_INNER_SPLITS:
        raise ValueError("Phase A requires exactly four inner splits")
    if isinstance(seed, bool) or seed != _PHASE_A_SEED:
        raise ValueError("Phase A requires seed 20260723")
    if not callable(trainer):
        raise ValueError("trainer must be callable")
    groups, target, weight = _validate_frame(frame)
    features = _finite_process_matrix(process_features, name="process_features")
    if len(features) != len(frame):
        raise ValueError("process_features rows must match frame length")

    splits = make_group_inner_splits(frame, n_splits=n_splits, seed=seed)
    if len(splits) != _PHASE_A_INNER_SPLITS:
        raise ValueError("Phase A split generator must return four inner splits")

    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    assignment_count = np.zeros(len(frame), dtype=np.int64)
    inner_fold = np.full(len(frame), -1, dtype=np.int64)
    best_epochs: list[int] = []
    for fold_number, (raw_train, raw_valid) in enumerate(splits):
        train_index = _validated_indices(
            raw_train, name="inner train index", row_count=len(frame)
        )
        valid_index = _validated_indices(
            raw_valid, name="inner validation index", row_count=len(frame)
        )
        if np.intersect1d(train_index, valid_index).size:
            raise ValueError("Inner train and validation indices overlap")
        if np.union1d(train_index, valid_index).size != len(frame):
            raise ValueError(
                "Each inner fold must partition and cover every outer-train row"
            )
        train_groups = groups[train_index]
        valid_groups = groups[valid_index]
        overlap = sorted(set(train_groups) & set(valid_groups))
        if overlap:
            raise ValueError(f"Inner group leakage between train and validation: {overlap[:10]}")

        scaler = fit_process_scaler(features, train_index)
        fold_prediction, best_epoch = _call_trainer(
            trainer,
            scaler.transform(features[train_index]),
            target[train_index].copy(),
            weight[train_index].copy(),
            scaler.transform(features[valid_index]),
            train_groups.copy(),
            valid_groups.copy(),
            target[valid_index].copy(),
            weight[valid_index].copy(),
        )
        prediction[valid_index] = fold_prediction
        assignment_count[valid_index] += 1
        inner_fold[valid_index] = fold_number
        best_epochs.append(best_epoch)

    if not np.all(assignment_count == 1):
        bad = np.flatnonzero(assignment_count != 1)
        raise ValueError(
            "Every row must be assigned exactly once; invalid row positions: "
            f"{bad[:10].tolist()}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Every row must receive one finite process prediction")
    if np.any(inner_fold < 0):
        raise ValueError("Every row must have an auditable inner-fold assignment")
    residual = target - prediction
    if not np.isfinite(residual).all():
        raise ValueError("Every process residual must be finite")
    return ProcessOOFResult(
        prediction=prediction,
        residual=residual,
        assignment_count=assignment_count,
        best_epochs=tuple(best_epochs),
        inner_fold=inner_fold,
    )


def median_best_epoch(best_epochs: Sequence[int]) -> int:
    """Return the rounded median of the four registered inner-fold epochs."""
    values = tuple(best_epochs)
    if len(values) != _PHASE_A_INNER_SPLITS:
        raise ValueError("median_best_epoch requires exactly four epochs")
    epochs = np.asarray([_positive_epoch(value) for value in values], dtype=np.int64)
    return max(1, int(np.rint(np.median(epochs))))


def _validate_fold_training_inputs(
    x_train: Any,
    y_train: Any,
    w_train: Any,
    x_valid: Any,
    train_groups: Any,
    valid_groups: Any,
    y_valid: Any,
    w_valid: Any,
) -> tuple[np.ndarray, ...]:
    train_x = _finite_process_matrix(x_train, name="x_train")
    valid_x = _finite_process_matrix(x_valid, name="x_valid")
    if len(train_x) == 0 or len(valid_x) == 0:
        raise ValueError("inner train and validation data must not be empty")
    train_y = _finite_vector(y_train, name="y_train", length=len(train_x))
    train_w = _finite_vector(w_train, name="w_train", length=len(train_x))
    valid_y = _finite_vector(y_valid, name="y_valid", length=len(valid_x))
    valid_w = _finite_vector(w_valid, name="w_valid", length=len(valid_x))
    if np.any(train_w <= 0) or np.any(valid_w <= 0):
        raise ValueError("training and validation weights must be positive")
    train_group_array = np.asarray(train_groups)
    valid_group_array = np.asarray(valid_groups)
    if train_group_array.ndim != 1 or len(train_group_array) != len(train_x):
        raise ValueError("train_groups must align with x_train")
    if valid_group_array.ndim != 1 or len(valid_group_array) != len(valid_x):
        raise ValueError("valid_groups must align with x_valid")
    if pd.isna(train_group_array).any() or pd.isna(valid_group_array).any():
        raise ValueError("train_groups and valid_groups must not be missing")
    train_group_array = train_group_array.astype(str)
    valid_group_array = valid_group_array.astype(str)
    if set(train_group_array) & set(valid_group_array):
        raise ValueError("Inner group leakage between train and validation")
    return train_x, train_y, train_w, valid_x, valid_y, valid_w


def fit_process_inner_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    train_groups: np.ndarray,
    valid_groups: np.ndarray,
    *,
    y_valid: np.ndarray | None = None,
    w_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Fit P1 and restore the epoch with minimum validation weighted Huber."""
    if y_valid is None or w_valid is None:
        raise ValueError("y_valid and w_valid validation targets are required for selection")
    train_x, train_y, train_w, valid_x, valid_y, valid_w = _validate_fold_training_inputs(
        x_train,
        y_train,
        w_train,
        x_valid,
        train_groups,
        valid_groups,
        y_valid,
        w_valid,
    )

    torch.manual_seed(_PHASE_A_SEED)
    model = ProcessMLP()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=_PROCESS_LEARNING_RATE, weight_decay=_WEIGHT_DECAY
    )
    train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
    train_w_tensor = torch.as_tensor(train_w, dtype=torch.float32)
    valid_x_tensor = torch.as_tensor(valid_x, dtype=torch.float32)
    valid_y_tensor = torch.as_tensor(valid_y, dtype=torch.float32)
    valid_w_tensor = torch.as_tensor(valid_w, dtype=torch.float32)

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, _MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = weighted_huber(
            model(train_x_tensor), train_y_tensor, train_w_tensor, _HUBER_DELTA
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("Process training loss must be finite")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = weighted_huber(
                model(valid_x_tensor), valid_y_tensor, valid_w_tensor, _HUBER_DELTA
            )
        numeric_loss = float(validation_loss.item())
        if not np.isfinite(numeric_loss):
            raise ValueError("Process validation loss must be finite")
        if numeric_loss < best_loss:
            best_loss = numeric_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= _PATIENCE:
                break

    if best_state is None or best_epoch <= 0:
        raise ValueError("Process training did not select a finite best epoch")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(valid_x_tensor).cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(prediction).all():
        raise ValueError("Process validation predictions must be finite")
    return prediction, best_epoch


__all__ = [
    "ProcessOOFResult",
    "ProcessScaler",
    "ProcessTrainer",
    "fit_process_inner_fold",
    "fit_process_scaler",
    "generate_process_oof",
    "median_best_epoch",
]
