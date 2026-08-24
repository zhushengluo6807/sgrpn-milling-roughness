"""Phase B probability training.

This module deliberately starts with the scale-head fitting primitive.  The
nested calibration and outer-fold orchestration APIs are declared here so their
public contracts are stable, but are implemented by the subsequent Task 6B
batch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .models import (
    ModelOutput,
    SelectiveGatedModel,
    assert_mean_model_unchanged,
    average_swap_predictions,
    build_scale_features,
    repeated_gaussian_nll,
)
from .probability import GroupConformalResult


# The two registered Phase B scale variants.  Their order is also the persisted
# output order used by the fold runner implemented in the next batch.
SCALE_MODELS = ("heteroscedastic", "homoscedastic")

CALIBRATION_SCORE_COLUMNS = (
    "group_id",
    "region_count",
    "reading_count",
    "score",
)

PHASE_B_MEAN_COLUMNS = (
    "sample_id",
    "group_id",
    "fold",
    "seed",
    "model",
    "mu",
)

PHASE_B_PREDICTION_COLUMNS = (
    "sample_id",
    "group_id",
    "fold",
    "seed",
    "scale_model",
    "target_mean",
    "ra_1",
    "ra_2",
    "ra_3",
    "mu",
    "sigma",
    "raw_lower_90",
    "raw_upper_90",
    "raw_lower_95",
    "raw_upper_95",
    "conformal_lower_90",
    "conformal_upper_90",
    "conformal_lower_95",
    "conformal_upper_95",
)


@dataclass(frozen=True)
class ScaleFit:
    model: nn.Module
    best_epoch: int
    history: pd.DataFrame
    mean_state_sha256: str


@dataclass(frozen=True)
class CalibrationArtifacts:
    predictions: pd.DataFrame
    group_scores: pd.DataFrame
    quantiles: Mapping[float, GroupConformalResult]
    inner_fold_definitions: pd.DataFrame


@dataclass(frozen=True)
class PhaseBRunFingerprint:
    value: str
    config_sha256: str
    phase_a_acceptance_sha256: str
    phase_a_run_manifest_sha256: str
    phase_a_training_fingerprint: str
    manifest_sha256: str
    folds_sha256: str
    cache_sha256: str


@dataclass(frozen=True)
class PhaseBFoldArtifacts:
    predictions: pd.DataFrame
    calibration: Mapping[str, CalibrationArtifacts]
    checkpoint_paths: Mapping[str, Path]
    history_paths: Mapping[str, Path]
    fingerprint: PhaseBRunFingerprint


@dataclass(frozen=True)
class PhaseBRunArtifacts:
    probability_predictions: pd.DataFrame
    mean_predictions: pd.DataFrame
    fold_artifacts: Mapping[tuple[int, int], PhaseBFoldArtifacts]
    fingerprints: Mapping[tuple[int, int], PhaseBRunFingerprint]


def _set_scale_seed(seed: int) -> None:
    """Set deterministic RNG state without widening the Phase A seed policy."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _state_sha256(snapshot: Mapping[str, Tensor]) -> str:
    """Hash a full state_dict snapshot, including BatchNorm-style buffers."""
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        value = snapshot[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _freeze_mean_state(mean_model: SelectiveGatedModel) -> dict[str, Tensor]:
    """Freeze and snapshot every parameter/buffer without subclass train hooks.

    The production G1 overrides ``train`` to preserve its expert state.  Scale
    fitting freezes *all* of G1, so the base ``nn.Module`` transition is the
    appropriate operation and also supports lightweight G1-shaped test models
    that intentionally initialize only the fields used by ``forward``.
    """
    for parameter in mean_model.parameters():
        parameter.requires_grad_(False)
    nn.Module.train(mean_model, False)
    return {
        name: value.detach().cpu().clone()
        for name, value in mean_model.state_dict().items()
    }


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if not isinstance(batch, Mapping):
        raise ValueError("scale loaders must yield mapping batches")
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _required_scale_tensors(batch: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    try:
        process = batch["process"]
        quality = batch["quality"]
        readings = batch["readings"]
        weight = batch["sample_weight"]
    except KeyError as error:
        raise ValueError(f"scale batch is missing {error.args[0]!r}") from error
    if not all(isinstance(value, Tensor) for value in (process, quality, readings, weight)):
        raise ValueError("scale batch process, quality, readings, and sample_weight must be tensors")
    return process, quality, readings, weight


def _current_orientation_output(mean_model: SelectiveGatedModel, batch: Mapping[str, Any]) -> ModelOutput:
    """Call the frozen mean once, without test-time horizontal averaging."""
    try:
        output = mean_model(
            spectrum=batch["spectrum"],
            window_mask=batch["window_mask"],
            process=batch["process"],
            quality=batch["quality"],
        )
    except KeyError as error:
        raise ValueError(f"scale batch is missing {error.args[0]!r}") from error
    if not isinstance(output, ModelOutput):
        raise ValueError("mean_model must return ModelOutput")
    return output


def _validation_nll(
    mean_model: SelectiveGatedModel,
    scale_model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> float:
    """Evaluate the registered swap-averaged predictor with exact weights."""
    total = 0.0
    total_weight = 0.0
    batch_count = 0
    # Keep the frozen mean in evaluation mode without invoking G1's specialised
    # ``train`` override a second time (see ``_freeze_mean_state``).
    nn.Module.train(mean_model, False)
    scale_model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _device_batch(raw_batch, device)
            process, quality, readings, weight = _required_scale_tensors(batch)
            # ``readings`` belongs solely to the scale loss.  The registered
            # swap helper accepts the exact mean-dataset key set, so preserve
            # its strict batch contract at this mean/scale boundary.
            mean_batch = {key: value for key, value in batch.items() if key != "readings"}
            output = average_swap_predictions(mean_model, mean_batch)
            features = build_scale_features(output, process, quality)
            loss = repeated_gaussian_nll(output.prediction, scale_model(features), readings, weight)
            weight_sum = float(weight.detach().sum().cpu())
            total += float(loss.detach().cpu()) * weight_sum
            total_weight += weight_sum
            batch_count += 1
    if batch_count == 0 or not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("validation loader must contain positive sample weight")
    value = total / total_weight
    if not math.isfinite(value):
        raise ValueError("validation NLL must be finite")
    return value


def fit_scale_model(
    *,
    mean_model: SelectiveGatedModel,
    scale_model: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    valid_loader: Iterable[Mapping[str, Any]],
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device,
    seed: int,
) -> ScaleFit:
    """Fit only a scale head while proving the complete G1 mean state is fixed.

    Optimizer batches use the original orientation, matching Phase A training.
    Epoch selection uses the deployed original/swapped average, so the selected
    scale estimates match evaluation and saved Phase B inference exactly.
    """
    if not isinstance(mean_model, SelectiveGatedModel):
        raise ValueError("mean_model must be a SelectiveGatedModel")
    if not isinstance(scale_model, nn.Module):
        raise ValueError("scale_model must be an nn.Module")
    if type(max_epochs) is not int or max_epochs < 1:
        raise ValueError("max_epochs must be a positive integer")
    if type(patience) is not int or patience < 1:
        raise ValueError("patience must be a positive integer")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in (learning_rate, weight_decay)
    ) or float(learning_rate) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative finite values")

    selected_device = torch.device(device)
    _set_scale_seed(seed)
    mean_snapshot = _freeze_mean_state(mean_model)
    mean_hash = _state_sha256(mean_snapshot)
    mean_model.to(selected_device)
    scale_model.to(selected_device)
    parameters = [parameter for parameter in scale_model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("scale_model must expose trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    best_epoch = 0
    best_validation = math.inf
    best_state: dict[str, Tensor] | None = None
    stalled_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        scale_model.train()
        train_total = 0.0
        train_weight = 0.0
        batch_count = 0
        for raw_batch in train_loader:
            batch = _device_batch(raw_batch, selected_device)
            process, quality, readings, weight = _required_scale_tensors(batch)
            with torch.no_grad():
                output = _current_orientation_output(mean_model, batch)
                features = build_scale_features(output, process, quality)
            optimizer.zero_grad(set_to_none=True)
            sigma = scale_model(features)
            loss = repeated_gaussian_nll(output.prediction, sigma, readings, weight)
            if not bool(torch.isfinite(loss)):
                raise ValueError("training NLL must be finite")
            loss.backward()
            optimizer.step()
            weight_sum = float(weight.detach().sum().cpu())
            train_total += float(loss.detach().cpu()) * weight_sum
            train_weight += weight_sum
            batch_count += 1
        if batch_count == 0 or not math.isfinite(train_weight) or train_weight <= 0.0:
            raise ValueError("training loader must contain positive sample weight")
        train_nll = train_total / train_weight
        if not math.isfinite(train_nll):
            raise ValueError("training NLL must be finite")

        validation_nll = _validation_nll(mean_model, scale_model, valid_loader, selected_device)
        history_rows.append(
            {"epoch": epoch, "train_nll": train_nll, "validation_nll": validation_nll}
        )
        if validation_nll < best_validation:
            best_epoch = epoch
            best_validation = validation_nll
            best_state = deepcopy(scale_model.state_dict())
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= patience:
                break

    if best_state is None:
        raise ValueError("no finite validation NLL was observed")
    scale_model.load_state_dict(best_state)
    assert_mean_model_unchanged(mean_model, mean_snapshot)
    return ScaleFit(
        model=scale_model,
        best_epoch=best_epoch,
        history=pd.DataFrame(history_rows, columns=["epoch", "train_nll", "validation_nll"]),
        mean_state_sha256=mean_hash,
    )


def build_nested_calibration(*args: Any, **kwargs: Any) -> Mapping[str, CalibrationArtifacts]:
    """Reserved for Task 6B's group-confined nested calibration implementation."""
    del args, kwargs
    raise NotImplementedError("build_nested_calibration is implemented in Task 6B")


def run_phase_b_fold(*args: Any, **kwargs: Any) -> PhaseBFoldArtifacts:
    """Reserved for Task 6B's outer-fold assembly implementation."""
    del args, kwargs
    raise NotImplementedError("run_phase_b_fold is implemented in Task 6B")


def run_phase_b(*args: Any, **kwargs: Any) -> PhaseBRunArtifacts:
    """Reserved for Task 7's all-fold/all-seed orchestration implementation."""
    del args, kwargs
    raise NotImplementedError("run_phase_b is implemented in Task 7")


__all__ = [
    "CALIBRATION_SCORE_COLUMNS",
    "PHASE_B_MEAN_COLUMNS",
    "PHASE_B_PREDICTION_COLUMNS",
    "SCALE_MODELS",
    "CalibrationArtifacts",
    "PhaseBFoldArtifacts",
    "PhaseBRunArtifacts",
    "PhaseBRunFingerprint",
    "ScaleFit",
    "build_nested_calibration",
    "fit_scale_model",
    "run_phase_b",
    "run_phase_b_fold",
]
