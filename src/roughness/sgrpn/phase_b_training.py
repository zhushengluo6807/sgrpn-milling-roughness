"""Phase B nested probability training, persistence, resume, and orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import random
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from roughness.scheme1.crossfit import make_group_inner_splits

from .config import PhaseAHandoff, PhaseBConfig, SGRPNConfig, validate_phase_b_output_root
from .crossfit import median_best_epoch
from .data import DataBundle, build_process_features, outer_indices, repeat_measure_batch
from .dataset import OrderBagDataset, collate_order_bags
from .models import (
    GlobalScale,
    ModelOutput,
    SelectiveGatedModel,
    VarianceHead,
    assert_mean_model_unchanged,
    average_swap_predictions,
    build_scale_features,
    repeated_gaussian_nll,
)
from .order_spectrum import OrderSpectrumCache
from .probability import (
    GroupConformalResult,
    conformal_interval,
    finite_sample_group_quantile,
    group_conformal_scores,
    raw_gaussian_interval,
)
from .training import MeanPathArtifacts, TrainingBackend, fit_g1_mean_path


# The two registered Phase B scale variants. Their order is persisted verbatim.
SCALE_MODELS = ("heteroscedastic", "homoscedastic")
MEAN_MODELS = ("P1", "R1", "G1")
PHASE_B_MODELS = (*MEAN_MODELS, *SCALE_MODELS)
PHASE_B_PROTOCOL = "sgrpn-phase-b-v1"
PHASE_B_STATE_ORDER = (
    "mean",
    "heteroscedastic",
    "homoscedastic",
    "calibration",
    "prediction",
    "complete",
)
PHASE_B_SEEDS = (20260723, 20260724, 20260725)
PHASE_B_ALPHAS = (0.10, 0.05)

CALIBRATION_SCORE_COLUMNS = (
    "group_id",
    "outer_fold",
    "inner_fold",
    "seed",
    "scale_model",
    "score",
    "region_count",
    "reading_count",
)

PHASE_B_MEAN_COLUMNS = (
    "sample_id",
    "group_id",
    "version",
    "fold",
    "seed",
    "model",
    "target_mean",
    "prediction",
    "sample_weight",
    "process_mean",
    "residual",
    "gate",
)

PHASE_B_PREDICTION_COLUMNS = (
    "sample_id",
    "group_id",
    "version",
    "fold",
    "seed",
    "scale_model",
    "target_mean",
    "ra_1",
    "ra_2",
    "ra_3",
    "sample_weight",
    "mu",
    "sigma",
    "gate",
    "correction",
    "raw_lower_90",
    "raw_upper_90",
    "raw_lower_95",
    "raw_upper_95",
    "conformal_q_90",
    "conformal_lower_90",
    "conformal_upper_90",
    "conformal_q_95",
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


def _refit_scale_model_exact_epochs(
    *,
    mean_model: SelectiveGatedModel,
    scale_model: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device,
    seed: int,
) -> ScaleFit:
    """Train a fresh scale head for the registered epoch count without selection."""
    if not isinstance(mean_model, SelectiveGatedModel):
        raise ValueError("mean_model must be a SelectiveGatedModel")
    if not isinstance(scale_model, nn.Module):
        raise ValueError("scale_model must be an nn.Module")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in (learning_rate, weight_decay)
    ) or float(learning_rate) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError(
            "learning_rate must be positive and weight_decay non-negative finite values"
        )

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
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
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
            loss = repeated_gaussian_nll(
                output.prediction, scale_model(features), readings, weight
            )
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
        history_rows.append({"epoch": epoch, "train_nll": train_nll})

    assert_mean_model_unchanged(mean_model, mean_snapshot)
    return ScaleFit(
        model=scale_model,
        best_epoch=epochs,
        history=pd.DataFrame(history_rows, columns=["epoch", "train_nll"]),
        mean_state_sha256=mean_hash,
    )


def build_nested_calibration(
    config: PhaseBConfig,
    phase_a_config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    fold: int,
    seed: int,
    *,
    device: str | torch.device,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
) -> Mapping[str, CalibrationArtifacts]:
    """Build group-confined OOF calibration scores for the two scale variants.

    This deliberately stops at calibration: it neither fits the final outer
    models nor creates any artifact on disk.  Each OOF block is produced by a
    fresh G1 path whose *entire* fitting universe is the corresponding inner
    training partition.  In particular, the held-out block's labels are only
    touched after its mean and scale predictions have been materialised.
    """
    if not isinstance(config, PhaseBConfig) or not isinstance(phase_a_config, SGRPNConfig):
        raise ValueError("nested calibration requires Phase B and Phase A configurations")
    if not isinstance(bundle, DataBundle) or not isinstance(cache, OrderSpectrumCache):
        raise ValueError("nested calibration requires a DataBundle and OrderSpectrumCache")
    if type(fold) is not int:
        raise ValueError("fold must be an integer")
    if type(seed) is not int or seed not in (20260723, 20260724, 20260725) or seed not in tuple(config.seeds):
        raise ValueError("seed must be a registered Phase B seed")
    if type(config.inner_splits) is not int or config.inner_splits != 4:
        raise ValueError("nested calibration requires exactly four inner splits")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    outer_train_index, _ = outer_indices(bundle, int(fold))
    outer_frame = bundle.manifest.iloc[outer_train_index].reset_index(drop=True).copy()
    outer_frame["sample_id"] = outer_frame["sample_id"].astype(str)
    outer_frame["group_id"] = outer_frame["group_id"].astype(str)
    group_count = outer_frame["group_id"].nunique()
    # ceil((n + 1) * .95) is an observed order statistic only when n >= 19.
    if group_count < 19:
        raise ValueError("nested calibration requires at least 19 calibration groups")

    splits = make_group_inner_splits(outer_frame, 4, seed)
    _validate_calibration_splits(outer_frame, splits)
    definition_rows: list[dict[str, Any]] = []
    predictions_by_scale: dict[str, list[pd.DataFrame]] = {
        name: [] for name in SCALE_MODELS
    }

    for predictor_fold, (train_rows, valid_rows) in enumerate(splits):
        train_frame = outer_frame.iloc[train_rows].reset_index(drop=True).copy()
        valid_frame = outer_frame.iloc[valid_rows].reset_index(drop=True).copy()
        train_ids = tuple(train_frame["sample_id"])
        valid_ids = tuple(valid_frame["sample_id"])
        train_groups = tuple(train_frame["group_id"])
        valid_groups = tuple(valid_frame["group_id"])
        _assert_disjoint_groups(train_groups, valid_groups, "nested predictor")

        # The exact sequence is part of the protocol boundary: fit_g1_mean_path
        # checks it against the registered outer-train universe itself.
        mean_path = fit_g1_mean_path(
            phase_a_config,
            bundle,
            cache,
            int(fold),
            seed,
            train_ids,
            device=device,
            backend=backend,
            batch_size=batch_size,
        )
        if tuple(mean_path.train_sample_ids) != train_ids:
            raise ValueError("mean path did not retain the confined inner-train IDs")

        train_batches = _calibration_batches(
            train_frame, cache, mean_path, batch_size=batch_size
        )
        valid_batches = _calibration_batches(
            valid_frame, cache, mean_path, batch_size=batch_size
        )
        scale_models: dict[str, nn.Module] = {}
        for scale_name, factory in _scale_factories().items():
            epoch_splits = make_group_inner_splits(train_frame, 4, seed)
            _validate_calibration_splits(train_frame, epoch_splits)
            selected_epochs: list[int] = []
            for scale_fold, (scale_train_rows, scale_valid_rows) in enumerate(epoch_splits):
                scale_train_frame = train_frame.iloc[scale_train_rows].reset_index(drop=True)
                scale_valid_frame = train_frame.iloc[scale_valid_rows].reset_index(drop=True)
                scale_train_groups = tuple(scale_train_frame["group_id"].astype(str))
                scale_valid_groups = tuple(scale_valid_frame["group_id"].astype(str))
                _assert_disjoint_groups(scale_train_groups, scale_valid_groups, "scale epoch selection")
                selected = fit_scale_model(
                    mean_model=mean_path.model,
                    scale_model=factory(),
                    train_loader=_calibration_batches(
                        scale_train_frame, cache, mean_path, batch_size=batch_size
                    ),
                    valid_loader=_calibration_batches(
                        scale_valid_frame, cache, mean_path, batch_size=batch_size
                    ),
                    max_epochs=config.max_epochs,
                    patience=config.patience,
                    learning_rate=config.variance_learning_rate,
                    weight_decay=config.weight_decay,
                    device=device,
                    seed=seed,
                )
                selected_epochs.append(selected.best_epoch)
                definition_rows.append(
                    _calibration_definition(
                        record_type="scale_selection",
                        fold=int(fold),
                        seed=seed,
                        predictor_fold=predictor_fold,
                        scale_model=scale_name,
                        train_sample_ids=tuple(scale_train_frame["sample_id"].astype(str)),
                        validation_sample_ids=tuple(scale_valid_frame["sample_id"].astype(str)),
                        train_group_ids=scale_train_groups,
                        validation_group_ids=scale_valid_groups,
                        source_sample_ids=train_ids,
                        source_group_ids=train_groups,
                        selected_epoch=selected.best_epoch,
                        scale_fold=scale_fold,
                    )
                )
            refit_epochs = median_best_epoch(selected_epochs)
            refit = _refit_scale_model_exact_epochs(
                mean_model=mean_path.model,
                scale_model=factory(),
                train_loader=train_batches,
                epochs=refit_epochs,
                learning_rate=config.variance_learning_rate,
                weight_decay=config.weight_decay,
                device=device,
                seed=seed,
            )
            scale_models[scale_name] = refit.model

        definition_rows.append(
            _calibration_definition(
                record_type="predictor",
                fold=int(fold),
                seed=seed,
                predictor_fold=predictor_fold,
                scale_model=None,
                train_sample_ids=train_ids,
                validation_sample_ids=valid_ids,
                train_group_ids=train_groups,
                validation_group_ids=valid_groups,
                source_sample_ids=train_ids,
                source_group_ids=train_groups,
                selected_epoch=None,
                scale_fold=None,
            )
        )
        for scale_name, scale_model in scale_models.items():
            predictions_by_scale[scale_name].append(
                _calibration_predictions(
                    valid_batches,
                    scale_model=scale_model,
                    mean_model=mean_path.model,
                    fold=int(fold),
                    seed=seed,
                    inner_fold=predictor_fold,
                    scale_name=scale_name,
                    device=torch.device(device),
                )
            )

    definitions = pd.DataFrame(definition_rows)
    results: dict[str, CalibrationArtifacts] = {}
    expected_ids = tuple(outer_frame["sample_id"])
    for scale_name, blocks in predictions_by_scale.items():
        predictions = pd.concat(blocks, ignore_index=True)
        _validate_calibration_oof(predictions, expected_ids)
        group_scores = _calibration_group_scores(predictions)
        quantiles = {
            float(alpha): finite_sample_group_quantile(group_scores, alpha=float(alpha))
            for alpha in config.alphas
        }
        results[scale_name] = CalibrationArtifacts(
            predictions=predictions,
            group_scores=group_scores,
            quantiles=quantiles,
            inner_fold_definitions=definitions.copy(),
        )
    return results


def _scale_factories() -> Mapping[str, type[nn.Module]]:
    return {"heteroscedastic": VarianceHead, "homoscedastic": GlobalScale}


def _assert_disjoint_groups(train_groups: Sequence[str], valid_groups: Sequence[str], scope: str) -> None:
    overlap = set(map(str, train_groups)) & set(map(str, valid_groups))
    if overlap:
        raise ValueError(f"{scope} groups overlap: {sorted(overlap)}")


def _validate_calibration_splits(
    frame: pd.DataFrame, splits: Sequence[tuple[np.ndarray, np.ndarray]]
) -> None:
    if len(splits) != 4:
        raise ValueError("nested calibration requires exactly four group splits")
    assigned = np.zeros(len(frame), dtype=np.int64)
    for train_rows, valid_rows in splits:
        train = np.asarray(train_rows, dtype=np.int64)
        valid = np.asarray(valid_rows, dtype=np.int64)
        if train.ndim != 1 or valid.ndim != 1 or len(train) == 0 or len(valid) == 0:
            raise ValueError("nested calibration splits must be nonempty one-dimensional rows")
        if np.intersect1d(train, valid).size:
            raise ValueError("nested calibration train and validation rows overlap")
        _assert_disjoint_groups(
            tuple(frame.iloc[train]["group_id"].astype(str)),
            tuple(frame.iloc[valid]["group_id"].astype(str)),
            "nested calibration",
        )
        assigned[valid] += 1
    if not np.all(assigned == 1):
        raise ValueError("each calibration row must be held out exactly once")


def _calibration_batches(
    frame: pd.DataFrame,
    cache: OrderSpectrumCache,
    mean_path: MeanPathArtifacts,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Build in-memory, non-augmented scale batches from confined scalers."""
    reset = frame.reset_index(drop=True).copy()
    repeats = repeat_measure_batch(reset)
    dataset = OrderBagDataset(
        reset,
        cache,
        mean_path.spectrum_scaler,
        mean_path.quality_scaler,
        mean_path.process_scaler.transform(build_process_features(reset)),
        repeats.mean,
        augment_horizontal_swap=False,
    )
    batches: list[dict[str, Any]] = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = collate_order_bags([dataset[index] for index in range(start, stop)])
        batch["readings"] = torch.as_tensor(repeats.readings[start:stop], dtype=torch.float32)
        batches.append(batch)
    if not batches:
        raise ValueError("calibration frame must not be empty")
    return batches


def _calibration_predictions(
    batches: Iterable[Mapping[str, Any]],
    *,
    scale_model: nn.Module,
    mean_model: SelectiveGatedModel,
    fold: int,
    seed: int,
    inner_fold: int,
    scale_name: str,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nn.Module.train(mean_model, False)
    scale_model.to(device).eval()
    with torch.no_grad():
        for raw_batch in batches:
            batch = _device_batch(raw_batch, device)
            mean_batch = {key: value for key, value in batch.items() if key != "readings"}
            output = average_swap_predictions(mean_model, mean_batch)
            sigma = scale_model(build_scale_features(output, batch["process"], batch["quality"]))
            mu_values = output.prediction.detach().cpu().numpy().astype(np.float64, copy=False)
            sigma_values = sigma.detach().cpu().numpy().astype(np.float64, copy=False)
            if not np.isfinite(mu_values).all() or not np.isfinite(sigma_values).all() or np.any(sigma_values <= 0):
                raise ValueError("calibration predictions must have finite mean and positive scale")
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "group_id": str(batch["group_id"][index]),
                        "fold": fold,
                        "seed": seed,
                        "inner_fold": inner_fold,
                        "scale_model": scale_name,
                        "target_mean": float(batch["target"][index].detach().cpu()),
                        "ra_1": float(batch["readings"][index, 0].detach().cpu()),
                        "ra_2": float(batch["readings"][index, 1].detach().cpu()),
                        "ra_3": float(batch["readings"][index, 2].detach().cpu()),
                        "mu": float(mu_values[index]),
                        "sigma": float(sigma_values[index]),
                    }
                )
    return pd.DataFrame(rows)


def _calibration_definition(
    *,
    record_type: str,
    fold: int,
    seed: int,
    predictor_fold: int,
    scale_model: str | None,
    train_sample_ids: tuple[str, ...],
    validation_sample_ids: tuple[str, ...],
    train_group_ids: tuple[str, ...],
    validation_group_ids: tuple[str, ...],
    source_sample_ids: tuple[str, ...],
    source_group_ids: tuple[str, ...],
    selected_epoch: int | None,
    scale_fold: int | None,
) -> dict[str, Any]:
    """Make the persisted-in-memory leakage audit unambiguous and redundant."""
    return {
        "record_type": record_type,
        "fold": fold,
        "seed": seed,
        "predictor_fold": predictor_fold,
        "scale_model": scale_model,
        "scale_fold": scale_fold,
        "selected_epoch": selected_epoch,
        "train_sample_ids": train_sample_ids,
        "validation_sample_ids": validation_sample_ids,
        "train_group_ids": train_group_ids,
        "validation_group_ids": validation_group_ids,
        "scaler_source_sample_ids": source_sample_ids,
        "scaler_source_group_ids": source_group_ids,
        "mean_fit_sample_ids": source_sample_ids,
        "mean_fit_group_ids": source_group_ids,
        "scale_fit_sample_ids": source_sample_ids,
        "scale_fit_group_ids": source_group_ids,
        "epoch_selection_sample_ids": source_sample_ids,
        "epoch_selection_group_ids": source_group_ids,
        "residual_source_sample_ids": source_sample_ids,
        "residual_source_group_ids": source_group_ids,
    }


def _validate_calibration_oof(predictions: pd.DataFrame, expected_ids: Sequence[str]) -> None:
    if tuple(predictions["sample_id"]) and predictions["sample_id"].duplicated().any():
        raise ValueError("calibration OOF predictions contain duplicate sample IDs")
    if set(predictions["sample_id"].astype(str)) != set(map(str, expected_ids)) or len(predictions) != len(expected_ids):
        raise ValueError("calibration OOF predictions must cover each outer-train sample exactly once")
    values = predictions[["mu", "sigma"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values[:, 1] <= 0.0):
        raise ValueError("calibration OOF predictions must be finite with positive sigma")


def _calibration_group_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    readings = predictions[["ra_1", "ra_2", "ra_3"]].to_numpy(dtype=np.float64)
    base = group_conformal_scores(
        predictions["group_id"].astype(str),
        predictions["mu"].to_numpy(dtype=np.float64),
        predictions["sigma"].to_numpy(dtype=np.float64),
        readings,
    )
    grouped = predictions.groupby("group_id", sort=True, observed=True)
    provenance_columns = ("fold", "inner_fold", "seed", "scale_model")
    if any((grouped[column].nunique(dropna=False) != 1).any() for column in provenance_columns):
        raise ValueError("calibration group provenance must be unique within each group")
    provenance = grouped[list(provenance_columns)].first().reset_index()
    provenance = provenance.rename(columns={"fold": "outer_fold"})
    counts = grouped.size().reset_index(name="region_count")
    scored = provenance.merge(counts, on="group_id", how="inner", validate="one_to_one")
    scored = scored.merge(base, on="group_id", how="inner", validate="one_to_one")
    scored["reading_count"] = 3 * scored["region_count"]
    scored = scored.loc[:, CALIBRATION_SCORE_COLUMNS]
    if (
        scored["group_id"].duplicated().any()
        or not np.isfinite(scored["score"].to_numpy(dtype=np.float64)).all()
        or (scored["score"] < 0.0).any()
    ):
        raise ValueError("calibration group scores must be unique, finite, and non-negative")
    return scored


def _select_and_refit_outer_scales(
    config: PhaseBConfig,
    outer_train: pd.DataFrame,
    cache: OrderSpectrumCache,
    mean_path: MeanPathArtifacts,
    *,
    seed: int,
    device: str | torch.device,
    batch_size: int,
) -> tuple[dict[str, nn.Module], dict[str, pd.DataFrame]]:
    """Select each fresh scale head inside outer train and refit it once."""
    splits = make_group_inner_splits(outer_train, 4, seed)
    _validate_calibration_splits(outer_train, splits)
    final_batches = _calibration_batches(outer_train, cache, mean_path, batch_size=batch_size)
    fitted: dict[str, nn.Module] = {}
    histories: dict[str, pd.DataFrame] = {}
    for scale_name, factory in _scale_factories().items():
        selected_epochs: list[int] = []
        for train_rows, valid_rows in splits:
            train_frame = outer_train.iloc[train_rows].reset_index(drop=True)
            valid_frame = outer_train.iloc[valid_rows].reset_index(drop=True)
            _assert_disjoint_groups(
                tuple(train_frame["group_id"].astype(str)),
                tuple(valid_frame["group_id"].astype(str)),
                "final scale epoch selection",
            )
            selected = fit_scale_model(
                mean_model=mean_path.model,
                scale_model=factory(),
                train_loader=_calibration_batches(train_frame, cache, mean_path, batch_size=batch_size),
                valid_loader=_calibration_batches(valid_frame, cache, mean_path, batch_size=batch_size),
                max_epochs=config.max_epochs,
                patience=config.patience,
                learning_rate=config.variance_learning_rate,
                weight_decay=config.weight_decay,
                device=device,
                seed=seed,
            )
            selected_epochs.append(selected.best_epoch)
        refit_epochs = median_best_epoch(selected_epochs)
        refit = _refit_scale_model_exact_epochs(
            mean_model=mean_path.model,
            scale_model=factory(),
            train_loader=final_batches,
            epochs=refit_epochs,
            learning_rate=config.variance_learning_rate,
            weight_decay=config.weight_decay,
            device=device,
            seed=seed,
        )
        for parameter in refit.model.parameters():
            parameter.requires_grad_(False)
        refit.model.eval()
        refit.model._phase_b_best_epochs = tuple(map(int, selected_epochs))
        refit.model._phase_b_refit_epoch = int(refit_epochs)
        fitted[scale_name] = refit.model
        histories[scale_name] = refit.history
    return fitted, histories


def _inference_batches(
    frame: pd.DataFrame,
    cache: OrderSpectrumCache,
    mean_path: MeanPathArtifacts,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Build feature-only outer-test batches without accessing response columns."""
    reset = frame.reset_index(drop=True).copy()
    dataset = OrderBagDataset(
        reset,
        cache,
        mean_path.spectrum_scaler,
        mean_path.quality_scaler,
        mean_path.process_scaler.transform(build_process_features(reset)),
        np.zeros(len(reset), dtype=np.float32),
        augment_horizontal_swap=False,
    )
    batches = [
        collate_order_bags([dataset[index] for index in range(start, min(start + batch_size, len(dataset)))])
        for start in range(0, len(dataset), batch_size)
    ]
    if not batches:
        raise ValueError("outer-test frame must not be empty")
    return batches


def _outer_inference(
    batches: Iterable[Mapping[str, Any]],
    *,
    mean_model: SelectiveGatedModel,
    scale_models: Mapping[str, nn.Module],
    calibration: Mapping[str, CalibrationArtifacts],
    fold: int,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the registered swap average and both frozen scale heads once."""
    probability_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    nn.Module.train(mean_model, False)
    for scale in scale_models.values():
        scale.to(device).eval()
    with torch.no_grad():
        for raw_batch in batches:
            batch = _device_batch(raw_batch, device)
            output = average_swap_predictions(mean_model, batch)
            if output.process_mean is None or output.residual is None or output.gate is None:
                raise ValueError(
                    "final G1 output must contain process mean, residual, and gate"
                )
            process_mean = output.process_mean.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            residual = output.residual.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            gate = output.gate.detach().cpu().numpy().astype(np.float64, copy=False)
            prediction = output.prediction.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            if not np.isfinite(
                np.column_stack((process_mean, residual, gate, prediction))
            ).all():
                raise ValueError("final mean predictions must be finite")
            zeros = np.zeros(len(prediction), dtype=np.float64)
            ones = np.ones(len(prediction), dtype=np.float64)
            values = {
                "P1": (process_mean, process_mean, zeros, zeros),
                "R1": (process_mean + residual, process_mean, residual, ones),
                "G1": (prediction, process_mean, residual, gate),
            }
            weights = batch["sample_weight"].detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            for model_name, components in values.items():
                model_prediction, model_process, model_residual, model_gate = components
                mean_rows.extend(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "group_id": str(batch["group_id"][index]),
                        "fold": fold,
                        "seed": seed,
                        "model": model_name,
                        "prediction": float(model_prediction[index]),
                        "sample_weight": float(weights[index]),
                        "process_mean": float(model_process[index]),
                        "residual": float(model_residual[index]),
                        "gate": float(model_gate[index]),
                    }
                    for index in range(len(prediction))
                )
            features = build_scale_features(output, batch["process"], batch["quality"])
            mu = prediction
            correction = prediction - process_mean
            for scale_name in SCALE_MODELS:
                scale_values = scale_models[scale_name](features).detach().cpu().numpy().astype(np.float64, copy=False)
                if not np.isfinite(scale_values).all() or np.any(scale_values <= 0.0):
                    raise ValueError("outer-test scale predictions must be finite and positive")
                for index in range(len(mu)):
                    probability_rows.append(
                        {
                            "sample_id": str(batch["sample_id"][index]),
                            "group_id": str(batch["group_id"][index]),
                            "fold": fold,
                            "seed": seed,
                            "scale_model": scale_name,
                            "sample_weight": float(weights[index]),
                            "mu": float(mu[index]),
                            "sigma": float(scale_values[index]),
                            "gate": float(gate[index]),
                            "correction": float(correction[index]),
                        }
                    )
    probabilities = pd.DataFrame(probability_rows)
    for scale_name in SCALE_MODELS:
        mask = probabilities["scale_model"] == scale_name
        rows = probabilities.loc[mask]
        for alpha, suffix in ((0.10, "90"), (0.05, "95")):
            lower, upper = raw_gaussian_interval(rows["mu"], rows["sigma"], alpha=alpha)
            probabilities.loc[mask, f"raw_lower_{suffix}"] = lower
            probabilities.loc[mask, f"raw_upper_{suffix}"] = upper
            conformal_lower, conformal_upper = conformal_interval(
                rows["mu"], rows["sigma"], quantile=calibration[scale_name].quantiles[alpha]
            )
            probabilities.loc[mask, f"conformal_q_{suffix}"] = calibration[
                scale_name
            ].quantiles[alpha].quantile
            probabilities.loc[mask, f"conformal_lower_{suffix}"] = conformal_lower
            probabilities.loc[mask, f"conformal_upper_{suffix}"] = conformal_upper
    return probabilities, pd.DataFrame(mean_rows)


def _phase_b_fingerprint(
    config: PhaseBConfig,
    handoff: PhaseAHandoff,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    *,
    fold: int,
    seed: int,
) -> PhaseBRunFingerprint:
    """Bind one Phase B unit to its protocol, handoff, and consumed inputs."""
    from .training import _cache_sha256, _frame_sha256

    def jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, Mapping):
            return {str(name): jsonable(item) for name, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [jsonable(item) for item in value]
        return value

    def canonical(payload: Any) -> bytes:
        return json.dumps(
            jsonable(payload),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    config_payload = jsonable(asdict(config))
    config_sha = hashlib.sha256(
        canonical(config_payload)
    ).hexdigest()
    manifest_sha = _frame_sha256(bundle.manifest)
    folds_sha = _frame_sha256(bundle.folds)
    cache_sha = _cache_sha256(cache)
    value = hashlib.sha256(
        canonical(
            {
                "protocol": PHASE_B_PROTOCOL,
                "phase_b_config": config_payload,
                "phase_b_config_sha256": config_sha,
                "phase_b_seeds": list(config.seeds),
                "phase_b_alphas": list(config.alphas),
                "models": [*MEAN_MODELS, *SCALE_MODELS],
                "fold": fold,
                "seed": seed,
                "phase_a_handoff": asdict(handoff),
                "manifest_sha256": manifest_sha,
                "folds_sha256": folds_sha,
                "cache_sha256": cache_sha,
            }
        )
    ).hexdigest()
    return PhaseBRunFingerprint(
        value=value,
        config_sha256=config_sha,
        phase_a_acceptance_sha256=handoff.acceptance_sha256,
        phase_a_run_manifest_sha256=handoff.run_manifest_sha256,
        phase_a_training_fingerprint=handoff.training_fingerprint,
        manifest_sha256=manifest_sha,
        folds_sha256=folds_sha,
        cache_sha256=cache_sha,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")


def _atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        temporary.write_bytes(content)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return _atomic_write_bytes(
        Path(path),
        json.dumps(
            _jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8"),
    )


def _atomic_write_frame(path: str | Path, frame: pd.DataFrame) -> Path:
    return _atomic_write_bytes(Path(path), frame.to_csv(index=False).encode("utf-8"))


def _atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> Path:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return _atomic_write_bytes(Path(path), buffer.getvalue())


def _atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return _atomic_write_bytes(Path(path), buffer.getvalue())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_b_artifact_paths(fold_dir: Path) -> dict[str, Path]:
    relatives = [
        *(f"mean/checkpoints/{model}.pt" for model in MEAN_MODELS),
        *(f"mean/history/{model}.csv" for model in MEAN_MODELS),
        *(f"scale/checkpoints/{model}.pt" for model in SCALE_MODELS),
        *(f"scale/history/{model}.csv" for model in SCALE_MODELS),
        "scalers/process.npz",
        "scalers/spectrum.npz",
        "scalers/quality.npz",
        "calibration/inner_folds.csv",
        "calibration/oof_predictions.csv",
        "calibration/group_scores.csv",
        "calibration/quantiles.json",
        "predictions.csv",
        "state.json",
    ]
    return {relative: fold_dir / relative for relative in relatives}


def _phase_b_state_payload(
    fingerprint: str,
    fold: int,
    seed: int,
    completed_stages: Sequence[str],
    *,
    status: str = "in_progress",
) -> dict[str, Any]:
    return {
        "protocol": PHASE_B_PROTOCOL,
        "fingerprint": fingerprint,
        "fold": int(fold),
        "seed": int(seed),
        "models": list(PHASE_B_MODELS),
        "completed_stages": list(completed_stages),
        "status": status,
    }


def _write_phase_b_state(
    fold_dir: Path,
    fingerprint: str,
    fold: int,
    seed: int,
    completed_stages: Sequence[str],
) -> Path:
    status = "complete" if tuple(completed_stages) == PHASE_B_STATE_ORDER else "in_progress"
    return _atomic_write_json(
        fold_dir / "state.json",
        _phase_b_state_payload(
            fingerprint, fold, seed, completed_stages, status=status
        ),
    )


def _phase_b_identity_matches(
    payload: Any,
    fingerprint: str,
    fold: int,
    seed: int,
    *,
    complete: bool,
) -> bool:
    stages = list(PHASE_B_STATE_ORDER if complete else PHASE_B_STATE_ORDER[:-1])
    return bool(
        isinstance(payload, dict)
        and payload.get("protocol") == PHASE_B_PROTOCOL
        and payload.get("fingerprint") == fingerprint
        and isinstance(payload.get("fold"), int)
        and not isinstance(payload.get("fold"), bool)
        and payload.get("fold") == fold
        and isinstance(payload.get("seed"), int)
        and not isinstance(payload.get("seed"), bool)
        and payload.get("seed") == seed
        and payload.get("models") == list(PHASE_B_MODELS)
        and payload.get("completed_stages") == stages
        and payload.get("status") == ("complete" if complete else "in_progress")
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _validate_phase_b_checkpoint(
    path: Path,
    *,
    model: str,
    fingerprint: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Phase B checkpoint {model} cannot be loaded") from error
    best_epochs = payload.get("best_epochs") if isinstance(payload, dict) else None
    refit_epoch = payload.get("refit_epoch") if isinstance(payload, dict) else None
    mean_hash = payload.get("mean_state_sha256") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != PHASE_B_PROTOCOL
        or payload.get("fingerprint") != fingerprint
        or payload.get("fold") != fold
        or isinstance(payload.get("fold"), bool)
        or payload.get("seed") != seed
        or isinstance(payload.get("seed"), bool)
        or payload.get("model") != model
        or not isinstance(best_epochs, list)
        or len(best_epochs) != 4
        or any(type(epoch) is not int or epoch < 1 for epoch in best_epochs)
        or type(refit_epoch) is not int
        or refit_epoch != median_best_epoch(best_epochs)
        or not isinstance(mean_hash, str)
        or len(mean_hash) != 64
        or any(character not in "0123456789abcdef" for character in mean_hash)
    ):
        raise ValueError(f"Phase B checkpoint {model} metadata is incompatible")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"Phase B checkpoint {model} model state is missing")
    if any(
        not isinstance(value, Tensor) or not bool(torch.isfinite(value).all())
        for value in model_state.values()
    ):
        raise ValueError(f"Phase B checkpoint {model} model state is invalid")
    return payload


def _validate_phase_b_history(
    path: Path, *, model: str, checkpoint: Mapping[str, Any]
) -> None:
    try:
        history = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as error:
        raise ValueError(f"Phase B history {model} cannot be loaded") from error
    metadata_columns = {
        "epoch",
        "protocol",
        "fingerprint",
        "fold",
        "seed",
        "model",
        "best_epochs",
        "refit_epoch",
        "mean_state_sha256",
    }
    if history.empty or not metadata_columns.issubset(history.columns):
        raise ValueError(f"Phase B history {model} schema is incompatible")
    try:
        epochs = pd.to_numeric(history["epoch"], errors="raise").to_numpy(dtype=np.float64)
        folds = pd.to_numeric(history["fold"], errors="raise").to_numpy(dtype=np.float64)
        seeds = pd.to_numeric(history["seed"], errors="raise").to_numpy(dtype=np.float64)
        refits = pd.to_numeric(history["refit_epoch"], errors="raise").to_numpy(
            dtype=np.float64
        )
        parsed_best = [json.loads(value) for value in history["best_epochs"]]
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Phase B history {model} values are incompatible") from error
    if (
        not np.isfinite(np.concatenate((epochs, folds, seeds, refits))).all()
        or np.any(epochs < 1)
        or np.any(epochs != np.floor(epochs))
        or set(history["protocol"]) != {PHASE_B_PROTOCOL}
        or set(history["fingerprint"]) != {str(checkpoint["fingerprint"])}
        or set(folds) != {float(checkpoint["fold"])}
        or set(seeds) != {float(checkpoint["seed"])}
        or set(history["model"]) != {model}
        or any(value != checkpoint["best_epochs"] for value in parsed_best)
        or set(refits) != {float(checkpoint["refit_epoch"])}
        or set(history["mean_state_sha256"]) != {str(checkpoint["mean_state_sha256"])}
    ):
        raise ValueError(f"Phase B history {model} metadata is incompatible")
    for column in history.columns:
        if column in metadata_columns or column in {"phase"}:
            continue
        try:
            values = pd.to_numeric(history[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Phase B history {model} numeric values are invalid") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Phase B history {model} numeric values are invalid")


def _validate_phase_b_scaler(
    path: Path,
    *,
    name: str,
    fingerprint: str,
    fold: int,
    seed: int,
) -> None:
    expected_shapes = {"process": (9,), "spectrum": (3, 361), "quality": (7,)}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"mean", "scale", "metadata_json"}:
                raise ValueError("schema")
            mean = archive["mean"]
            scale = archive["scale"]
            metadata_value = archive["metadata_json"]
    except Exception as error:
        raise ValueError(f"Phase B scaler {name} cannot be loaded") from error
    try:
        metadata = json.loads(str(metadata_value.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Phase B scaler {name} metadata is invalid") from error
    if (
        mean.shape != expected_shapes[name]
        or scale.shape != expected_shapes[name]
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
        or metadata
        != {
            "protocol": PHASE_B_PROTOCOL,
            "fingerprint": fingerprint,
            "fold": fold,
            "seed": seed,
            "scaler": name,
        }
    ):
        raise ValueError(f"Phase B scaler {name} is incompatible")


def _read_phase_b_probability_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "group_id": str,
            "version": str,
            "scale_model": str,
        },
    )
    if tuple(frame.columns) != PHASE_B_PREDICTION_COLUMNS or frame.empty:
        raise ValueError("Phase B persisted predictions schema is incompatible")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or frame.duplicated(["sample_id", "seed", "scale_model"]).any()
        or set(frame["scale_model"]) != set(SCALE_MODELS)
        or (frame["sigma"] <= 0.0).any()
    ):
        raise ValueError("Phase B persisted predictions values are incompatible")
    for suffix in ("90", "95"):
        if (
            frame[f"raw_lower_{suffix}"] > frame[f"raw_upper_{suffix}"]
        ).any() or (
            frame[f"conformal_lower_{suffix}"] > frame[f"conformal_upper_{suffix}"]
        ).any():
            raise ValueError("Phase B persisted prediction bounds are incompatible")
    return frame


def _phase_b_mean_frame_from_checkpoint(payload: Mapping[str, Any]) -> pd.DataFrame:
    raw = payload.get("mean_predictions")
    if not isinstance(raw, dict) or set(raw) != set(PHASE_B_MEAN_COLUMNS):
        raise ValueError("Phase B G1 checkpoint mean predictions are missing")
    frame = pd.DataFrame(raw, columns=PHASE_B_MEAN_COLUMNS)
    if frame.empty or frame.duplicated(["sample_id", "seed", "model"]).any():
        raise ValueError("Phase B persisted mean predictions are incompatible")
    if set(frame["model"]) != set(MEAN_MODELS):
        raise ValueError("Phase B persisted mean model coverage is incompatible")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Phase B persisted mean predictions must be finite")
    return frame


def _validate_phase_b_calibration(
    fold_dir: Path, *, fingerprint: str, fold: int, seed: int
) -> None:
    calibration_dir = fold_dir / "calibration"
    definitions = pd.read_csv(calibration_dir / "inner_folds.csv")
    definition_columns = {
        "record_type",
        "fold",
        "seed",
        "predictor_fold",
        "scale_model",
        "train_sample_ids",
        "validation_sample_ids",
        "train_group_ids",
        "validation_group_ids",
    }
    if (
        definitions.empty
        or not definition_columns.issubset(definitions.columns)
        or set(pd.to_numeric(definitions["fold"], errors="raise")) != {fold}
        or set(pd.to_numeric(definitions["seed"], errors="raise")) != {seed}
        or set(definitions["record_type"].dropna().astype(str))
        != {"predictor", "scale_selection"}
        or set(definitions["scale_model"].dropna().astype(str)) != set(SCALE_MODELS)
    ):
        raise ValueError("Phase B inner-fold definitions are incompatible")
    for column in (
        "train_sample_ids",
        "validation_sample_ids",
        "train_group_ids",
        "validation_group_ids",
    ):
        try:
            parsed = [json.loads(str(value)) for value in definitions[column]]
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Phase B inner-fold definition IDs are invalid") from error
        if any(not isinstance(value, list) for value in parsed):
            raise ValueError("Phase B inner-fold definition IDs are invalid")

    oof = pd.read_csv(
        calibration_dir / "oof_predictions.csv",
        dtype={"sample_id": str, "group_id": str, "scale_model": str},
    )
    expected_oof_columns = (
        "sample_id",
        "group_id",
        "fold",
        "seed",
        "inner_fold",
        "scale_model",
        "target_mean",
        "ra_1",
        "ra_2",
        "ra_3",
        "mu",
        "sigma",
    )
    if (
        tuple(oof.columns) != expected_oof_columns
        or oof.empty
        or set(oof["fold"]) != {fold}
        or set(oof["seed"]) != {seed}
        or set(oof["scale_model"]) != set(SCALE_MODELS)
        or oof.duplicated(["sample_id", "scale_model"]).any()
        or not np.isfinite(
            oof.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ).all()
        or (oof["sigma"] <= 0.0).any()
    ):
        raise ValueError("Phase B calibration OOF predictions are incompatible")
    sample_sets = {
        model: set(oof.loc[oof["scale_model"] == model, "sample_id"])
        for model in SCALE_MODELS
    }
    if sample_sets[SCALE_MODELS[0]] != sample_sets[SCALE_MODELS[1]]:
        raise ValueError("Phase B calibration model sample coverage is incompatible")

    scores = pd.read_csv(
        calibration_dir / "group_scores.csv",
        dtype={"group_id": str, "scale_model": str},
    )
    if (
        tuple(scores.columns) != CALIBRATION_SCORE_COLUMNS
        or scores.empty
        or set(scores["outer_fold"]) != {fold}
        or set(scores["seed"]) != {seed}
        or set(scores["scale_model"]) != set(SCALE_MODELS)
        or scores.duplicated(["group_id", "scale_model"]).any()
        or not np.isfinite(
            scores.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ).all()
        or (scores["score"] < 0.0).any()
        or (scores["reading_count"] != 3 * scores["region_count"]).any()
    ):
        raise ValueError("Phase B calibration group scores are incompatible")

    quantiles = _load_json_object(calibration_dir / "quantiles.json")
    if (
        quantiles.get("protocol") != PHASE_B_PROTOCOL
        or quantiles.get("fingerprint") != fingerprint
        or quantiles.get("fold") != fold
        or quantiles.get("seed") != seed
        or quantiles.get("models") != list(SCALE_MODELS)
        or quantiles.get("alphas") != list(PHASE_B_ALPHAS)
        or not isinstance(quantiles.get("quantiles"), dict)
        or set(quantiles["quantiles"]) != set(SCALE_MODELS)
    ):
        raise ValueError("Phase B calibration quantiles metadata is incompatible")
    expected_alpha_keys = {f"{alpha:.2f}" for alpha in PHASE_B_ALPHAS}
    for model in SCALE_MODELS:
        model_quantiles = quantiles["quantiles"][model]
        if not isinstance(model_quantiles, dict) or set(model_quantiles) != expected_alpha_keys:
            raise ValueError("Phase B calibration quantile coverage is incompatible")
        group_count = len(scores.loc[scores["scale_model"] == model])
        for alpha_key, raw in model_quantiles.items():
            alpha = float(alpha_key)
            if (
                not isinstance(raw, dict)
                or raw.get("alpha") != alpha
                or raw.get("group_count") != group_count
                or type(raw.get("order_index")) is not int
                or not 1 <= raw["order_index"] <= group_count
                or isinstance(raw.get("quantile"), bool)
                or not isinstance(raw.get("quantile"), (int, float))
                or not math.isfinite(float(raw["quantile"]))
                or float(raw["quantile"]) < 0.0
            ):
                raise ValueError("Phase B calibration quantile values are incompatible")


def _validate_completed_phase_b_artifacts(
    fold_dir: Path,
    marker: Mapping[str, Any],
    fingerprint: str,
    fold: int,
    seed: int,
    *,
    marker_published: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _phase_b_identity_matches(marker, fingerprint, fold, seed, complete=True):
        raise ValueError("Phase B completion identity is incompatible")
    expected = _phase_b_artifact_paths(fold_dir)
    hashes = marker.get("artifacts")
    if not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise ValueError("Phase B completion artifact set is incompatible")
    actual_files = {
        path.relative_to(fold_dir).as_posix()
        for path in fold_dir.rglob("*")
        if path.is_file()
    }
    expected_files = set(expected)
    if marker_published:
        expected_files.add("complete.json")
    if actual_files != expected_files:
        raise ValueError("Phase B completion directory tree is incompatible")
    for relative, path in expected.items():
        if not path.is_file() or hashes.get(relative) != _sha256_file(path):
            raise ValueError(f"Phase B completion artifact hash mismatch: {relative}")

    state = _load_json_object(fold_dir / "state.json")
    if not _phase_b_identity_matches(state, fingerprint, fold, seed, complete=True):
        raise ValueError("Phase B completed state is incompatible")

    checkpoints: dict[str, dict[str, Any]] = {}
    for model in PHASE_B_MODELS:
        family = "mean" if model in MEAN_MODELS else "scale"
        checkpoint = _validate_phase_b_checkpoint(
            fold_dir / family / "checkpoints" / f"{model}.pt",
            model=model,
            fingerprint=fingerprint,
            fold=fold,
            seed=seed,
        )
        checkpoints[model] = checkpoint
        _validate_phase_b_history(
            fold_dir / family / "history" / f"{model}.csv",
            model=model,
            checkpoint=checkpoint,
        )
    mean_hash = _state_sha256(checkpoints["G1"]["model_state"])
    if any(checkpoint["mean_state_sha256"] != mean_hash for checkpoint in checkpoints.values()):
        raise ValueError("Phase B checkpoint mean-state binding is incompatible")

    for scaler in ("process", "spectrum", "quality"):
        _validate_phase_b_scaler(
            fold_dir / "scalers" / f"{scaler}.npz",
            name=scaler,
            fingerprint=fingerprint,
            fold=fold,
            seed=seed,
        )
    _validate_phase_b_calibration(
        fold_dir, fingerprint=fingerprint, fold=fold, seed=seed
    )
    predictions = _read_phase_b_probability_predictions(fold_dir / "predictions.csv")
    mean_predictions = _phase_b_mean_frame_from_checkpoint(checkpoints["G1"])
    if (
        (
            "probability_rows" in marker
            and (
                type(marker.get("probability_rows")) is not int
                or marker.get("probability_rows") != len(predictions)
            )
        )
        or (
            "mean_rows" in marker
            and (
                type(marker.get("mean_rows")) is not int
                or marker.get("mean_rows") != len(mean_predictions)
            )
        )
        or set(predictions["fold"]) != {fold}
        or set(predictions["seed"]) != {seed}
        or set(mean_predictions["fold"]) != {fold}
        or set(mean_predictions["seed"]) != {seed}
        or set(predictions["sample_id"]) != set(mean_predictions["sample_id"])
        or len(predictions) * len(MEAN_MODELS)
        != len(mean_predictions) * len(SCALE_MODELS)
    ):
        raise ValueError("Phase B persisted outer prediction coverage is incompatible")
    return predictions, mean_predictions


def completed_phase_b_fold_matches(
    marker: str | Path,
    fingerprint: str | PhaseBRunFingerprint,
    *,
    fold: int,
    seed: int,
) -> bool:
    """Return true only for an exact, deeply valid completed Phase B unit."""
    if type(fold) is not int or type(seed) is not int:
        return False
    expected_fingerprint = (
        fingerprint.value if isinstance(fingerprint, PhaseBRunFingerprint) else str(fingerprint)
    )
    marker_path = Path(marker)
    try:
        payload = _load_json_object(marker_path)
        _validate_completed_phase_b_artifacts(
            marker_path.parent,
            payload,
            expected_fingerprint,
            fold,
            seed,
            marker_published=True,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def _validate_fold_predictions(
    predictions: pd.DataFrame,
    outer_test: pd.DataFrame,
    *,
    fold: int,
    seed: int,
) -> None:
    if list(predictions.columns) != list(PHASE_B_PREDICTION_COLUMNS):
        raise ValueError("Phase B prediction columns are incompatible")
    expected_ids = tuple(outer_test["sample_id"].astype(str))
    if len(predictions) != len(expected_ids) * len(SCALE_MODELS):
        raise ValueError("Phase B predictions must cover every test sample and scale model")
    if predictions.duplicated(["sample_id", "seed", "scale_model"]).any():
        raise ValueError("Phase B predictions contain duplicate sample/seed/model rows")
    if set(predictions["sample_id"].astype(str)) != set(expected_ids):
        raise ValueError("Phase B predictions must cover exactly the outer-test samples")
    if set(predictions["scale_model"]) != set(SCALE_MODELS):
        raise ValueError("Phase B predictions must include both registered scale models")
    if set(predictions["fold"]) != {fold} or set(predictions["seed"]) != {seed}:
        raise ValueError("Phase B prediction fold or seed is incompatible")
    numeric = predictions.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (predictions["sigma"] <= 0.0).any():
        raise ValueError("Phase B predictions must be finite with positive sigma")
    for suffix in ("90", "95"):
        if (
            predictions[f"raw_lower_{suffix}"] > predictions[f"raw_upper_{suffix}"]
        ).any() or (
            predictions[f"conformal_lower_{suffix}"] > predictions[f"conformal_upper_{suffix}"]
        ).any():
            raise ValueError("Phase B intervals must have ordered bounds")


def _checkpoint_metadata(
    *,
    fingerprint: str,
    fold: int,
    seed: int,
    model: str,
    best_epochs: Sequence[int],
    refit_epoch: int,
    mean_state_sha256: str,
) -> dict[str, Any]:
    epochs = list(map(int, best_epochs))
    if len(epochs) != 4 or any(epoch < 1 for epoch in epochs):
        raise ValueError(f"Phase B {model} requires four positive best epochs")
    if int(refit_epoch) != median_best_epoch(epochs):
        raise ValueError(f"Phase B {model} refit epoch is incompatible")
    return {
        "protocol": PHASE_B_PROTOCOL,
        "fingerprint": fingerprint,
        "fold": int(fold),
        "seed": int(seed),
        "model": model,
        "best_epochs": epochs,
        "refit_epoch": int(refit_epoch),
        "mean_state_sha256": mean_state_sha256,
    }


def _history_with_metadata(
    history: pd.DataFrame, metadata: Mapping[str, Any]
) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame) or history.empty or "epoch" not in history:
        raise ValueError(f"Phase B {metadata['model']} history must be non-empty")
    persisted = history.reset_index(drop=True).copy()
    persisted["protocol"] = metadata["protocol"]
    persisted["fingerprint"] = metadata["fingerprint"]
    persisted["fold"] = metadata["fold"]
    persisted["seed"] = metadata["seed"]
    persisted["model"] = metadata["model"]
    persisted["best_epochs"] = json.dumps(
        metadata["best_epochs"], separators=(",", ":")
    )
    persisted["refit_epoch"] = metadata["refit_epoch"]
    persisted["mean_state_sha256"] = metadata["mean_state_sha256"]
    return persisted


def _serialize_inner_fold_definitions(frame: pd.DataFrame) -> pd.DataFrame:
    persisted = frame.reset_index(drop=True).copy()
    for column in persisted.columns:
        if column.endswith("_ids"):
            persisted[column] = persisted[column].map(
                lambda value: json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
            )
    return persisted


def _mean_model_state(
    full_state: Mapping[str, Tensor], model: str
) -> dict[str, Tensor]:
    prefix = {"P1": "process_expert.", "R1": "residual_expert."}.get(model)
    if prefix is None:
        selected = dict(full_state)
    else:
        selected = {
            name[len(prefix) :]: value for name, value in full_state.items() if name.startswith(prefix)
        }
        if not selected:
            selected = dict(full_state)
    return {name: value.detach().cpu().clone() for name, value in selected.items()}


def _publish_phase_b_fold(
    *,
    fold_dir: Path,
    fingerprint: PhaseBRunFingerprint,
    fold: int,
    seed: int,
    mean_path: MeanPathArtifacts,
    scale_models: Mapping[str, nn.Module],
    scale_histories: Mapping[str, pd.DataFrame],
    calibration: Mapping[str, CalibrationArtifacts],
    predictions: pd.DataFrame,
    mean_predictions: pd.DataFrame,
) -> None:
    if set(scale_models) != set(SCALE_MODELS) or set(scale_histories) != set(SCALE_MODELS):
        raise ValueError("Phase B final scale model set is incompatible")
    if set(calibration) != set(SCALE_MODELS):
        raise ValueError("Phase B calibration model set is incompatible")
    fold_dir.mkdir(parents=True, exist_ok=True)
    full_mean_state = {
        name: value.detach().cpu().clone()
        for name, value in mean_path.model.state_dict().items()
    }
    if not full_mean_state:
        raise ValueError("Phase B final mean model state is empty")
    mean_hash = _state_sha256(full_mean_state)

    for model in MEAN_MODELS:
        best_epochs = tuple(map(int, mean_path.best_epochs[model]))
        refit_epoch = median_best_epoch(best_epochs)
        metadata = _checkpoint_metadata(
            fingerprint=fingerprint.value,
            fold=fold,
            seed=seed,
            model=model,
            best_epochs=best_epochs,
            refit_epoch=refit_epoch,
            mean_state_sha256=mean_hash,
        )
        checkpoint = {
            **metadata,
            "model_state": _mean_model_state(full_mean_state, model),
        }
        if model == "G1":
            checkpoint["mean_predictions"] = mean_predictions.to_dict(orient="list")
        _atomic_torch_save(
            fold_dir / "mean" / "checkpoints" / f"{model}.pt", checkpoint
        )
        _atomic_write_frame(
            fold_dir / "mean" / "history" / f"{model}.csv",
            _history_with_metadata(mean_path.histories[model], metadata),
        )

    scaler_metadata = {
        "protocol": PHASE_B_PROTOCOL,
        "fingerprint": fingerprint.value,
        "fold": fold,
        "seed": seed,
    }
    for name, scaler in (
        ("process", mean_path.process_scaler),
        ("spectrum", mean_path.spectrum_scaler),
        ("quality", mean_path.quality_scaler),
    ):
        metadata_json = json.dumps(
            {**scaler_metadata, "scaler": name},
            sort_keys=True,
            separators=(",", ":"),
        )
        _atomic_save_npz(
            fold_dir / "scalers" / f"{name}.npz",
            mean=np.asarray(scaler.mean),
            scale=np.asarray(scaler.scale),
            metadata_json=np.asarray(metadata_json),
        )
    _write_phase_b_state(fold_dir, fingerprint.value, fold, seed, ("mean",))

    for index, model in enumerate(SCALE_MODELS):
        history = scale_histories[model]
        if not isinstance(history, pd.DataFrame) or history.empty:
            raise ValueError(f"Phase B {model} history must be non-empty")
        refit_epoch = int(getattr(scale_models[model], "_phase_b_refit_epoch", 0))
        if refit_epoch < 1:
            refit_epoch = int(pd.to_numeric(history["epoch"], errors="raise").max())
        best_epochs = tuple(
            map(
                int,
                getattr(scale_models[model], "_phase_b_best_epochs", (refit_epoch,) * 4),
            )
        )
        metadata = _checkpoint_metadata(
            fingerprint=fingerprint.value,
            fold=fold,
            seed=seed,
            model=model,
            best_epochs=best_epochs,
            refit_epoch=refit_epoch,
            mean_state_sha256=mean_hash,
        )
        _atomic_torch_save(
            fold_dir / "scale" / "checkpoints" / f"{model}.pt",
            {
                **metadata,
                "model_state": {
                    name: value.detach().cpu().clone()
                    for name, value in scale_models[model].state_dict().items()
                },
            },
        )
        _atomic_write_frame(
            fold_dir / "scale" / "history" / f"{model}.csv",
            _history_with_metadata(history, metadata),
        )
        _write_phase_b_state(
            fold_dir,
            fingerprint.value,
            fold,
            seed,
            PHASE_B_STATE_ORDER[: 2 + index],
        )

    definitions = calibration[SCALE_MODELS[0]].inner_fold_definitions
    for model in SCALE_MODELS[1:]:
        if not definitions.equals(calibration[model].inner_fold_definitions):
            raise ValueError("Phase B calibration inner-fold definitions disagree")
    oof_predictions = pd.concat(
        [calibration[model].predictions for model in SCALE_MODELS], ignore_index=True
    )
    group_scores = pd.concat(
        [calibration[model].group_scores for model in SCALE_MODELS], ignore_index=True
    )
    quantiles_payload = {
        "protocol": PHASE_B_PROTOCOL,
        "fingerprint": fingerprint.value,
        "fold": fold,
        "seed": seed,
        "models": list(SCALE_MODELS),
        "alphas": list(PHASE_B_ALPHAS),
        "quantiles": {
            model: {
                f"{alpha:.2f}": asdict(calibration[model].quantiles[alpha])
                for alpha in PHASE_B_ALPHAS
            }
            for model in SCALE_MODELS
        },
    }
    _atomic_write_frame(
        fold_dir / "calibration" / "inner_folds.csv",
        _serialize_inner_fold_definitions(definitions),
    )
    _atomic_write_frame(
        fold_dir / "calibration" / "oof_predictions.csv", oof_predictions
    )
    _atomic_write_frame(
        fold_dir / "calibration" / "group_scores.csv", group_scores
    )
    _atomic_write_json(fold_dir / "calibration" / "quantiles.json", quantiles_payload)
    _write_phase_b_state(
        fold_dir, fingerprint.value, fold, seed, PHASE_B_STATE_ORDER[:4]
    )

    _atomic_write_frame(fold_dir / "predictions.csv", predictions)
    _write_phase_b_state(
        fold_dir, fingerprint.value, fold, seed, PHASE_B_STATE_ORDER[:5]
    )
    _write_phase_b_state(
        fold_dir, fingerprint.value, fold, seed, PHASE_B_STATE_ORDER
    )
    marker = _phase_b_state_payload(
        fingerprint.value,
        fold,
        seed,
        PHASE_B_STATE_ORDER,
        status="complete",
    )
    marker["artifacts"] = {
        relative: _sha256_file(path)
        for relative, path in _phase_b_artifact_paths(fold_dir).items()
    }
    marker["probability_rows"] = len(predictions)
    marker["mean_rows"] = len(mean_predictions)
    _validate_completed_phase_b_artifacts(
        fold_dir,
        marker,
        fingerprint.value,
        fold,
        seed,
        marker_published=False,
    )
    _atomic_write_json(fold_dir / "complete.json", marker)


def _load_phase_b_calibration(fold_dir: Path) -> dict[str, CalibrationArtifacts]:
    calibration_dir = fold_dir / "calibration"
    definitions = pd.read_csv(calibration_dir / "inner_folds.csv")
    for column in definitions.columns:
        if column.endswith("_ids"):
            definitions[column] = definitions[column].map(
                lambda value: tuple(json.loads(str(value)))
            )
    oof = pd.read_csv(
        calibration_dir / "oof_predictions.csv",
        dtype={"sample_id": str, "group_id": str, "scale_model": str},
    )
    scores = pd.read_csv(
        calibration_dir / "group_scores.csv",
        dtype={"group_id": str, "scale_model": str},
    )
    raw_quantiles = _load_json_object(calibration_dir / "quantiles.json")["quantiles"]
    return {
        model: CalibrationArtifacts(
            predictions=oof.loc[oof["scale_model"] == model].reset_index(drop=True),
            group_scores=scores.loc[scores["scale_model"] == model].reset_index(drop=True),
            quantiles={
                float(alpha): GroupConformalResult(**payload)
                for alpha, payload in raw_quantiles[model].items()
            },
            inner_fold_definitions=definitions.copy(),
        )
        for model in SCALE_MODELS
    }


def _load_completed_phase_b_fold(
    fold_dir: Path,
    fingerprint: PhaseBRunFingerprint,
    fold: int,
    seed: int,
) -> PhaseBFoldArtifacts:
    marker = _load_json_object(fold_dir / "complete.json")
    predictions, mean_predictions = _validate_completed_phase_b_artifacts(
        fold_dir,
        marker,
        fingerprint.value,
        fold,
        seed,
        marker_published=True,
    )
    predictions.attrs["mean_predictions"] = mean_predictions
    return PhaseBFoldArtifacts(
        predictions=predictions,
        calibration=_load_phase_b_calibration(fold_dir),
        checkpoint_paths={
            model: fold_dir / "scale" / "checkpoints" / f"{model}.pt"
            for model in SCALE_MODELS
        },
        history_paths={
            model: fold_dir / "scale" / "history" / f"{model}.csv"
            for model in SCALE_MODELS
        },
        fingerprint=fingerprint,
    )


def run_phase_b_fold(
    config: PhaseBConfig,
    handoff: PhaseAHandoff,
    phase_a_config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    fold: int,
    seed: int,
    *,
    device: str | torch.device | None = None,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
    output_root: str | Path | None = None,
) -> PhaseBFoldArtifacts:
    """Train one probability fold entirely in memory.

    Calibration deliberately completes before this function asks the final G1
    path to see all outer-train groups.  The outer-test rows are represented by
    feature-only batches until both scale models and conformal quantiles are
    final, which keeps their labels on the scoring side of the boundary.
    """
    if not isinstance(config, PhaseBConfig) or not isinstance(handoff, PhaseAHandoff):
        raise ValueError("run_phase_b_fold requires Phase B configuration and Phase A handoff")
    if not isinstance(phase_a_config, SGRPNConfig):
        raise ValueError("run_phase_b_fold requires a Phase A configuration")
    if not isinstance(bundle, DataBundle) or not isinstance(cache, OrderSpectrumCache):
        raise ValueError("run_phase_b_fold requires a DataBundle and OrderSpectrumCache")
    if type(fold) is not int:
        raise ValueError("fold must be an integer")
    if type(seed) is not int or seed not in tuple(config.seeds):
        raise ValueError("seed must be a registered Phase B seed")
    if tuple(config.seeds) != PHASE_B_SEEDS:
        raise ValueError("Phase B seeds must match the exact registered sequence")
    if tuple(map(float, config.alphas)) != PHASE_B_ALPHAS:
        raise ValueError("Phase B alphas must match the exact registered sequence")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    selected_device: str | torch.device = "cpu" if device is None else device
    root = validate_phase_b_output_root(config.output_dir, output_root=output_root)
    fingerprint = _phase_b_fingerprint(
        config, handoff, bundle, cache, fold=fold, seed=seed
    )
    fold_dir = root / "folds" / f"fold_{fold}" / f"seed_{seed}"
    marker_path = fold_dir / "complete.json"
    if marker_path.exists():
        if not completed_phase_b_fold_matches(
            marker_path, fingerprint, fold=fold, seed=seed
        ):
            raise ValueError("incompatible completed Phase B fold")
        return _load_completed_phase_b_fold(fold_dir, fingerprint, fold, seed)
    if fold_dir.exists() and any(fold_dir.iterdir()):
        raise ValueError("incompatible partial Phase B fold artifacts")

    outer_train_index, outer_test_index = outer_indices(bundle, fold)
    outer_train = bundle.manifest.iloc[outer_train_index].reset_index(drop=True).copy()
    outer_train["sample_id"] = outer_train["sample_id"].astype(str)

    # This is intentionally first: no final model is selected from a path that
    # had access to an outer-test reading.
    calibration = build_nested_calibration(
        config,
        phase_a_config,
        bundle,
        cache,
        fold,
        seed,
        device=selected_device,
        backend=backend,
        batch_size=batch_size,
    )
    final_mean = fit_g1_mean_path(
        phase_a_config,
        bundle,
        cache,
        fold,
        seed,
        tuple(outer_train["sample_id"]),
        device=selected_device,
        backend=backend,
        batch_size=batch_size,
    )
    final_scales, scale_histories = _select_and_refit_outer_scales(
        config,
        outer_train,
        cache,
        final_mean,
        seed=seed,
        device=selected_device,
        batch_size=batch_size,
    )

    # Do not call repeat_measure_batch on this frame before all fitted state and
    # quantiles are final.  Feature-only bags use dummy targets and carry no
    # readings at all.
    outer_test_features = bundle.manifest.iloc[outer_test_index].reset_index(drop=True).copy()
    inference_batches = _inference_batches(
        outer_test_features, cache, final_mean, batch_size=batch_size
    )
    inferred, mean_predictions = _outer_inference(
        inference_batches,
        mean_model=final_mean.model,
        scale_models=final_scales,
        calibration=calibration,
        fold=fold,
        seed=seed,
        device=torch.device(selected_device),
    )

    # This is the sole outer-test readout.  It is joined after inference and
    # therefore cannot influence a fitted model, scaler, or quantile.
    outer_labels = repeat_measure_batch(outer_test_features)
    label_frame = outer_test_features.loc[
        :, ["sample_id", "group_id", "version"]
    ].copy()
    label_frame["sample_id"] = label_frame["sample_id"].astype(str)
    label_frame["group_id"] = label_frame["group_id"].astype(str)
    label_frame["version"] = label_frame["version"].astype(str)
    label_frame["target_mean"] = outer_labels.mean
    label_frame["ra_1"] = outer_labels.readings[:, 0]
    label_frame["ra_2"] = outer_labels.readings[:, 1]
    label_frame["ra_3"] = outer_labels.readings[:, 2]
    predictions = inferred.merge(
        label_frame, on=["sample_id", "group_id"], how="inner", validate="many_to_one"
    )
    predictions = predictions.loc[:, PHASE_B_PREDICTION_COLUMNS]
    _validate_fold_predictions(predictions, outer_test_features, fold=fold, seed=seed)
    mean_predictions = mean_predictions.merge(
        label_frame.loc[:, ["sample_id", "group_id", "version", "target_mean"]],
        on=["sample_id", "group_id"],
        how="inner",
        validate="many_to_one",
    )
    mean_predictions = mean_predictions.loc[:, PHASE_B_MEAN_COLUMNS]
    predictions.attrs["mean_predictions"] = mean_predictions

    _publish_phase_b_fold(
        fold_dir=fold_dir,
        fingerprint=fingerprint,
        fold=fold,
        seed=seed,
        mean_path=final_mean,
        scale_models=final_scales,
        scale_histories=scale_histories,
        calibration=calibration,
        predictions=predictions,
        mean_predictions=mean_predictions,
    )
    checkpoint_paths = {
        name: fold_dir / "scale" / "checkpoints" / f"{name}.pt"
        for name in SCALE_MODELS
    }
    history_paths = {
        name: fold_dir / "scale" / "history" / f"{name}.csv"
        for name in SCALE_MODELS
    }
    return PhaseBFoldArtifacts(
        predictions=predictions,
        calibration=calibration,
        checkpoint_paths=checkpoint_paths,
        history_paths=history_paths,
        fingerprint=fingerprint,
    )


def _validate_phase_b_oof_cartesian(
    probability: pd.DataFrame,
    mean: pd.DataFrame,
    bundle: DataBundle,
) -> None:
    if tuple(probability.columns) != PHASE_B_PREDICTION_COLUMNS:
        raise ValueError("combined Phase B probability columns are incompatible")
    if tuple(mean.columns) != PHASE_B_MEAN_COLUMNS:
        raise ValueError("combined Phase B mean columns are incompatible")
    if len(bundle.manifest) != 586 or len(probability) != 3516 or len(mean) != 5274:
        raise ValueError("combined Phase B OOF row counts are incompatible")
    if (
        probability.duplicated(["sample_id", "seed", "scale_model"]).any()
        or mean.duplicated(["sample_id", "seed", "model"]).any()
        or set(probability["seed"]) != set(PHASE_B_SEEDS)
        or set(mean["seed"]) != set(PHASE_B_SEEDS)
        or set(probability["scale_model"]) != set(SCALE_MODELS)
        or set(mean["model"]) != set(MEAN_MODELS)
    ):
        raise ValueError("combined Phase B OOF Cartesian keys are incompatible")
    expected_ids = set(bundle.manifest["sample_id"].astype(str))
    if set(probability["sample_id"].astype(str)) != expected_ids or set(
        mean["sample_id"].astype(str)
    ) != expected_ids:
        raise ValueError("combined Phase B OOF sample coverage is incompatible")
    for seed in PHASE_B_SEEDS:
        for model in SCALE_MODELS:
            rows = probability.loc[
                (probability["seed"] == seed) & (probability["scale_model"] == model)
            ]
            if set(rows["sample_id"].astype(str)) != expected_ids or len(rows) != 586:
                raise ValueError("combined Phase B probability Cartesian coverage is incomplete")
        for model in MEAN_MODELS:
            rows = mean.loc[(mean["seed"] == seed) & (mean["model"] == model)]
            if set(rows["sample_id"].astype(str)) != expected_ids or len(rows) != 586:
                raise ValueError("combined Phase B mean Cartesian coverage is incomplete")

    fold_rows = bundle.folds.loc[:, ["sample_id", "fold"]].copy()
    fold_rows["sample_id"] = fold_rows["sample_id"].astype(str)
    if fold_rows["sample_id"].duplicated().any() or set(fold_rows["sample_id"]) != expected_ids:
        raise ValueError("combined Phase B fold assignments are incompatible")
    expected_fold = fold_rows.set_index("sample_id")["fold"].astype(int)
    for frame in (probability, mean):
        actual_fold = frame["sample_id"].astype(str).map(expected_fold)
        if not np.array_equal(
            pd.to_numeric(frame["fold"], errors="raise").to_numpy(dtype=np.int64),
            actual_fold.to_numpy(dtype=np.int64),
        ):
            raise ValueError("combined Phase B row fold binding is incompatible")
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("combined Phase B OOF numeric values must be finite")


def run_phase_b(
    config: PhaseBConfig,
    handoff: PhaseAHandoff,
    phase_a_config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    *,
    device: str | torch.device | None = None,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
    output_root: str | Path | None = None,
) -> PhaseBRunArtifacts:
    """Run the fixed five-fold, three-seed Phase B protocol without selection."""
    if not isinstance(config, PhaseBConfig) or not isinstance(handoff, PhaseAHandoff):
        raise ValueError("run_phase_b requires Phase B configuration and Phase A handoff")
    if not isinstance(phase_a_config, SGRPNConfig):
        raise ValueError("run_phase_b requires a Phase A configuration")
    if not isinstance(bundle, DataBundle) or not isinstance(cache, OrderSpectrumCache):
        raise ValueError("run_phase_b requires a DataBundle and OrderSpectrumCache")
    if tuple(config.seeds) != PHASE_B_SEEDS or tuple(map(float, config.alphas)) != PHASE_B_ALPHAS:
        raise ValueError("run_phase_b requires the exact registered seeds and alphas")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    root = validate_phase_b_output_root(config.output_dir, output_root=output_root)
    fold_values = pd.to_numeric(bundle.folds["fold"], errors="raise")
    if (
        not np.equal(fold_values, np.floor(fold_values)).all()
        or set(fold_values.astype(int)) != set(range(5))
    ):
        raise ValueError("run_phase_b requires exactly outer folds 0 through 4")

    fold_artifacts: dict[tuple[int, int], PhaseBFoldArtifacts] = {}
    fingerprints: dict[tuple[int, int], PhaseBRunFingerprint] = {}
    for fold in range(5):
        for seed in PHASE_B_SEEDS:
            artifact = run_phase_b_fold(
                config,
                handoff,
                phase_a_config,
                bundle,
                cache,
                fold,
                seed,
                device=device,
                backend=backend,
                batch_size=batch_size,
                output_root=root,
            )
            key = (fold, seed)
            fold_artifacts[key] = artifact
            fingerprints[key] = artifact.fingerprint

    probability_blocks: list[pd.DataFrame] = []
    for key in fold_artifacts:
        block = fold_artifacts[key].predictions.copy()
        block.attrs = {}
        probability_blocks.append(block)
    probability = pd.concat(probability_blocks, ignore_index=True)
    mean = pd.concat(
        [
            fold_artifacts[key].predictions.attrs["mean_predictions"]
            for key in fold_artifacts
        ],
        ignore_index=True,
    )
    probability = probability.loc[:, PHASE_B_PREDICTION_COLUMNS]
    mean = mean.loc[:, PHASE_B_MEAN_COLUMNS]
    _validate_phase_b_oof_cartesian(probability, mean, bundle)
    prediction_dir = root / "predictions"
    _atomic_write_frame(
        prediction_dir / "oof_probability_predictions.csv", probability
    )
    _atomic_write_frame(prediction_dir / "oof_mean_predictions.csv", mean)
    return PhaseBRunArtifacts(
        probability_predictions=probability,
        mean_predictions=mean,
        fold_artifacts=fold_artifacts,
        fingerprints=fingerprints,
    )


__all__ = [
    "CALIBRATION_SCORE_COLUMNS",
    "MEAN_MODELS",
    "PHASE_B_MEAN_COLUMNS",
    "PHASE_B_MODELS",
    "PHASE_B_PREDICTION_COLUMNS",
    "PHASE_B_PROTOCOL",
    "PHASE_B_STATE_ORDER",
    "SCALE_MODELS",
    "CalibrationArtifacts",
    "PhaseBFoldArtifacts",
    "PhaseBRunArtifacts",
    "PhaseBRunFingerprint",
    "ScaleFit",
    "build_nested_calibration",
    "completed_phase_b_fold_matches",
    "fit_scale_model",
    "run_phase_b",
    "run_phase_b_fold",
]
