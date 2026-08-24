"""Phase B probability training.

This module deliberately starts with the scale-head fitting primitive.  The
nested calibration and outer-fold orchestration APIs are declared here so their
public contracts are stable, but are implemented by the subsequent Task 6B
batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
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
from .training import MeanPathArtifacts, fit_g1_mean_path


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
    """Build group-confined OOF calibration scores for the two scale variants.

    This deliberately stops at calibration: it neither fits the final outer
    models nor creates any artifact on disk.  Each OOF block is produced by a
    fresh G1 path whose *entire* fitting universe is the corresponding inner
    training partition.  In particular, the held-out block's labels are only
    touched after its mean and scale predictions have been materialised.
    """
    config, phase_a, bundle, cache, fold, seed, device, batch_size = _nested_arguments(
        *args, **kwargs
    )
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
            phase_a,
            bundle,
            cache,
            int(fold),
            seed,
            train_ids,
            device=device,
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
            refit = fit_scale_model(
                mean_model=mean_path.model,
                scale_model=factory(),
                train_loader=train_batches,
                valid_loader=train_batches,
                max_epochs=refit_epochs,
                patience=max(1, min(config.patience, refit_epochs)),
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


def _nested_arguments(*args: Any, **kwargs: Any) -> tuple[
    PhaseBConfig, SGRPNConfig, DataBundle, OrderSpectrumCache, int, int, str | torch.device, int
]:
    """Normalize the deliberately compact public calibration call signature."""
    names = ("config", "phase_a", "bundle", "cache", "fold", "seed")
    if len(args) > len(names):
        raise TypeError("build_nested_calibration received too many positional arguments")
    values = dict(zip(names, args))
    for name in names:
        if name in kwargs:
            if name in values:
                raise TypeError(f"build_nested_calibration received {name!r} twice")
            values[name] = kwargs.pop(name)
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"build_nested_calibration is missing required arguments: {missing}")
    device = kwargs.pop("device", None)
    batch_size = kwargs.pop("batch_size", 8)
    if kwargs:
        raise TypeError(f"unexpected build_nested_calibration arguments: {sorted(kwargs)}")
    config = values["config"]
    phase_a = values["phase_a"]
    bundle = values["bundle"]
    cache = values["cache"]
    if not isinstance(config, PhaseBConfig) or not isinstance(phase_a, SGRPNConfig):
        raise ValueError("nested calibration requires Phase B and Phase A configurations")
    if not isinstance(bundle, DataBundle) or not isinstance(cache, OrderSpectrumCache):
        raise ValueError("nested calibration requires a DataBundle and OrderSpectrumCache")
    if type(values["fold"]) is not int:
        raise ValueError("fold must be an integer")
    return config, phase_a, bundle, cache, values["fold"], values["seed"], ("cpu" if device is None else device), batch_size


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
    counts = predictions.groupby("group_id", as_index=False, sort=True, observed=True).size()
    counts = counts.rename(columns={"size": "region_count"})
    scored = counts.merge(base, on="group_id", how="inner", validate="one_to_one")
    scored["reading_count"] = 3 * scored["region_count"]
    scored = scored.loc[:, CALIBRATION_SCORE_COLUMNS]
    if (
        scored["group_id"].duplicated().any()
        or not np.isfinite(scored["score"].to_numpy(dtype=np.float64)).all()
        or (scored["score"] < 0.0).any()
    ):
        raise ValueError("calibration group scores must be unique, finite, and non-negative")
    return scored


def _fold_arguments(*args: Any, **kwargs: Any) -> tuple[
    PhaseBConfig,
    PhaseAHandoff,
    SGRPNConfig,
    DataBundle,
    OrderSpectrumCache,
    int,
    int,
    str | torch.device,
    int,
    str | Path | None,
]:
    names = ("config", "handoff", "phase_a", "bundle", "cache")
    if len(args) > len(names):
        raise TypeError("run_phase_b_fold received too many positional arguments")
    values = dict(zip(names, args))
    for name in names:
        if name in kwargs:
            if name in values:
                raise TypeError(f"run_phase_b_fold received {name!r} twice")
            values[name] = kwargs.pop(name)
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"run_phase_b_fold is missing required arguments: {missing}")
    fold = kwargs.pop("fold", None)
    seed = kwargs.pop("seed", None)
    device = kwargs.pop("device", "cpu")
    batch_size = kwargs.pop("batch_size", 8)
    output_root = kwargs.pop("output_root", None)
    if kwargs:
        raise TypeError(f"unexpected run_phase_b_fold arguments: {sorted(kwargs)}")
    if not isinstance(values["config"], PhaseBConfig) or not isinstance(values["handoff"], PhaseAHandoff):
        raise ValueError("run_phase_b_fold requires Phase B configuration and Phase A handoff")
    if not isinstance(values["phase_a"], SGRPNConfig):
        raise ValueError("run_phase_b_fold requires a Phase A configuration")
    if not isinstance(values["bundle"], DataBundle) or not isinstance(values["cache"], OrderSpectrumCache):
        raise ValueError("run_phase_b_fold requires a DataBundle and OrderSpectrumCache")
    if type(fold) is not int:
        raise ValueError("fold must be an integer")
    return (
        values["config"], values["handoff"], values["phase_a"], values["bundle"], values["cache"],
        fold, seed, device, batch_size, output_root,
    )


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
        refit = fit_scale_model(
            mean_model=mean_path.model,
            scale_model=factory(),
            train_loader=final_batches,
            valid_loader=final_batches,
            max_epochs=median_best_epoch(selected_epochs),
            patience=max(1, min(config.patience, median_best_epoch(selected_epochs))),
            learning_rate=config.variance_learning_rate,
            weight_decay=config.weight_decay,
            device=device,
            seed=seed,
        )
        for parameter in refit.model.parameters():
            parameter.requires_grad_(False)
        refit.model.eval()
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
            if output.process_mean is None or output.residual is None:
                raise ValueError("final G1 output must contain process mean and residual")
            values = {
                "P1": output.process_mean,
                "R1": output.process_mean + output.residual,
                "G1": output.prediction,
            }
            for model_name, model_values in values.items():
                result = model_values.detach().cpu().numpy().astype(np.float64, copy=False)
                if not np.isfinite(result).all():
                    raise ValueError("final mean predictions must be finite")
                mean_rows.extend(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "group_id": str(batch["group_id"][index]),
                        "fold": fold,
                        "seed": seed,
                        "model": model_name,
                        "mu": float(result[index]),
                    }
                    for index in range(len(result))
                )
            features = build_scale_features(output, batch["process"], batch["quality"])
            mu = output.prediction.detach().cpu().numpy().astype(np.float64, copy=False)
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
                            "mu": float(mu[index]),
                            "sigma": float(scale_values[index]),
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
            probabilities.loc[mask, f"conformal_lower_{suffix}"] = conformal_lower
            probabilities.loc[mask, f"conformal_upper_{suffix}"] = conformal_upper
    return probabilities, pd.DataFrame(mean_rows)


def _phase_b_fingerprint(
    config: PhaseBConfig,
    handoff: PhaseAHandoff,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
) -> PhaseBRunFingerprint:
    """Bind the in-memory fold to its immutable handoff and consumed inputs."""
    from .training import _cache_sha256, _frame_sha256

    config_payload = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(config).items()
    }
    config_sha = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest_sha = _frame_sha256(bundle.manifest)
    folds_sha = _frame_sha256(bundle.folds)
    cache_sha = _cache_sha256(cache)
    value = hashlib.sha256(
        json.dumps(
            {
                "config": config_sha,
                "phase_a_acceptance": handoff.acceptance_sha256,
                "phase_a_run_manifest": handoff.run_manifest_sha256,
                "phase_a_training": handoff.training_fingerprint,
                "manifest": manifest_sha,
                "folds": folds_sha,
                "cache": cache_sha,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
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


def run_phase_b_fold(*args: Any, **kwargs: Any) -> PhaseBFoldArtifacts:
    """Train one probability fold entirely in memory.

    Calibration deliberately completes before this function asks the final G1
    path to see all outer-train groups.  The outer-test rows are represented by
    feature-only batches until both scale models and conformal quantiles are
    final, which keeps their labels on the scoring side of the boundary.
    """
    config, handoff, phase_a, bundle, cache, fold, seed, device, batch_size, output_root = (
        _fold_arguments(*args, **kwargs)
    )
    if type(seed) is not int or seed not in tuple(config.seeds):
        raise ValueError("seed must be a registered Phase B seed")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    root = validate_phase_b_output_root(config.output_dir, output_root=output_root)

    outer_train_index, outer_test_index = outer_indices(bundle, fold)
    outer_train = bundle.manifest.iloc[outer_train_index].reset_index(drop=True).copy()
    outer_train["sample_id"] = outer_train["sample_id"].astype(str)

    # This is intentionally first: no final model is selected from a path that
    # had access to an outer-test reading.
    calibration = build_nested_calibration(
        config, phase_a, bundle, cache, fold, seed, device=device, batch_size=batch_size
    )
    final_mean = fit_g1_mean_path(
        phase_a,
        bundle,
        cache,
        fold,
        seed,
        tuple(outer_train["sample_id"]),
        device=device,
        batch_size=batch_size,
    )
    final_scales, scale_histories = _select_and_refit_outer_scales(
        config, outer_train, cache, final_mean, seed=seed, device=device, batch_size=batch_size
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
        device=torch.device(device),
    )

    # This is the sole outer-test readout.  It is joined after inference and
    # therefore cannot influence a fitted model, scaler, or quantile.
    outer_labels = repeat_measure_batch(outer_test_features)
    label_frame = outer_test_features.loc[:, ["sample_id", "group_id"]].copy()
    label_frame["sample_id"] = label_frame["sample_id"].astype(str)
    label_frame["group_id"] = label_frame["group_id"].astype(str)
    label_frame["target_mean"] = outer_labels.mean
    label_frame["ra_1"] = outer_labels.readings[:, 0]
    label_frame["ra_2"] = outer_labels.readings[:, 1]
    label_frame["ra_3"] = outer_labels.readings[:, 2]
    predictions = inferred.merge(
        label_frame, on=["sample_id", "group_id"], how="inner", validate="many_to_one"
    )
    predictions = predictions.loc[:, PHASE_B_PREDICTION_COLUMNS]
    _validate_fold_predictions(predictions, outer_test_features, fold=fold, seed=seed)
    mean_predictions = mean_predictions.loc[:, PHASE_B_MEAN_COLUMNS]
    predictions.attrs["mean_predictions"] = mean_predictions

    fingerprint = _phase_b_fingerprint(config, handoff, bundle, cache)
    fold_dir = root / "folds" / f"fold_{fold}" / f"seed_{seed}"
    checkpoint_paths = {
        name: fold_dir / "checkpoints" / f"{name}.pt" for name in SCALE_MODELS
    }
    history_paths = {
        name: fold_dir / "history" / f"{name}.csv" for name in SCALE_MODELS
    }
    # The histories remain deliberately in-memory for Task 6; retain them in
    # local scope to make the no-write contract explicit until Task 7 persists.
    del scale_histories
    return PhaseBFoldArtifacts(
        predictions=predictions,
        calibration=calibration,
        checkpoint_paths=checkpoint_paths,
        history_paths=history_paths,
        fingerprint=fingerprint,
    )


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
