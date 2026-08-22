"""Leakage-safe nested Phase A training and exact fold resume."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import inspect
import io
import json
from pathlib import Path
import random
from typing import Any, Protocol
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from roughness.scheme1.crossfit import make_group_inner_splits

from .config import SGRPNConfig
from .crossfit import (
    ProcessScaler,
    ValidationAwareTrainer,
    fit_process_scaler,
    generate_process_oof,
    median_best_epoch,
)
from .data import DataBundle, build_process_features, outer_indices, validate_group_split
from .dataset import OrderBagDataset, collate_order_bags
from .models import (
    DirectFusionModel,
    ModelOutput,
    ProcessMLP,
    ResidualExpert,
    ResidualFusionModel,
    SelectiveGatedModel,
    VibrationOnlyModel,
    average_swap_predictions,
    sgrpn_loss,
    weighted_huber,
)
from .order_spectrum import (
    OrderSpectrumCache,
    QualityScaler,
    SpectrumScaler,
    fit_quality_scaler,
    fit_spectrum_scaler,
)


MODEL_SEQUENCE = ("P1", "V1", "F1", "R1", "G1")
OOF_COLUMNS = (
    "sample_id", "group_id", "version", "fold", "seed", "model",
    "target", "prediction", "sample_weight", "process_mean", "residual", "gate",
)


@dataclass(frozen=True)
class RunFingerprint:
    value: str
    config_sha256: str
    manifest_sha256: str
    folds_sha256: str
    cache_sha256: str


@dataclass(frozen=True)
class EpochSelection:
    best_epochs: Sequence[int]

    @property
    def refit_epochs(self) -> int:
        return median_best_epoch(tuple(self.best_epochs))


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    best_epoch: int
    history: pd.DataFrame


@dataclass(frozen=True)
class FoldArtifacts:
    predictions: pd.DataFrame
    checkpoint_paths: dict[str, Path]
    history_paths: dict[str, Path]
    fingerprint: RunFingerprint


class TrainingBackend(Protocol):
    def fit(self, **kwargs: Any) -> TrainingResult: ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return _sha256_bytes(frame.to_csv(index=False).encode("utf-8"))


def _cache_sha256(cache: OrderSpectrumCache) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(cache.segment_ids), ensure_ascii=False).encode("utf-8"))
    for array in (cache.spectra, cache.offsets, cache.quality, cache.durations_s):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_run_fingerprint(
    config: SGRPNConfig, bundle: DataBundle, cache: OrderSpectrumCache
) -> RunFingerprint:
    config_sha = _sha256_bytes(
        json.dumps(_jsonable(asdict(config)), sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    manifest_sha = (
        _sha256_file(Path(config.manifest_path))
        if Path(config.manifest_path).is_file()
        else _frame_sha256(bundle.manifest)
    )
    folds_sha = (
        _sha256_file(Path(config.folds_path))
        if Path(config.folds_path).is_file()
        else _frame_sha256(bundle.folds)
    )
    cache_sha = _cache_sha256(cache)
    value = _sha256_bytes(
        json.dumps(
            {
                "protocol": "sgrpn-phase-a-v1",
                "config": config_sha,
                "manifest": manifest_sha,
                "folds": folds_sha,
                "cache": cache_sha,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return RunFingerprint(value, config_sha, manifest_sha, folds_sha, cache_sha)


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")


def _atomic_write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    return _atomic_write_bytes(
        path,
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> Path:
    return _atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> Path:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return _atomic_write_bytes(path, buffer.getvalue())


def _atomic_save_scalers(path: Path, **arrays: np.ndarray) -> Path:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return _atomic_write_bytes(path, buffer.getvalue())


def completed_fold_matches(
    marker: str | Path,
    fingerprint: str,
    *,
    fold: int | None = None,
    seed: int | None = None,
    models: Sequence[str] = MODEL_SEQUENCE,
) -> bool:
    try:
        raw = json.loads(Path(marker).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    expected_models = list(models)
    return bool(
        raw.get("status") == "complete"
        and raw.get("fingerprint") == str(fingerprint)
        and raw.get("models") == expected_models
        and raw.get("completed_stages") == expected_models
        and (fold is None or raw.get("fold") == int(fold))
        and (seed is None or raw.get("seed") == int(seed))
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(value: str | torch.device | None) -> torch.device:
    selected = torch.device(value) if value is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Phase A device must be cpu or cuda")
    return selected


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _model_output(stage: str, model: nn.Module, batch: dict[str, Any]) -> ModelOutput:
    if stage == "P1":
        prediction = model(batch["process"])
        return ModelOutput(prediction=prediction, process_mean=prediction)
    if stage == "V1":
        return model(batch["spectrum"], batch["window_mask"])
    if stage == "F1" or isinstance(model, ResidualFusionModel):
        return model(batch["spectrum"], batch["window_mask"], batch["process"])
    if stage == "R1":
        return model(batch["spectrum"], batch["window_mask"])
    if stage == "G1":
        return model(
            batch["spectrum"], batch["window_mask"], batch["process"], batch["quality"]
        )
    raise ValueError(f"Unknown Phase A stage: {stage}")


def _batch_loss(
    stage: str, output: ModelOutput, batch: dict[str, Any], delta: float
) -> torch.Tensor:
    if stage == "G1":
        return sgrpn_loss(output, batch["target"], batch["sample_weight"], delta)
    return weighted_huber(
        output.prediction, batch["target"], batch["sample_weight"], delta
    )


def _validation_loss(
    stage: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    delta: float,
) -> float:
    model.eval()
    outputs: list[ModelOutput] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            outputs.append(_model_output(stage, model, batch))
            targets.append(batch["target"])
            weights.append(batch["sample_weight"])
    if not outputs:
        raise ValueError("validation loader must not be empty")
    prediction = torch.cat([item.prediction for item in outputs])
    target = torch.cat(targets)
    weight = torch.cat(weights)
    if stage == "G1":
        gate = torch.cat([item.gate for item in outputs if item.gate is not None])
        residual = torch.cat([item.residual for item in outputs if item.residual is not None])
        loss = sgrpn_loss(ModelOutput(prediction, residual=residual, gate=gate), target, weight, delta)
    else:
        loss = weighted_huber(prediction, target, weight, delta)
    return float(loss.item())


class TorchTrainingBackend:
    """Configured deterministic trainer shared by every Phase A component."""

    def fit(
        self,
        *,
        stage: str,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader | None,
        max_epochs: int,
        patience: int,
        learning_rate: float,
        weight_decay: float,
        loss_name: str,
        device: torch.device,
        seed: int,
    ) -> TrainingResult:
        del loss_name
        if max_epochs < 1 or patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        set_global_seed(seed)
        model.to(device)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError(f"{stage} has no trainable parameters")
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_numerator = 0.0
            train_weight = 0.0
            for raw_batch in train_loader:
                batch = _move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = _model_output(stage, model, batch)
                loss = _batch_loss(stage, output, batch, 0.10)
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"{stage} training loss must be finite")
                loss.backward()
                optimizer.step()
                weight_sum = float(batch["sample_weight"].sum().item())
                train_numerator += float(loss.detach().item()) * weight_sum
                train_weight += weight_sum
            if train_weight <= 0:
                raise ValueError("training loader must contain positive sample weight")
            validation = (
                train_numerator / train_weight
                if validation_loader is None
                else _validation_loss(stage, model, validation_loader, device, 0.10)
            )
            improved = validation_loader is None or validation < best_loss
            if improved:
                best_loss = validation
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_numerator / train_weight,
                    "validation_loss": validation,
                    "is_best": improved,
                }
            )
            if validation_loader is not None and bad_epochs >= patience:
                break
        if best_state is None or best_epoch < 1:
            raise ValueError(f"{stage} did not select a finite epoch")
        model.load_state_dict(best_state)
        return TrainingResult(model, best_epoch, pd.DataFrame.from_records(history))


def _factory_call(factory: Callable[..., Any], *arguments: Any) -> Any:
    signature = inspect.signature(factory)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return factory(*arguments)
    return factory(*arguments[: len(positional)])


def select_epochs_group_cv(
    component_factory: Callable[..., nn.Module],
    dataset_factory: Callable[..., tuple[Any, Any]],
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    train_step: Callable[..., TrainingResult | int],
    validation_step: Callable[..., Any] | None,
) -> EpochSelection:
    """Select a component epoch count on four preflighted group folds.

    ``dataset_factory`` receives train index, validation index, and fold number;
    it owns scaler fitting so the only IDs it may use are the supplied train IDs.
    The compact callback form keeps injected tests on the production path.
    """
    if len(inner_splits) != 4:
        raise ValueError("Phase A requires exactly four inner splits")
    epochs: list[int] = []
    for fold_number, (raw_train, raw_valid) in enumerate(inner_splits):
        train_index = np.asarray(raw_train, dtype=np.int64)
        valid_index = np.asarray(raw_valid, dtype=np.int64)
        if train_index.ndim != 1 or valid_index.ndim != 1:
            raise ValueError("inner split indices must be one-dimensional")
        if not len(train_index) or not len(valid_index):
            raise ValueError("inner train and validation indices must not be empty")
        if np.intersect1d(train_index, valid_index).size:
            raise ValueError("inner train and validation indices overlap")
        train_data, valid_data = _factory_call(
            dataset_factory, train_index, valid_index, fold_number
        )
        model = _factory_call(component_factory, fold_number)
        result = _factory_call(
            train_step, model, train_data, valid_data, validation_step, fold_number
        )
        epoch = result.best_epoch if isinstance(result, TrainingResult) else result
        if isinstance(epoch, (bool, np.bool_)) or not isinstance(epoch, (int, np.integer)) or int(epoch) < 1:
            raise ValueError("train_step must return a positive best epoch")
        epochs.append(int(epoch))
    return EpochSelection(tuple(epochs))


def refit_component(
    component_factory: Callable[..., nn.Module],
    dataset_factory: Callable[..., Any],
    epoch_selection: EpochSelection,
    train_step: Callable[..., TrainingResult | nn.Module],
) -> nn.Module:
    """Train a fresh component on complete outer-train data for median epochs."""
    model = _factory_call(component_factory)
    dataset = _factory_call(dataset_factory)
    result = _factory_call(train_step, model, dataset, epoch_selection.refit_epochs)
    if isinstance(result, TrainingResult):
        return result.model
    if isinstance(result, nn.Module):
        return result
    raise ValueError("refit train_step must return TrainingResult or nn.Module")


class _ProcessDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        process: np.ndarray,
        target: np.ndarray,
    ) -> None:
        if len(frame) != len(process) or len(frame) != len(target):
            raise ValueError("process dataset arrays must align")
        self.frame = frame.reset_index(drop=True)
        self.process = np.asarray(process, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "process": torch.from_numpy(self.process[index].copy()),
            "target": torch.tensor(self.target[index], dtype=torch.float32),
            "sample_weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "group_id": str(row["group_id"]),
        }


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    order_bags: bool = False,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=0,
        collate_fn=collate_order_bags if order_bags else None,
    )


@dataclass
class _SignalFit:
    selection: EpochSelection
    inner_models: list[nn.Module]
    inner_process_scalers: list[ProcessScaler]
    inner_spectrum_scalers: list[SpectrumScaler]
    inner_quality_scalers: list[QualityScaler]
    model: nn.Module
    process_scaler: ProcessScaler
    spectrum_scaler: SpectrumScaler
    quality_scaler: QualityScaler
    history: pd.DataFrame


def _learning_rate(config: SGRPNConfig, stage: str) -> float:
    if stage in {"P1", "G1"}:
        return config.process_learning_rate if stage == "P1" else config.gate_learning_rate
    return config.signal_learning_rate


def _fit_backend(
    backend: TrainingBackend,
    *,
    stage: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    max_epochs: int,
    config: SGRPNConfig,
    device: torch.device,
    seed: int,
) -> TrainingResult:
    result = backend.fit(
        stage=stage,
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        max_epochs=max_epochs,
        patience=config.patience,
        learning_rate=_learning_rate(config, stage),
        weight_decay=config.weight_decay,
        loss_name="sgrpn" if stage == "G1" else "weighted_huber",
        device=device,
        seed=seed,
    )
    if not isinstance(result, TrainingResult):
        raise ValueError("training backend must return TrainingResult")
    if result.best_epoch < 1 or result.best_epoch > max_epochs:
        raise ValueError("training backend returned an invalid best epoch")
    return result


def _validate_inner_splits(
    frame: pd.DataFrame, splits: Sequence[tuple[np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(splits) != 4:
        raise ValueError("Phase A requires exactly four inner splits")
    assignment = np.zeros(len(frame), dtype=np.int64)
    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    groups = frame["group_id"].astype(str).to_numpy()
    for raw_train, raw_valid in splits:
        train_index = np.asarray(raw_train, dtype=np.int64)
        valid_index = np.asarray(raw_valid, dtype=np.int64)
        if (
            train_index.ndim != 1
            or valid_index.ndim != 1
            or not len(train_index)
            or not len(valid_index)
            or np.intersect1d(train_index, valid_index).size
            or np.union1d(train_index, valid_index).size != len(frame)
        ):
            raise ValueError("each inner split must be a disjoint complete partition")
        validate_group_split(groups[train_index], groups[valid_index])
        assignment[valid_index] += 1
        normalized.append((train_index, valid_index))
    if not np.all(assignment == 1):
        raise ValueError("inner validation folds must assign every row exactly once")
    return normalized


def _signal_dataset(
    frame: pd.DataFrame,
    cache: OrderSpectrumCache,
    spectrum_scaler: SpectrumScaler,
    quality_scaler: QualityScaler,
    process: np.ndarray,
    target: np.ndarray,
    *,
    augment: bool,
) -> OrderBagDataset:
    return OrderBagDataset(
        frame,
        cache,
        spectrum_scaler,
        quality_scaler,
        np.asarray(process, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        augment_horizontal_swap=augment,
    )


def _expert_state(model: SelectiveGatedModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("process_expert.", "residual_expert."))
    }


def _assert_experts_unchanged(
    before: dict[str, torch.Tensor], model: SelectiveGatedModel
) -> None:
    after = model.state_dict()
    if set(before) != {
        name
        for name in after
        if name.startswith(("process_expert.", "residual_expert."))
    }:
        raise RuntimeError("G1 expert state keys changed during gate training")
    if not all(torch.equal(expected, after[name].detach().cpu()) for name, expected in before.items()):
        raise RuntimeError("G1 training changed a frozen expert state tensor")


def _fit_signal_stage(
    *,
    stage: str,
    config: SGRPNConfig,
    frame: pd.DataFrame,
    cache: OrderSpectrumCache,
    features: np.ndarray,
    targets: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[int | None], nn.Module],
    backend: TrainingBackend,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> _SignalFit:
    inner_models: list[nn.Module] = []
    process_scalers: list[ProcessScaler] = []
    spectrum_scalers: list[SpectrumScaler] = []
    quality_scalers: list[QualityScaler] = []
    epochs: list[int] = []
    histories: list[pd.DataFrame] = []
    sample_ids = frame["sample_id"].astype(str).to_numpy()
    for fold_number, (train_index, valid_index) in enumerate(inner_splits):
        train_ids = sample_ids[train_index].tolist()
        process_scaler = fit_process_scaler(features, train_index)
        spectrum_scaler = fit_spectrum_scaler(cache, train_ids)
        quality_scaler = fit_quality_scaler(cache, train_ids)
        train_data = _signal_dataset(
            frame.iloc[train_index], cache, spectrum_scaler, quality_scaler,
            process_scaler.transform(features[train_index]), targets[train_index], augment=True,
        )
        valid_data = _signal_dataset(
            frame.iloc[valid_index], cache, spectrum_scaler, quality_scaler,
            process_scaler.transform(features[valid_index]), targets[valid_index], augment=False,
        )
        model = model_factory(fold_number)
        before = None
        if stage == "G1":
            if not isinstance(model, SelectiveGatedModel):
                raise ValueError("G1 factory must return SelectiveGatedModel")
            model.freeze_experts()
            before = _expert_state(model)
        result = _fit_backend(
            backend,
            stage=stage,
            model=model,
            train_loader=_loader(
                train_data, batch_size=batch_size, shuffle=True,
                seed=seed, order_bags=True,
            ),
            validation_loader=_loader(
                valid_data, batch_size=batch_size, shuffle=False,
                seed=seed, order_bags=True,
            ),
            max_epochs=config.max_epochs,
            config=config,
            device=device,
            seed=seed,
        )
        if before is not None:
            _assert_experts_unchanged(before, result.model)
        inner_models.append(result.model)
        process_scalers.append(process_scaler)
        spectrum_scalers.append(spectrum_scaler)
        quality_scalers.append(quality_scaler)
        epochs.append(result.best_epoch)
        history = result.history.copy()
        history.insert(0, "inner_fold", fold_number)
        history.insert(0, "phase", "selection")
        histories.append(history)

    selection = EpochSelection(tuple(epochs))
    all_index = np.arange(len(frame), dtype=np.int64)
    all_ids = sample_ids.tolist()
    process_scaler = fit_process_scaler(features, all_index)
    spectrum_scaler = fit_spectrum_scaler(cache, all_ids)
    quality_scaler = fit_quality_scaler(cache, all_ids)
    refit_data = _signal_dataset(
        frame, cache, spectrum_scaler, quality_scaler,
        process_scaler.transform(features), targets, augment=True,
    )
    refit_model = model_factory(None)
    before = None
    if stage == "G1":
        if not isinstance(refit_model, SelectiveGatedModel):
            raise ValueError("G1 factory must return SelectiveGatedModel")
        refit_model.freeze_experts()
        before = _expert_state(refit_model)
    refit = _fit_backend(
        backend,
        stage=stage,
        model=refit_model,
        train_loader=_loader(
            refit_data, batch_size=batch_size, shuffle=True,
            seed=seed, order_bags=True,
        ),
        validation_loader=None,
        max_epochs=selection.refit_epochs,
        config=config,
        device=device,
        seed=seed,
    )
    if before is not None:
        _assert_experts_unchanged(before, refit.model)
    refit_history = refit.history.copy()
    refit_history.insert(0, "inner_fold", -1)
    refit_history.insert(0, "phase", "refit")
    histories.append(refit_history)
    return _SignalFit(
        selection=selection,
        inner_models=inner_models,
        inner_process_scalers=process_scalers,
        inner_spectrum_scalers=spectrum_scalers,
        inner_quality_scalers=quality_scalers,
        model=refit.model,
        process_scaler=process_scaler,
        spectrum_scaler=spectrum_scaler,
        quality_scaler=quality_scaler,
        history=pd.concat(histories, ignore_index=True),
    )


def _fold_dir(config: SGRPNConfig, fold: int, seed: int) -> Path:
    return Path(config.output_dir) / "folds" / f"fold_{fold}" / f"seed_{seed}"


def _stage_paths(fold_dir: Path, stage: str) -> tuple[Path, Path, Path]:
    return (
        fold_dir / "checkpoints" / f"{stage}.pt",
        fold_dir / "history" / f"{stage}.csv",
        fold_dir / "scalers" / f"{stage}_scalers.npz",
    )


def _state_payload(
    fingerprint: RunFingerprint,
    fold: int,
    seed: int,
    completed: Sequence[str],
    device: torch.device,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "status": status,
        "fingerprint": fingerprint.value,
        "fold": int(fold),
        "seed": int(seed),
        "models": list(MODEL_SEQUENCE),
        "completed_stages": list(completed),
        "next_stage": None if len(completed) == len(MODEL_SEQUENCE) else MODEL_SEQUENCE[len(completed)],
        "device": str(device),
    }


def _validate_partial_state(
    state_path: Path, fingerprint: RunFingerprint, fold: int, seed: int
) -> None:
    if not state_path.exists():
        return
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("incompatible or corrupt partial fold state") from error
    completed = raw.get("completed_stages")
    if (
        raw.get("fingerprint") != fingerprint.value
        or raw.get("fold") != int(fold)
        or raw.get("seed") != int(seed)
        or raw.get("models") != list(MODEL_SEQUENCE)
        or not isinstance(completed, list)
        or completed != list(MODEL_SEQUENCE[: len(completed)])
        or raw.get("next_stage")
        != (None if len(completed) == len(MODEL_SEQUENCE) else MODEL_SEQUENCE[len(completed)])
    ):
        raise ValueError("incompatible fingerprint, fold, seed, model set, or stage state")


def _checkpoint_payload(
    *,
    fingerprint: RunFingerprint,
    fold: int,
    seed: int,
    stage: str,
    selection: EpochSelection,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "protocol": "sgrpn-phase-a-v1",
        "fingerprint": fingerprint.value,
        "fold": int(fold),
        "seed": int(seed),
        "model": stage,
        "completed_stage": stage,
        "best_epochs": list(map(int, selection.best_epochs)),
        "refit_epochs": selection.refit_epochs,
        "device": str(device),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }


def _persist_stage(
    *,
    fold_dir: Path,
    fingerprint: RunFingerprint,
    fold: int,
    seed: int,
    stage: str,
    completed: list[str],
    selection: EpochSelection,
    model: nn.Module,
    history: pd.DataFrame,
    scalers: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[Path, Path]:
    checkpoint, history_path, scaler_path = _stage_paths(fold_dir, stage)
    _atomic_torch_save(
        checkpoint,
        _checkpoint_payload(
            fingerprint=fingerprint, fold=fold, seed=seed, stage=stage,
            selection=selection, model=model, device=device,
        ),
    )
    _atomic_write_frame(history_path, history)
    _atomic_save_scalers(scaler_path, **scalers)
    completed.append(stage)
    _atomic_write_json(
        fold_dir / "state.json",
        _state_payload(fingerprint, fold, seed, completed, device),
    )
    return checkpoint, history_path


def _read_outer_targets(frame: pd.DataFrame) -> np.ndarray:
    """The single auditable outer-test label read, called only after all fits."""
    values = frame["ra_mean"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("outer-test targets must be finite")
    return values


def _predict_process(
    model: nn.Module,
    frame: pd.DataFrame,
    process: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> ModelOutput:
    dataset = _ProcessDataset(frame, process, np.zeros(len(frame), dtype=np.float32))
    predictions: list[torch.Tensor] = []
    model.to(device).eval()
    with torch.no_grad():
        for raw_batch in _loader(dataset, batch_size=batch_size, shuffle=False, seed=0):
            batch = _move_batch(raw_batch, device)
            predictions.append(model(batch["process"]).detach().cpu())
    prediction = torch.cat(predictions)
    return ModelOutput(prediction=prediction, process_mean=prediction)


def _predict_signal(
    *,
    stage: str,
    model: nn.Module,
    frame: pd.DataFrame,
    cache: OrderSpectrumCache,
    spectrum_scaler: SpectrumScaler,
    quality_scaler: QualityScaler,
    process: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> ModelOutput:
    dataset = _signal_dataset(
        frame, cache, spectrum_scaler, quality_scaler, process,
        np.zeros(len(frame), dtype=np.float32), augment=False,
    )
    collected: dict[str, list[torch.Tensor]] = {
        "prediction": [], "process_mean": [], "residual": [], "gate": [], "embedding": []
    }
    model.to(device).eval()
    with torch.no_grad():
        for raw_batch in _loader(
            dataset, batch_size=batch_size, shuffle=False, seed=0, order_bags=True
        ):
            batch = _move_batch(raw_batch, device)
            output = average_swap_predictions(model, batch)
            for name in collected:
                value = getattr(output, name)
                if value is not None:
                    collected[name].append(value.detach().cpu())
    if not collected["prediction"]:
        raise ValueError(f"{stage} produced no predictions")
    payload: dict[str, torch.Tensor | None] = {}
    for name, values in collected.items():
        payload[name] = torch.cat(values) if values else None
    return ModelOutput(**payload)


def _numpy_component(value: torch.Tensor | None, length: int) -> np.ndarray:
    if value is None:
        return np.full(length, np.nan, dtype=np.float64)
    array = value.detach().cpu().numpy().astype(np.float64, copy=False)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError("prediction component must be a finite vector")
    return array


def _prediction_frame(
    frame: pd.DataFrame,
    outputs: dict[str, ModelOutput],
    targets: np.ndarray,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in MODEL_SEQUENCE:
        output = outputs[stage]
        prediction = _numpy_component(output.prediction, len(frame))
        process_mean = _numpy_component(output.process_mean, len(frame))
        residual = _numpy_component(output.residual, len(frame))
        gate = _numpy_component(output.gate, len(frame))
        for index, row in enumerate(frame.itertuples(index=False)):
            rows.append(
                {
                    "sample_id": str(row.sample_id),
                    "group_id": str(row.group_id),
                    "version": str(row.version),
                    "fold": int(fold),
                    "seed": int(seed),
                    "model": stage,
                    "target": float(targets[index]),
                    "prediction": float(prediction[index]),
                    "sample_weight": float(row.sample_weight),
                    "process_mean": float(process_mean[index]),
                    "residual": float(residual[index]),
                    "gate": float(gate[index]),
                }
            )
    predictions = pd.DataFrame.from_records(rows, columns=OOF_COLUMNS)
    if (
        len(predictions) != len(frame) * len(MODEL_SEQUENCE)
        or predictions.duplicated(["sample_id", "model"]).any()
        or set(predictions["model"]) != set(MODEL_SEQUENCE)
        or not np.isfinite(predictions[["target", "prediction", "sample_weight"]]).all().all()
    ):
        raise ValueError("fold predictions are incomplete, duplicated, or non-finite")
    g1_gate = predictions.loc[predictions["model"] == "G1", "gate"]
    if not g1_gate.between(0.0, 1.0).all():
        raise ValueError("G1 gates must lie in [0, 1]")
    return predictions


def _validate_protocol(
    config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tuple(config.seeds) != (20260723,) or int(seed) != 20260723:
        raise ValueError("Phase A requires exactly seed 20260723")
    if config.inner_splits != 4:
        raise ValueError("Phase A requires exactly four inner splits")
    if not 1 <= config.max_epochs <= 200 or not 1 <= config.patience <= 20:
        raise ValueError("Phase A epochs/patience must be positive and at most 200/20")
    if (
        config.sample_rate_hz != 25600
        or config.window_samples != 25600
        or (config.order_min, config.order_max, config.order_step) != (0.0, 90.0, 0.25)
    ):
        raise ValueError("Phase A sample and order-grid protocol is locked")
    if not np.isclose(config.huber_delta_um, 0.10):
        raise ValueError("Phase A Huber delta must be 0.10")
    locked_numeric = (
        (config.process_learning_rate, 1e-3, "process learning rate"),
        (config.signal_learning_rate, 3e-4, "signal learning rate"),
        (config.gate_learning_rate, 1e-3, "gate learning rate"),
        (config.weight_decay, 1e-4, "weight decay"),
        (config.gate_penalty, 1e-3, "gate penalty"),
        (config.correction_penalty, 1e-2, "correction penalty"),
    )
    for actual, expected, name in locked_numeric:
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-15):
            raise ValueError(f"Phase A {name} is locked to {expected}")
    weights = bundle.manifest["sample_weight"].to_numpy(dtype=np.float64)
    split_counts = bundle.manifest["split_count"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(split_counts).all()
        or np.any(weights <= 0)
        or np.any(split_counts <= 0)
        or not np.allclose(weights, 1.0 / split_counts, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Phase A sample_weight must equal 1/split_count")
    parts = [part.lower() for part in Path(config.output_dir).parts]
    if "scheme1" in parts or "scheme1_physics" in parts:
        raise ValueError("legacy output directories are read-only")
    if len(parts) < 2 or parts[-2:] != ["sgrpn", "phase_a"]:
        raise ValueError("Phase A outputs must be written under outputs/sgrpn/phase_a")
    if len(cache.segment_ids) != len(bundle.manifest) or tuple(
        bundle.manifest["sample_id"].astype(str)
    ) != tuple(cache.segment_ids):
        raise ValueError("cache segment IDs must exactly match manifest order")
    train_index, test_index = outer_indices(bundle, fold)
    if not len(train_index) or not len(test_index):
        raise ValueError("outer fold must contain train and test rows")
    validate_group_split(
        bundle.manifest.iloc[train_index]["group_id"].to_numpy(),
        bundle.manifest.iloc[test_index]["group_id"].to_numpy(),
    )
    return train_index, test_index


def _load_completed_fold(
    fold_dir: Path, fingerprint: RunFingerprint, fold: int, seed: int
) -> FoldArtifacts:
    marker = json.loads((fold_dir / "complete.json").read_text(encoding="utf-8"))
    predictions_path = fold_dir / "oof_predictions.csv"
    if (
        not predictions_path.is_file()
        or marker.get("predictions_sha256") != _sha256_file(predictions_path)
    ):
        raise ValueError("completed fold prediction fingerprint is missing or incompatible")
    predictions = pd.read_csv(predictions_path)
    if tuple(predictions.columns) != OOF_COLUMNS:
        raise ValueError("completed fold OOF schema is incompatible")
    checkpoints = {stage: _stage_paths(fold_dir, stage)[0] for stage in MODEL_SEQUENCE}
    histories = {stage: _stage_paths(fold_dir, stage)[1] for stage in MODEL_SEQUENCE}
    if not all(path.is_file() for path in (*checkpoints.values(), *histories.values())):
        raise ValueError("completed fold is missing checkpoint or history artifacts")
    return FoldArtifacts(predictions, checkpoints, histories, fingerprint)


def run_phase_a_fold(
    config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    fold: int,
    seed: int,
    device: str | torch.device | None = None,
    *,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
) -> FoldArtifacts:
    """Train one outer fold in the exact P1→V1→F1→R1→G1 order."""
    train_index, test_index = _validate_protocol(config, bundle, cache, fold, seed)
    selected_device = _device(device)
    set_global_seed(seed)
    fingerprint = build_run_fingerprint(config, bundle, cache)
    fold_dir = _fold_dir(config, fold, seed)
    marker_path = fold_dir / "complete.json"
    if marker_path.exists():
        if not completed_fold_matches(marker_path, fingerprint.value, fold=fold, seed=seed):
            raise ValueError("incompatible completed-fold fingerprint, fold, seed, or model state")
        return _load_completed_fold(fold_dir, fingerprint, fold, seed)
    if fold_dir.exists() and any(fold_dir.iterdir()) and not (fold_dir / "state.json").is_file():
        raise ValueError("incompatible partial fold artifacts have no exact stage state")
    _validate_partial_state(fold_dir / "state.json", fingerprint, fold, seed)
    fold_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        fold_dir / "state.json",
        _state_payload(fingerprint, fold, seed, [], selected_device),
    )

    trainer: TrainingBackend = TorchTrainingBackend() if backend is None else backend
    train_frame = bundle.manifest.iloc[train_index].reset_index(drop=True).copy()
    # Materialize only non-label held-out fields before the trained-stage
    # barrier. No ra column is selected or copied through this frame.
    test_columns = [
        column
        for column in bundle.manifest.columns
        if column not in {"ra_1", "ra_2", "ra_3", "ra_mean"}
    ]
    test_frame = (
        bundle.manifest.loc[:, test_columns]
        .iloc[test_index]
        .reset_index(drop=True)
        .copy()
    )
    train_features = build_process_features(train_frame)
    test_features = build_process_features(test_frame)
    train_targets = train_frame["ra_mean"].to_numpy(dtype=np.float64, copy=True)
    inner_splits = _validate_inner_splits(
        train_frame,
        make_group_inner_splits(train_frame, n_splits=4, seed=seed),
    )
    checkpoints: dict[str, Path] = {}
    histories: dict[str, Path] = {}
    completed: list[str] = []

    # P1: the Task 6 boundary owns canonical feature validation, split/scaler
    # fitting, and OOF assignment. This callback only supplies configured fitting.
    p1_inner_models: list[ProcessMLP] = []
    p1_call = 0

    def configured_p1_trainer(
        x_train: np.ndarray,
        y_train: np.ndarray,
        w_train: np.ndarray,
        x_valid: np.ndarray,
        train_groups: np.ndarray,
        valid_groups: np.ndarray,
        *,
        y_valid: np.ndarray,
        w_valid: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        nonlocal p1_call
        train_rows, valid_rows = inner_splits[p1_call]
        model = ProcessMLP()
        train_rows_frame = train_frame.iloc[train_rows].copy()
        valid_rows_frame = train_frame.iloc[valid_rows].copy()
        # Task 6 already proved these arrays/groups correspond; retain explicit
        # equality checks at the adapter boundary before any optimizer call.
        if not np.array_equal(train_rows_frame["group_id"].astype(str), train_groups.astype(str)):
            raise ValueError("P1 inner train group order is incompatible")
        if not np.array_equal(valid_rows_frame["group_id"].astype(str), valid_groups.astype(str)):
            raise ValueError("P1 inner validation group order is incompatible")
        train_rows_frame.loc[:, "sample_weight"] = w_train
        valid_rows_frame.loc[:, "sample_weight"] = w_valid
        result = _fit_backend(
            trainer,
            stage="P1",
            model=model,
            train_loader=_loader(
                _ProcessDataset(train_rows_frame, x_train, y_train),
                batch_size=batch_size, shuffle=True, seed=seed,
            ),
            validation_loader=_loader(
                _ProcessDataset(valid_rows_frame, x_valid, y_valid),
                batch_size=batch_size, shuffle=False, seed=seed,
            ),
            max_epochs=config.max_epochs,
            config=config,
            device=selected_device,
            seed=seed,
        )
        result.model.to(selected_device).eval()
        with torch.no_grad():
            prediction = result.model(
                torch.as_tensor(x_valid, dtype=torch.float32, device=selected_device)
            ).detach().cpu().numpy()
        p1_inner_models.append(result.model)
        p1_histories.append(result.history.assign(phase="selection", inner_fold=p1_call))
        p1_call += 1
        return prediction.astype(np.float64, copy=False), result.best_epoch

    p1_histories: list[pd.DataFrame] = []
    p1_oof = generate_process_oof(
        train_frame,
        train_features,
        ValidationAwareTrainer(configured_p1_trainer),
        feature_sample_ids=train_frame["sample_id"].astype(str).to_numpy(),
        n_splits=4,
        seed=seed,
    )
    p1_selection = EpochSelection(tuple(p1_oof.best_epochs))
    all_train = np.arange(len(train_frame), dtype=np.int64)
    outer_process_scaler = fit_process_scaler(train_features, all_train)
    p1_refit = _fit_backend(
        trainer,
        stage="P1",
        model=ProcessMLP(),
        train_loader=_loader(
            _ProcessDataset(
                train_frame, outer_process_scaler.transform(train_features), train_targets
            ),
            batch_size=batch_size, shuffle=True, seed=seed,
        ),
        validation_loader=None,
        max_epochs=p1_selection.refit_epochs,
        config=config,
        device=selected_device,
        seed=seed,
    )
    p1_histories.append(p1_refit.history.assign(phase="refit", inner_fold=-1))
    checkpoints["P1"], histories["P1"] = _persist_stage(
        fold_dir=fold_dir, fingerprint=fingerprint, fold=fold, seed=seed, stage="P1",
        completed=completed, selection=p1_selection, model=p1_refit.model,
        history=pd.concat(p1_histories, ignore_index=True),
        scalers={"process_mean": outer_process_scaler.mean, "process_scale": outer_process_scaler.scale},
        device=selected_device,
    )

    # V1: direct vibration comparator.
    v1 = _fit_signal_stage(
        stage="V1", config=config, frame=train_frame, cache=cache,
        features=train_features, targets=train_targets, inner_splits=inner_splits,
        model_factory=lambda _fold: VibrationOnlyModel(), backend=trainer,
        device=selected_device, seed=seed, batch_size=batch_size,
    )
    checkpoints["V1"], histories["V1"] = _persist_stage(
        fold_dir=fold_dir, fingerprint=fingerprint, fold=fold, seed=seed, stage="V1",
        completed=completed, selection=v1.selection, model=v1.model, history=v1.history,
        scalers={
            "process_mean": v1.process_scaler.mean, "process_scale": v1.process_scaler.scale,
            "spectrum_mean": v1.spectrum_scaler.mean, "spectrum_scale": v1.spectrum_scaler.scale,
            "quality_mean": v1.quality_scaler.mean, "quality_scale": v1.quality_scaler.scale,
        }, device=selected_device,
    )

    # F1: direct process/vibration fusion comparator.
    f1 = _fit_signal_stage(
        stage="F1", config=config, frame=train_frame, cache=cache,
        features=train_features, targets=train_targets, inner_splits=inner_splits,
        model_factory=lambda _fold: DirectFusionModel(), backend=trainer,
        device=selected_device, seed=seed, batch_size=batch_size,
    )
    checkpoints["F1"], histories["F1"] = _persist_stage(
        fold_dir=fold_dir, fingerprint=fingerprint, fold=fold, seed=seed, stage="F1",
        completed=completed, selection=f1.selection, model=f1.model, history=f1.history,
        scalers={
            "process_mean": f1.process_scaler.mean, "process_scale": f1.process_scaler.scale,
            "spectrum_mean": f1.spectrum_scaler.mean, "spectrum_scale": f1.spectrum_scaler.scale,
            "quality_mean": f1.quality_scaler.mean, "quality_scale": f1.quality_scaler.scale,
        }, device=selected_device,
    )

    # R1: residual expert target is exactly Task 6's outer-train P1 OOF residual.
    r1 = _fit_signal_stage(
        stage="R1", config=config, frame=train_frame, cache=cache,
        features=train_features, targets=np.asarray(p1_oof.residual, dtype=np.float64),
        inner_splits=inner_splits,
        model_factory=lambda _fold: ResidualExpert(), backend=trainer,
        device=selected_device, seed=seed, batch_size=batch_size,
    )
    checkpoints["R1"], histories["R1"] = _persist_stage(
        fold_dir=fold_dir, fingerprint=fingerprint, fold=fold, seed=seed, stage="R1",
        completed=completed, selection=r1.selection, model=r1.model, history=r1.history,
        scalers={
            "process_mean": r1.process_scaler.mean, "process_scale": r1.process_scaler.scale,
            "spectrum_mean": r1.spectrum_scaler.mean, "spectrum_scale": r1.spectrum_scaler.scale,
            "quality_mean": r1.quality_scaler.mean, "quality_scale": r1.quality_scaler.scale,
        }, device=selected_device,
    )

    # G1: use fold-matched upstream experts for inner selection, then fresh
    # copies of the two complete-outer-train refits. Only the gate is trainable.
    def gate_factory(inner_fold: int | None) -> SelectiveGatedModel:
        if inner_fold is None:
            process_expert = deepcopy(p1_refit.model).cpu()
            residual_expert = deepcopy(r1.model).cpu()
        else:
            process_expert = deepcopy(p1_inner_models[inner_fold]).cpu()
            residual_expert = deepcopy(r1.inner_models[inner_fold]).cpu()
        model = SelectiveGatedModel(process_expert, residual_expert)
        model.freeze_experts()
        return model

    g1 = _fit_signal_stage(
        stage="G1", config=config, frame=train_frame, cache=cache,
        features=train_features, targets=train_targets, inner_splits=inner_splits,
        model_factory=gate_factory, backend=trainer,
        device=selected_device, seed=seed, batch_size=batch_size,
    )
    checkpoints["G1"], histories["G1"] = _persist_stage(
        fold_dir=fold_dir, fingerprint=fingerprint, fold=fold, seed=seed, stage="G1",
        completed=completed, selection=g1.selection, model=g1.model, history=g1.history,
        scalers={
            "process_mean": g1.process_scaler.mean, "process_scale": g1.process_scaler.scale,
            "spectrum_mean": g1.spectrum_scaler.mean, "spectrum_scale": g1.spectrum_scaler.scale,
            "quality_mean": g1.quality_scaler.mean, "quality_scale": g1.quality_scaler.scale,
        }, device=selected_device,
    )
    if completed != list(MODEL_SEQUENCE):
        raise RuntimeError("Phase A stage state machine did not complete in registered order")

    # All model fitting is complete before this block constructs any held-out
    # label array. Inference itself uses dummy targets and swap averaging.
    process_test = outer_process_scaler.transform(test_features)
    outputs: dict[str, ModelOutput] = {
        "P1": _predict_process(
            p1_refit.model, test_frame, process_test, selected_device, batch_size
        )
    }
    outputs["V1"] = _predict_signal(
        stage="V1", model=v1.model, frame=test_frame, cache=cache,
        spectrum_scaler=v1.spectrum_scaler, quality_scaler=v1.quality_scaler,
        process=v1.process_scaler.transform(test_features), device=selected_device,
        batch_size=batch_size,
    )
    outputs["F1"] = _predict_signal(
        stage="F1", model=f1.model, frame=test_frame, cache=cache,
        spectrum_scaler=f1.spectrum_scaler, quality_scaler=f1.quality_scaler,
        process=f1.process_scaler.transform(test_features), device=selected_device,
        batch_size=batch_size,
    )
    r1_model = ResidualFusionModel(
        deepcopy(p1_refit.model).cpu(), deepcopy(r1.model).cpu()
    )
    outputs["R1"] = _predict_signal(
        stage="R1", model=r1_model, frame=test_frame, cache=cache,
        spectrum_scaler=r1.spectrum_scaler, quality_scaler=r1.quality_scaler,
        process=process_test, device=selected_device, batch_size=batch_size,
    )
    outputs["G1"] = _predict_signal(
        stage="G1", model=g1.model, frame=test_frame, cache=cache,
        spectrum_scaler=g1.spectrum_scaler, quality_scaler=g1.quality_scaler,
        process=g1.process_scaler.transform(test_features), device=selected_device,
        batch_size=batch_size,
    )
    outer_label_frame = bundle.manifest.iloc[test_index].reset_index(drop=True)
    outer_targets = _read_outer_targets(outer_label_frame)
    predictions = _prediction_frame(test_frame, outputs, outer_targets, fold, seed)
    predictions_path = fold_dir / "oof_predictions.csv"
    _atomic_write_frame(predictions_path, predictions)
    marker = _state_payload(
        fingerprint, fold, seed, completed, selected_device, status="complete"
    )
    marker["predictions_sha256"] = _sha256_file(predictions_path)
    marker["prediction_rows"] = len(predictions)
    _atomic_write_json(marker_path, marker)
    _atomic_write_json(
        fold_dir / "state.json",
        _state_payload(
            fingerprint, fold, seed, completed, selected_device, status="complete"
        ),
    )
    return FoldArtifacts(predictions, checkpoints, histories, fingerprint)


def run_phase_a(
    config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    *,
    device: str | torch.device | None = None,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
) -> FoldArtifacts:
    """Run every registered outer fold and atomically persist combined OOF rows."""
    fold_values = sorted(pd.to_numeric(bundle.folds["fold"], errors="raise").astype(int).unique())
    if fold_values != [0, 1, 2, 3, 4]:
        raise ValueError("Phase A requires the existing five outer folds 0..4")
    artifacts = [
        run_phase_a_fold(
            config, bundle, cache, fold, 20260723, device,
            backend=backend, batch_size=batch_size,
        )
        for fold in fold_values
    ]
    predictions = pd.concat([artifact.predictions for artifact in artifacts], ignore_index=True)
    expected_rows = len(bundle.manifest) * len(MODEL_SEQUENCE)
    if (
        len(predictions) != expected_rows
        or predictions.duplicated(["sample_id", "model"]).any()
        or set(predictions["model"]) != set(MODEL_SEQUENCE)
    ):
        raise ValueError("combined Phase A OOF predictions are incomplete or duplicated")
    _atomic_write_frame(Path(config.output_dir) / "oof_predictions.csv", predictions)
    checkpoints = {
        f"fold_{fold}:{stage}": path
        for fold, artifact in zip(fold_values, artifacts, strict=True)
        for stage, path in artifact.checkpoint_paths.items()
    }
    histories = {
        f"fold_{fold}:{stage}": path
        for fold, artifact in zip(fold_values, artifacts, strict=True)
        for stage, path in artifact.history_paths.items()
    }
    return FoldArtifacts(predictions, checkpoints, histories, artifacts[0].fingerprint)


__all__ = [
    "EpochSelection",
    "FoldArtifacts",
    "MODEL_SEQUENCE",
    "RunFingerprint",
    "TorchTrainingBackend",
    "TrainingBackend",
    "TrainingResult",
    "build_run_fingerprint",
    "completed_fold_matches",
    "refit_component",
    "run_phase_a",
    "run_phase_a_fold",
    "select_epochs_group_cv",
    "set_global_seed",
]
