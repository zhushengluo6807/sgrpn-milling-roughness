"""Leakage-safe nested Phase A training and exact fold resume."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import io
import json
from pathlib import Path
import random
from numbers import Real
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
    ProcessOOFResult,
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


PHASE_A_PROTOCOL = "sgrpn-phase-a-v2"
MODEL_SEQUENCE = ("P1", "V1", "F1", "R1", "G1")
OOF_COLUMNS = (
    "sample_id", "group_id", "version", "fold", "seed", "model",
    "target", "prediction", "sample_weight", "process_mean", "residual", "gate",
)
_PROJECT_PHASE_A_OUTPUT = (
    Path(__file__).resolve().parents[3] / "outputs" / "sgrpn" / "phase_a"
).resolve()


def validate_phase_a_output_root(
    configured_output: str | Path,
    *,
    output_root: str | Path | None = None,
) -> Path:
    """Require the one production root, with an explicit exact test injection."""
    configured = Path(configured_output).resolve()
    allowed = (
        _PROJECT_PHASE_A_OUTPUT
        if output_root is None
        else Path(output_root).resolve()
    )
    if configured != allowed:
        raise ValueError(
            "Phase A requires the exact project output root outputs/sgrpn/phase_a; "
            "temporary roots require explicit exact output_root injection"
        )
    return configured


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
    _selection_models: tuple[nn.Module, ...] = field(
        default=(), repr=False, compare=False
    )

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


@dataclass(frozen=True)
class ComponentFoldData:
    train_data: Any
    validation_data: Any
    train_sample_ids: Sequence[str]
    validation_sample_ids: Sequence[str]
    scaler_source_ids: Sequence[str]


@dataclass(frozen=True)
class ComponentRefitData:
    train_data: Any
    sample_ids: Sequence[str]
    scaler_source_ids: Sequence[str]


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
    # Scientific identity follows the canonical in-memory frames actually
    # consumed. Source bytes are bound separately as provenance, never used as
    # a substitute for the consumed state.
    manifest_sha = _frame_sha256(bundle.manifest)
    folds_sha = _frame_sha256(bundle.folds)
    manifest_source_sha = (
        _sha256_file(Path(config.manifest_path))
        if Path(config.manifest_path).is_file()
        else None
    )
    folds_source_sha = (
        _sha256_file(Path(config.folds_path))
        if Path(config.folds_path).is_file()
        else None
    )
    cache_sha = _cache_sha256(cache)
    value = _sha256_bytes(
        json.dumps(
            {
                "protocol": PHASE_A_PROTOCOL,
                "config": config_sha,
                "manifest": manifest_sha,
                "folds": folds_sha,
                "manifest_source": manifest_source_sha,
                "folds_source": folds_source_sha,
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
    if (
        fold is not None
        and (
            isinstance(fold, (bool, np.bool_))
            or not isinstance(fold, (int, np.integer))
        )
    ) or (
        seed is not None
        and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
        )
    ):
        return False
    try:
        raw = json.loads(Path(marker).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    expected_models = list(models)
    raw_fold = raw.get("fold")
    raw_seed = raw.get("seed")
    return bool(
        raw.get("status") == "complete"
        and raw.get("protocol") == PHASE_A_PROTOCOL
        and raw.get("fingerprint") == str(fingerprint)
        and raw.get("models") == expected_models
        and raw.get("completed_stages") == expected_models
        and (
            fold is None
            or (
                isinstance(raw_fold, int)
                and not isinstance(raw_fold, bool)
                and raw_fold == int(fold)
            )
        )
        and (
            seed is None
            or (
                isinstance(raw_seed, int)
                and not isinstance(raw_seed, bool)
                and raw_seed == int(seed)
            )
        )
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _context_seed(
    base_seed: int,
    *,
    outer_fold: int,
    stage: str,
    inner_fold: int,
    substage: str,
) -> int:
    payload = json.dumps(
        {
            "seed": int(base_seed),
            "outer_fold": int(outer_fold),
            "stage": str(stage),
            "inner_fold": int(inner_fold),
            "substage": str(substage),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _construct_seeded_model(
    factory: Callable[..., nn.Module],
    *factory_arguments: Any,
    base_seed: int,
    outer_fold: int,
    stage: str,
    inner_fold: int,
    substage: str,
) -> nn.Module:
    """Construct one model without reading or perturbing ambient RNG state."""
    seed = _context_seed(
        base_seed,
        outer_fold=outer_fold,
        stage=stage,
        inner_fold=inner_fold,
        substage=substage,
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = _factory_call(factory, *factory_arguments)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    if not isinstance(model, nn.Module):
        raise ValueError("component_factory must return a fresh nn.Module")
    return model


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
            outputs.append(
                _model_output(stage, model, batch)
                if stage == "P1"
                else average_swap_predictions(model, batch)
            )
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
    *,
    base_seed: int = 0,
    outer_fold: int = -1,
    stage: str = "component",
    substage: str = "selection",
) -> EpochSelection:
    """Own split preflight and audited scaler boundaries for four group folds."""
    frame = getattr(dataset_factory, "audit_frame", None)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("dataset_factory must expose an audit_frame DataFrame")
    splits = _validate_inner_splits(frame, inner_splits)
    if "sample_id" not in frame or frame["sample_id"].isna().any():
        raise ValueError("audit_frame must contain non-missing sample_id")
    sample_ids = frame["sample_id"].astype(str).to_numpy()
    epochs: list[int] = []
    selection_models: list[nn.Module] = []
    for fold_number, (train_index, valid_index) in enumerate(splits):
        fold_data = _factory_call(
            dataset_factory, train_index, valid_index, fold_number
        )
        if not isinstance(fold_data, ComponentFoldData):
            raise ValueError("dataset_factory must return ComponentFoldData")
        expected_train = tuple(sample_ids[train_index])
        expected_valid = tuple(sample_ids[valid_index])
        if (
            tuple(map(str, fold_data.train_sample_ids)) != expected_train
            or tuple(map(str, fold_data.validation_sample_ids)) != expected_valid
            or tuple(map(str, fold_data.scaler_source_ids)) != expected_train
            or len(set(map(str, fold_data.scaler_source_ids))) != len(expected_train)
        ):
            raise ValueError("dataset/scaler source IDs must exactly match the inner-train boundary")
        model = _construct_seeded_model(
            component_factory,
            fold_number,
            base_seed=base_seed,
            outer_fold=outer_fold,
            stage=stage,
            inner_fold=fold_number,
            substage=substage,
        )
        if any(model is previous for previous in selection_models):
            raise ValueError("component_factory reused a model instance; every fold must be fresh")
        selection_models.append(model)
        result = _factory_call(
            train_step, model, fold_data, validation_step, fold_number
        )
        if isinstance(result, TrainingResult) and result.model is not model:
            raise ValueError("selection callback substituted the exact seeded model instance")
        epoch = result.best_epoch if isinstance(result, TrainingResult) else result
        if isinstance(epoch, (bool, np.bool_)) or not isinstance(epoch, (int, np.integer)) or int(epoch) < 1:
            raise ValueError("train_step must return a positive best epoch")
        epochs.append(int(epoch))
    return EpochSelection(tuple(epochs), tuple(selection_models))


def refit_component(
    component_factory: Callable[..., nn.Module],
    dataset_factory: Callable[..., Any],
    epoch_selection: EpochSelection,
    train_step: Callable[..., TrainingResult | nn.Module],
    *,
    base_seed: int = 0,
    outer_fold: int = -1,
    stage: str = "component",
    substage: str = "refit",
) -> nn.Module:
    """Train a fresh component on complete outer-train data for median epochs."""
    frame = getattr(dataset_factory, "audit_frame", None)
    if not isinstance(frame, pd.DataFrame) or "sample_id" not in frame:
        raise ValueError("dataset_factory must expose an audit_frame with sample_id")
    expected_ids = tuple(frame["sample_id"].astype(str))
    payload = _factory_call(dataset_factory)
    if not isinstance(payload, ComponentRefitData):
        raise ValueError("refit dataset_factory must return ComponentRefitData")
    if (
        tuple(map(str, payload.sample_ids)) != expected_ids
        or tuple(map(str, payload.scaler_source_ids)) != expected_ids
        or len(set(map(str, payload.scaler_source_ids))) != len(expected_ids)
    ):
        raise ValueError("refit data and fresh scaler must use every outer-train row exactly once")
    model = _construct_seeded_model(
        component_factory,
        base_seed=base_seed,
        outer_fold=outer_fold,
        stage=stage,
        inner_fold=-1,
        substage=substage,
    )
    if any(model is previous for previous in epoch_selection._selection_models):
        raise ValueError("refit model must be fresh and distinct from every selection model")
    result = _factory_call(
        train_step, model, payload.train_data, epoch_selection.refit_epochs
    )
    if isinstance(result, TrainingResult):
        if result.model is not model:
            raise ValueError("refit callback substituted the exact seeded model instance")
        return result.model
    if isinstance(result, nn.Module):
        if result is not model:
            raise ValueError("refit callback substituted the exact seeded model instance")
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


@dataclass
class _P1Fit:
    oof: ProcessOOFResult
    selection: EpochSelection
    inner_models: list[ProcessMLP]
    model: ProcessMLP
    scaler: ProcessScaler
    history: pd.DataFrame


@dataclass(frozen=True)
class _ConfinedP1CacheEntry:
    sample_ids: tuple[str, ...]
    fit: _P1Fit


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
    if result.model is not model:
        raise ValueError("training backend substituted the exact seeded model instance")
    if result.best_epoch < 1 or result.best_epoch > max_epochs:
        raise ValueError("training backend returned an invalid best epoch")
    _validate_backend_history(result.history, result.best_epoch, max_epochs)
    return result


def _validate_backend_history(
    history: pd.DataFrame, best_epoch: int, max_epochs: int
) -> None:
    required = {"epoch", "train_loss", "validation_loss"}
    if not isinstance(history, pd.DataFrame) or history.empty or not required.issubset(history):
        raise ValueError("training backend history schema is incompatible")
    raw_epochs = history["epoch"].to_numpy(copy=False)
    epochs: list[int] = []
    for value in raw_epochs:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not float(value).is_integer()
        ):
            raise ValueError("training backend history epoch must be integral")
        epochs.append(int(value))
    losses = history.loc[:, ["train_loss", "validation_loss"]].to_numpy(
        dtype=np.float64
    )
    if (
        len(set(epochs)) != len(epochs)
        or epochs != list(range(1, len(epochs) + 1))
        or epochs[-1] > int(max_epochs)
        or int(best_epoch) not in epochs
        or not np.isfinite(losses).all()
    ):
        raise ValueError("training backend history epochs/losses are inconsistent")


def _normalize_inner_index(raw: Any, *, name: str, row_count: int) -> np.ndarray:
    values = np.asarray(raw)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"inner {name} indices must be a nonempty vector")
    normalized: list[int] = []
    for value in values:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not float(value).is_integer()
        ):
            raise ValueError(f"inner {name} indices must be finite integral non-boolean values")
        index = int(value)
        if index < 0 or index >= row_count:
            raise ValueError(f"inner {name} index is outside row bounds")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"inner {name} indices must be unique")
    return np.asarray(normalized, dtype=np.int64)


def _validate_inner_splits(
    frame: pd.DataFrame, splits: Sequence[tuple[np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(splits) != 4:
        raise ValueError("Phase A requires exactly four inner splits")
    assignment = np.zeros(len(frame), dtype=np.int64)
    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    groups = frame["group_id"].astype(str).to_numpy()
    for raw_train, raw_valid in splits:
        train_index = _normalize_inner_index(
            raw_train, name="train", row_count=len(frame)
        )
        valid_index = _normalize_inner_index(
            raw_valid, name="validation", row_count=len(frame)
        )
        if (
            np.intersect1d(train_index, valid_index).size
            or not np.array_equal(
                np.sort(np.concatenate((train_index, valid_index))),
                np.arange(len(frame), dtype=np.int64),
            )
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


def _fit_p1_subset(
    *,
    config: SGRPNConfig,
    frame: pd.DataFrame,
    features: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    backend: TrainingBackend,
    device: torch.device,
    seed: int,
    outer_fold: int,
    context_stage: str,
    context_inner_fold: int = -1,
    batch_size: int,
) -> _P1Fit:
    """Cross-fit and refit P1 using only the supplied group-confined frame."""
    splits = _validate_inner_splits(frame, inner_splits)
    inner_models: list[ProcessMLP] = []
    histories: list[pd.DataFrame] = []
    call_number = 0

    def callback(
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
        nonlocal call_number
        train_rows, valid_rows = splits[call_number]
        train_frame = frame.iloc[train_rows].copy()
        valid_frame = frame.iloc[valid_rows].copy()
        if not np.array_equal(train_frame["group_id"].astype(str), train_groups.astype(str)):
            raise ValueError("P1 inner train group order is incompatible")
        if not np.array_equal(valid_frame["group_id"].astype(str), valid_groups.astype(str)):
            raise ValueError("P1 inner validation group order is incompatible")
        train_frame.loc[:, "sample_weight"] = w_train
        valid_frame.loc[:, "sample_weight"] = w_valid
        result = _fit_backend(
            backend,
            stage="P1",
            model=_construct_seeded_model(
                ProcessMLP,
                base_seed=seed,
                outer_fold=outer_fold,
                stage=context_stage,
                inner_fold=context_inner_fold,
                substage=f"p1-selection-{call_number}",
            ),
            train_loader=_loader(
                _ProcessDataset(train_frame, x_train, y_train),
                batch_size=batch_size, shuffle=True, seed=seed,
            ),
            validation_loader=_loader(
                _ProcessDataset(valid_frame, x_valid, y_valid),
                batch_size=batch_size, shuffle=False, seed=seed,
            ),
            max_epochs=config.max_epochs,
            config=config,
            device=device,
            seed=seed,
        )
        result.model.to(device).eval()
        with torch.no_grad():
            prediction = result.model(
                torch.as_tensor(x_valid, dtype=torch.float32, device=device)
            ).detach().cpu().numpy()
        inner_models.append(result.model)
        history = result.history.copy()
        history.insert(0, "inner_fold", call_number)
        history.insert(0, "phase", "selection")
        histories.append(history)
        call_number += 1
        return prediction.astype(np.float64, copy=False), result.best_epoch

    oof = generate_process_oof(
        frame,
        features,
        ValidationAwareTrainer(callback),
        feature_sample_ids=frame["sample_id"].astype(str).to_numpy(),
        n_splits=4,
        seed=seed,
    )
    selection = EpochSelection(tuple(oof.best_epochs))
    all_rows = np.arange(len(frame), dtype=np.int64)
    scaler = fit_process_scaler(features, all_rows)
    refit = _fit_backend(
        backend,
        stage="P1",
        model=_construct_seeded_model(
            ProcessMLP,
            base_seed=seed,
            outer_fold=outer_fold,
            stage=context_stage,
            inner_fold=context_inner_fold,
            substage="p1-refit",
        ),
        train_loader=_loader(
            _ProcessDataset(
                frame, scaler.transform(features),
                frame["ra_mean"].to_numpy(dtype=np.float64, copy=True),
            ),
            batch_size=batch_size, shuffle=True, seed=seed,
        ),
        validation_loader=None,
        max_epochs=selection.refit_epochs,
        config=config,
        device=device,
        seed=seed,
    )
    refit_history = refit.history.copy()
    refit_history.insert(0, "inner_fold", -1)
    refit_history.insert(0, "phase", "refit")
    histories.append(refit_history)
    return _P1Fit(
        oof=oof,
        selection=selection,
        inner_models=inner_models,
        model=refit.model,
        scaler=scaler,
        history=pd.concat(histories, ignore_index=True),
    )


def _process_prediction_array(
    model: nn.Module,
    scaler: ProcessScaler,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.to(device).eval()
    with torch.no_grad():
        values = model(
            torch.as_tensor(scaler.transform(features), dtype=torch.float32, device=device)
        ).detach().cpu().numpy().astype(np.float64, copy=False)
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("P1 subset prediction must be a finite vector")
    return values


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
    outer_fold: int,
    context_stage: str | None = None,
    context_inner_fold: int = -1,
    batch_size: int,
) -> _SignalFit:
    inner_models: list[nn.Module] = []
    process_scalers: list[ProcessScaler] = []
    spectrum_scalers: list[SpectrumScaler] = []
    quality_scalers: list[QualityScaler] = []
    histories: list[pd.DataFrame] = []
    sample_ids = frame["sample_id"].astype(str).to_numpy()

    class SelectionDatasets:
        audit_frame = frame

        def __call__(self, train_index, valid_index, fold_number):
            train_ids = sample_ids[train_index].tolist()
            process_scaler = fit_process_scaler(features, train_index)
            spectrum_scaler = fit_spectrum_scaler(cache, train_ids)
            quality_scaler = fit_quality_scaler(cache, train_ids)
            process_scalers.append(process_scaler)
            spectrum_scalers.append(spectrum_scaler)
            quality_scalers.append(quality_scaler)
            return ComponentFoldData(
                train_data=_signal_dataset(
                    frame.iloc[train_index], cache, spectrum_scaler, quality_scaler,
                    process_scaler.transform(features[train_index]), targets[train_index],
                    augment=True,
                ),
                validation_data=_signal_dataset(
                    frame.iloc[valid_index], cache, spectrum_scaler, quality_scaler,
                    process_scaler.transform(features[valid_index]), targets[valid_index],
                    augment=False,
                ),
                train_sample_ids=tuple(sample_ids[train_index]),
                validation_sample_ids=tuple(sample_ids[valid_index]),
                scaler_source_ids=tuple(train_ids),
            )

    def selection_step(model, fold_data, _validation_step, fold_number):
        before = None
        if stage == "G1":
            if not isinstance(model, SelectiveGatedModel):
                raise ValueError("G1 factory must return SelectiveGatedModel")
            model.freeze_experts()
            before = _expert_state(model)
        result = _fit_backend(
            backend, stage=stage, model=model,
            train_loader=_loader(
                fold_data.train_data, batch_size=batch_size, shuffle=True,
                seed=seed, order_bags=True,
            ),
            validation_loader=_loader(
                fold_data.validation_data, batch_size=batch_size, shuffle=False,
                seed=seed, order_bags=True,
            ),
            max_epochs=config.max_epochs, config=config, device=device, seed=seed,
        )
        if before is not None:
            _assert_experts_unchanged(before, result.model)
        inner_models.append(result.model)
        history = result.history.copy()
        history.insert(0, "inner_fold", fold_number)
        history.insert(0, "phase", "selection")
        histories.append(history)
        return result

    selection = select_epochs_group_cv(
        lambda fold_number: model_factory(fold_number),
        SelectionDatasets(), inner_splits, selection_step, None,
        base_seed=seed,
        outer_fold=outer_fold,
        stage=stage if context_stage is None else context_stage,
        substage=f"selection-within-{context_inner_fold}",
    )

    class RefitDataset:
        audit_frame = frame

        def __init__(self):
            self.process_scaler: ProcessScaler | None = None
            self.spectrum_scaler: SpectrumScaler | None = None
            self.quality_scaler: QualityScaler | None = None

        def __call__(self):
            all_index = np.arange(len(frame), dtype=np.int64)
            all_ids = sample_ids.tolist()
            self.process_scaler = fit_process_scaler(features, all_index)
            self.spectrum_scaler = fit_spectrum_scaler(cache, all_ids)
            self.quality_scaler = fit_quality_scaler(cache, all_ids)
            data = _signal_dataset(
                frame, cache, self.spectrum_scaler, self.quality_scaler,
                self.process_scaler.transform(features), targets, augment=True,
            )
            return ComponentRefitData(data, tuple(all_ids), tuple(all_ids))

    refit_dataset = RefitDataset()
    refit_results: list[TrainingResult] = []

    def refit_step(model, dataset, epochs):
        before = None
        if stage == "G1":
            if not isinstance(model, SelectiveGatedModel):
                raise ValueError("G1 factory must return SelectiveGatedModel")
            model.freeze_experts()
            before = _expert_state(model)
        result = _fit_backend(
            backend, stage=stage, model=model,
            train_loader=_loader(
                dataset, batch_size=batch_size, shuffle=True, seed=seed,
                order_bags=True,
            ),
            validation_loader=None, max_epochs=epochs, config=config,
            device=device, seed=seed,
        )
        if before is not None:
            _assert_experts_unchanged(before, result.model)
        refit_results.append(result)
        return result

    refit_model = refit_component(
        lambda: model_factory(None), refit_dataset, selection, refit_step,
        base_seed=seed,
        outer_fold=outer_fold,
        stage=stage if context_stage is None else context_stage,
        substage=f"refit-within-{context_inner_fold}",
    )
    refit = refit_results[0]
    history = refit.history.copy()
    history.insert(0, "inner_fold", -1)
    history.insert(0, "phase", "refit")
    histories.append(history)
    if (
        refit_dataset.process_scaler is None
        or refit_dataset.spectrum_scaler is None
        or refit_dataset.quality_scaler is None
    ):
        raise RuntimeError("refit scaler factory did not produce complete scalers")
    return _SignalFit(
        selection=selection,
        inner_models=inner_models,
        inner_process_scalers=process_scalers,
        inner_spectrum_scalers=spectrum_scalers,
        inner_quality_scalers=quality_scalers,
        model=refit_model,
        process_scaler=refit_dataset.process_scaler,
        spectrum_scaler=refit_dataset.spectrum_scaler,
        quality_scaler=refit_dataset.quality_scaler,
        history=pd.concat(histories, ignore_index=True),
    )


def _fit_r1_nested_stage(
    *,
    config: SGRPNConfig,
    frame: pd.DataFrame,
    cache: OrderSpectrumCache,
    features: np.ndarray,
    outer_p1_oof: ProcessOOFResult,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    backend: TrainingBackend,
    device: torch.device,
    seed: int,
    outer_fold: int,
    batch_size: int,
) -> tuple[_SignalFit, dict[int, _ConfinedP1CacheEntry]]:
    """Select R1 with fold-confined P1 residual targets, then refit outer-train."""
    epochs: list[int] = []
    histories: list[pd.DataFrame] = []
    inner_models: list[nn.Module] = []
    process_scalers: list[ProcessScaler] = []
    spectrum_scalers: list[SpectrumScaler] = []
    quality_scalers: list[QualityScaler] = []
    p1_cache: dict[int, _ConfinedP1CacheEntry] = {}
    target = frame["ra_mean"].to_numpy(dtype=np.float64, copy=True)
    for fold_number, (train_index, valid_index) in enumerate(inner_splits):
        subset = frame.iloc[train_index].reset_index(drop=True).copy()
        subset_features = features[train_index]
        subset_splits = _validate_inner_splits(
            subset,
            make_group_inner_splits(subset, n_splits=4, seed=seed),
        )
        p1_subset = _fit_p1_subset(
            config=config, frame=subset, features=subset_features,
            inner_splits=subset_splits, backend=backend, device=device,
            seed=seed, outer_fold=outer_fold, context_stage="R1-upstream-P1",
            context_inner_fold=fold_number, batch_size=batch_size,
        )
        p1_cache[fold_number] = _ConfinedP1CacheEntry(
            tuple(subset["sample_id"].astype(str)), p1_subset
        )
        valid_residual = target[valid_index] - _process_prediction_array(
            p1_subset.model, p1_subset.scaler, features[valid_index], device
        )
        train_ids = subset["sample_id"].astype(str).tolist()
        spectrum_scaler = fit_spectrum_scaler(cache, train_ids)
        quality_scaler = fit_quality_scaler(cache, train_ids)
        train_data = _signal_dataset(
            subset, cache, spectrum_scaler, quality_scaler,
            p1_subset.scaler.transform(subset_features), p1_subset.oof.residual,
            augment=True,
        )
        valid_data = _signal_dataset(
            frame.iloc[valid_index], cache, spectrum_scaler, quality_scaler,
            p1_subset.scaler.transform(features[valid_index]), valid_residual,
            augment=False,
        )
        result = _fit_backend(
            backend, stage="R1", model=_construct_seeded_model(
                ResidualExpert,
                base_seed=seed,
                outer_fold=outer_fold,
                stage="R1",
                inner_fold=fold_number,
                substage="selection",
            ),
            train_loader=_loader(
                train_data, batch_size=batch_size, shuffle=True, seed=seed,
                order_bags=True,
            ),
            validation_loader=_loader(
                valid_data, batch_size=batch_size, shuffle=False, seed=seed,
                order_bags=True,
            ),
            max_epochs=config.max_epochs, config=config, device=device, seed=seed,
        )
        epochs.append(result.best_epoch)
        inner_models.append(result.model)
        process_scalers.append(p1_subset.scaler)
        spectrum_scalers.append(spectrum_scaler)
        quality_scalers.append(quality_scaler)
        history = result.history.copy()
        history.insert(0, "inner_fold", fold_number)
        history.insert(0, "phase", "selection")
        histories.append(history)

    selection = EpochSelection(tuple(epochs))
    all_rows = np.arange(len(frame), dtype=np.int64)
    process_scaler = fit_process_scaler(features, all_rows)
    all_ids = frame["sample_id"].astype(str).tolist()
    spectrum_scaler = fit_spectrum_scaler(cache, all_ids)
    quality_scaler = fit_quality_scaler(cache, all_ids)
    refit_data = _signal_dataset(
        frame, cache, spectrum_scaler, quality_scaler,
        process_scaler.transform(features), outer_p1_oof.residual, augment=True,
    )
    refit = _fit_backend(
        backend, stage="R1", model=_construct_seeded_model(
            ResidualExpert,
            base_seed=seed,
            outer_fold=outer_fold,
            stage="R1",
            inner_fold=-1,
            substage="refit",
        ),
        train_loader=_loader(
            refit_data, batch_size=batch_size, shuffle=True, seed=seed,
            order_bags=True,
        ),
        validation_loader=None, max_epochs=selection.refit_epochs,
        config=config, device=device, seed=seed,
    )
    refit_history = refit.history.copy()
    refit_history.insert(0, "inner_fold", -1)
    refit_history.insert(0, "phase", "refit")
    histories.append(refit_history)
    return (
        _SignalFit(
            selection, inner_models, process_scalers, spectrum_scalers,
            quality_scalers, refit.model, process_scaler, spectrum_scaler,
            quality_scaler, pd.concat(histories, ignore_index=True),
        ),
        p1_cache,
    )


def _fit_g1_nested_stage(
    *,
    config: SGRPNConfig,
    frame: pd.DataFrame,
    cache: OrderSpectrumCache,
    features: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    outer_p1: _P1Fit,
    outer_r1: _SignalFit,
    p1_cache: dict[int, _ConfinedP1CacheEntry],
    backend: TrainingBackend,
    device: torch.device,
    seed: int,
    outer_fold: int,
    batch_size: int,
) -> _SignalFit:
    """Select gates with upstream experts rebuilt entirely without fold k."""
    target = frame["ra_mean"].to_numpy(dtype=np.float64, copy=True)
    epochs: list[int] = []
    histories: list[pd.DataFrame] = []
    inner_models: list[nn.Module] = []
    process_scalers: list[ProcessScaler] = []
    spectrum_scalers: list[SpectrumScaler] = []
    quality_scalers: list[QualityScaler] = []
    for fold_number, (train_index, valid_index) in enumerate(inner_splits):
        subset = frame.iloc[train_index].reset_index(drop=True).copy()
        subset_features = features[train_index]
        subset_target = target[train_index]
        subset_splits = _validate_inner_splits(
            subset,
            make_group_inner_splits(subset, n_splits=4, seed=seed),
        )
        try:
            cache_entry = p1_cache[fold_number]
        except KeyError as error:
            raise RuntimeError("missing fold-confined P1 cache entry") from error
        if cache_entry.sample_ids != tuple(subset["sample_id"].astype(str)):
            raise RuntimeError("fold-confined P1 cache boundary is incompatible")
        p1_subset = cache_entry.fit
        cached_p1_state = {
            name: value.detach().cpu().clone()
            for name, value in p1_subset.model.state_dict().items()
        }
        residual_subset = _fit_signal_stage(
            stage="R1", config=config, frame=subset, cache=cache,
            features=subset_features, targets=p1_subset.oof.residual,
            inner_splits=subset_splits, model_factory=lambda _fold: ResidualExpert(),
            backend=backend, device=device, seed=seed, outer_fold=outer_fold,
            context_stage="G1-upstream-R1", context_inner_fold=fold_number,
            batch_size=batch_size,
        )
        model = _construct_seeded_model(
            lambda: SelectiveGatedModel(
                deepcopy(p1_subset.model).cpu(), deepcopy(residual_subset.model).cpu()
            ),
            base_seed=seed,
            outer_fold=outer_fold,
            stage="G1",
            inner_fold=fold_number,
            substage="selection",
        )
        model.freeze_experts()
        before = _expert_state(model)
        train_data = _signal_dataset(
            subset, cache, residual_subset.spectrum_scaler,
            residual_subset.quality_scaler,
            p1_subset.scaler.transform(subset_features), subset_target,
            augment=True,
        )
        valid_data = _signal_dataset(
            frame.iloc[valid_index], cache, residual_subset.spectrum_scaler,
            residual_subset.quality_scaler,
            p1_subset.scaler.transform(features[valid_index]), target[valid_index],
            augment=False,
        )
        result = _fit_backend(
            backend, stage="G1", model=model,
            train_loader=_loader(
                train_data, batch_size=batch_size, shuffle=True, seed=seed,
                order_bags=True,
            ),
            validation_loader=_loader(
                valid_data, batch_size=batch_size, shuffle=False, seed=seed,
                order_bags=True,
            ),
            max_epochs=config.max_epochs, config=config, device=device, seed=seed,
        )
        _assert_experts_unchanged(before, result.model)
        if not all(
            torch.equal(value, p1_subset.model.state_dict()[name].detach().cpu())
            for name, value in cached_p1_state.items()
        ):
            raise RuntimeError("G1 mutated a cached fold-confined P1 expert")
        epochs.append(result.best_epoch)
        inner_models.append(result.model)
        process_scalers.append(p1_subset.scaler)
        spectrum_scalers.append(residual_subset.spectrum_scaler)
        quality_scalers.append(residual_subset.quality_scaler)
        history = result.history.copy()
        history.insert(0, "inner_fold", fold_number)
        history.insert(0, "phase", "selection")
        histories.append(history)

    selection = EpochSelection(tuple(epochs))
    model = _construct_seeded_model(
        lambda: SelectiveGatedModel(
            deepcopy(outer_p1.model).cpu(), deepcopy(outer_r1.model).cpu()
        ),
        base_seed=seed,
        outer_fold=outer_fold,
        stage="G1",
        inner_fold=-1,
        substage="refit",
    )
    model.freeze_experts()
    before = _expert_state(model)
    refit_data = _signal_dataset(
        frame, cache, outer_r1.spectrum_scaler, outer_r1.quality_scaler,
        outer_p1.scaler.transform(features), target, augment=True,
    )
    refit = _fit_backend(
        backend, stage="G1", model=model,
        train_loader=_loader(
            refit_data, batch_size=batch_size, shuffle=True, seed=seed,
            order_bags=True,
        ),
        validation_loader=None, max_epochs=selection.refit_epochs,
        config=config, device=device, seed=seed,
    )
    _assert_experts_unchanged(before, refit.model)
    refit_history = refit.history.copy()
    refit_history.insert(0, "inner_fold", -1)
    refit_history.insert(0, "phase", "refit")
    histories.append(refit_history)
    return _SignalFit(
        selection, inner_models, process_scalers, spectrum_scalers,
        quality_scalers, refit.model, outer_p1.scaler,
        outer_r1.spectrum_scaler, outer_r1.quality_scaler,
        pd.concat(histories, ignore_index=True),
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
        "protocol": PHASE_A_PROTOCOL,
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
        raw.get("protocol") != PHASE_A_PROTOCOL
        or raw.get("fingerprint") != fingerprint.value
        or not isinstance(raw.get("fold"), int)
        or isinstance(raw.get("fold"), bool)
        or raw.get("fold") != int(fold)
        or not isinstance(raw.get("seed"), int)
        or isinstance(raw.get("seed"), bool)
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
        "protocol": PHASE_A_PROTOCOL,
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
    metadata = {
        "protocol": PHASE_A_PROTOCOL,
        "fingerprint": fingerprint.value,
        "fold": int(fold),
        "seed": int(seed),
        "stage": stage,
        "best_epochs": list(map(int, selection.best_epochs)),
        "refit_epochs": selection.refit_epochs,
    }
    _atomic_torch_save(
        checkpoint,
        _checkpoint_payload(
            fingerprint=fingerprint, fold=fold, seed=seed, stage=stage,
            selection=selection, model=model, device=device,
        ),
    )
    bound_history = history.copy()
    bound_history["protocol"] = metadata["protocol"]
    bound_history["fingerprint"] = metadata["fingerprint"]
    bound_history["fold"] = metadata["fold"]
    bound_history["seed"] = metadata["seed"]
    bound_history["stage"] = metadata["stage"]
    bound_history["best_epochs"] = json.dumps(
        metadata["best_epochs"], separators=(",", ":")
    )
    bound_history["refit_epochs"] = metadata["refit_epochs"]
    _atomic_write_frame(history_path, bound_history)
    _atomic_save_scalers(
        scaler_path,
        **scalers,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    checkpoint_payload = _validate_checkpoint(
        checkpoint,
        stage=stage,
        fingerprint=fingerprint,
        fold=fold,
        seed=seed,
    )
    persisted_metadata = _artifact_metadata(
        fingerprint=fingerprint,
        fold=fold,
        seed=seed,
        stage=stage,
        best_epochs=checkpoint_payload["best_epochs"],
        refit_epochs=checkpoint_payload["refit_epochs"],
    )
    _validate_history(history_path, stage, persisted_metadata)
    _validate_scalers(scaler_path, stage, persisted_metadata)
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
    _validate_oof_frame(predictions, frame, fold, seed)
    return predictions


def _validate_outer_fold_definitions(folds: pd.DataFrame) -> list[int]:
    if "fold" not in folds or folds["fold"].isna().any():
        raise ValueError("outer fold definitions must contain non-missing fold values")
    values: list[int] = []
    for value in folds["fold"]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("outer fold definitions must use canonical integral values")
        values.append(int(value))
    unique = sorted(set(values))
    if unique != [0, 1, 2, 3, 4]:
        raise ValueError("Phase A requires the existing five outer folds 0..4")
    return values


def _validate_oof_frame(
    predictions: pd.DataFrame,
    expected_frame: pd.DataFrame,
    fold: int,
    seed: int,
) -> None:
    if tuple(predictions.columns) != OOF_COLUMNS:
        raise ValueError("OOF prediction schema is incompatible")
    normalized = predictions.copy()
    for column in ("sample_id", "group_id", "version", "model"):
        if normalized[column].isna().any():
            raise ValueError(f"OOF prediction {column} must not be missing")
        normalized.loc[:, column] = normalized[column].astype(str)
    expected = expected_frame.copy()
    for column in ("sample_id", "group_id", "version"):
        expected.loc[:, column] = expected[column].astype(str)
    expected_ids = expected["sample_id"].tolist()
    expected_pairs = {
        (sample_id, model) for sample_id in expected_ids for model in MODEL_SEQUENCE
    }
    actual_pairs = set(zip(normalized["sample_id"], normalized["model"], strict=True))
    if (
        len(normalized) != len(expected_pairs)
        or normalized.duplicated(["sample_id", "model"]).any()
        or actual_pairs != expected_pairs
    ):
        raise ValueError("OOF predictions must equal the exact sample/model Cartesian product")
    def exact_integer_column(column: str) -> np.ndarray:
        values: list[int] = []
        for value in normalized[column]:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"OOF {column} must use canonical integral values")
            if isinstance(value, (int, np.integer)):
                values.append(int(value))
                continue
            if isinstance(value, str) and _canonical_integer_text(value, signed=True):
                values.append(int(value))
                continue
            raise ValueError(f"OOF {column} must use canonical integral values")
        return np.asarray(values, dtype=np.int64)

    try:
        fold_values = exact_integer_column("fold")
        seed_values = exact_integer_column("seed")
        numeric = normalized.loc[:, ["target", "prediction", "sample_weight"]].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("OOF prediction numeric columns are incompatible") from error
    if (
        not np.isfinite(fold_values).all()
        or not np.isfinite(seed_values).all()
        or not np.all(fold_values == int(fold))
        or not np.all(seed_values == int(seed))
        or not np.isfinite(numeric).all()
        or np.any(numeric[:, 2] <= 0)
    ):
        raise ValueError("OOF fold/seed/target/prediction/weight values are incompatible")
    expected_by_id = expected.set_index("sample_id")
    for row in normalized.itertuples(index=False):
        source = expected_by_id.loc[str(row.sample_id)]
        if (
            str(row.group_id) != str(source["group_id"])
            or str(row.version) != str(source["version"])
            or not np.isclose(float(row.sample_weight), float(source["sample_weight"]), rtol=0.0, atol=1e-12)
            or (
                "ra_mean" in expected_by_id.columns
                and not np.isclose(
                    float(row.target), float(source["ra_mean"]), rtol=0.0, atol=1e-12
                )
            )
        ):
            raise ValueError("OOF ID/group/version/target/weight mapping is incompatible")
    semantics = {
        "P1": (("process_mean",), ("residual", "gate")),
        "V1": ((), ("process_mean", "residual", "gate")),
        "F1": ((), ("process_mean", "residual", "gate")),
        "R1": (("process_mean", "residual"), ("gate",)),
        "G1": (("process_mean", "residual", "gate"), ()),
    }
    for model, (finite_columns, nan_columns) in semantics.items():
        rows = normalized.loc[normalized["model"] == model]
        if any(not np.isfinite(rows[column].to_numpy(dtype=np.float64)).all() for column in finite_columns):
            raise ValueError(f"OOF {model} populated components must be finite")
        if any(not rows[column].isna().all() for column in nan_columns):
            raise ValueError(f"OOF {model} absent components must be NaN")
    gates = normalized.loc[normalized["model"] == "G1", "gate"].to_numpy(dtype=np.float64)
    if np.any((gates < 0.0) | (gates > 1.0)):
        raise ValueError("OOF G1 gates must lie in [0, 1]")


def _validate_protocol(
    config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    fold: int,
    seed: int,
    output_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or tuple(config.seeds) != (20260723,)
        or int(seed) != 20260723
    ):
        raise ValueError("Phase A requires exactly seed 20260723")
    if (
        isinstance(fold, (bool, np.bool_))
        or not isinstance(fold, (int, np.integer))
    ):
        raise ValueError("Phase A outer fold must be an integral fold index")
    _validate_outer_fold_definitions(bundle.folds)
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
    validate_phase_a_output_root(config.output_dir, output_root=output_root)
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


def _load_and_validate_persisted_oof(
    fold_dir: Path,
    marker: dict[str, Any],
    fold: int,
    seed: int,
    expected_frame: pd.DataFrame,
) -> pd.DataFrame:
    predictions_path = fold_dir / "oof_predictions.csv"
    if (
        not predictions_path.is_file()
        or marker.get("predictions_sha256") != _sha256_file(predictions_path)
    ):
        raise ValueError("completed fold prediction fingerprint is missing or incompatible")
    predictions = pd.read_csv(
        predictions_path,
        dtype={
            "sample_id": str, "group_id": str, "version": str, "model": str,
            "fold": str, "seed": str,
        },
    )
    _validate_oof_frame(predictions, expected_frame, fold, seed)
    predictions["fold"] = predictions["fold"].astype(np.int64)
    predictions["seed"] = predictions["seed"].astype(np.int64)
    for column in (
        "target", "prediction", "sample_weight", "process_mean", "residual", "gate"
    ):
        predictions[column] = pd.to_numeric(
            predictions[column], errors="raise"
        ).astype(np.float64)
    return predictions


def _load_completed_fold(
    fold_dir: Path,
    fingerprint: RunFingerprint,
    fold: int,
    seed: int,
    expected_frame: pd.DataFrame,
) -> FoldArtifacts:
    marker = json.loads((fold_dir / "complete.json").read_text(encoding="utf-8"))
    _validate_completed_artifacts(fold_dir, marker, fingerprint, fold, seed)
    predictions = _load_and_validate_persisted_oof(
        fold_dir, marker, fold, seed, expected_frame
    )
    checkpoints = {stage: _stage_paths(fold_dir, stage)[0] for stage in MODEL_SEQUENCE}
    histories = {stage: _stage_paths(fold_dir, stage)[1] for stage in MODEL_SEQUENCE}
    if not all(path.is_file() for path in (*checkpoints.values(), *histories.values())):
        raise ValueError("completed fold is missing checkpoint or history artifacts")
    return FoldArtifacts(predictions, checkpoints, histories, fingerprint)


def _checkpoint_model(stage: str) -> nn.Module:
    if stage == "P1":
        return ProcessMLP()
    if stage == "V1":
        return VibrationOnlyModel()
    if stage == "F1":
        return DirectFusionModel()
    if stage == "R1":
        return ResidualExpert()
    if stage == "G1":
        return SelectiveGatedModel()
    raise ValueError(f"unknown checkpoint stage: {stage}")


def _expected_artifact_paths(fold_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for stage in MODEL_SEQUENCE:
        checkpoint, history, scaler = _stage_paths(fold_dir, stage)
        for path in (checkpoint, history, scaler):
            paths[path.relative_to(fold_dir).as_posix()] = path
    return paths


def _validate_checkpoint(
    path: Path,
    *,
    stage: str,
    fingerprint: RunFingerprint,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"checkpoint {stage} cannot be loaded") from error
    epochs = payload.get("best_epochs") if isinstance(payload, dict) else None
    refit_epochs = payload.get("refit_epochs") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("protocol") != PHASE_A_PROTOCOL
        or payload.get("fingerprint") != fingerprint.value
        or not isinstance(payload.get("fold"), int)
        or isinstance(payload.get("fold"), bool)
        or payload.get("fold") != int(fold)
        or not isinstance(payload.get("seed"), int)
        or isinstance(payload.get("seed"), bool)
        or payload.get("seed") != int(seed)
        or payload.get("model") != stage
        or payload.get("completed_stage") != stage
        or not isinstance(epochs, list)
        or len(epochs) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in epochs)
        or not isinstance(refit_epochs, int)
        or isinstance(refit_epochs, bool)
        or refit_epochs != EpochSelection(tuple(epochs)).refit_epochs
    ):
        raise ValueError(f"checkpoint {stage} metadata is incompatible")
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint {stage} model state is missing")
    for value in state.values():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint {stage} model state must contain finite tensors")
    try:
        _construct_seeded_model(
            _checkpoint_model,
            stage,
            base_seed=seed,
            outer_fold=fold,
            stage=f"validate-{stage}",
            inner_fold=-1,
            substage="checkpoint",
        ).load_state_dict(state, strict=True)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError(f"checkpoint {stage} model state is invalid") from error
    return payload


def _canonical_integer_text(value: Any, *, signed: bool = False) -> bool:
    original = str(value)
    negative = signed and original.startswith("-")
    text = original[1:] if negative else original
    return bool(text) and text.isascii() and text.isdigit() and (
        text == "0" or not text.startswith("0")
    ) and not (negative and text == "0")


def _artifact_metadata(
    *, fingerprint: RunFingerprint, fold: int, seed: int, stage: str,
    best_epochs: Sequence[int], refit_epochs: int,
) -> dict[str, Any]:
    return {
        "protocol": PHASE_A_PROTOCOL,
        "fingerprint": fingerprint.value,
        "fold": int(fold),
        "seed": int(seed),
        "stage": stage,
        "best_epochs": list(map(int, best_epochs)),
        "refit_epochs": int(refit_epochs),
    }


def _validate_history(
    path: Path,
    stage: str,
    expected_metadata: dict[str, Any],
) -> None:
    try:
        history = pd.read_csv(
            path,
            dtype={
                "inner_fold": str,
                "epoch": str,
                "fold": str,
                "seed": str,
                "refit_epochs": str,
                "protocol": str,
                "fingerprint": str,
                "stage": str,
                "best_epochs": str,
            },
        )
    except Exception as error:
        raise ValueError(f"history {stage} cannot be loaded") from error
    required = {
        "phase", "inner_fold", "epoch", "train_loss", "validation_loss",
        "protocol", "fingerprint", "fold", "seed", "stage", "best_epochs",
        "refit_epochs",
    }
    if history.empty or not required.issubset(history.columns):
        raise ValueError(f"history {stage} schema is incompatible")
    if not all(_canonical_integer_text(value, signed=True) for value in history["inner_fold"]):
        raise ValueError(f"history {stage} inner_fold must be integral")
    if not all(_canonical_integer_text(value) for value in history["epoch"]):
        raise ValueError(f"history {stage} epoch must be integral")
    if not all(_canonical_integer_text(value) for value in history["fold"]):
        raise ValueError(f"history {stage} fold must be integral")
    if not all(_canonical_integer_text(value) for value in history["seed"]):
        raise ValueError(f"history {stage} seed must be integral")
    if not all(_canonical_integer_text(value) for value in history["refit_epochs"]):
        raise ValueError(f"history {stage} refit epoch must be integral")
    inner_fold = history["inner_fold"].astype(np.int64)
    epochs = history["epoch"].astype(np.int64)
    numeric = history.loc[:, ["train_loss", "validation_loss"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (epochs < 1).any():
        raise ValueError(f"history {stage} values must be finite and valid")
    for key in ("protocol", "fingerprint", "stage"):
        if set(history[key]) != {str(expected_metadata[key])}:
            raise ValueError(f"history {stage} metadata {key} is incompatible")
    for key in ("fold", "seed", "refit_epochs"):
        if set(history[key].astype(np.int64)) != {int(expected_metadata[key])}:
            raise ValueError(f"history {stage} metadata {key} is incompatible")
    expected_best = expected_metadata["best_epochs"]
    try:
        parsed_best = [json.loads(value) for value in history["best_epochs"]]
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"history {stage} best epochs metadata is invalid") from error
    if any(value != expected_best for value in parsed_best):
        raise ValueError(f"history {stage} best epochs metadata is incompatible")
    if set(history["phase"].astype(str)) != {"selection", "refit"}:
        raise ValueError(f"history {stage} must contain selection and refit phases")
    selection_folds = set(
        inner_fold.loc[history["phase"].astype(str) == "selection"]
    )
    if selection_folds != {0, 1, 2, 3}:
        raise ValueError(f"history {stage} inner-fold coverage is incompatible")
    if set(inner_fold.loc[history["phase"].astype(str) == "refit"]) != {-1}:
        raise ValueError(f"history {stage} refit marker is incompatible")
    for fold_number, best_epoch in enumerate(expected_best):
        selected_epochs = set(
            epochs.loc[
                (history["phase"].astype(str) == "selection")
                & (inner_fold == fold_number)
            ]
        )
        if int(best_epoch) not in selected_epochs:
            raise ValueError(f"history {stage} checkpoint best epoch is inconsistent")
    if int(expected_metadata["refit_epochs"]) not in set(
        epochs.loc[history["phase"].astype(str) == "refit"]
    ):
        raise ValueError(f"history {stage} checkpoint refit epoch is inconsistent")


def _validate_scalers(
    path: Path, stage: str, expected_metadata: dict[str, Any]
) -> None:
    expected = {
        "process_mean": (9,),
        "process_scale": (9,),
    }
    if stage != "P1":
        expected |= {
            "spectrum_mean": (3, 361),
            "spectrum_scale": (3, 361),
            "quality_mean": (7,),
            "quality_scale": (7,),
        }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected) | {"metadata_json"}:
                raise ValueError(f"scaler {stage} schema is incompatible")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "schema" in str(error):
            raise
        raise ValueError(f"scaler {stage} cannot be loaded") from error
    metadata_array = arrays.pop("metadata_json")
    if metadata_array.shape != () or metadata_array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"scaler {stage} metadata schema is incompatible")
    try:
        metadata = json.loads(str(metadata_array.item()))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"scaler {stage} metadata is invalid") from error
    metadata_integer_keys = ("fold", "seed", "refit_epochs")
    if (
        not isinstance(metadata, dict)
        or any(
            not isinstance(metadata.get(key), int)
            or isinstance(metadata.get(key), bool)
            for key in metadata_integer_keys
        )
        or not isinstance(metadata.get("best_epochs"), list)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in metadata.get("best_epochs", [])
        )
        or metadata != expected_metadata
    ):
        raise ValueError(f"scaler {stage} metadata is incompatible")
    for name, shape in expected.items():
        values = arrays[name]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"scaler {stage} {name} shape/finiteness is incompatible")
        if name.endswith("_scale") and np.any(values <= 0):
            raise ValueError(f"scaler {stage} scales must be positive")


def _validate_completed_artifacts(
    fold_dir: Path,
    marker: dict[str, Any],
    fingerprint: RunFingerprint,
    fold: int,
    seed: int,
) -> None:
    expected = _expected_artifact_paths(fold_dir)
    hashes = marker.get("artifacts")
    if not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise ValueError("completed artifact set is missing or incompatible")
    for relative, path in expected.items():
        if not path.is_file() or hashes.get(relative) != _sha256_file(path):
            raise ValueError(f"completed artifact hash mismatch: {relative}")
    checkpoints: dict[str, dict[str, Any]] = {}
    for stage in MODEL_SEQUENCE:
        checkpoint, history, scaler = _stage_paths(fold_dir, stage)
        payload = _validate_checkpoint(
            checkpoint, stage=stage, fingerprint=fingerprint, fold=fold, seed=seed
        )
        checkpoints[stage] = payload
        metadata = _artifact_metadata(
            fingerprint=fingerprint,
            fold=fold,
            seed=seed,
            stage=stage,
            best_epochs=payload["best_epochs"],
            refit_epochs=payload["refit_epochs"],
        )
        _validate_history(history, stage, metadata)
        _validate_scalers(scaler, stage, metadata)
    g1_state = checkpoints["G1"]["model_state"]
    for expert_stage, prefix in (("P1", "process_expert."), ("R1", "residual_expert.")):
        expert_state = checkpoints[expert_stage]["model_state"]
        embedded = {
            name[len(prefix):]: value
            for name, value in g1_state.items()
            if name.startswith(prefix)
        }
        if set(embedded) != set(expert_state) or any(
            not torch.equal(embedded[name], expected)
            for name, expected in expert_state.items()
        ):
            raise ValueError(f"G1 embedded {expert_stage} expert checkpoint is incompatible")


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
    output_root: str | Path | None = None,
) -> FoldArtifacts:
    """Train one outer fold in the exact P1→V1→F1→R1→G1 order."""
    train_index, test_index = _validate_protocol(
        config, bundle, cache, fold, seed, output_root
    )
    selected_device = _device(device)
    set_global_seed(seed)
    fingerprint = build_run_fingerprint(config, bundle, cache)
    fold_dir = _fold_dir(config, fold, seed)
    marker_path = fold_dir / "complete.json"
    if marker_path.exists():
        if not completed_fold_matches(marker_path, fingerprint.value, fold=fold, seed=seed):
            raise ValueError("incompatible completed-fold fingerprint, fold, seed, or model state")
        expected_frame = bundle.manifest.loc[
            :, ["sample_id", "group_id", "version", "sample_weight", "ra_mean"]
        ].iloc[test_index].reset_index(drop=True)
        return _load_completed_fold(
            fold_dir, fingerprint, fold, seed, expected_frame
        )
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

    # P1: Task 6 owns canonical feature validation, group cross-fitting and
    # inner-train process scalers; Task 7 supplies the configured backend/refit.
    p1 = _fit_p1_subset(
        config=config, frame=train_frame, features=train_features,
        inner_splits=inner_splits, backend=trainer, device=selected_device,
        seed=seed, outer_fold=fold, context_stage="P1", batch_size=batch_size,
    )
    p1_oof = p1.oof
    p1_selection = p1.selection
    p1_refit = TrainingResult(p1.model, p1.selection.refit_epochs, p1.history)
    outer_process_scaler = p1.scaler
    checkpoints["P1"], histories["P1"] = _persist_stage(
        fold_dir=fold_dir, fingerprint=fingerprint, fold=fold, seed=seed, stage="P1",
        completed=completed, selection=p1_selection, model=p1.model,
        history=p1.history,
        scalers={"process_mean": outer_process_scaler.mean, "process_scale": outer_process_scaler.scale},
        device=selected_device,
    )

    # V1: direct vibration comparator.
    v1 = _fit_signal_stage(
        stage="V1", config=config, frame=train_frame, cache=cache,
        features=train_features, targets=train_targets, inner_splits=inner_splits,
        model_factory=lambda _fold: VibrationOnlyModel(), backend=trainer,
        device=selected_device, seed=seed, outer_fold=fold, batch_size=batch_size,
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
        device=selected_device, seed=seed, outer_fold=fold, batch_size=batch_size,
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

    # R1: every selection fold rebuilds P1 inside its own training groups, so
    # its residual training targets cannot encode the held-out fold's labels.
    r1, confined_p1_cache = _fit_r1_nested_stage(
        config=config, frame=train_frame, cache=cache, features=train_features,
        outer_p1_oof=p1_oof, inner_splits=inner_splits, backend=trainer,
        device=selected_device, seed=seed, outer_fold=fold, batch_size=batch_size,
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

    # G1: for each fold k, rebuild P1 OOF/refit and ResidualExpert
    # selection/refit using only k's training groups, then freeze those experts.
    g1 = _fit_g1_nested_stage(
        config=config, frame=train_frame, cache=cache, features=train_features,
        inner_splits=inner_splits, outer_p1=p1, outer_r1=r1,
        p1_cache=confined_p1_cache,
        backend=trainer, device=selected_device, seed=seed, outer_fold=fold,
        batch_size=batch_size,
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
    marker["artifacts"] = {
        relative: _sha256_file(path)
        for relative, path in _expected_artifact_paths(fold_dir).items()
    }
    # Treat completion as a commit boundary: validate the exact persisted
    # candidate with the same deep checks used by resume before publishing the
    # atomic marker that makes the fold reusable.
    _validate_completed_artifacts(fold_dir, marker, fingerprint, fold, seed)
    _load_and_validate_persisted_oof(
        fold_dir,
        marker,
        fold,
        seed,
        outer_label_frame.loc[
            :, ["sample_id", "group_id", "version", "sample_weight", "ra_mean"]
        ],
    )
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
    output_root: str | Path | None = None,
) -> FoldArtifacts:
    """Run every registered outer fold and atomically persist combined OOF rows."""
    canonical_folds = _validate_outer_fold_definitions(bundle.folds)
    fold_values = sorted(set(canonical_folds))
    artifacts = [
        run_phase_a_fold(
            config, bundle, cache, fold, 20260723, device,
            backend=backend, batch_size=batch_size,
            output_root=output_root,
        )
        for fold in fold_values
    ]
    predictions = pd.concat([artifact.predictions for artifact in artifacts], ignore_index=True)
    for fold in fold_values:
        assigned = bundle.folds.loc[
            np.asarray(canonical_folds, dtype=np.int64) == fold,
            "sample_id",
        ].astype(str)
        expected = bundle.manifest.loc[
            bundle.manifest["sample_id"].astype(str).isin(set(assigned)),
            ["sample_id", "group_id", "version", "sample_weight"],
        ]
        _validate_oof_frame(
            predictions.loc[predictions["fold"].astype(int) == fold].reset_index(drop=True),
            expected.reset_index(drop=True), fold, 20260723,
        )
    if len(predictions) != len(bundle.manifest) * len(MODEL_SEQUENCE):
        raise ValueError("combined Phase A OOF Cartesian coverage is incompatible")
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
    "ComponentFoldData",
    "ComponentRefitData",
    "EpochSelection",
    "FoldArtifacts",
    "MODEL_SEQUENCE",
    "PHASE_A_PROTOCOL",
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
    "validate_phase_a_output_root",
]
