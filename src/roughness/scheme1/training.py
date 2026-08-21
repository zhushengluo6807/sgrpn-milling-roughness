from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import Scheme1Config
from .dataset import SegmentBagDataset, collate_segment_bags
from .models import (
    CNNEncoder,
    CNNTCNEncoder,
    SegmentFusionRegressor,
    SegmentSignalRegressor,
)
from .signals import ChannelStats


def compute_run_fingerprint(
    payload: dict,
    input_paths: list[str | Path],
) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    for raw_path in sorted(Path(path).resolve() for path in input_paths):
        if not raw_path.is_file():
            raise ValueError(f"Fingerprint input is not a file: {raw_path}")
        digest.update(str(raw_path).encode("utf-8"))
        with raw_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def scheme1_run_fingerprint(
    config: Scheme1Config,
    model_name: str,
    outer_fold: int,
    seed: int,
    requested_max_epochs: int,
    batch_size: int,
    validation_fraction: float,
    fusion_encoder: str = "cnn",
) -> str:
    inputs = [
        config.manifest_path,
        config.folds_path,
        config.output_dir / "window_index.csv",
        config.output_dir
        / "folds"
        / f"fold_{outer_fold}_channel_stats.json",
    ]
    if model_name == "N4":
        inputs.append(
            config.output_dir / "classic" / "m0_baseline_provenance.csv"
        )
    payload = {
        "model": model_name,
        "outer_fold": int(outer_fold),
        "seed": int(seed),
        "requested_max_epochs": int(requested_max_epochs),
        "batch_size": int(batch_size),
        "validation_fraction": float(validation_fraction),
        "sample_rate_hz": config.sample_rate_hz,
        "window_samples": config.window_samples,
        "stride_samples": config.stride_samples,
        "patience": config.patience,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if model_name in {"N3", "N4"}:
        payload["fusion_encoder"] = fusion_encoder
    return compute_run_fingerprint(payload, inputs)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_scheme1_model(
    model_name: str,
    fusion_encoder: str = "cnn",
) -> nn.Module:
    if model_name == "N1":
        return SegmentSignalRegressor(CNNEncoder())
    if model_name == "N2":
        return SegmentSignalRegressor(CNNTCNEncoder())
    if model_name == "N3":
        if fusion_encoder not in {"cnn", "cnn_tcn"}:
            raise ValueError("fusion_encoder must be cnn or cnn_tcn")
        return SegmentFusionRegressor(
            CNNEncoder() if fusion_encoder == "cnn" else CNNTCNEncoder(),
            process_dim=3,
            physics_dim=0,
            mode="direct",
        )
    if model_name == "N4":
        if fusion_encoder not in {"cnn", "cnn_tcn"}:
            raise ValueError("fusion_encoder must be cnn or cnn_tcn")
        return SegmentFusionRegressor(
            CNNEncoder() if fusion_encoder == "cnn" else CNNTCNEncoder(),
            process_dim=3,
            physics_dim=0,
            mode="residual",
        )
    raise ValueError(f"Unsupported Scheme 1 model: {model_name}")


def scheme1_forward(model: nn.Module, batch: dict) -> torch.Tensor:
    if isinstance(model, SegmentSignalRegressor):
        return model(batch["signal"], batch["window_mask"]).prediction
    if isinstance(model, SegmentFusionRegressor):
        return model(
            batch["signal"],
            batch["window_mask"],
            process=batch["process"],
            physics=(
                batch["physics"] if model.physics_dim > 0 else None
            ),
            base_ra=batch["base_ra"] if model.mode != "direct" else None,
        ).prediction
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def weighted_mae_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    if not (predicted.shape == target.shape == sample_weight.shape):
        raise ValueError("prediction, target and weight shapes must match")
    denominator = sample_weight.sum()
    if denominator <= 0:
        raise ValueError("sample weights must have positive sum")
    return (torch.abs(predicted - target) * sample_weight).sum() / denominator


def make_group_train_validation_split(
    frame: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "group_id" not in frame:
        raise ValueError("frame must contain group_id")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = np.asarray(sorted(frame["group_id"].astype(str).unique()))
    if len(groups) < 2:
        raise ValueError("At least two groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation_count = max(1, int(round(len(groups) * validation_fraction)))
    validation_count = min(validation_count, len(groups) - 1)
    validation_groups = set(groups[:validation_count])
    is_validation = frame["group_id"].astype(str).isin(validation_groups).to_numpy()
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = int(patience)
        self.best_metric = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self._best_state: dict | None = None

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def update(self, metric: float, model: nn.Module, epoch: int) -> bool:
        if not np.isfinite(metric):
            raise ValueError("Early-stopping metric must be finite")
        if metric < self.best_metric:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self._best_state = deepcopy(model.state_dict())
            return True
        self.bad_epochs += 1
        return False

    def restore(self, model: nn.Module) -> None:
        if self._best_state is None:
            raise ValueError("No best state has been recorded")
        model.load_state_dict(self._best_state)

    @property
    def best_state(self) -> dict:
        if self._best_state is None:
            raise ValueError("No best state has been recorded")
        return self._best_state


@dataclass(frozen=True)
class TrainResult:
    best_epoch: int
    best_validation_mae: float
    epochs_run: int
    checkpoint_path: Path
    log_path: Path


@dataclass(frozen=True)
class FoldRunResult:
    model_name: str
    outer_fold: int
    seed: int
    best_epoch: int
    best_validation_mae: float
    checkpoint_path: Path
    log_path: Path
    oof_path: Path


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _evaluate(
    model: nn.Module,
    loader,
    forward_fn: Callable[[nn.Module, dict], torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    weighted_error = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            predicted = forward_fn(model, batch)
            error = torch.abs(predicted - batch["target"])
            weighted_error += float((error * batch["sample_weight"]).sum())
            weight_sum += float(batch["sample_weight"].sum())
    if weight_sum <= 0:
        raise ValueError("Validation loader has no positive sample weight")
    return weighted_error / weight_sum


def train_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    forward_fn: Callable[[nn.Module, dict], torch.Tensor],
    checkpoint_path: str | Path,
    log_path: str | Path,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> TrainResult:
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    stopper = EarlyStopper(patience)
    history = []
    for epoch in range(max_epochs):
        model.train()
        weighted_error = 0.0
        weight_sum = 0.0
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            predicted = forward_fn(model, batch)
            loss = weighted_mae_loss(
                predicted,
                batch["target"],
                batch["sample_weight"],
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                weighted_error += float(
                    (
                        torch.abs(predicted - batch["target"])
                        * batch["sample_weight"]
                    ).sum()
                )
                weight_sum += float(batch["sample_weight"].sum())
        validation_mae = _evaluate(
            model, validation_loader, forward_fn, device
        )
        is_best = stopper.update(validation_mae, model, epoch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": weighted_error / weight_sum,
                "validation_mae": validation_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": is_best,
            }
        )
        if stopper.should_stop:
            break

    stopper.restore(model)
    checkpoint = Path(checkpoint_path)
    log = Path(log_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": stopper.best_state,
            "best_epoch": stopper.best_epoch,
            "best_validation_mae": stopper.best_metric,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(log, index=False)
    return TrainResult(
        best_epoch=stopper.best_epoch,
        best_validation_mae=stopper.best_metric,
        epochs_run=len(history),
        checkpoint_path=checkpoint,
        log_path=log,
    )


def _load_channel_stats(path: Path) -> ChannelStats:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ChannelStats(
        mean=np.asarray(raw["mean"], dtype=np.float64),
        std=np.asarray(raw["std"], dtype=np.float64),
        sample_count=int(raw["sample_count"]),
        source_segment_ids=tuple(map(str, raw["source_segment_ids"])),
    )


def _scale_process(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    columns = ["n_rpm", "fz_mm_per_tooth", "ap_mm"]
    mean = train[columns].mean()
    scale = train[columns].std(ddof=0)
    scale = scale.mask(scale <= 0, 1.0)
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled.loc[:, columns] = (train[columns] - mean) / scale
    test_scaled.loc[:, columns] = (test[columns] - mean) / scale
    return (
        train_scaled,
        test_scaled,
        {
            "columns": columns,
            "mean": mean.to_dict(),
            "scale": scale.to_dict(),
            "source_segment_ids": train["sample_id"].astype(str).tolist(),
        },
    )


def train_one_fold(
    model_name: str,
    outer_fold: int,
    seed: int,
    config: Scheme1Config,
    batch_size: int = 4,
    validation_fraction: float = 0.2,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    fusion_encoder: str = "cnn",
) -> FoldRunResult:
    if model_name not in {"N1", "N2", "N3", "N4"}:
        raise ValueError("train_one_fold supports N1, N2, N3 and N4")
    set_global_seed(seed)
    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    manifest = pd.read_csv(config.manifest_path)
    folds = pd.read_csv(config.folds_path)
    data = manifest.merge(
        folds[["sample_id", "fold"]],
        on="sample_id",
        validate="one_to_one",
    )
    train_raw = data[data["fold"].astype(int) != outer_fold].reset_index(
        drop=True
    )
    test_raw = data[data["fold"].astype(int) == outer_fold].reset_index(
        drop=True
    )
    if train_raw.empty or test_raw.empty:
        raise ValueError(f"Outer fold {outer_fold} has empty train or test data")
    train_frame, test_frame, process_stats = _scale_process(
        train_raw, test_raw
    )
    base_ra_column = None
    if model_name == "N4":
        baseline = pd.read_csv(
            config.output_dir / "classic" / "m0_baseline_provenance.csv"
        )
        current = baseline[baseline["outer_fold"].astype(int) == outer_fold]
        train_base = current[current["role"] == "train"][
            ["sample_id", "base_ra"]
        ]
        test_base = current[current["role"] == "test"][
            ["sample_id", "base_ra"]
        ]
        train_frame = train_frame.merge(
            train_base, on="sample_id", validate="one_to_one"
        )
        test_frame = test_frame.merge(
            test_base, on="sample_id", validate="one_to_one"
        )
        base_ra_column = "base_ra"

    window_index = pd.read_csv(config.output_dir / "window_index.csv")
    stats = _load_channel_stats(
        config.output_dir
        / "folds"
        / f"fold_{outer_fold}_channel_stats.json"
    )
    train_dataset = SegmentBagDataset(
        train_frame,
        window_index,
        stats,
        base_ra_column=base_ra_column,
        cache_signals=True,
    )
    test_dataset = SegmentBagDataset(
        test_frame,
        window_index,
        stats,
        base_ra_column=base_ra_column,
        cache_signals=True,
    )
    train_indices, validation_indices = make_group_train_validation_split(
        train_frame,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_segment_bags,
        "num_workers": 0,
        "pin_memory": selected_device.type == "cuda",
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices.tolist()),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        Subset(train_dataset, validation_indices.tolist()),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    run_dir = (
        config.output_dir
        / "neural"
        / model_name
        / f"fold_{outer_fold}"
        / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_max_epochs = max_epochs or config.max_epochs
    run_fingerprint = scheme1_run_fingerprint(
        config,
        model_name,
        outer_fold,
        seed,
        effective_max_epochs,
        batch_size,
        validation_fraction,
        fusion_encoder,
    )
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "status": "running",
                "model": model_name,
                "outer_fold": int(outer_fold),
                "seed": int(seed),
                "requested_max_epochs": int(effective_max_epochs),
                "batch_size": int(batch_size),
                "validation_fraction": float(validation_fraction),
                "fusion_encoder": fusion_encoder,
                "run_fingerprint": run_fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "process_scaler.json").write_text(
        json.dumps(process_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model = build_scheme1_model(model_name, fusion_encoder=fusion_encoder)
    trained = train_model(
        model,
        train_loader,
        validation_loader,
        forward_fn=scheme1_forward,
        checkpoint_path=run_dir / "best.pt",
        log_path=run_dir / "history.csv",
        max_epochs=effective_max_epochs,
        patience=config.patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=selected_device,
    )

    model.eval()
    prediction_rows = []
    with torch.no_grad():
        for raw_batch in test_loader:
            batch = _move_batch(raw_batch, selected_device)
            predicted = scheme1_forward(model, batch).detach().cpu().numpy()
            targets = batch["target"].detach().cpu().numpy()
            weights = batch["sample_weight"].detach().cpu().numpy()
            prediction_rows.extend(
                {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "model": model_name,
                    "fold": int(outer_fold),
                    "seed": int(seed),
                    "y_true": float(target),
                    "y_pred": float(prediction),
                    "sample_weight": float(weight),
                }
                for sample_id, group_id, target, prediction, weight in zip(
                    raw_batch["segment_id"],
                    raw_batch["group_id"],
                    targets,
                    predicted,
                    weights,
                    strict=True,
                )
            )
    oof_path = run_dir / "oof_predictions.csv"
    pd.DataFrame.from_records(prediction_rows).to_csv(oof_path, index=False)
    if len(prediction_rows) != len(test_frame):
        raise AssertionError("Outer-test prediction count is incomplete")
    metadata_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "model": model_name,
                "outer_fold": int(outer_fold),
                "seed": int(seed),
                "requested_max_epochs": int(effective_max_epochs),
                "batch_size": int(batch_size),
                "validation_fraction": float(validation_fraction),
                "fusion_encoder": fusion_encoder,
                "run_fingerprint": run_fingerprint,
                "best_epoch": int(trained.best_epoch),
                "best_validation_mae": float(trained.best_validation_mae),
                "outer_test_rows": len(prediction_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return FoldRunResult(
        model_name=model_name,
        outer_fold=int(outer_fold),
        seed=int(seed),
        best_epoch=trained.best_epoch,
        best_validation_mae=trained.best_validation_mae,
        checkpoint_path=trained.checkpoint_path,
        log_path=trained.log_path,
        oof_path=oof_path,
    )
