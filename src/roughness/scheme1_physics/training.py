from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from roughness.scheme1.training import (
    compute_run_fingerprint,
    set_global_seed,
    train_model,
)

from .config import Scheme1PhysicsConfig
from .data import build_physics_loaders, prepare_physics_fold
from .models import (
    PhysicsResidualRegressor,
    assert_frozen_state_unchanged,
    freeze_residual_for_gate,
    load_ordinary_into_gated,
)


_MODEL_MAP = {
    "PW0": ("word", "baseline"),
    "PW1": ("word", "ordinary"),
    "PW2": ("word", "gated"),
    "PE0": ("exact", "baseline"),
    "PE1": ("exact", "ordinary"),
    "PE2": ("exact", "gated"),
}


def _model_entry(model_name: str) -> tuple[str, str]:
    try:
        return _MODEL_MAP[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown physics model: {model_name}") from error


def physics_model_formula(model_name: str) -> str:
    return _model_entry(model_name)[0]


def physics_model_mode(model_name: str) -> str:
    return _model_entry(model_name)[1]


def ordinary_parent_model(model_name: str) -> str:
    parents = {"PW2": "PW1", "PE2": "PE1"}
    try:
        return parents[model_name]
    except KeyError as error:
        raise ValueError(f"{model_name} is not a gated model") from error


def write_physics_baseline_oof(
    config: Scheme1PhysicsConfig,
    force: bool = False,
    folds=None,
    seeds=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = config.output_dir / "physics"
    selected_folds = tuple(range(5) if folds is None else map(int, folds))
    selected_seeds = tuple(
        config.source.seeds if seeds is None else map(int, seeds)
    )
    full_request = set(selected_folds) == set(range(5)) and set(
        selected_seeds
    ) == set(config.source.seeds)
    suffix = "" if full_request else "_partial"
    prediction_path = output_dir / f"oof_predictions{suffix}.csv"
    selection_path = output_dir / f"selections{suffix}.csv"
    audit_path = output_dir / f"candidate_audit{suffix}.csv"
    paths = (prediction_path, selection_path, audit_path)
    if not force and all(path.is_file() for path in paths):
        return tuple(pd.read_csv(path) for path in paths)

    prediction_parts = []
    selections = []
    audit_parts = []
    for formula, model_name in (("word", "PW0"), ("exact", "PE0")):
        for outer_fold in selected_folds:
            for seed in selected_seeds:
                prepared = prepare_physics_fold(
                    config,
                    formula=formula,
                    outer_fold=outer_fold,
                    seed=seed,
                )
                test = prepared.test_raw
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "sample_id": test["sample_id"].astype(str),
                            "group_id": test["group_id"].astype(str),
                            "model": model_name,
                            "formula": formula,
                            "radius_mm": prepared.selection.radius_mm,
                            "fold": int(outer_fold),
                            "seed": int(seed),
                            "y_true": test["ra_mean"].astype(float),
                            "y_pred": test["base_ra"].astype(float),
                            "base_ra": test["base_ra"].astype(float),
                            "sample_weight": test["sample_weight"].astype(
                                float
                            ),
                        }
                    )
                )
                selections.append(
                    {
                        "model": model_name,
                        "formula": formula,
                        "outer_fold": int(outer_fold),
                        "seed": int(seed),
                        "radius_mm": prepared.selection.radius_mm,
                        "inner_weighted_mae": (
                            prepared.selection.inner_weighted_mae
                        ),
                        "source_sample_ids": "|".join(
                            prepared.selection.source_sample_ids
                        ),
                    }
                )
                current_audit = prepared.selection_audit.copy()
                current_audit.insert(0, "model", model_name)
                audit_parts.append(current_audit)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    selection_frame = pd.DataFrame.from_records(selections)
    audits = pd.concat(audit_parts, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    selection_frame.to_csv(selection_path, index=False)
    audits.to_csv(audit_path, index=False)
    return predictions, selection_frame, audits


def physics_forward(
    model: PhysicsResidualRegressor, batch: dict
) -> torch.Tensor:
    return model(
        signal=batch["signal"],
        window_mask=batch["window_mask"],
        process=batch["process"],
        physics=batch["physics"],
        base_ra=batch["base_ra"],
    ).prediction


def physics_run_fingerprint(
    config: Scheme1PhysicsConfig,
    model_name: str,
    outer_fold: int,
    seed: int,
    radius_mm: float,
    requested_max_epochs: int,
    batch_size: int,
    validation_fraction: float,
    parent_checkpoint: str | Path | None = None,
) -> str:
    inputs = [
        config.source.manifest_path,
        config.source.folds_path,
        config.source.output_dir / "window_index.csv",
        config.source.output_dir
        / "folds"
        / f"fold_{outer_fold}_channel_stats.json",
    ]
    if parent_checkpoint is not None:
        inputs.append(Path(parent_checkpoint))
    payload = {
        "model": model_name,
        "formula": physics_model_formula(model_name),
        "outer_fold": int(outer_fold),
        "seed": int(seed),
        "radius_mm": float(radius_mm),
        "re_candidates_mm": list(config.source.re_candidates_mm),
        "inner_splits": config.source.inner_splits,
        "requested_max_epochs": int(requested_max_epochs),
        "batch_size": int(batch_size),
        "validation_fraction": float(validation_fraction),
        "patience": config.source.patience,
        "learning_rate": config.source.learning_rate,
        "weight_decay": config.source.weight_decay,
        "dropout": config.residual_dropout,
    }
    return compute_run_fingerprint(payload, inputs)


@dataclass(frozen=True)
class PhysicsFoldRunResult:
    model_name: str
    formula: str
    outer_fold: int
    seed: int
    radius_mm: float
    best_epoch: int
    best_validation_mae: float
    checkpoint_path: Path
    log_path: Path
    oof_path: Path


def _predict_physics_loader(
    model: PhysicsResidualRegressor,
    loader,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = {
                key: value.to(device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in raw_batch.items()
            }
            output = model(
                signal=batch["signal"],
                window_mask=batch["window_mask"],
                process=batch["process"],
                physics=batch["physics"],
                base_ra=batch["base_ra"],
            )
            for index, sample_id in enumerate(batch["segment_id"]):
                row = {
                    "sample_id": str(sample_id),
                    "group_id": str(batch["group_id"][index]),
                    "y_true": float(batch["target"][index].cpu()),
                    "y_pred": float(output.prediction[index].cpu()),
                    "base_ra": float(batch["base_ra"][index].cpu()),
                    "residual": float(output.residual[index].cpu()),
                    "sample_weight": float(
                        batch["sample_weight"][index].cpu()
                    ),
                }
                if output.gate is not None:
                    row["gate"] = float(output.gate[index].cpu())
                rows.append(row)
    return pd.DataFrame.from_records(rows)


def _extend_checkpoint(path: Path, metadata: dict) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.update(metadata)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_metadata(
    run_dir: Path,
    *,
    model_name: str,
    formula: str,
    outer_fold: int,
    seed: int,
    radius_mm: float,
    run_fingerprint: str,
    parent_checkpoint_fingerprint: str | None = None,
) -> None:
    payload = {
        "status": "complete",
        "model": model_name,
        "formula": formula,
        "outer_fold": int(outer_fold),
        "seed": int(seed),
        "radius_mm": float(radius_mm),
        "run_fingerprint": run_fingerprint,
        "parent_checkpoint_fingerprint": parent_checkpoint_fingerprint,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def completed_physics_run_matches(
    run_dir: str | Path,
    expected_fingerprint: str,
    expected_parent_checkpoint_fingerprint: str | None = None,
) -> bool:
    directory = Path(run_dir)
    metadata_path = directory / "run_metadata.json"
    oof_path = directory / "oof_predictions.csv"
    if not metadata_path.is_file() or not oof_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        oof = pd.read_csv(oof_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if metadata.get("status") != "complete":
        return False
    if metadata.get("run_fingerprint") != expected_fingerprint:
        return False
    if (
        expected_parent_checkpoint_fingerprint is not None
        and metadata.get("parent_checkpoint_fingerprint")
        != expected_parent_checkpoint_fingerprint
    ):
        return False
    if oof.empty:
        return False
    expected_columns = {"model", "formula", "fold", "seed", "radius_mm"}
    if not expected_columns <= set(oof):
        return False
    scalar_checks = {
        "model": str(metadata.get("model")),
        "formula": str(metadata.get("formula")),
        "fold": int(metadata.get("outer_fold")),
        "seed": int(metadata.get("seed")),
    }
    for column, expected in scalar_checks.items():
        if not oof[column].eq(expected).all():
            return False
    return bool(
        np.isclose(
            oof["radius_mm"].to_numpy(dtype=float),
            float(metadata.get("radius_mm")),
        ).all()
    )


def train_physics_fold(
    model_name: str,
    outer_fold: int,
    seed: int,
    config: Scheme1PhysicsConfig,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
) -> PhysicsFoldRunResult:
    if physics_model_mode(model_name) != "ordinary":
        raise ValueError("train_physics_fold supports PW1 and PE1")
    set_global_seed(seed)
    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    formula = physics_model_formula(model_name)
    prepared = prepare_physics_fold(
        config, formula=formula, outer_fold=outer_fold, seed=seed
    )
    actual_batch_size = int(batch_size or config.batch_size)
    actual_max_epochs = int(max_epochs or config.source.max_epochs)
    loaders = build_physics_loaders(
        prepared,
        config,
        batch_size=actual_batch_size,
        validation_fraction=config.validation_fraction,
        seed=seed,
        device=selected_device,
    )
    model = PhysicsResidualRegressor(
        "ordinary", dropout=config.residual_dropout
    )
    run_dir = (
        config.output_dir
        / "neural"
        / model_name
        / f"fold_{outer_fold}"
        / f"seed_{seed}"
    )
    checkpoint_path = run_dir / "best.pt"
    log_path = run_dir / "training_log.csv"
    fingerprint = physics_run_fingerprint(
        config,
        model_name,
        outer_fold,
        seed,
        prepared.selection.radius_mm,
        actual_max_epochs,
        actual_batch_size,
        config.validation_fraction,
    )
    trained = train_model(
        model,
        loaders.train_loader,
        loaders.validation_loader,
        physics_forward,
        checkpoint_path,
        log_path,
        max_epochs=actual_max_epochs,
        patience=config.source.patience,
        learning_rate=config.source.learning_rate,
        weight_decay=config.source.weight_decay,
        device=selected_device,
    )
    _extend_checkpoint(
        trained.checkpoint_path,
        {
            "model": model_name,
            "formula": formula,
            "outer_fold": int(outer_fold),
            "seed": int(seed),
            "radius_mm": float(prepared.selection.radius_mm),
            "run_fingerprint": fingerprint,
        },
    )
    oof = _predict_physics_loader(model, loaders.test_loader, selected_device)
    oof.insert(2, "model", model_name)
    oof.insert(3, "formula", formula)
    oof.insert(4, "radius_mm", prepared.selection.radius_mm)
    oof.insert(5, "fold", int(outer_fold))
    oof.insert(6, "seed", int(seed))
    oof_path = run_dir / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    _write_run_metadata(
        run_dir,
        model_name=model_name,
        formula=formula,
        outer_fold=outer_fold,
        seed=seed,
        radius_mm=prepared.selection.radius_mm,
        run_fingerprint=fingerprint,
    )
    return PhysicsFoldRunResult(
        model_name=model_name,
        formula=formula,
        outer_fold=int(outer_fold),
        seed=int(seed),
        radius_mm=float(prepared.selection.radius_mm),
        best_epoch=trained.best_epoch,
        best_validation_mae=trained.best_validation_mae,
        checkpoint_path=trained.checkpoint_path,
        log_path=trained.log_path,
        oof_path=oof_path,
    )


def train_gated_physics_fold(
    model_name: str,
    outer_fold: int,
    seed: int,
    config: Scheme1PhysicsConfig,
    parent_checkpoint: str | Path,
    device: torch.device | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
) -> PhysicsFoldRunResult:
    if physics_model_mode(model_name) != "gated":
        raise ValueError("train_gated_physics_fold supports PW2 and PE2")
    set_global_seed(seed)
    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    formula = physics_model_formula(model_name)
    prepared = prepare_physics_fold(
        config, formula=formula, outer_fold=outer_fold, seed=seed
    )
    parent_path = Path(parent_checkpoint)
    parent = torch.load(
        parent_path, map_location="cpu", weights_only=False
    )
    expected = {
        "model": ordinary_parent_model(model_name),
        "formula": formula,
        "outer_fold": int(outer_fold),
        "seed": int(seed),
        "radius_mm": float(prepared.selection.radius_mm),
    }
    for key, value in expected.items():
        actual = parent.get(key)
        matches = (
            np.isclose(float(actual), value)
            if key == "radius_mm" and actual is not None
            else actual == value
        )
        if not matches:
            label = "formula mismatch" if key == "formula" else f"{key} mismatch"
            raise ValueError(
                f"Parent checkpoint {label}: expected {value}, got {actual}"
            )
    if "model_state" not in parent:
        raise ValueError("Parent checkpoint has no model_state")

    actual_batch_size = int(batch_size or config.batch_size)
    actual_max_epochs = int(max_epochs or config.source.max_epochs)
    loaders = build_physics_loaders(
        prepared,
        config,
        batch_size=actual_batch_size,
        validation_fraction=config.validation_fraction,
        seed=seed,
        device=selected_device,
    )
    model = PhysicsResidualRegressor(
        "gated", dropout=config.residual_dropout
    )
    load_ordinary_into_gated(model, parent["model_state"])
    frozen_snapshot = freeze_residual_for_gate(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if {id(parameter) for parameter in trainable} != {
        id(model.gate_layer.weight),
        id(model.gate_layer.bias),
    }:
        raise AssertionError("only gate parameters may be trainable")

    run_dir = (
        config.output_dir
        / "neural"
        / model_name
        / f"fold_{outer_fold}"
        / f"seed_{seed}"
    )
    checkpoint_path = run_dir / "best.pt"
    log_path = run_dir / "training_log.csv"
    parent_fingerprint = _file_sha256(parent_path)
    fingerprint = physics_run_fingerprint(
        config,
        model_name,
        outer_fold,
        seed,
        prepared.selection.radius_mm,
        actual_max_epochs,
        actual_batch_size,
        config.validation_fraction,
        parent_checkpoint=parent_path,
    )
    trained = train_model(
        model,
        loaders.train_loader,
        loaders.validation_loader,
        physics_forward,
        checkpoint_path,
        log_path,
        max_epochs=actual_max_epochs,
        patience=config.source.patience,
        learning_rate=config.source.learning_rate,
        weight_decay=config.source.weight_decay,
        device=selected_device,
    )
    assert_frozen_state_unchanged(model, frozen_snapshot)
    _extend_checkpoint(
        trained.checkpoint_path,
        {
            "model": model_name,
            "formula": formula,
            "outer_fold": int(outer_fold),
            "seed": int(seed),
            "radius_mm": float(prepared.selection.radius_mm),
            "run_fingerprint": fingerprint,
            "parent_checkpoint_fingerprint": parent_fingerprint,
        },
    )
    audit = {
        "frozen_parameter_count": int(
            sum(value.numel() for value in frozen_snapshot.values())
        ),
        "gate_parameter_count": int(
            sum(parameter.numel() for parameter in model.gate_layer.parameters())
        ),
        "parent_checkpoint_fingerprint": parent_fingerprint,
        "unchanged": True,
    }
    (run_dir / "frozen_parameter_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    oof = _predict_physics_loader(model, loaders.test_loader, selected_device)
    oof.insert(2, "model", model_name)
    oof.insert(3, "formula", formula)
    oof.insert(4, "radius_mm", prepared.selection.radius_mm)
    oof.insert(5, "fold", int(outer_fold))
    oof.insert(6, "seed", int(seed))
    oof_path = run_dir / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    _write_run_metadata(
        run_dir,
        model_name=model_name,
        formula=formula,
        outer_fold=outer_fold,
        seed=seed,
        radius_mm=prepared.selection.radius_mm,
        run_fingerprint=fingerprint,
        parent_checkpoint_fingerprint=parent_fingerprint,
    )
    return PhysicsFoldRunResult(
        model_name=model_name,
        formula=formula,
        outer_fold=int(outer_fold),
        seed=int(seed),
        radius_mm=float(prepared.selection.radius_mm),
        best_epoch=trained.best_epoch,
        best_validation_mae=trained.best_validation_mae,
        checkpoint_path=trained.checkpoint_path,
        log_path=trained.log_path,
        oof_path=oof_path,
    )
