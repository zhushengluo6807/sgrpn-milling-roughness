"""Post-audit group split-conformal probability analysis for SGRPN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import torch
import yaml

from roughness.scheme1.crossfit import make_group_inner_splits

from . import phase_b_training
from .config import PhaseBConfig, SGRPNConfig
from .crossfit import ProcessScaler, fit_process_scaler
from .data import DataBundle, build_process_features, outer_indices, repeat_measure_batch
from .models import GlobalScale, SelectiveGatedModel, VarianceHead
from .order_spectrum import (
    OrderSpectrumCache,
    QualityScaler,
    SpectrumScaler,
    fit_quality_scaler,
    fit_spectrum_scaler,
)
from .phase_b_training import CalibrationArtifacts
from .probability import finite_sample_group_quantile
from .training import MeanPathArtifacts, TrainingBackend


CORRECTIVE_PROTOCOL = "sgrpn-phase-b-group-split-conformal-corrective-v1"
_PROJECT_CORRECTIVE_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "sgrpn"
    / "phase_b_split_conformal_corrective"
).resolve()
_CORRECTIVE_KEYS = {
    "phase_b_config_path",
    "phase_b_config_file_sha256",
    "phase_b_run_manifest_path",
    "phase_b_run_manifest_sha256",
    "phase_b_immutable_after_path",
    "phase_b_immutable_after_sha256",
    "output_dir",
    "split_seed",
    "calibration_fold",
}


@dataclass(frozen=True)
class CorrectiveConfig:
    phase_b_config_path: Path
    phase_b_config_file_sha256: str
    phase_b_run_manifest_path: Path
    phase_b_run_manifest_sha256: str
    phase_b_immutable_after_path: Path
    phase_b_immutable_after_sha256: str
    output_dir: Path
    split_seed: int
    calibration_fold: int


@dataclass(frozen=True)
class CorrectiveSplit:
    """One fixed proper-training/calibration partition of an outer train set."""

    proper_train_sample_ids: tuple[str, ...]
    calibration_sample_ids: tuple[str, ...]
    proper_train_group_ids: tuple[str, ...]
    calibration_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class CorrectiveFittedPredictor:
    split: CorrectiveSplit
    mean_path: MeanPathArtifacts
    scale_models: dict[str, object]
    scale_histories: dict[str, pd.DataFrame]
    calibration: dict[str, CalibrationArtifacts]


@dataclass(frozen=True)
class CorrectiveFoldResult:
    predictions: pd.DataFrame
    mean_predictions: pd.DataFrame
    fitted: CorrectiveFittedPredictor


@dataclass(frozen=True)
class PersistedCorrectiveFold:
    predictions: pd.DataFrame
    mean_predictions: pd.DataFrame
    calibration_predictions: pd.DataFrame
    group_scores: pd.DataFrame
    quantiles: dict[str, object]
    marker: dict[str, object]


@dataclass(frozen=True)
class CorrectiveRunResult:
    probability_predictions: pd.DataFrame
    mean_predictions: pd.DataFrame
    calibration_predictions: pd.DataFrame
    group_scores: pd.DataFrame
    quantiles: pd.DataFrame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"required corrective input is missing or unreadable: {path}") from error
    return digest.hexdigest()


def _resolved(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_corrective_config(
    path: str | Path, *, output_root: str | Path | None = None
) -> CorrectiveConfig:
    """Load the frozen post-audit protocol without creating its output root."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("corrective config must be a mapping")
    missing = sorted(_CORRECTIVE_KEYS - set(raw))
    extra = sorted(set(raw) - _CORRECTIVE_KEYS)
    if missing or extra:
        raise ValueError(f"corrective config keys are incompatible: missing={missing}, extra={extra}")

    base = config_path.parent
    input_specs = (
        ("phase_b_config_path", "phase_b_config_file_sha256"),
        ("phase_b_run_manifest_path", "phase_b_run_manifest_sha256"),
        ("phase_b_immutable_after_path", "phase_b_immutable_after_sha256"),
    )
    resolved_inputs: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for path_field, hash_field in input_specs:
        source = _resolved(base, raw[path_field], path_field)
        registered = raw[hash_field]
        if (
            not isinstance(registered, str)
            or len(registered) != 64
            or any(character not in "0123456789abcdef" for character in registered)
        ):
            raise ValueError(f"{hash_field} must be 64 lowercase hexadecimal characters")
        if _sha256_file(source) != registered:
            raise ValueError(f"{hash_field} does not match the frozen prior artifact")
        resolved_inputs[path_field] = source
        hashes[hash_field] = registered

    configured_output = _resolved(base, raw["output_dir"], "output_dir")
    allowed_output = (
        _PROJECT_CORRECTIVE_OUTPUT
        if output_root is None
        else Path(output_root).resolve()
    )
    if configured_output != allowed_output:
        raise ValueError("corrective config requires the exact independent output root")
    if configured_output == resolved_inputs["phase_b_run_manifest_path"].parent:
        raise ValueError("corrective output root must not be the frozen Phase B root")
    if type(raw["split_seed"]) is not int or raw["split_seed"] != 20260723:
        raise ValueError("corrective split_seed must be exactly 20260723")
    if type(raw["calibration_fold"]) is not int or raw["calibration_fold"] != 0:
        raise ValueError("corrective calibration_fold must be exactly 0")

    return CorrectiveConfig(
        **resolved_inputs,
        **hashes,
        output_dir=configured_output,
        split_seed=20260723,
        calibration_fold=0,
    )


def make_corrective_split(
    outer_train: pd.DataFrame,
    *,
    split_seed: int,
    calibration_fold: int,
) -> CorrectiveSplit:
    """Reserve one whole group fold for calibration using a fixed split seed."""
    if not isinstance(outer_train, pd.DataFrame) or outer_train.empty:
        raise ValueError("outer_train must be a non-empty DataFrame")
    required = {"sample_id", "group_id"}
    missing = sorted(required - set(outer_train.columns))
    if missing:
        raise ValueError(f"outer_train missing columns: {missing}")
    if type(split_seed) is not int:
        raise ValueError("split_seed must be an integer")
    if type(calibration_fold) is not int or calibration_fold not in range(4):
        raise ValueError("calibration_fold must be one of 0,1,2,3")

    frame = outer_train.reset_index(drop=True).copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    if (
        frame["sample_id"].duplicated().any()
        or (frame["sample_id"].str.len() == 0).any()
        or (frame["group_id"].str.len() == 0).any()
    ):
        raise ValueError("outer_train identifiers must be non-empty and sample-unique")

    proper_rows, calibration_rows = make_group_inner_splits(
        frame, 4, split_seed
    )[calibration_fold]
    proper = frame.iloc[proper_rows]
    calibration = frame.iloc[calibration_rows]
    proper_groups = tuple(sorted(proper["group_id"].unique()))
    calibration_groups = tuple(sorted(calibration["group_id"].unique()))
    if len(calibration_groups) < 19:
        raise ValueError("corrective split requires at least 19 calibration groups")
    if set(proper_groups) & set(calibration_groups):
        raise ValueError("proper-training and calibration groups overlap")
    if set(proper["sample_id"]) | set(calibration["sample_id"]) != set(
        frame["sample_id"]
    ):
        raise ValueError("corrective split does not cover every outer-train sample")

    return CorrectiveSplit(
        proper_train_sample_ids=tuple(proper["sample_id"]),
        calibration_sample_ids=tuple(calibration["sample_id"]),
        proper_train_group_ids=proper_groups,
        calibration_group_ids=calibration_groups,
    )


def build_fixed_calibration(
    phase_b_config: object,
    calibration_frame: pd.DataFrame,
    cache: object,
    mean_path: object,
    scale_models: dict[str, object],
    *,
    fold: int,
    seed: int,
    device: str | torch.device,
    batch_size: int = 8,
) -> dict[str, CalibrationArtifacts]:
    """Score a reserved group set with one already-fitted mean/scale predictor."""
    if not isinstance(calibration_frame, pd.DataFrame) or calibration_frame.empty:
        raise ValueError("calibration_frame must be a non-empty DataFrame")
    if set(scale_models) != set(phase_b_training.SCALE_MODELS):
        raise ValueError("corrective calibration requires both registered scale models")
    alphas = tuple(map(float, getattr(phase_b_config, "alphas", ())))
    if alphas != phase_b_training.PHASE_B_ALPHAS:
        raise ValueError("corrective calibration requires the registered alphas")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    batches = phase_b_training._calibration_batches(
        calibration_frame,
        cache,
        mean_path,
        batch_size=batch_size,
    )
    expected_ids = tuple(calibration_frame["sample_id"].astype(str))
    results: dict[str, CalibrationArtifacts] = {}
    for scale_name in phase_b_training.SCALE_MODELS:
        predictions = phase_b_training._calibration_predictions(
            batches,
            scale_model=scale_models[scale_name],
            mean_model=mean_path.model,
            fold=fold,
            seed=seed,
            inner_fold=0,
            scale_name=scale_name,
            device=torch.device(device),
        )
        phase_b_training._validate_calibration_oof(predictions, expected_ids)
        group_scores = phase_b_training._calibration_group_scores(predictions)
        quantiles = {
            alpha: finite_sample_group_quantile(group_scores, alpha=alpha)
            for alpha in alphas
        }
        definitions = pd.DataFrame(
            [
                {
                    "record_type": "fixed_split_predictor",
                    "fold": fold,
                    "seed": seed,
                    "scale_model": scale_name,
                    "calibration_sample_ids": expected_ids,
                    "calibration_group_ids": tuple(
                        calibration_frame["group_id"].astype(str)
                    ),
                }
            ]
        )
        results[scale_name] = CalibrationArtifacts(
            predictions=predictions,
            group_scores=group_scores,
            quantiles=quantiles,
            inner_fold_definitions=definitions,
        )
    return results


def fit_corrective_predictor(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    phase_a_config: SGRPNConfig,
    bundle: DataBundle,
    outer_train: pd.DataFrame,
    cache: OrderSpectrumCache,
    *,
    fold: int,
    seed: int,
    device: str | torch.device,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
) -> CorrectiveFittedPredictor:
    """Fit once on proper train, freeze, then calibrate on reserved groups."""
    split = make_corrective_split(
        outer_train,
        split_seed=corrective_config.split_seed,
        calibration_fold=corrective_config.calibration_fold,
    )
    indexed = outer_train.assign(
        sample_id=outer_train["sample_id"].astype(str)
    ).set_index("sample_id", drop=False)
    proper_train = indexed.loc[list(split.proper_train_sample_ids)].reset_index(drop=True)
    calibration_frame = indexed.loc[list(split.calibration_sample_ids)].reset_index(drop=True)

    mean_path = phase_b_training.fit_g1_mean_path(
        phase_a_config,
        bundle,
        cache,
        fold,
        seed,
        split.proper_train_sample_ids,
        device=device,
        backend=backend,
        batch_size=batch_size,
    )
    scale_models, scale_histories = phase_b_training._select_and_refit_outer_scales(
        phase_b_config,
        proper_train,
        cache,
        mean_path,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    calibration = build_fixed_calibration(
        phase_b_config,
        calibration_frame,
        cache,
        mean_path,
        scale_models,
        fold=fold,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    return CorrectiveFittedPredictor(
        split=split,
        mean_path=mean_path,
        scale_models=scale_models,
        scale_histories=scale_histories,
        calibration=calibration,
    )


def run_corrective_fold_in_memory(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    phase_a_config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    *,
    fold: int,
    seed: int,
    device: str | torch.device,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
) -> CorrectiveFoldResult:
    """Fit/calibrate on outer train roles and score the untouched outer fold once."""
    if type(fold) is not int or fold not in range(5):
        raise ValueError("corrective fold must be one of 0,1,2,3,4")
    if type(seed) is not int or seed not in phase_b_training.PHASE_B_SEEDS:
        raise ValueError("corrective seed must be one of the registered Phase B seeds")
    outer_train_index, outer_test_index = outer_indices(bundle, fold)
    outer_train = bundle.manifest.iloc[outer_train_index].reset_index(drop=True).copy()
    outer_test = bundle.manifest.iloc[outer_test_index].reset_index(drop=True).copy()
    for frame in (outer_train, outer_test):
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["group_id"] = frame["group_id"].astype(str)

    fitted = fit_corrective_predictor(
        corrective_config,
        phase_b_config,
        phase_a_config,
        bundle,
        outer_train,
        cache,
        fold=fold,
        seed=seed,
        device=device,
        backend=backend,
        batch_size=batch_size,
    )
    test_groups = set(outer_test["group_id"])
    proper_groups = set(fitted.split.proper_train_group_ids)
    calibration_groups = set(fitted.split.calibration_group_ids)
    if (
        proper_groups & calibration_groups
        or proper_groups & test_groups
        or calibration_groups & test_groups
        or proper_groups | calibration_groups != set(outer_train["group_id"])
    ):
        raise ValueError("corrective proper-training/calibration/test groups are incompatible")

    inference_batches = phase_b_training._inference_batches(
        outer_test,
        cache,
        fitted.mean_path,
        batch_size=batch_size,
    )
    inferred, mean_predictions = phase_b_training._outer_inference(
        inference_batches,
        mean_model=fitted.mean_path.model,
        scale_models=fitted.scale_models,
        calibration=fitted.calibration,
        fold=fold,
        seed=seed,
        device=torch.device(device),
    )

    labels = repeat_measure_batch(outer_test)
    label_frame = outer_test.loc[:, ["sample_id", "group_id", "version"]].copy()
    label_frame["version"] = label_frame["version"].astype(str)
    label_frame["target_mean"] = labels.mean
    label_frame["ra_1"] = labels.readings[:, 0]
    label_frame["ra_2"] = labels.readings[:, 1]
    label_frame["ra_3"] = labels.readings[:, 2]
    predictions = inferred.merge(
        label_frame,
        on=["sample_id", "group_id"],
        how="inner",
        validate="many_to_one",
    ).loc[:, phase_b_training.PHASE_B_PREDICTION_COLUMNS]
    phase_b_training._validate_fold_predictions(
        predictions,
        outer_test,
        fold=fold,
        seed=seed,
    )
    mean = mean_predictions.merge(
        label_frame.loc[:, ["sample_id", "group_id", "version", "target_mean"]],
        on=["sample_id", "group_id"],
        how="inner",
        validate="many_to_one",
    ).loc[:, phase_b_training.PHASE_B_MEAN_COLUMNS]
    if (
        len(mean) != 3 * len(outer_test)
        or mean.duplicated(["sample_id", "seed", "model"]).any()
        or set(mean["model"]) != set(phase_b_training.MEAN_MODELS)
    ):
        raise ValueError("corrective mean predictions are incomplete")
    return CorrectiveFoldResult(
        predictions=predictions,
        mean_predictions=mean,
        fitted=fitted,
    )


def _corrective_fold_dir(root: Path, fold: int, seed: int) -> Path:
    return root / "folds" / f"fold_{fold}" / f"seed_{seed}"


def _assert_corrective_training_open(root: Path) -> None:
    if (root / "training_seal.json").exists():
        raise ValueError("corrective training root is sealed")
    if (root / "evaluation" / "evaluation_invocation.json").exists():
        raise ValueError("corrective evaluation has started; training is locked")


def _acquire_corrective_training_invocation(
    root: Path, *, fold: int, seed: int, fingerprint: str
) -> Path:
    path = root / ".invocations" / f"fold_{fold}-seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("corrective training unit has already been invoked") from error
    payload = {
        "protocol": CORRECTIVE_PROTOCOL,
        "fold": fold,
        "seed": seed,
        "fingerprint": fingerprint,
        "status": "started",
    }
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _canonical_json_bytes(value: object) -> bytes:
    def convert(item: object) -> object:
        if isinstance(item, Path):
            return str(item.resolve())
        if isinstance(item, dict):
            return {str(key): convert(content) for key, content in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(content) for content in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        convert(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def corrective_fingerprint(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    *,
    fold: int,
    seed: int,
) -> str:
    """Bind a corrective unit to its frozen sources, data, split, and model seed."""
    from .training import _cache_sha256, _frame_sha256

    payload = {
        "protocol": CORRECTIVE_PROTOCOL,
        "corrective_config": asdict(corrective_config),
        "phase_b_config": asdict(phase_b_config),
        "manifest_sha256": _frame_sha256(bundle.manifest),
        "folds_sha256": _frame_sha256(bundle.folds),
        "cache_sha256": _cache_sha256(cache),
        "fold": fold,
        "seed": seed,
        "scale_models": list(phase_b_training.SCALE_MODELS),
        "alphas": list(phase_b_training.PHASE_B_ALPHAS),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _corrective_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "complete.json"
    }


def publish_corrective_fold(
    output_root: str | Path,
    result: CorrectiveFoldResult,
    *,
    fingerprint: str,
    fold: int,
    seed: int,
    selected_device: str,
) -> Path:
    """Atomically publish one completed corrective unit; marker appears last."""
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("corrective fingerprint must be 64 lowercase hexadecimal characters")
    if selected_device not in {"cpu", "cuda"}:
        raise ValueError("corrective selected_device must be cpu or cuda")
    root = Path(output_root).resolve()
    _assert_corrective_training_open(root)
    destination = _corrective_fold_dir(root, fold, seed)
    if destination.exists():
        marker = destination / "complete.json"
        if marker.is_file():
            load_validated_corrective_fold(
                destination, fingerprint=fingerprint, fold=fold, seed=seed
            )
            return marker
        raise ValueError("partial corrective fold directory already exists")

    staging = root / ".staging" / f"fold_{fold}-seed_{seed}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    fitted = result.fitted
    try:
        phase_b_training._atomic_torch_save(
            staging / "checkpoints" / "mean.pt",
            {
                "protocol": CORRECTIVE_PROTOCOL,
                "fingerprint": fingerprint,
                "fold": fold,
                "seed": seed,
                "role": "proper_train_fixed_predictor",
                "train_sample_ids": list(fitted.split.proper_train_sample_ids),
                "model_state": fitted.mean_path.model.state_dict(),
            },
        )
        for scale_name in phase_b_training.SCALE_MODELS:
            phase_b_training._atomic_torch_save(
                staging / "checkpoints" / f"{scale_name}.pt",
                {
                    "protocol": CORRECTIVE_PROTOCOL,
                    "fingerprint": fingerprint,
                    "fold": fold,
                    "seed": seed,
                    "role": "proper_train_fixed_predictor",
                    "train_sample_ids": list(fitted.split.proper_train_sample_ids),
                    "model_state": fitted.scale_models[scale_name].state_dict(),
                },
            )
            phase_b_training._atomic_write_frame(
                staging / "history" / f"{scale_name}.csv",
                fitted.scale_histories[scale_name],
            )
        for name, scaler in (
            ("process", fitted.mean_path.process_scaler),
            ("spectrum", fitted.mean_path.spectrum_scaler),
            ("quality", fitted.mean_path.quality_scaler),
        ):
            phase_b_training._atomic_save_npz(
                staging / "scalers" / f"{name}.npz",
                mean=np.asarray(scaler.mean),
                scale=np.asarray(scaler.scale),
            )

        split_rows = [
            *(
                {"sample_id": sample_id, "role": "proper_train"}
                for sample_id in fitted.split.proper_train_sample_ids
            ),
            *(
                {"sample_id": sample_id, "role": "calibration"}
                for sample_id in fitted.split.calibration_sample_ids
            ),
        ]
        phase_b_training._atomic_write_frame(
            staging / "split.csv", pd.DataFrame(split_rows)
        )
        calibration_predictions = pd.concat(
            [
                fitted.calibration[name].predictions
                for name in phase_b_training.SCALE_MODELS
            ],
            ignore_index=True,
        )
        group_scores = pd.concat(
            [
                fitted.calibration[name].group_scores
                for name in phase_b_training.SCALE_MODELS
            ],
            ignore_index=True,
        )
        quantiles = {
            name: {
                f"{alpha:.2f}": asdict(fitted.calibration[name].quantiles[alpha])
                for alpha in phase_b_training.PHASE_B_ALPHAS
            }
            for name in phase_b_training.SCALE_MODELS
        }
        phase_b_training._atomic_write_frame(
            staging / "calibration" / "predictions.csv", calibration_predictions
        )
        phase_b_training._atomic_write_frame(
            staging / "calibration" / "group_scores.csv", group_scores
        )
        phase_b_training._atomic_write_json(
            staging / "calibration" / "quantiles.json", quantiles
        )
        phase_b_training._atomic_write_frame(
            staging / "predictions.csv", result.predictions
        )
        phase_b_training._atomic_write_frame(
            staging / "mean_predictions.csv", result.mean_predictions
        )
        checkpoint_hashes = {
            name: _sha256_file(staging / "checkpoints" / f"{name}.pt")
            for name in ("mean", *phase_b_training.SCALE_MODELS)
        }
        phase_b_training._atomic_write_json(
            staging / "provenance.json",
            {
                "protocol": CORRECTIVE_PROTOCOL,
                "fingerprint": fingerprint,
                "fold": fold,
                "seed": seed,
                "selected_device": selected_device,
                "proper_train_group_count": len(fitted.split.proper_train_group_ids),
                "calibration_group_count": len(fitted.split.calibration_group_ids),
                "checkpoint_sha256": checkpoint_hashes,
                "calibration_and_test_share_checkpoint": True,
            },
        )
        artifacts = _corrective_artifact_hashes(staging)
        marker_payload = {
            "protocol": CORRECTIVE_PROTOCOL,
            "fingerprint": fingerprint,
            "fold": fold,
            "seed": seed,
            "selected_device": selected_device,
            "status": "complete",
            "artifacts": artifacts,
            "probability_rows": len(result.predictions),
            "mean_rows": len(result.mean_predictions),
            "calibration_rows": len(calibration_predictions),
        }
        phase_b_training._atomic_write_json(
            staging / "complete.json", marker_payload
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
    except Exception:
        raise
    return destination / "complete.json"


def load_validated_corrective_fold(
    fold_dir: str | Path,
    *,
    fingerprint: str,
    fold: int,
    seed: int,
) -> PersistedCorrectiveFold:
    """Load one unit only after identity and every published artifact hash match."""
    root = Path(fold_dir).resolve()
    marker_path = root / "complete.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("corrective completion marker is missing or invalid") from error
    if (
        not isinstance(marker, dict)
        or marker.get("protocol") != CORRECTIVE_PROTOCOL
        or marker.get("fingerprint") != fingerprint
        or marker.get("fold") != fold
        or marker.get("seed") != seed
        or marker.get("status") != "complete"
        or marker.get("selected_device") not in {"cpu", "cuda"}
        or not isinstance(marker.get("artifacts"), dict)
    ):
        raise ValueError("corrective completion marker identity is incompatible")
    actual = _corrective_artifact_hashes(root)
    if actual != marker["artifacts"]:
        raise ValueError("corrective artifact hash closure is incompatible")

    predictions = pd.read_csv(root / "predictions.csv", float_precision="round_trip")
    mean = pd.read_csv(root / "mean_predictions.csv", float_precision="round_trip")
    calibration = pd.read_csv(
        root / "calibration" / "predictions.csv", float_precision="round_trip"
    )
    scores = pd.read_csv(
        root / "calibration" / "group_scores.csv", float_precision="round_trip"
    )
    quantiles = json.loads(
        (root / "calibration" / "quantiles.json").read_text(encoding="utf-8")
    )
    if (
        tuple(predictions.columns) != phase_b_training.PHASE_B_PREDICTION_COLUMNS
        or tuple(mean.columns) != phase_b_training.PHASE_B_MEAN_COLUMNS
        or len(predictions) != marker.get("probability_rows")
        or len(mean) != marker.get("mean_rows")
        or len(calibration) != marker.get("calibration_rows")
    ):
        raise ValueError("corrective persisted table schema or count is incompatible")
    _validate_corrective_semantics(
        root,
        predictions=predictions,
        calibration_predictions=calibration,
        group_scores=scores,
        quantiles=quantiles,
        fold=fold,
        seed=seed,
    )
    return PersistedCorrectiveFold(
        predictions=predictions,
        mean_predictions=mean,
        calibration_predictions=calibration,
        group_scores=scores,
        quantiles=quantiles,
        marker=marker,
    )


def _compare_recomputed_frame(
    saved: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    label: str,
) -> None:
    left = saved.sort_values(list(keys), kind="stable").reset_index(drop=True)
    right = recomputed.loc[:, left.columns].sort_values(
        list(keys), kind="stable"
    ).reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"corrective {label} row count does not recompute")
    text_columns = tuple(
        column
        for column in left.columns
        if column in {"sample_id", "group_id", "version", "scale_model", "model"}
    )
    if any(
        not left[column].astype(str).equals(right[column].astype(str))
        for column in text_columns
    ):
        raise ValueError(f"corrective {label} identity does not match source inference")
    numeric_columns = tuple(column for column in left.columns if column not in text_columns)
    if not np.allclose(
        left.loc[:, numeric_columns].to_numpy(dtype=np.float64),
        right.loc[:, numeric_columns].to_numpy(dtype=np.float64),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(f"corrective {label} values do not match source inference")


def validate_corrective_fold_against_sources(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    phase_a_config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    *,
    fold: int,
    seed: int,
    device: str | torch.device,
    batch_size: int = 8,
) -> PersistedCorrectiveFold:
    """Rebuild source roles and replay locked checkpoints for one completed unit."""
    fingerprint = corrective_fingerprint(
        corrective_config, phase_b_config, bundle, cache, fold=fold, seed=seed
    )
    root = _corrective_fold_dir(corrective_config.output_dir, fold, seed)
    unit = load_validated_corrective_fold(
        root, fingerprint=fingerprint, fold=fold, seed=seed
    )
    outer_train_index, outer_test_index = outer_indices(bundle, fold)
    outer_train = bundle.manifest.iloc[outer_train_index].reset_index(drop=True).copy()
    outer_test = bundle.manifest.iloc[outer_test_index].reset_index(drop=True).copy()
    for frame in (outer_train, outer_test):
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["group_id"] = frame["group_id"].astype(str)
    rebuilt_split = make_corrective_split(
        outer_train,
        split_seed=corrective_config.split_seed,
        calibration_fold=corrective_config.calibration_fold,
    )
    expected_proper_groups = 126 if fold < 2 else 127
    expected_test_groups = 43 if fold < 2 else 42
    if (
        len(rebuilt_split.calibration_group_ids) != 43
        or len(rebuilt_split.proper_train_group_ids) != expected_proper_groups
        or outer_test["group_id"].nunique() != expected_test_groups
    ):
        raise ValueError("corrective frozen group counts are incompatible")
    saved_split = pd.read_csv(root / "split.csv", dtype=str, keep_default_na=False)
    saved_roles = {
        role: tuple(saved_split.loc[saved_split["role"] == role, "sample_id"])
        for role in ("proper_train", "calibration")
    }
    if (
        saved_roles["proper_train"] != rebuilt_split.proper_train_sample_ids
        or saved_roles["calibration"] != rebuilt_split.calibration_sample_ids
        or set(unit.predictions["sample_id"].astype(str))
        != set(outer_test["sample_id"])
    ):
        raise ValueError("corrective persisted roles do not match the frozen source split")

    indexed = outer_train.set_index("sample_id", drop=False)
    proper = indexed.loc[list(rebuilt_split.proper_train_sample_ids)].reset_index(drop=True)
    calibration_frame = indexed.loc[
        list(rebuilt_split.calibration_sample_ids)
    ].reset_index(drop=True)
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    expected_checkpoint_hashes = {
        name: _sha256_file(root / "checkpoints" / f"{name}.pt")
        for name in ("mean", *phase_b_training.SCALE_MODELS)
    }
    if provenance.get("checkpoint_sha256") != expected_checkpoint_hashes:
        raise ValueError("corrective checkpoint provenance hashes are incompatible")

    checkpoint_payloads = {
        name: torch.load(
            root / "checkpoints" / f"{name}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for name in ("mean", *phase_b_training.SCALE_MODELS)
    }
    for name, payload in checkpoint_payloads.items():
        if (
            payload.get("protocol") != CORRECTIVE_PROTOCOL
            or payload.get("fingerprint") != fingerprint
            or payload.get("fold") != fold
            or payload.get("seed") != seed
            or payload.get("role") != "proper_train_fixed_predictor"
            or tuple(map(str, payload.get("train_sample_ids", ())))
            != rebuilt_split.proper_train_sample_ids
            or not isinstance(payload.get("model_state"), dict)
        ):
            raise ValueError(f"corrective {name} checkpoint metadata is incompatible")
    mean_model = SelectiveGatedModel()
    mean_model.load_state_dict(checkpoint_payloads["mean"]["model_state"], strict=True)
    mean_model.to(torch.device(device))
    scale_models = {
        "heteroscedastic": VarianceHead(),
        "homoscedastic": GlobalScale(),
    }
    for name, model in scale_models.items():
        model.load_state_dict(checkpoint_payloads[name]["model_state"], strict=True)

    scaler_types = {
        "process": ProcessScaler,
        "spectrum": SpectrumScaler,
        "quality": QualityScaler,
    }
    scalers = {}
    for name, scaler_type in scaler_types.items():
        with np.load(root / "scalers" / f"{name}.npz", allow_pickle=False) as payload:
            if set(payload.files) != {"mean", "scale"}:
                raise ValueError(f"corrective {name} scaler schema is incompatible")
            scalers[name] = scaler_type(
                mean=np.asarray(payload["mean"]), scale=np.asarray(payload["scale"])
            )
    recomputed_scalers = {
        "process": fit_process_scaler(
            build_process_features(proper), np.arange(len(proper), dtype=np.int64)
        ),
        "spectrum": fit_spectrum_scaler(cache, tuple(proper["sample_id"])),
        "quality": fit_quality_scaler(cache, tuple(proper["sample_id"])),
    }
    for name in scaler_types:
        if not np.array_equal(scalers[name].mean, recomputed_scalers[name].mean) or not np.array_equal(
            scalers[name].scale, recomputed_scalers[name].scale
        ):
            raise ValueError(f"corrective {name} scaler does not derive from proper train")

    mean_path = MeanPathArtifacts(
        fold=fold,
        seed=seed,
        train_sample_ids=rebuilt_split.proper_train_sample_ids,
        model=mean_model,
        process_scaler=scalers["process"],
        spectrum_scaler=scalers["spectrum"],
        quality_scaler=scalers["quality"],
        inner_splits=(),
        inner_models=(),
        inner_scalers=(),
        best_epochs={},
        histories={},
    )
    replay_calibration = build_fixed_calibration(
        phase_b_config,
        calibration_frame,
        cache,
        mean_path,
        scale_models,
        fold=fold,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    replay_calibration_predictions = pd.concat(
        [replay_calibration[name].predictions for name in phase_b_training.SCALE_MODELS],
        ignore_index=True,
    )
    _compare_recomputed_frame(
        unit.calibration_predictions,
        replay_calibration_predictions,
        keys=("sample_id", "scale_model"),
        label="calibration predictions",
    )
    inference_batches = phase_b_training._inference_batches(
        outer_test, cache, mean_path, batch_size=batch_size
    )
    replay_probability, replay_mean = phase_b_training._outer_inference(
        inference_batches,
        mean_model=mean_model,
        scale_models=scale_models,
        calibration=replay_calibration,
        fold=fold,
        seed=seed,
        device=torch.device(device),
    )
    labels = repeat_measure_batch(outer_test)
    label_frame = outer_test.loc[:, ["sample_id", "group_id", "version"]].copy()
    label_frame["version"] = label_frame["version"].astype(str)
    label_frame["target_mean"] = labels.mean
    label_frame["ra_1"] = labels.readings[:, 0]
    label_frame["ra_2"] = labels.readings[:, 1]
    label_frame["ra_3"] = labels.readings[:, 2]
    replay_probability = replay_probability.merge(
        label_frame,
        on=["sample_id", "group_id"],
        how="inner",
        validate="many_to_one",
    ).loc[:, phase_b_training.PHASE_B_PREDICTION_COLUMNS]
    replay_mean = replay_mean.merge(
        label_frame.loc[:, ["sample_id", "group_id", "version", "target_mean"]],
        on=["sample_id", "group_id"],
        how="inner",
        validate="many_to_one",
    ).loc[:, phase_b_training.PHASE_B_MEAN_COLUMNS]
    _compare_recomputed_frame(
        unit.predictions,
        replay_probability,
        keys=("sample_id", "scale_model"),
        label="outer-test predictions",
    )
    _compare_recomputed_frame(
        unit.mean_predictions,
        replay_mean,
        keys=("sample_id", "model"),
        label="outer-test mean predictions",
    )
    from .evaluation import (
        _phase_b_validate_probability_values,
        _validate_phase_b_mean_semantics,
    )

    _phase_b_validate_probability_values(unit.predictions, enforce_registered_intervals=True)
    _validate_phase_b_mean_semantics(unit.mean_predictions, probability=unit.predictions)
    return unit


def run_corrective_fold(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    phase_a_config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    *,
    fold: int,
    seed: int,
    device: str | torch.device,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
) -> PersistedCorrectiveFold:
    """Resume a hash-valid unit or train and atomically publish it once."""
    root = Path(corrective_config.output_dir).resolve()
    _assert_corrective_training_open(root)
    fingerprint = corrective_fingerprint(
        corrective_config,
        phase_b_config,
        bundle,
        cache,
        fold=fold,
        seed=seed,
    )
    destination = _corrective_fold_dir(root, fold, seed)
    marker = destination / "complete.json"
    if marker.is_file():
        return load_validated_corrective_fold(
            destination,
            fingerprint=fingerprint,
            fold=fold,
            seed=seed,
        )
    if destination.exists():
        raise ValueError("partial corrective fold directory already exists")
    _acquire_corrective_training_invocation(
        root, fold=fold, seed=seed, fingerprint=fingerprint
    )
    result = run_corrective_fold_in_memory(
        corrective_config,
        phase_b_config,
        phase_a_config,
        bundle,
        cache,
        fold=fold,
        seed=seed,
        device=device,
        backend=backend,
        batch_size=batch_size,
    )
    selected_device = torch.device(device).type
    publish_corrective_fold(
        corrective_config.output_dir,
        result,
        fingerprint=fingerprint,
        fold=fold,
        seed=seed,
        selected_device=selected_device,
    )
    return load_validated_corrective_fold(
        destination,
        fingerprint=fingerprint,
        fold=fold,
        seed=seed,
    )


def run_corrective_all(
    corrective_config: CorrectiveConfig,
    phase_b_config: PhaseBConfig,
    phase_a_config: SGRPNConfig,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
    *,
    device: str | torch.device,
    backend: TrainingBackend | None = None,
    batch_size: int = 8,
) -> CorrectiveRunResult:
    """Run or resume the fixed five-fold by three-seed corrective Cartesian."""
    if tuple(phase_b_config.seeds) != phase_b_training.PHASE_B_SEEDS:
        raise ValueError("corrective run requires the three registered Phase B seeds")
    root = Path(corrective_config.output_dir).resolve()
    _assert_corrective_training_open(root)
    units: list[PersistedCorrectiveFold] = []
    quantile_rows: list[dict[str, object]] = []
    for fold in range(5):
        for seed in phase_b_training.PHASE_B_SEEDS:
            unit = run_corrective_fold(
                corrective_config,
                phase_b_config,
                phase_a_config,
                bundle,
                cache,
                fold=fold,
                seed=seed,
                device=device,
                backend=backend,
                batch_size=batch_size,
            )
            units.append(unit)
            for scale_name in phase_b_training.SCALE_MODELS:
                for alpha in phase_b_training.PHASE_B_ALPHAS:
                    quantile_rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "scale_model": scale_name,
                            **unit.quantiles[scale_name][f"{alpha:.2f}"],
                        }
                    )
    probability = pd.concat(
        [unit.predictions for unit in units], ignore_index=True
    ).loc[:, phase_b_training.PHASE_B_PREDICTION_COLUMNS]
    mean = pd.concat(
        [unit.mean_predictions for unit in units], ignore_index=True
    ).loc[:, phase_b_training.PHASE_B_MEAN_COLUMNS]
    calibration = pd.concat(
        [unit.calibration_predictions for unit in units], ignore_index=True
    )
    scores = pd.concat([unit.group_scores for unit in units], ignore_index=True)
    quantiles = pd.DataFrame(quantile_rows).loc[
        :, ("fold", "seed", "scale_model", "alpha", "group_count", "order_index", "quantile")
    ]
    phase_b_training._validate_phase_b_oof_cartesian(probability, mean, bundle)

    paths = {
        "probability": root / "predictions" / "oof_probability_predictions.csv",
        "mean": root / "predictions" / "oof_mean_predictions.csv",
        "calibration": root / "calibration" / "predictions.csv",
        "scores": root / "calibration" / "group_scores.csv",
        "quantiles": root / "calibration" / "quantiles.csv",
    }
    for name, frame in (
        ("probability", probability),
        ("mean", mean),
        ("calibration", calibration),
        ("scores", scores),
        ("quantiles", quantiles),
    ):
        phase_b_training._atomic_write_frame(paths[name], frame)
    phase_b_training._atomic_write_json(
        root / "run_manifest.json",
        {
            "protocol": CORRECTIVE_PROTOCOL,
            "training_status": "complete",
            "completed_units": len(units),
            "selected_device": torch.device(device).type,
            "split_seed": corrective_config.split_seed,
            "calibration_fold": corrective_config.calibration_fold,
            "phase_b_config_file_sha256": corrective_config.phase_b_config_file_sha256,
            "phase_b_run_manifest_sha256": corrective_config.phase_b_run_manifest_sha256,
            "phase_b_immutable_after_sha256": corrective_config.phase_b_immutable_after_sha256,
            "combined_artifacts": {
                path.relative_to(root).as_posix(): _sha256_file(path)
                for path in paths.values()
            },
        },
    )
    return CorrectiveRunResult(
        probability_predictions=probability,
        mean_predictions=mean,
        calibration_predictions=calibration,
        group_scores=scores,
        quantiles=quantiles,
    )


def _validate_corrective_semantics(
    root: Path,
    *,
    predictions: pd.DataFrame,
    calibration_predictions: pd.DataFrame,
    group_scores: pd.DataFrame,
    quantiles: object,
    fold: int,
    seed: int,
) -> None:
    if tuple(group_scores.columns) != phase_b_training.CALIBRATION_SCORE_COLUMNS:
        raise ValueError("corrective calibration score schema is incompatible")
    required_calibration = {
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
    }
    if not required_calibration.issubset(calibration_predictions.columns):
        raise ValueError("corrective calibration prediction schema is incompatible")
    split = pd.read_csv(root / "split.csv", dtype=str, keep_default_na=False)
    if tuple(split.columns) != ("sample_id", "role") or set(split["role"]) != {
        "proper_train",
        "calibration",
    }:
        raise ValueError("corrective split artifact is incompatible")
    proper_ids = set(split.loc[split["role"] == "proper_train", "sample_id"])
    calibration_ids = set(split.loc[split["role"] == "calibration", "sample_id"])
    if (
        proper_ids & calibration_ids
        or calibration_ids != set(calibration_predictions["sample_id"].astype(str))
    ):
        raise ValueError("corrective split and calibration artifacts disagree")
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if (
        provenance.get("protocol") != CORRECTIVE_PROTOCOL
        or provenance.get("fold") != fold
        or provenance.get("seed") != seed
        or provenance.get("calibration_and_test_share_checkpoint") is not True
    ):
        raise ValueError("corrective predictor provenance is incompatible")
    checkpoint_hashes = {
        name: _sha256_file(root / "checkpoints" / f"{name}.pt")
        for name in ("mean", *phase_b_training.SCALE_MODELS)
    }
    if provenance.get("checkpoint_sha256") != checkpoint_hashes:
        raise ValueError("corrective checkpoint provenance hashes are incompatible")
    if not isinstance(quantiles, dict) or set(quantiles) != set(
        phase_b_training.SCALE_MODELS
    ):
        raise ValueError("corrective quantile schema is incompatible")

    for scale_name in phase_b_training.SCALE_MODELS:
        predicted = calibration_predictions.loc[
            calibration_predictions["scale_model"].astype(str) == scale_name
        ].copy()
        saved_scores = group_scores.loc[
            group_scores["scale_model"].astype(str) == scale_name
        ].copy()
        if (
            predicted.empty
            or len(predicted) != len(calibration_ids)
            or set(predicted["fold"]) != {fold}
            or set(predicted["seed"]) != {seed}
            or set(predicted["inner_fold"]) != {0}
        ):
            raise ValueError("corrective calibration prediction identity is incompatible")
        rebuilt = phase_b_training._calibration_group_scores(predicted)
        rebuilt = rebuilt.sort_values("group_id").reset_index(drop=True)
        saved_scores = saved_scores.sort_values("group_id").reset_index(drop=True)
        if tuple(saved_scores.columns) != tuple(rebuilt.columns) or len(saved_scores) != len(
            rebuilt
        ):
            raise ValueError("corrective calibration scores do not derive from predictions")
        text_columns = ("group_id", "scale_model")
        numeric_columns = tuple(
            column for column in rebuilt.columns if column not in text_columns
        )
        if any(
            not saved_scores[column].astype(str).equals(rebuilt[column].astype(str))
            for column in text_columns
        ) or not np.allclose(
            saved_scores.loc[:, numeric_columns].to_numpy(dtype=np.float64),
            rebuilt.loc[:, numeric_columns].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("corrective calibration scores do not derive from predictions")

        model_quantiles = quantiles[scale_name]
        if not isinstance(model_quantiles, dict) or set(model_quantiles) != {
            "0.10",
            "0.05",
        }:
            raise ValueError("corrective quantile schema is incompatible")
        test_rows = predictions.loc[predictions["scale_model"].astype(str) == scale_name]
        for alpha, suffix in ((0.10, "90"), (0.05, "95")):
            rebuilt_quantile = finite_sample_group_quantile(rebuilt, alpha=alpha)
            stored = model_quantiles[f"{alpha:.2f}"]
            expected = asdict(rebuilt_quantile)
            if not isinstance(stored, dict) or any(
                not np.isclose(float(stored[key]), float(value), rtol=0.0, atol=1e-12)
                for key, value in expected.items()
            ):
                raise ValueError("corrective quantiles do not derive from calibration scores")
            q = rebuilt_quantile.quantile
            if (
                not np.allclose(test_rows[f"conformal_q_{suffix}"], q, rtol=0.0, atol=1e-12)
                or not np.allclose(
                    test_rows[f"conformal_lower_{suffix}"],
                    test_rows["mu"] - q * test_rows["sigma"],
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.allclose(
                    test_rows[f"conformal_upper_{suffix}"],
                    test_rows["mu"] + q * test_rows["sigma"],
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError("corrective test intervals do not derive from calibration quantiles")


__all__ = [
    "CORRECTIVE_PROTOCOL",
    "CorrectiveConfig",
    "CorrectiveFittedPredictor",
    "CorrectiveFoldResult",
    "CorrectiveRunResult",
    "CorrectiveSplit",
    "PersistedCorrectiveFold",
    "build_fixed_calibration",
    "corrective_fingerprint",
    "fit_corrective_predictor",
    "load_corrective_config",
    "load_validated_corrective_fold",
    "make_corrective_split",
    "publish_corrective_fold",
    "run_corrective_fold",
    "run_corrective_all",
    "run_corrective_fold_in_memory",
    "validate_corrective_fold_against_sources",
]
