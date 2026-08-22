"""Command-line orchestration for SGRPN Phase A only."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any
import uuid

import numpy as np
import pandas as pd
import torch

from .config import SGRPNConfig, config_fingerprint, load_sgrpn_config
from .data import load_data_bundle
from .evaluation import PREDICTION_COLUMNS, validate_prediction_cartesian
from .order_spectrum import (
    build_order_cache,
    load_order_cache,
    save_order_cache,
)
from .reporting import write_phase_a_report
from .training import (
    MODEL_SEQUENCE,
    RunFingerprint,
    _canonical_integer_text,
    _load_and_validate_persisted_oof,
    _validate_completed_artifacts,
    build_run_fingerprint,
    run_phase_a,
    run_phase_a_fold,
    validate_phase_a_output_root,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    return _atomic_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )


def _validate_output_dir(config: SGRPNConfig) -> Path:
    return validate_phase_a_output_root(config.output_dir)


def _selected_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return requested


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _validate_formal_run_manifest(
    output: Path,
    config: SGRPNConfig,
    expected_fingerprint: RunFingerprint,
) -> dict[str, Any]:
    manifest = _read_json(output / "run_manifest.json", "formal run manifest")
    required_status = {
        "audit_status": "complete",
        "feature_status": "complete",
        "training_status": "complete",
    }
    if any(manifest.get(key) != value for key, value in required_status.items()):
        raise ValueError("formal run manifest is incomplete")
    if (
        manifest.get("config_fingerprint") != config_fingerprint(config, "phase_a")
        or manifest.get("training_fingerprint") != expected_fingerprint.value
        or manifest.get("selected_device") not in {"cpu", "cuda"}
    ):
        raise ValueError("formal run manifest fingerprint/device is incompatible")
    paths = {
        "manifest": Path(config.manifest_path),
        "folds": Path(config.folds_path),
        "window_index": Path(config.window_index_path),
        "m0_oof": Path(config.m0_oof_path),
    }
    recorded_inputs = manifest.get("input_sha256")
    current_inputs = {name: _sha256(path) for name, path in paths.items()}
    if not isinstance(recorded_inputs, dict) or recorded_inputs != current_inputs:
        raise ValueError("formal run manifest input fingerprint mismatch")
    duration = output / "audit" / "duration_audit.csv"
    duration_record = manifest.get("duration_audit")
    if (
        not isinstance(duration_record, dict)
        or duration_record.get("provenance") != "canonical_current_data"
        or not duration.is_file()
        or duration_record.get("sha256") != _sha256(duration)
    ):
        raise ValueError("formal run manifest duration audit is incomplete or incompatible")
    return manifest


def _merge_run_manifest(output: Path, updates: dict[str, Any]) -> Path:
    path = output / "run_manifest.json"
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = _read_json(path, "run manifest")
    payload.update(updates)
    payload.setdefault("schema_version", "sgrpn-phase-a-run-v1")
    payload["phase_b_executed"] = False
    payload.pop("proceed_to_phase_b", None)
    try:
        installed_version = package_version("roughness-training")
    except PackageNotFoundError:
        installed_version = "not-installed"
    existing_environment = payload.get("environment", {})
    environment = dict(existing_environment) if isinstance(existing_environment, dict) else {}
    environment.update(
        {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "package_version": installed_version,
            "device": payload.get(
                "selected_device", environment.get("device", "not-selected")
            ),
        }
    )
    payload["environment"] = environment
    return _atomic_json(path, payload)


def _audit(config: SGRPNConfig) -> None:
    output = _validate_output_dir(config)
    bundle = load_data_bundle(config)
    duration_path = _atomic_bytes(
        output / "audit" / "duration_audit.csv",
        bundle.duration_audit.to_csv(index=False).encode("utf-8"),
    )
    _merge_run_manifest(
        output,
        {
            "audit_status": "complete",
            "config_fingerprint": config_fingerprint(config, "phase_a"),
            "seed": 20260723,
            "seed_count": 1,
            "fold_audit": bundle.fold_audit,
            "duration_audit": {
                "provenance": "canonical_current_data",
                "row_count": int(len(bundle.duration_audit)),
                "sha256": _sha256(duration_path),
            },
            "input_sha256": {
                "manifest": _sha256(Path(config.manifest_path)),
                "folds": _sha256(Path(config.folds_path)),
                "window_index": _sha256(Path(config.window_index_path)),
                "m0_oof": _sha256(Path(config.m0_oof_path)),
            },
        },
    )
    print(json.dumps(bundle.fold_audit, ensure_ascii=False, sort_keys=True))


def _features(config: SGRPNConfig) -> None:
    output = _validate_output_dir(config)
    bundle = load_data_bundle(config)
    cache = build_order_cache(bundle, config)
    staging = output / f".features-staging-{uuid.uuid4().hex}"
    staged_config = replace(config, output_dir=staging)
    try:
        staged_npz, staged_json = save_order_cache(cache, bundle, staged_config)
        destination = output / "features"
        destination.mkdir(parents=True, exist_ok=True)
        staged_npz.replace(destination / "order_spectrum_cache.npz")
        staged_json.replace(destination / "order_spectrum_cache.json")
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)
    # Deep-load the published bytes before recording completion.
    load_order_cache(bundle, config)
    _merge_run_manifest(output, {"feature_status": "complete"})
    print(output / "features" / "order_spectrum_cache.npz")


def _train(
    config: SGRPNConfig,
    *,
    fold: int | None = None,
    device: str = "auto",
    resume: bool = False,
) -> None:
    del resume  # Task 7 itself accepts only exact complete artifacts for reuse.
    output = _validate_output_dir(config)
    if fold is not None and fold not in range(5):
        raise ValueError("Phase A fold must be one of 0,1,2,3,4")
    bundle = load_data_bundle(config)
    cache = load_order_cache(bundle, config)
    selected = _selected_device(device)
    if fold is None:
        result = run_phase_a(config, bundle, cache, device=selected)
        status = "complete"
    else:
        result = run_phase_a_fold(config, bundle, cache, fold, 20260723, selected)
        status = "partial"
    _merge_run_manifest(
        output,
        {
            "training_status": status,
            "selected_device": selected,
            "training_fingerprint": result.fingerprint.value,
            "trained_fold": fold,
            "model_sequence": list(MODEL_SEQUENCE),
        },
    )
    print(json.dumps({"training_status": status, "device": selected, "fold": fold}))


def _typed_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"formal OOF predictions are missing: {path}")
    try:
        return pd.read_csv(
            path,
            dtype={
                "sample_id": str,
                "group_id": str,
                "version": str,
                "model": str,
                "fold": str,
                "seed": str,
            },
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"formal OOF predictions cannot be loaded: {path}") from error


def _verify_fold_artifacts(
    output: Path,
    fold: int,
    expected_frame: pd.DataFrame,
    expected_fingerprint: RunFingerprint,
) -> tuple[str, str, pd.DataFrame]:
    fold_dir = output / "folds" / f"fold_{fold}" / "seed_20260723"
    marker = _read_json(fold_dir / "complete.json", f"fold {fold} completion marker")
    if (
        marker.get("status") != "complete"
        or marker.get("fold") != fold
        or marker.get("seed") != 20260723
        or marker.get("models") != list(MODEL_SEQUENCE)
        or marker.get("completed_stages") != list(MODEL_SEQUENCE)
        or marker.get("fingerprint") != expected_fingerprint.value
    ):
        raise ValueError(
            f"fold {fold} marker does not match the current recomputed fingerprint"
        )
    prediction_path = fold_dir / "oof_predictions.csv"
    if marker.get("predictions_sha256") != _sha256(prediction_path):
        raise ValueError(f"fold {fold} OOF fingerprint mismatch")
    expected_artifacts = {
        f"checkpoints/{stage}.pt" for stage in MODEL_SEQUENCE
    } | {
        f"history/{stage}.csv" for stage in MODEL_SEQUENCE
    } | {
        f"scalers/{stage}_scalers.npz" for stage in MODEL_SEQUENCE
    }
    hashes = marker.get("artifacts")
    if not isinstance(hashes, dict) or set(hashes) != expected_artifacts:
        raise ValueError(f"fold {fold} formal artifact set is incomplete")
    _validate_completed_artifacts(
        fold_dir, marker, expected_fingerprint, fold, 20260723
    )
    fold_predictions = _load_and_validate_persisted_oof(
        fold_dir, marker, fold, 20260723, expected_frame
    )
    state = _read_json(fold_dir / "state.json", f"fold {fold} state")
    device = state.get("device")
    if state.get("status") != "complete" or state.get("fingerprint") != marker["fingerprint"] or device not in {"cpu", "cuda"}:
        raise ValueError(f"fold {fold} state is incomplete or incompatible")
    return expected_fingerprint.value, str(device), fold_predictions


def _load_m0(config: SGRPNConfig, manifest: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    source_hash = _sha256(Path(config.m0_oof_path))
    m0 = pd.read_csv(
        config.m0_oof_path,
        dtype={"sample_id": str, "group_id": str, "model": str, "fold": str, "seed": str},
    )
    required = {"sample_id", "group_id", "model", "fold", "seed", "y_true", "y_pred", "sample_weight"}
    missing = sorted(required - set(m0.columns))
    if missing:
        raise ValueError(f"M0 OOF missing columns: {missing}")
    m0 = m0.loc[m0["model"] == "M0"].copy()

    def exact_integer_column(column: str) -> np.ndarray:
        parsed: list[int] = []
        for value in m0[column]:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"M0 {column} must use canonical integral values")
            if isinstance(value, (int, np.integer)):
                parsed.append(int(value))
            elif isinstance(value, str) and _canonical_integer_text(value, signed=True):
                parsed.append(int(value))
            else:
                raise ValueError(f"M0 {column} must use canonical integral values")
        return np.asarray(parsed, dtype=np.int64)

    m0["fold"] = exact_integer_column("fold")
    m0["seed"] = exact_integer_column("seed")
    m0 = m0.loc[m0["seed"] == 20260723].copy()
    expected_ids = tuple(manifest["sample_id"].astype(str))
    if len(m0) != len(expected_ids) or m0["sample_id"].duplicated().any() or set(m0["sample_id"]) != set(expected_ids):
        raise ValueError("M0 OOF Cartesian coverage is incomplete or incompatible")
    expected = manifest.set_index("sample_id")
    fold_map = folds.set_index("sample_id")["fold"]
    indexed = m0.set_index("sample_id").reindex(expected_ids)
    numeric = indexed.loc[:, ["fold", "seed", "y_true", "y_pred", "sample_weight"]].apply(pd.to_numeric, errors="raise")
    if (
        not np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
        or not np.array_equal(numeric["fold"].astype(int), fold_map.reindex(expected_ids).astype(int))
        or not np.all(numeric["seed"].astype(int) == 20260723)
        or not np.allclose(numeric["y_true"], expected.reindex(expected_ids)["ra_mean"], rtol=0, atol=1e-12)
        or not np.allclose(numeric["sample_weight"], expected.reindex(expected_ids)["sample_weight"], rtol=0, atol=1e-12)
        or not np.array_equal(indexed["group_id"].astype(str), expected.reindex(expected_ids)["group_id"].astype(str))
    ):
        raise ValueError("M0 OOF ID/group/fold/target/weight mapping is incompatible")
    if _sha256(Path(config.m0_oof_path)) != source_hash:
        raise RuntimeError("read-only M0 OOF changed during evaluation")
    return pd.DataFrame(
        {
            "sample_id": expected_ids,
            "group_id": expected.reindex(expected_ids)["group_id"].astype(str).to_numpy(),
            "version": expected.reindex(expected_ids)["version"].astype(str).to_numpy(),
            "fold": numeric["fold"].astype(int).to_numpy(),
            "seed": numeric["seed"].astype(int).to_numpy(),
            "model": "M0",
            "target": numeric["y_true"].to_numpy(dtype=np.float64),
            "prediction": numeric["y_pred"].to_numpy(dtype=np.float64),
            "sample_weight": numeric["sample_weight"].to_numpy(dtype=np.float64),
            "process_mean": np.nan,
            "residual": np.nan,
            "gate": np.nan,
        },
        columns=PREDICTION_COLUMNS,
    )


def load_formal_phase_a_predictions(
    config: SGRPNConfig,
    *,
    output_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load complete formal OOF artifacts; this function cannot train models."""
    output = validate_phase_a_output_root(config.output_dir, output_root=output_root)
    bundle = load_data_bundle(config)
    cache = load_order_cache(bundle, config)
    expected_fingerprint = build_run_fingerprint(config, bundle, cache)
    manifest = bundle.manifest.copy()
    folds = bundle.folds.copy()
    fingerprints: set[str] = set()
    devices: set[str] = set()
    fold_predictions: list[pd.DataFrame] = []
    fold_map = folds.set_index("sample_id")["fold"].astype(int)
    for fold in range(5):
        expected_ids = fold_map.index[fold_map == fold]
        expected_frame = manifest.loc[
            manifest["sample_id"].isin(expected_ids),
            ["sample_id", "group_id", "version", "sample_weight", "ra_mean"],
        ].reset_index(drop=True)
        fingerprint, device, persisted = _verify_fold_artifacts(
            output, fold, expected_frame, expected_fingerprint
        )
        fingerprints.add(fingerprint)
        devices.add(device)
        fold_predictions.append(persisted)
    if fingerprints != {expected_fingerprint.value}:
        raise ValueError("formal fold fingerprints do not match")
    _validate_formal_run_manifest(output, config, expected_fingerprint)
    combined_path = output / "oof_predictions.csv"
    trained = _typed_predictions(combined_path)
    if tuple(trained.columns) != PREDICTION_COLUMNS:
        raise ValueError("trained OOF prediction schema is incompatible")
    # Add M0 only after the exact Task 7 five-model product has been checked.
    task7_expected = len(manifest) * len(MODEL_SEQUENCE)
    if len(trained) != task7_expected or set(trained["model"].astype(str)) != set(MODEL_SEQUENCE):
        raise ValueError("trained OOF Cartesian coverage is incomplete or incompatible")
    trained = validate_prediction_cartesian(
        trained,
        expected_sample_ids=tuple(manifest["sample_id"].astype(str)),
        models=MODEL_SEQUENCE,
    )
    normalized_trained = trained.copy()
    normalized_trained["fold"] = pd.to_numeric(normalized_trained["fold"], errors="raise").astype(np.int64)
    normalized_trained["seed"] = pd.to_numeric(normalized_trained["seed"], errors="raise").astype(np.int64)
    for column in ("target", "prediction", "sample_weight", "process_mean", "residual", "gate"):
        normalized_trained[column] = pd.to_numeric(normalized_trained[column], errors="raise").astype(np.float64)
    keys = ["sample_id", "model"]
    normalized_trained = normalized_trained.sort_values(keys).reset_index(drop=True)
    persisted_combined = pd.concat(fold_predictions, ignore_index=True).sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            normalized_trained,
            persisted_combined,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("combined OOF does not exactly match signed fold artifacts") from error
    trained = normalized_trained
    m0 = _load_m0(config, manifest, folds)
    predictions = pd.concat([m0, trained], ignore_index=True)
    predictions = validate_prediction_cartesian(
        predictions,
        expected_sample_ids=tuple(manifest["sample_id"].astype(str)),
    )
    mapped = predictions["sample_id"].map(fold_map)
    if not np.array_equal(pd.to_numeric(predictions["fold"]).astype(int), mapped.astype(int)):
        raise ValueError("formal OOF fold mapping is incompatible")
    quality = pd.DataFrame(
        {
            "sample_id": tuple(cache.segment_ids),
            **{
                f"quality_{index}": cache.quality[:, index]
                for index in range(cache.quality.shape[1])
            },
        }
    )
    manifest.attrs["duration_audit"] = bundle.duration_audit.copy()
    manifest.attrs["quality_features"] = quality
    return predictions, manifest, {
        "training_fingerprint": expected_fingerprint.value,
        "config_sha256": expected_fingerprint.config_sha256,
        "manifest_sha256": expected_fingerprint.manifest_sha256,
        "folds_sha256": expected_fingerprint.folds_sha256,
        "cache_sha256": expected_fingerprint.cache_sha256,
        "selected_devices": sorted(devices),
        "duration_audit": {
            "provenance": "canonical_current_data",
            "row_count": int(len(bundle.duration_audit)),
        },
    }


def _evaluate(config: SGRPNConfig) -> None:
    output = _validate_output_dir(config)
    if config.bootstrap_repetitions != 10_000:
        raise ValueError("formal Phase A requires exactly 10000 bootstrap repetitions")
    predictions, manifest, provenance = load_formal_phase_a_predictions(config)
    quality = manifest.attrs["quality_features"]
    duration = manifest.attrs["duration_audit"]
    written = write_phase_a_report(
        output,
        predictions,
        manifest,
        duration_audit=duration,
        quality_features=quality,
        bootstrap_repetitions=config.bootstrap_repetitions,
        bootstrap_seed=20260723,
        run_manifest=provenance,
    )
    acceptance = _read_json(written["acceptance"], "acceptance")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roughness-sgrpn")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "features", "evaluate-phase-a"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    train = subparsers.add_parser("train-phase-a")
    train.add_argument("--config", required=True)
    train.add_argument("--fold", type=int, choices=range(5))
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--resume", action="store_true")
    run = subparsers.add_parser("run-phase-a")
    run.add_argument("--config", required=True)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = load_sgrpn_config(args.config)
        if args.command == "audit":
            _audit(config)
        elif args.command == "features":
            _features(config)
        elif args.command == "train-phase-a":
            _train(
                config,
                fold=args.fold,
                device=args.device,
                resume=args.resume,
            )
        elif args.command == "evaluate-phase-a":
            _evaluate(config)
        elif args.command == "run-phase-a":
            _audit(config)
            _features(config)
            _train(config, device=args.device, resume=args.resume)
            _evaluate(config)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "load_formal_phase_a_predictions", "main"]
