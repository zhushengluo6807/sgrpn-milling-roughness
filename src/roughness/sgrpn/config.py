from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import yaml


@dataclass(frozen=True)
class SGRPNConfig:
    manifest_path: Path
    folds_path: Path
    window_index_path: Path
    m0_oof_path: Path
    output_dir: Path
    sample_rate_hz: int
    window_samples: int
    order_min: float
    order_max: float
    order_step: float
    seeds: Sequence[int]
    inner_splits: int
    max_epochs: int
    patience: int
    process_learning_rate: float
    signal_learning_rate: float
    gate_learning_rate: float
    weight_decay: float
    huber_delta_um: float
    gate_penalty: float
    correction_penalty: float
    bootstrap_repetitions: int


@dataclass(frozen=True)
class PhaseBConfig:
    phase_a_config_path: Path
    phase_a_config_file_sha256: str
    phase_a_acceptance_path: Path
    phase_a_run_manifest_path: Path
    output_dir: Path
    seeds: Sequence[int]
    alphas: Sequence[float]
    inner_splits: int
    max_epochs: int
    patience: int
    variance_learning_rate: float
    weight_decay: float
    bootstrap_repetitions: int


@dataclass(frozen=True)
class PhaseAHandoff:
    acceptance_sha256: str
    run_manifest_sha256: str
    phase_a_config_sha256: str
    training_fingerprint: str
    cache_sha256: str
    input_sha256: Mapping[str, str]


_REQUIRED_KEYS = {
    "manifest_path",
    "folds_path",
    "window_index_path",
    "m0_oof_path",
    "output_dir",
    "sample_rate_hz",
    "window_samples",
    "order_min",
    "order_max",
    "order_step",
    "seeds",
    "inner_splits",
    "max_epochs",
    "patience",
    "process_learning_rate",
    "signal_learning_rate",
    "gate_learning_rate",
    "weight_decay",
    "huber_delta_um",
    "gate_penalty",
    "correction_penalty",
    "bootstrap_repetitions",
}

_PHASE_B_REQUIRED_KEYS = {
    "phase_a_config_path",
    "phase_a_config_file_sha256",
    "phase_a_acceptance_path",
    "phase_a_run_manifest_path",
    "output_dir",
    "seeds",
    "alphas",
    "inner_splits",
    "max_epochs",
    "patience",
    "variance_learning_rate",
    "weight_decay",
    "bootstrap_repetitions",
}
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROJECT_PHASE_B_OUTPUT = (
    Path(__file__).resolve().parents[3] / "outputs" / "sgrpn" / "phase_b"
).resolve()


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def load_sgrpn_config(path: str | Path) -> SGRPNConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    missing = sorted(_REQUIRED_KEYS - raw.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    seeds = tuple(int(seed) for seed in raw["seeds"])
    if seeds != (20260723,):
        raise ValueError("Phase A requires exactly seed 20260723")
    sample_rate_hz = raw["sample_rate_hz"]
    if type(sample_rate_hz) is not int or sample_rate_hz != 25600:
        raise ValueError("Phase A sample_rate_hz must be exactly 25600")

    order_min = float(raw["order_min"])
    order_max = float(raw["order_max"])
    order_step = float(raw["order_step"])
    order_count = (order_max - order_min) / order_step + 1
    if order_count != 361:
        raise ValueError("Phase A order grid must contain exactly 361 values")

    base = config_path.parent
    inputs = {
        name: _resolve(base, raw[name])
        for name in ("manifest_path", "folds_path", "window_index_path", "m0_oof_path")
    }
    for name, input_path in inputs.items():
        if not input_path.is_file():
            raise ValueError(f"{name} is not an existing file: {input_path}")

    return SGRPNConfig(
        **inputs,
        output_dir=_resolve(base, raw["output_dir"]),
        sample_rate_hz=25600,
        window_samples=int(raw["window_samples"]),
        order_min=order_min,
        order_max=order_max,
        order_step=order_step,
        seeds=seeds,
        inner_splits=int(raw["inner_splits"]),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        process_learning_rate=float(raw["process_learning_rate"]),
        signal_learning_rate=float(raw["signal_learning_rate"]),
        gate_learning_rate=float(raw["gate_learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        huber_delta_um=float(raw["huber_delta_um"]),
        gate_penalty=float(raw["gate_penalty"]),
        correction_penalty=float(raw["correction_penalty"]),
        bootstrap_repetitions=int(raw["bootstrap_repetitions"]),
    )


def _require_exact_integer(raw: object, expected: int, field: str, description: str) -> int:
    if type(raw) is not int or raw != expected:
        raise ValueError(f"Phase B {field} must be exactly {description}")
    return expected


def _require_exact_real(raw: object, expected: float, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or float(raw) != expected:
        raise ValueError(f"Phase B {field} must be exactly {expected}")
    return expected


def validate_phase_b_output_root(
    path: str | Path, *, output_root: str | Path | None = None
) -> Path:
    """Require the exact Phase B root without creating it."""
    configured = Path(path).resolve()
    allowed = (
        _PROJECT_PHASE_B_OUTPUT
        if output_root is None
        else Path(output_root).resolve()
    )
    if configured != allowed:
        raise ValueError(
            "Phase B requires the exact Phase B output root outputs/sgrpn/phase_b; "
            "temporary roots require explicit exact output_root injection"
        )
    return configured


def load_phase_b_config(path: str | Path) -> PhaseBConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Phase B config must be a mapping")
    missing = sorted(_PHASE_B_REQUIRED_KEYS - raw.keys())
    extra = sorted(raw.keys() - _PHASE_B_REQUIRED_KEYS)
    if missing or extra:
        raise ValueError(f"Phase B config keys are incompatible: missing={missing}, extra={extra}")

    registered_hash = raw["phase_a_config_file_sha256"]
    if not isinstance(registered_hash, str) or not _LOWER_SHA256.fullmatch(registered_hash):
        raise ValueError("Phase B phase_a_config_file_sha256 must be 64 lowercase hex characters")

    seeds_raw = raw["seeds"]
    if (
        not isinstance(seeds_raw, list)
        or any(type(seed) is not int for seed in seeds_raw)
        or tuple(seeds_raw) != (20260723, 20260724, 20260725)
    ):
        raise ValueError("Phase B seeds must be exactly 20260723, 20260724, 20260725")
    alphas_raw = raw["alphas"]
    if (
        not isinstance(alphas_raw, list)
        or any(isinstance(alpha, bool) or not isinstance(alpha, (int, float)) for alpha in alphas_raw)
        or tuple(map(float, alphas_raw)) != (0.10, 0.05)
    ):
        raise ValueError("Phase B alphas must be exactly 0.10, 0.05")

    inner_splits = _require_exact_integer(raw["inner_splits"], 4, "inner_splits", "four")
    max_epochs = _require_exact_integer(raw["max_epochs"], 200, "max_epochs", "200")
    patience = _require_exact_integer(raw["patience"], 20, "patience", "20")
    variance_learning_rate = _require_exact_real(
        raw["variance_learning_rate"], 0.001, "variance_learning_rate"
    )
    weight_decay = _require_exact_real(raw["weight_decay"], 0.0001, "weight_decay")
    bootstrap_repetitions = _require_exact_integer(
        raw["bootstrap_repetitions"], 10000, "bootstrap_repetitions", "10000"
    )

    base = config_path.parent
    output_dir = _resolve(base, raw["output_dir"])
    validate_phase_b_output_root(output_dir)
    return PhaseBConfig(
        phase_a_config_path=_resolve(base, raw["phase_a_config_path"]),
        phase_a_config_file_sha256=registered_hash,
        phase_a_acceptance_path=_resolve(base, raw["phase_a_acceptance_path"]),
        phase_a_run_manifest_path=_resolve(base, raw["phase_a_run_manifest_path"]),
        output_dir=output_dir,
        seeds=tuple(seeds_raw),
        alphas=tuple(map(float, alphas_raw)),
        inner_splits=inner_splits,
        max_epochs=max_epochs,
        patience=patience,
        variance_learning_rate=variance_learning_rate,
        weight_decay=weight_decay,
        bootstrap_repetitions=bootstrap_repetitions,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"required artifact is missing or unreadable: {path}") from error
    return digest.hexdigest()


def _read_json_object(path: Path, name: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is missing or invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _validate_phase_a_completion_identity(
    payload: dict, *, name: str, fold: int, fingerprint: str
) -> None:
    expected_models = ["P1", "V1", "F1", "R1", "G1"]
    if payload.get("protocol") != "sgrpn-phase-a-v2":
        raise ValueError(f"Phase A {name} protocol is missing or incompatible")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(f"Phase A {name} fingerprint is incompatible")
    if (
        type(payload.get("fold")) is not int
        or payload["fold"] != fold
        or type(payload.get("seed")) is not int
        or payload["seed"] != 20260723
    ):
        raise ValueError(f"Phase A {name} fold/seed identity is incompatible")
    if (
        payload.get("status") != "complete"
        or payload.get("models") != expected_models
        or payload.get("completed_stages") != expected_models
        or payload.get("next_stage") is not None
    ):
        raise ValueError(f"Phase A {name} completion model state is incompatible")


def validate_phase_b_handoff(config: PhaseBConfig) -> PhaseAHandoff:
    """Deeply verify the immutable Phase A v2 handoff without writing output."""
    acceptance_path = Path(config.phase_a_acceptance_path).resolve()
    run_manifest_path = Path(config.phase_a_run_manifest_path).resolve()
    phase_a_config_path = Path(config.phase_a_config_path).resolve()

    acceptance = _read_json_object(acceptance_path, "Phase A acceptance")
    decision = acceptance.get("decision")
    if (
        acceptance.get("schema_version") != "sgrpn-phase-a-acceptance-v1"
        or acceptance.get("proceed_to_phase_b") is not True
        or not isinstance(decision, dict)
        or decision.get("proceed_to_phase_b") is not True
        or acceptance.get("subjective_override_allowed") is not False
        or acceptance.get("phase_b_executed") is not False
    ):
        raise ValueError("Phase A acceptance gate is closed or incompatible")

    run_manifest = _read_json_object(run_manifest_path, "Phase A run manifest")
    if run_manifest.get("phase_b_executed") is not False:
        raise ValueError("Phase A run manifest says Phase B was already executed")
    acceptance_sha256 = _sha256_file(acceptance_path)
    registered_artifacts = run_manifest.get("artifacts")
    if (
        not isinstance(registered_artifacts, dict)
        or registered_artifacts.get("evaluation/acceptance.json") != acceptance_sha256
    ):
        raise ValueError("Phase A acceptance hash mismatch")

    phase_a_config_sha256 = _sha256_file(phase_a_config_path)
    if phase_a_config_sha256 != config.phase_a_config_file_sha256:
        raise ValueError("Phase A config file SHA-256 mismatch")

    phase_a = load_sgrpn_config(phase_a_config_path)
    phase_a_output = Path(phase_a.output_dir).resolve()
    if (
        acceptance_path != phase_a_output / "evaluation" / "acceptance.json"
        or run_manifest_path != phase_a_output / "run_manifest.json"
    ):
        raise ValueError("Phase A handoff paths do not bind to the configured output root")

    input_paths = {
        "manifest": Path(phase_a.manifest_path),
        "folds": Path(phase_a.folds_path),
        "window_index": Path(phase_a.window_index_path),
        "m0_oof": Path(phase_a.m0_oof_path),
    }
    registered_inputs = run_manifest.get("input_sha256")
    if not isinstance(registered_inputs, dict) or set(registered_inputs) != set(input_paths):
        raise ValueError("Phase A input_sha256 registry is incompatible")
    for name, path in input_paths.items():
        if registered_inputs[name] != _sha256_file(path):
            raise ValueError(f"Phase A {name} input hash mismatch")

    cache_npz = phase_a_output / "features" / "order_spectrum_cache.npz"
    cache_json = phase_a_output / "features" / "order_spectrum_cache.json"
    cache_metadata = _read_json_object(cache_json, "Phase A cache metadata")
    if cache_metadata.get("cache_sha256") != _sha256_file(cache_npz):
        raise ValueError("Phase A cache NPZ hash mismatch")

    from .data import load_data_bundle, outer_indices
    from .order_spectrum import load_order_cache
    from .training import (
        MODEL_SEQUENCE,
        PHASE_A_PROTOCOL,
        _load_and_validate_persisted_oof,
        _validate_completed_artifacts,
        build_run_fingerprint,
    )

    if PHASE_A_PROTOCOL != "sgrpn-phase-a-v2" or list(MODEL_SEQUENCE) != [
        "P1", "V1", "F1", "R1", "G1"
    ]:
        raise ValueError("current Phase A protocol/model sequence is incompatible")
    bundle = load_data_bundle(phase_a)
    cache = load_order_cache(bundle, phase_a)
    fingerprint = build_run_fingerprint(phase_a, bundle, cache)
    component_fields = {
        "config_sha256": fingerprint.config_sha256,
        "manifest_sha256": fingerprint.manifest_sha256,
        "folds_sha256": fingerprint.folds_sha256,
        "cache_sha256": fingerprint.cache_sha256,
        "training_fingerprint": fingerprint.value,
    }
    for field, recomputed in component_fields.items():
        if run_manifest.get(field) != recomputed:
            raise ValueError(f"Phase A run manifest {field} mismatch")

    for fold in range(5):
        fold_dir = phase_a_output / "folds" / f"fold_{fold}" / "seed_20260723"
        marker = _read_json_object(fold_dir / "complete.json", "Phase A marker")
        state = _read_json_object(fold_dir / "state.json", "Phase A state")
        _validate_phase_a_completion_identity(
            marker, name="marker", fold=fold, fingerprint=fingerprint.value
        )
        _validate_phase_a_completion_identity(
            state, name="state", fold=fold, fingerprint=fingerprint.value
        )
        _validate_completed_artifacts(fold_dir, marker, fingerprint, fold, 20260723)
        _, test_index = outer_indices(bundle, fold)
        expected_frame = bundle.manifest.iloc[test_index].reset_index(drop=True)
        _load_and_validate_persisted_oof(
            fold_dir, marker, fold, 20260723, expected_frame
        )

    validate_phase_b_output_root(config.output_dir)
    return PhaseAHandoff(
        acceptance_sha256=acceptance_sha256,
        run_manifest_sha256=_sha256_file(run_manifest_path),
        phase_a_config_sha256=phase_a_config_sha256,
        training_fingerprint=fingerprint.value,
        cache_sha256=fingerprint.cache_sha256,
        input_sha256=dict(registered_inputs),
    )


def config_fingerprint(config: SGRPNConfig, phase: str) -> str:
    payload = {"phase": phase, **asdict(config)}
    normalized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
