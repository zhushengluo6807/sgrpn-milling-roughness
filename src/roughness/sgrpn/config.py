from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

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
    if int(raw["sample_rate_hz"]) != 25600:
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
        sample_rate_hz=int(raw["sample_rate_hz"]),
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


def config_fingerprint(config: SGRPNConfig, phase: str) -> str:
    payload = {"phase": phase, **asdict(config)}
    normalized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
