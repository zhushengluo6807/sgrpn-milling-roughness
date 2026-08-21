from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    labels: Path
    v3_records: Path
    v4_records: Path
    segments_root: Path
    output_dir: Path
    seed: int
    n_splits: int


def load_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {
        "labels",
        "v3_records",
        "v4_records",
        "segments_root",
        "output_dir",
        "seed",
        "n_splits",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    base = path.parent

    def resolved(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    inputs = {
        key: resolved(raw[key])
        for key in ["labels", "v3_records", "v4_records", "segments_root"]
    }
    for key, input_path in inputs.items():
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {key}={input_path}")

    return TrainingConfig(
        **inputs,
        output_dir=resolved(raw["output_dir"]),
        seed=int(raw["seed"]),
        n_splits=int(raw["n_splits"]),
    )
