from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import yaml

from roughness.scheme1.config import Scheme1Config, load_scheme1_config


EXPECTED_MODELS = ("PW0", "PW1", "PW2", "PE0", "PE1", "PE2")


@dataclass(frozen=True)
class Scheme1PhysicsConfig:
    source: Scheme1Config
    source_config_path: Path
    output_dir: Path
    models: tuple[str, ...]
    batch_size: int
    validation_fraction: float
    residual_dropout: float


def _resolve(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (base / candidate).resolve()
    )


def _validate_source_artifacts(source: Scheme1Config) -> None:
    required = [
        source.output_dir / "window_index.csv",
        source.output_dir / "classic" / "oof_predictions.csv",
        *[
            source.output_dir
            / "folds"
            / f"fold_{fold}_channel_stats.json"
            for fold in range(5)
        ],
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "Missing source Scheme 1 inputs: "
            + ", ".join(str(path) for path in missing)
        )


def load_scheme1_physics_config(
    path: str | Path,
) -> Scheme1PhysicsConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = {
        "scheme1_config_path",
        "output_dir",
        "models",
        "batch_size",
        "validation_fraction",
        "residual_dropout",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    source_config_path = _resolve(
        config_path.parent, raw["scheme1_config_path"]
    )
    source = load_scheme1_config(source_config_path)
    _validate_source_artifacts(source)
    output_dir = _resolve(config_path.parent, raw["output_dir"])
    if output_dir == source.output_dir:
        raise ValueError(
            "physics output_dir must differ from source output_dir"
        )

    models = tuple(map(str, raw["models"]))
    if models != EXPECTED_MODELS:
        raise ValueError(f"models must equal {EXPECTED_MODELS}")
    batch_size = int(raw["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validation_fraction = float(raw["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    residual_dropout = float(raw["residual_dropout"])
    if not 0.0 <= residual_dropout < 1.0:
        raise ValueError("residual_dropout must satisfy 0 <= value < 1")

    return Scheme1PhysicsConfig(
        source=source,
        source_config_path=source_config_path,
        output_dir=output_dir,
        models=models,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        residual_dropout=residual_dropout,
    )


def write_physics_protocol_checkpoint(
    config: Scheme1PhysicsConfig,
    output_path: str | Path | None = None,
) -> Path:
    destination = (
        Path(output_path)
        if output_path is not None
        else config.output_dir / "run_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    git_check = subprocess.run(
        ["git", "-C", str(config.source.manifest_path.parent), "rev-parse"],
        capture_output=True,
        check=False,
        text=True,
    )
    payload = {
        "source_config_path": str(config.source_config_path),
        "source_output_dir": str(config.source.output_dir),
        "output_dir": str(config.output_dir),
        "models": list(config.models),
        "re_candidates_mm": list(config.source.re_candidates_mm),
        "seeds": list(config.source.seeds),
        "inner_splits": config.source.inner_splits,
        "training": {
            "batch_size": config.batch_size,
            "validation_fraction": config.validation_fraction,
            "residual_dropout": config.residual_dropout,
            "max_epochs": config.source.max_epochs,
            "patience": config.source.patience,
            "learning_rate": config.source.learning_rate,
            "weight_decay": config.source.weight_decay,
        },
        "git_available": git_check.returncode == 0,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination

