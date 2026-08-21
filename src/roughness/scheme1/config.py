from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import yaml


@dataclass(frozen=True)
class Scheme1Config:
    segments_dir: Path
    manifest_path: Path
    folds_path: Path
    output_dir: Path
    sample_rate_hz: int
    window_samples: int
    stride_samples: int
    re_candidates_mm: tuple[float, ...]
    seeds: tuple[int, ...]
    inner_splits: int
    welch_nperseg: int
    welch_noverlap: int
    nominal_band_min_halfwidth_hz: float
    nominal_band_relative_halfwidth: float
    hf_band_hz: tuple[float, float]
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    bootstrap_repetitions: int


_REQUIRED_KEYS = {
    "segments_dir",
    "manifest_path",
    "folds_path",
    "output_dir",
    "sample_rate_hz",
    "window_samples",
    "stride_samples",
    "re_candidates_mm",
    "seeds",
    "inner_splits",
    "welch_nperseg",
    "welch_noverlap",
    "nominal_band_min_halfwidth_hz",
    "nominal_band_relative_halfwidth",
    "hf_band_hz",
    "max_epochs",
    "patience",
    "learning_rate",
    "weight_decay",
    "bootstrap_repetitions",
}


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def load_scheme1_config(path: str | Path) -> Scheme1Config:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    missing = sorted(_REQUIRED_KEYS - raw.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    base = config_path.parent
    segments_dir = _resolve(base, raw["segments_dir"])
    manifest_path = _resolve(base, raw["manifest_path"])
    folds_path = _resolve(base, raw["folds_path"])
    output_dir = _resolve(base, raw["output_dir"])

    for name, input_path, expected in (
        ("segments_dir", segments_dir, "directory"),
        ("manifest_path", manifest_path, "file"),
        ("folds_path", folds_path, "file"),
    ):
        valid = input_path.is_dir() if expected == "directory" else input_path.is_file()
        if not valid:
            raise ValueError(f"{name} is not an existing {expected}: {input_path}")

    welch_nperseg = int(raw["welch_nperseg"])
    welch_noverlap = int(raw["welch_noverlap"])
    if not 0 <= welch_noverlap < welch_nperseg:
        raise ValueError("welch_noverlap must satisfy 0 <= overlap < nperseg")

    re_candidates = tuple(float(value) for value in raw["re_candidates_mm"])
    if not re_candidates or any(value <= 0 for value in re_candidates):
        raise ValueError("re_candidates_mm must contain positive values")

    hf_band = tuple(float(value) for value in raw["hf_band_hz"])
    if len(hf_band) != 2 or not 0 <= hf_band[0] < hf_band[1]:
        raise ValueError("hf_band_hz must contain two increasing non-negative values")

    return Scheme1Config(
        segments_dir=segments_dir,
        manifest_path=manifest_path,
        folds_path=folds_path,
        output_dir=output_dir,
        sample_rate_hz=int(raw["sample_rate_hz"]),
        window_samples=int(raw["window_samples"]),
        stride_samples=int(raw["stride_samples"]),
        re_candidates_mm=re_candidates,
        seeds=tuple(int(value) for value in raw["seeds"]),
        inner_splits=int(raw["inner_splits"]),
        welch_nperseg=welch_nperseg,
        welch_noverlap=welch_noverlap,
        nominal_band_min_halfwidth_hz=float(
            raw["nominal_band_min_halfwidth_hz"]
        ),
        nominal_band_relative_halfwidth=float(
            raw["nominal_band_relative_halfwidth"]
        ),
        hf_band_hz=(hf_band[0], hf_band[1]),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        bootstrap_repetitions=int(raw["bootstrap_repetitions"]),
    )


def write_protocol_checkpoint(
    config: Scheme1Config,
    fold_audit: dict[str, int],
    output_path: str | Path | None = None,
) -> Path:
    destination = (
        Path(output_path)
        if output_path is not None
        else config.output_dir / "run_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    git_check = subprocess.run(
        ["git", "-C", str(config.manifest_path.parents[2]), "rev-parse", "--git-dir"],
        capture_output=True,
        check=False,
        text=True,
    )
    payload = {
        "paths": {
            "segments_dir": str(config.segments_dir),
            "manifest_path": str(config.manifest_path),
            "folds_path": str(config.folds_path),
            "output_dir": str(config.output_dir),
        },
        "protocol": {
            "sample_rate_hz": config.sample_rate_hz,
            "window_samples": config.window_samples,
            "stride_samples": config.stride_samples,
            "re_candidates_mm": list(config.re_candidates_mm),
            "seeds": list(config.seeds),
            "inner_splits": config.inner_splits,
        },
        "outer_folds": fold_audit,
        "git_available": git_check.returncode == 0,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination
