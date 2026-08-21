"""Fixed-grid order-spectrum features for SGRPN Phase A."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from roughness.scheme1.signals import load_signal_csv

from .config import SGRPNConfig
from .data import DataBundle


_WINDOW_SAMPLES = 25600
_CHANNEL_COUNT = 3
_QUALITY_WIDTH = 7
_GRID_MIN = 0.0
_GRID_MAX = 90.0
_GRID_STEP = 0.25
_GRID_COUNT = 361


def order_grid(order_min: float, order_max: float, order_step: float) -> np.ndarray:
    """Return the inclusive fixed order grid, rejecting imprecise grids."""
    start, stop, step = float(order_min), float(order_max), float(order_step)
    if not np.isfinite([start, stop, step]).all() or step <= 0 or stop < start:
        raise ValueError("order grid requires finite min/max and a positive step")
    count_float = (stop - start) / step
    count = round(count_float)
    if not np.isclose(count_float, count, rtol=0.0, atol=1e-10):
        raise ValueError("order grid endpoints must align with order_step")
    return start + step * np.arange(count + 1, dtype=np.float64)


def _require_phase_a_grid(order_min: float, order_max: float, order_step: float) -> np.ndarray:
    grid = order_grid(order_min, order_max, order_step)
    expected = order_grid(_GRID_MIN, _GRID_MAX, _GRID_STEP)
    if grid.shape != (_GRID_COUNT,) or not np.array_equal(grid, expected):
        raise ValueError("Phase A order grid must be 0–90 in 0.25-order steps")
    return grid


def _validate_window(window: np.ndarray) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    if values.shape != (_WINDOW_SAMPLES, _CHANNEL_COUNT):
        raise ValueError("window must have shape [25600, 3]")
    if not np.isfinite(values).all():
        raise ValueError("window contains non-finite values")
    return values


def window_to_order_spectrum(
    window: np.ndarray,
    n_rpm: float,
    sample_rate_hz: int,
    order_min: float,
    order_max: float,
    order_step: float,
) -> np.ndarray:
    """Convert one complete, three-channel window to log-power order spectra."""
    values = _validate_window(window)
    rpm = float(n_rpm)
    if not np.isfinite(rpm) or rpm <= 0:
        raise ValueError("n_rpm must be finite and positive")
    if int(sample_rate_hz) != _WINDOW_SAMPLES:
        raise ValueError("Phase A sample_rate_hz must be exactly 25600")
    grid = _require_phase_a_grid(order_min, order_max, order_step)

    centered = values - values.mean(axis=0, keepdims=True)
    tapered = centered * np.hanning(_WINDOW_SAMPLES)[:, None]
    power = np.abs(np.fft.rfft(tapered, axis=0)) ** 2
    frequencies_hz = np.fft.rfftfreq(_WINDOW_SAMPLES, d=1.0 / sample_rate_hz)
    orders = frequencies_hz / (rpm / 60.0)
    log_power = np.log1p(power)
    spectra = np.empty((_CHANNEL_COUNT, grid.size), dtype=np.float32)
    for channel in range(_CHANNEL_COUNT):
        spectra[channel] = np.interp(grid, orders, log_power[:, channel]).astype(np.float32)
    return spectra


def signal_quality_features(signal: np.ndarray) -> np.ndarray:
    """Return seven finite, horizontal-channel exchange-invariant quality features."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != _CHANNEL_COUNT:
        raise ValueError("signal must have shape [samples, 3] with at least one sample")
    if not np.isfinite(values).all():
        raise ValueError("signal contains non-finite values")

    horizontal = values[:, :2]
    vertical = values[:, 2]
    horizontal_rms = np.sqrt(np.mean(horizontal**2, axis=0))
    horizontal_std = horizontal.std(axis=0)
    features = np.array(
        [
            np.mean(np.abs(horizontal)),
            np.mean(horizontal_std),
            np.mean(horizontal_rms),
            np.sqrt(np.mean((horizontal[:, 0] - horizontal[:, 1]) ** 2)),
            np.max(np.abs(horizontal)),
            np.sqrt(np.mean(vertical**2)),
            vertical.std(),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(features).all():
        raise ValueError("signal quality features must be finite")
    return features


@dataclass(frozen=True)
class OrderSpectrumCache:
    segment_ids: Sequence[str]
    spectra: np.ndarray
    offsets: np.ndarray
    quality: np.ndarray
    durations_s: np.ndarray

    def __post_init__(self) -> None:
        segment_ids = tuple(map(str, self.segment_ids))
        spectra = np.asarray(self.spectra, dtype=np.float32)
        offsets = np.asarray(self.offsets, dtype=np.int64)
        quality = np.asarray(self.quality, dtype=np.float32)
        durations_s = np.asarray(self.durations_s, dtype=np.float64)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("segment_ids must be unique")
        if spectra.ndim != 3 or spectra.shape[1:] != (_CHANNEL_COUNT, _GRID_COUNT):
            raise ValueError("spectra must have shape [all_windows, 3, 361]")
        if offsets.shape != (len(segment_ids) + 1,) or offsets[0] != 0:
            raise ValueError("offsets must start at zero and have one row per segment")
        if np.any(np.diff(offsets) < 0) or offsets[-1] != len(spectra):
            raise ValueError("offsets must be non-decreasing and end at all_windows")
        if quality.shape != (len(segment_ids), _QUALITY_WIDTH):
            raise ValueError("quality must have shape [n_segments, 7]")
        if durations_s.shape != (len(segment_ids),):
            raise ValueError("durations_s must have shape [n_segments]")
        if not (np.isfinite(spectra).all() and np.isfinite(quality).all() and np.isfinite(durations_s).all()):
            raise ValueError("cache values must be finite")
        if np.any(durations_s < 0):
            raise ValueError("durations_s must be non-negative")
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "spectra", spectra)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "durations_s", durations_s)

    def _segment_index(self, segment_id: str) -> int:
        try:
            return self.segment_ids.index(str(segment_id))
        except ValueError as error:
            raise ValueError(f"Unknown segment ID: {segment_id}") from error

    def bag(self, segment_id: str) -> np.ndarray:
        index = self._segment_index(segment_id)
        return self.spectra[self.offsets[index] : self.offsets[index + 1]]

    def quality_row(self, segment_id: str) -> np.ndarray:
        return self.quality[self._segment_index(segment_id)]


@dataclass(frozen=True)
class SpectrumScaler:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class QualityScaler:
    mean: np.ndarray
    scale: np.ndarray


def _requested_indices(cache: OrderSpectrumCache, segment_ids: Sequence[str]) -> list[int]:
    requested = tuple(map(str, segment_ids))
    if not requested:
        raise ValueError("training segment IDs must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("training segment IDs must not contain duplicates")
    return [cache._segment_index(segment_id) for segment_id in requested]


def _nonzero_scale(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0, dtype=np.float64)
    return np.where(scale > 0.0, scale, 1.0).astype(np.float32)


def fit_spectrum_scaler(cache: OrderSpectrumCache, train_segment_ids: Sequence[str]) -> SpectrumScaler:
    indices = _requested_indices(cache, train_segment_ids)
    bags = [cache.spectra[cache.offsets[index] : cache.offsets[index + 1]] for index in indices]
    nonempty = [bag for bag in bags if len(bag)]
    if not nonempty:
        raise ValueError("requested training segments contain no windows")
    values = np.concatenate(nonempty, axis=0)
    return SpectrumScaler(
        mean=values.mean(axis=0, dtype=np.float64).astype(np.float32),
        scale=_nonzero_scale(values),
    )


def fit_quality_scaler(cache: OrderSpectrumCache, train_segment_ids: Sequence[str]) -> QualityScaler:
    indices = _requested_indices(cache, train_segment_ids)
    values = cache.quality[indices]
    return QualityScaler(
        mean=values.mean(axis=0, dtype=np.float64).astype(np.float32),
        scale=_nonzero_scale(values),
    )


def _manifest_index(bundle: DataBundle) -> pd.DataFrame:
    required = {"sample_id", "signal_path", "n_rpm", "duration_s"}
    missing = sorted(required - set(bundle.manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    indexed = bundle.manifest.assign(sample_id=bundle.manifest["sample_id"].astype(str)).set_index("sample_id", drop=False)
    if indexed.index.duplicated().any():
        raise ValueError("Manifest sample_id values must be unique")
    return indexed


def build_order_cache(bundle: DataBundle, config: SGRPNConfig) -> OrderSpectrumCache:
    """Build one spectrum bag per manifest segment using only indexed windows."""
    if config.sample_rate_hz != _WINDOW_SAMPLES or config.window_samples != _WINDOW_SAMPLES:
        raise ValueError("Phase A windows and sample rate must be exactly 25600")
    _require_phase_a_grid(config.order_min, config.order_max, config.order_step)
    required_windows = {"segment_id", "start_sample", "end_sample"}
    missing = sorted(required_windows - set(bundle.windows.columns))
    if missing:
        raise ValueError(f"Window index missing columns: {missing}")

    manifest = _manifest_index(bundle)
    segment_ids = tuple(manifest.index.tolist())
    indexed_windows = bundle.windows.assign(segment_id=bundle.windows["segment_id"].astype(str))
    unknown = sorted(set(indexed_windows["segment_id"]) - set(segment_ids))
    if unknown:
        raise ValueError(f"Window index contains unknown segment IDs: {unknown[:10]}")

    all_spectra: list[np.ndarray] = []
    offsets = [0]
    quality: list[np.ndarray] = []
    durations: list[float] = []
    for segment_id in segment_ids:
        row = manifest.loc[segment_id]
        signal = load_signal_csv(Path(row["signal_path"]))
        quality.append(signal_quality_features(signal))
        durations.append(float(row["duration_s"]))
        rows = indexed_windows.loc[indexed_windows["segment_id"] == segment_id]
        for window_row in rows.itertuples(index=False):
            start = int(getattr(window_row, "start_sample"))
            end = int(getattr(window_row, "end_sample"))
            if start < 0 or end - start != _WINDOW_SAMPLES or end > len(signal):
                raise ValueError(f"Window index must select a complete [25600, 3] window for {segment_id}")
            all_spectra.append(
                window_to_order_spectrum(
                    signal[start:end],
                    row["n_rpm"],
                    config.sample_rate_hz,
                    config.order_min,
                    config.order_max,
                    config.order_step,
                )
            )
        offsets.append(len(all_spectra))

    spectra = np.stack(all_spectra).astype(np.float32) if all_spectra else np.empty((0, _CHANNEL_COUNT, _GRID_COUNT), dtype=np.float32)
    return OrderSpectrumCache(
        segment_ids=segment_ids,
        spectra=spectra,
        offsets=np.asarray(offsets, dtype=np.int64),
        quality=np.asarray(quality, dtype=np.float32),
        durations_s=np.asarray(durations, dtype=np.float64),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_paths(config: SGRPNConfig) -> tuple[Path, Path]:
    features_dir = Path(config.output_dir) / "features"
    return features_dir / "order_spectrum_cache.npz", features_dir / "order_spectrum_cache.json"


def _mismatch_count(bundle: DataBundle) -> int:
    if "mismatch_over_1ms" not in bundle.duration_audit.columns:
        return 0
    return int(bundle.duration_audit["mismatch_over_1ms"].sum())


def _metadata(cache: OrderSpectrumCache, bundle: DataBundle, config: SGRPNConfig, cache_sha256: str) -> dict:
    return {
        "grid": order_grid(config.order_min, config.order_max, config.order_step).tolist(),
        "sample_rate_hz": int(config.sample_rate_hz),
        "manifest_sha256": _sha256_file(Path(config.manifest_path)),
        "window_index_sha256": _sha256_file(Path(config.window_index_path)),
        "segment_count": len(cache.segment_ids),
        "window_count": int(cache.spectra.shape[0]),
        "mismatch_count": _mismatch_count(bundle),
        "cache_sha256": cache_sha256,
    }


def save_order_cache(cache: OrderSpectrumCache, bundle: DataBundle, config: SGRPNConfig) -> tuple[Path, Path]:
    """Persist the cache and source fingerprints beneath ``output_dir/features``."""
    _require_phase_a_grid(config.order_min, config.order_max, config.order_step)
    npz_path, json_path = _feature_paths(config)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        segment_ids=np.asarray(cache.segment_ids, dtype=np.str_),
        spectra=cache.spectra,
        offsets=cache.offsets,
        quality=cache.quality,
        durations_s=cache.durations_s,
    )
    metadata = _metadata(cache, bundle, config, _sha256_file(npz_path))
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path, json_path


def load_order_cache(bundle: DataBundle, config: SGRPNConfig) -> OrderSpectrumCache:
    """Load a cache only when its grid, source fingerprints, and bytes match."""
    expected_grid = _require_phase_a_grid(config.order_min, config.order_max, config.order_step)
    npz_path, json_path = _feature_paths(config)
    if not npz_path.is_file() or not json_path.is_file():
        raise ValueError("Order spectrum cache files are missing")
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Order spectrum cache metadata is invalid") from error
    if metadata.get("sample_rate_hz") != int(config.sample_rate_hz) or metadata.get("grid") != expected_grid.tolist():
        raise ValueError("Order spectrum cache has an incompatible grid or sample rate")
    if metadata.get("manifest_sha256") != _sha256_file(Path(config.manifest_path)) or metadata.get("window_index_sha256") != _sha256_file(Path(config.window_index_path)):
        raise ValueError("Order spectrum cache source fingerprint does not match")
    if metadata.get("cache_sha256") != _sha256_file(npz_path):
        raise ValueError("Order spectrum cache fingerprint does not match cache bytes")
    with np.load(npz_path, allow_pickle=False) as archive:
        cache = OrderSpectrumCache(
            segment_ids=tuple(archive["segment_ids"].tolist()),
            spectra=archive["spectra"],
            offsets=archive["offsets"],
            quality=archive["quality"],
            durations_s=archive["durations_s"],
        )
    if metadata.get("segment_count") != len(cache.segment_ids) or metadata.get("window_count") != len(cache.spectra) or metadata.get("mismatch_count") != _mismatch_count(bundle):
        raise ValueError("Order spectrum cache metadata does not match bundle")
    return cache
