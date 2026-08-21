from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from .signals import load_signal_csv


def bottom_geometry_ra_um(fz_mm: float, re_mm: float) -> float:
    fz = float(fz_mm)
    radius = float(re_mm)
    if radius <= 0:
        raise ValueError("re_mm must be positive")
    if fz < 0 or fz > 2.0 * radius:
        raise ValueError("fz_mm must satisfy 0 <= fz_mm <= 2 * re_mm")
    radicand = radius**2 - (fz / 2.0) ** 2
    return float(1000.0 * (radius - math.sqrt(radicand)) / 4.0)


def _band_energy(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    if selected.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[selected], frequencies[selected]))


def _nominal_band(
    center_hz: float,
    min_halfwidth_hz: float,
    relative_halfwidth: float,
    nyquist_hz: float,
) -> tuple[float, float]:
    halfwidth = max(min_halfwidth_hz, relative_halfwidth * center_hz)
    return max(0.0, center_hz - halfwidth), min(
        nyquist_hz, center_hz + halfwidth
    )


def band_limited_displacement_proxy(
    acceleration: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = 20.0,
    high_hz: float = 10_000.0,
) -> float:
    values = np.asarray(acceleration, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("acceleration must be a one-dimensional signal")
    if not np.isfinite(values).all():
        raise ValueError("acceleration contains non-finite values")
    nyquist = sample_rate_hz / 2.0
    high = min(float(high_hz), nyquist)
    if not 0 < low_hz < high:
        raise ValueError("Displacement proxy band must exclude zero frequency")

    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(values)
    selected = (frequencies >= low_hz) & (frequencies <= high)
    displacement_spectrum = np.zeros_like(spectrum)
    angular_frequency = 2.0 * np.pi * frequencies[selected]
    displacement_spectrum[selected] = (
        -spectrum[selected] / angular_frequency**2
    )
    displacement = np.fft.irfft(displacement_spectrum, n=len(values))
    proxy = float(np.sqrt(np.mean(displacement**2)))
    if not np.isfinite(proxy):
        raise ValueError("Displacement proxy is non-finite")
    return proxy


def extract_channel_features(
    values: np.ndarray,
    sample_rate_hz: float,
    n_rpm: float,
    teeth: int = 3,
    welch_nperseg: int = 8192,
    welch_noverlap: int = 4096,
    nominal_band_min_halfwidth_hz: float = 5.0,
    nominal_band_relative_halfwidth: float = 0.05,
    hf_band_hz: tuple[float, float] = (1000.0, 10_000.0),
) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("values must be a one-dimensional signal")
    if not np.isfinite(x).all():
        raise ValueError("values contain non-finite data")
    if sample_rate_hz <= 0 or n_rpm <= 0 or teeth <= 0:
        raise ValueError("sample rate, spindle speed and teeth must be positive")

    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    nperseg = min(int(welch_nperseg), len(x))
    noverlap = min(int(welch_noverlap), nperseg - 1)
    frequencies, psd = scipy_signal.welch(
        x,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )
    total_energy = float(np.sum(psd))
    if total_energy <= 0:
        raise ValueError("Signal has zero spectral energy")
    probability = psd / total_energy
    positive = probability > 0
    entropy_denominator = math.log(len(probability)) if len(probability) > 1 else 1
    spectral_entropy = float(
        -np.sum(probability[positive] * np.log(probability[positive]))
        / entropy_denominator
    )
    spectral_centroid = float(
        np.sum(frequencies * psd) / np.sum(psd)
    )

    rotation_hz = n_rpm / 60.0
    tpf_hz = teeth * rotation_hz
    nyquist = sample_rate_hz / 2.0

    def nominal_energy(center_hz: float) -> float:
        low, high = _nominal_band(
            center_hz,
            nominal_band_min_halfwidth_hz,
            nominal_band_relative_halfwidth,
            nyquist,
        )
        return _band_energy(frequencies, psd, low, high)

    hf_low = max(0.0, float(hf_band_hz[0]))
    hf_high = min(nyquist, float(hf_band_hz[1]))
    hf_energy = _band_energy(frequencies, psd, hf_low, hf_high)
    integrated_energy = _band_energy(frequencies, psd, 0.0, nyquist)
    features = {
        "rms": rms,
        "std": float(np.std(x)),
        "ptp": float(np.ptp(x)),
        "kurtosis": float(scipy_stats.kurtosis(x, fisher=False, bias=False)),
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "rotation_energy": nominal_energy(rotation_hz),
        "tpf_energy": nominal_energy(tpf_hz),
        "tpf_h2_energy": nominal_energy(2.0 * tpf_hz),
        "tpf_h3_energy": nominal_energy(3.0 * tpf_hz),
        "tpf_h4_energy": nominal_energy(4.0 * tpf_hz),
        "hf_energy": hf_energy,
        "hf_energy_ratio": (
            hf_energy / integrated_energy if integrated_energy > 0 else 0.0
        ),
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "displacement_proxy_relative": band_limited_displacement_proxy(
            x,
            sample_rate_hz=sample_rate_hz,
            low_hz=20.0,
            high_hz=min(10_000.0, nyquist),
        ),
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("Extracted features contain non-finite values")
    return features


def horizontal_feature_views(
    ch9: np.ndarray,
    ch10: np.ndarray,
    **feature_kwargs,
) -> dict[str, float]:
    ch9_features = extract_channel_features(ch9, **feature_kwargs)
    ch10_features = extract_channel_features(ch10, **feature_kwargs)
    result: dict[str, float] = {}
    for name in sorted(ch9_features):
        first = ch9_features[name]
        second = ch10_features[name]
        result[f"A_X_{name}"] = first
        result[f"A_Y_{name}"] = second
        result[f"B_X_{name}"] = second
        result[f"B_Y_{name}"] = first
        result[f"H_sym_mean_{name}"] = (first + second) / 2.0
        result[f"H_sym_l2_{name}"] = math.sqrt(
            (first**2 + second**2) / 2.0
        )
        result[f"H_sym_max_{name}"] = max(first, second)
        result[f"H_sym_min_{name}"] = min(first, second)
    return result


def aggregate_window_features(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if not rows:
        raise ValueError("rows must not be empty")
    keys = sorted(rows[0])
    if any(set(row) != set(keys) for row in rows):
        raise ValueError("All feature rows must have identical keys")
    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {key} contains non-finite values")
        aggregated[f"{key}_mean"] = float(values.mean())
        aggregated[f"{key}_std"] = float(values.std(ddof=0))
        aggregated[f"{key}_max"] = float(values.max())
    return aggregated


def extract_segment_feature_row(
    signal: np.ndarray,
    windows: Sequence[tuple[int, int]],
    n_rpm: float,
    fz_mm: float,
    re_candidates_mm: Sequence[float],
    **feature_kwargs,
) -> dict[str, float]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("signal must have shape [samples, 3]")
    if not windows:
        raise ValueError("windows must not be empty")

    window_rows: list[dict[str, float]] = []
    for start, end in windows:
        if not 0 <= start < end <= len(values):
            raise ValueError("Window falls outside the segment")
        window = values[start:end]
        z_features = extract_channel_features(
            window[:, 2], n_rpm=n_rpm, **feature_kwargs
        )
        horizontal = horizontal_feature_views(
            window[:, 0],
            window[:, 1],
            n_rpm=n_rpm,
            **feature_kwargs,
        )
        window_rows.append(
            {
                **{f"Z_{name}": value for name, value in z_features.items()},
                **horizontal,
            }
        )

    result = aggregate_window_features(window_rows)
    for radius in re_candidates_mm:
        radius_name = f"{float(radius):.2f}".replace(".", "p")
        result[f"ra_geo_bottom_re_{radius_name}_um"] = bottom_geometry_ra_um(
            fz_mm=float(fz_mm),
            re_mm=float(radius),
        )
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Segment features contain non-finite values")
    return result


@dataclass(frozen=True)
class FeatureScaler:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    source_segment_ids: tuple[str, ...]

    def transform(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            array = values.loc[:, self.feature_names].to_numpy(dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("Feature array has incompatible shape")
        transformed = (array - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Scaled features contain non-finite values")
        return transformed


def fit_feature_scaler(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    train_segment_ids: set[str],
) -> FeatureScaler:
    required = {"sample_id", *feature_names}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature frame missing columns: {missing}")
    selected_ids = tuple(sorted(map(str, train_segment_ids)))
    selected = frame[
        frame["sample_id"].astype(str).isin(selected_ids)
    ].loc[:, feature_names]
    if len(selected) != len(selected_ids):
        found = set(
            frame.loc[
                frame["sample_id"].astype(str).isin(selected_ids), "sample_id"
            ].astype(str)
        )
        raise ValueError(
            f"Unknown or duplicate training segment IDs: "
            f"{sorted(set(selected_ids) - found)}"
        )
    values = selected.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Training features contain non-finite values")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("Training features contain a constant column")
    return FeatureScaler(
        feature_names=tuple(feature_names),
        mean=mean,
        scale=scale,
        source_segment_ids=selected_ids,
    )


def build_segment_feature_table(
    manifest: pd.DataFrame,
    window_index: pd.DataFrame,
    re_candidates_mm: Sequence[float],
    **feature_kwargs,
) -> pd.DataFrame:
    required_manifest = {
        "sample_id",
        "group_id",
        "signal_path",
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
        "ra_mean",
        "sample_weight",
    }
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Manifest missing columns: {missing_manifest}")
    required_windows = {"segment_id", "start_sample", "end_sample"}
    missing_windows = sorted(required_windows - set(window_index.columns))
    if missing_windows:
        raise ValueError(f"Window index missing columns: {missing_windows}")

    windows_by_segment = {
        str(segment_id): [
            (int(row.start_sample), int(row.end_sample))
            for row in group.sort_values("start_sample").itertuples(index=False)
        ]
        for segment_id, group in window_index.groupby("segment_id", sort=False)
    }
    records = []
    for row in manifest.itertuples(index=False):
        segment_id = str(row.sample_id)
        if segment_id not in windows_by_segment:
            raise ValueError(f"No windows for segment {segment_id}")
        signal = load_signal_csv(row.signal_path)
        features = extract_segment_feature_row(
            signal,
            windows=windows_by_segment[segment_id],
            n_rpm=float(row.n_rpm),
            fz_mm=float(row.fz_mm_per_tooth),
            re_candidates_mm=re_candidates_mm,
            **feature_kwargs,
        )
        records.append(
            {
                "sample_id": segment_id,
                "group_id": str(row.group_id),
                "n_rpm": float(row.n_rpm),
                "fz_mm_per_tooth": float(row.fz_mm_per_tooth),
                "ap_mm": float(row.ap_mm),
                "ra_mean": float(row.ra_mean),
                "sample_weight": float(row.sample_weight),
                **features,
            }
        )
    result = pd.DataFrame.from_records(records)
    if result["sample_id"].duplicated().any():
        raise ValueError("Duplicate segment feature rows")
    numeric = result.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Feature table contains non-finite values")
    return result


def _feature_schema(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    metadata = {
        "sample_id",
        "group_id",
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
        "ra_mean",
        "sample_weight",
    }
    schema: dict[str, dict[str, str]] = {}
    for column in frame.columns:
        if column in metadata:
            continue
        if column.startswith("ra_geo_bottom_re_"):
            unit = "um"
            kind = "bottom_geometry_proxy"
        elif "displacement_proxy_relative" in column:
            unit = "relative_unit"
            kind = "band_limited_displacement_proxy"
        else:
            unit = "derived_from_g"
            kind = "signal_feature"
        schema[column] = {
            "unit": unit,
            "kind": kind,
            "direction_invariant": str(column.startswith("H_sym_")).lower(),
        }
    return schema


def write_feature_artifacts(
    frame: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    features_path = destination / "segment_features.csv"
    schema_path = destination / "feature_schema.json"
    frame.to_csv(features_path, index=False)
    schema_path.write_text(
        json.dumps(_feature_schema(frame), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"features": features_path, "schema": schema_path}
