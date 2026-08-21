from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


SIGNAL_CHANNELS = ("Ch9_g", "Ch10_g", "Ch11_g")


@dataclass(frozen=True)
class ChannelStats:
    mean: np.ndarray
    std: np.ndarray
    sample_count: int
    source_segment_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "channels": list(SIGNAL_CHANNELS),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "sample_count": self.sample_count,
            "source_segment_ids": list(self.source_segment_ids),
        }


def _validate_columns(path: Path, columns: list[str]) -> None:
    missing = sorted(set(SIGNAL_CHANNELS) - set(columns))
    if missing:
        raise ValueError(f"Missing signal columns in {path}: {missing}")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate columns in signal file: {path}")


def load_signal_csv(path: str | Path) -> np.ndarray:
    signal_path = Path(path)
    header = pd.read_csv(signal_path, nrows=0)
    _validate_columns(signal_path, header.columns.astype(str).tolist())
    frame = pd.read_csv(signal_path, usecols=list(SIGNAL_CHANNELS))
    if frame.empty:
        raise ValueError(f"Empty signal file: {signal_path}")
    values = frame.loc[:, SIGNAL_CHANNELS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Signal contains non-finite values: {signal_path}")
    return values


def _require_manifest_columns(manifest: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")


def fit_channel_stats(
    manifest: pd.DataFrame,
    train_segment_ids: set[str],
    chunk_size: int = 100_000,
) -> ChannelStats:
    _require_manifest_columns(manifest, {"sample_id", "signal_path"})
    if not train_segment_ids:
        raise ValueError("train_segment_ids must not be empty")
    indexed = manifest.assign(
        sample_id=manifest["sample_id"].astype(str)
    ).set_index("sample_id", drop=False)
    unknown = sorted(set(map(str, train_segment_ids)) - set(indexed.index))
    if unknown:
        raise ValueError(f"Unknown training segment IDs: {unknown[:10]}")

    count = 0
    mean = np.zeros(len(SIGNAL_CHANNELS), dtype=np.float64)
    m2 = np.zeros(len(SIGNAL_CHANNELS), dtype=np.float64)
    source_ids = tuple(sorted(map(str, train_segment_ids)))
    for segment_id in source_ids:
        path = Path(indexed.at[segment_id, "signal_path"])
        header = pd.read_csv(path, nrows=0)
        _validate_columns(path, header.columns.astype(str).tolist())
        for chunk in pd.read_csv(
            path,
            usecols=list(SIGNAL_CHANNELS),
            chunksize=chunk_size,
        ):
            values = chunk.loc[:, SIGNAL_CHANNELS].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Signal contains non-finite values: {path}")
            if len(values) == 0:
                continue
            batch_count = len(values)
            batch_mean = values.mean(axis=0)
            batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
            total = count + batch_count
            delta = batch_mean - mean
            mean = mean + delta * batch_count / total
            m2 = m2 + batch_m2 + delta**2 * count * batch_count / total
            count = total

    if count == 0:
        raise ValueError("Training signals contain no samples")
    std = np.sqrt(m2 / count)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Training signals contain a constant channel")
    return ChannelStats(
        mean=mean,
        std=std,
        sample_count=count,
        source_segment_ids=source_ids,
    )


def standardize_signal(signal: np.ndarray, stats: ChannelStats) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(SIGNAL_CHANNELS):
        raise ValueError("signal must have shape [samples, 3]")
    scaled = (values - stats.mean) / stats.std
    if not np.isfinite(scaled).all():
        raise ValueError("Standardized signal contains non-finite values")
    return scaled


def audit_signal_files(
    manifest: pd.DataFrame,
    sample_rate_hz: int,
    clipping_fraction_threshold: float = 0.01,
) -> dict:
    _require_manifest_columns(
        manifest, {"sample_id", "signal_path", "duration_s"}
    )
    if not 0 < clipping_fraction_threshold < 1:
        raise ValueError("clipping_fraction_threshold must be between 0 and 1")

    total_samples = 0
    duration_mismatch_count = 0
    maximum_duration_difference_s = 0.0
    channel_min = np.full(len(SIGNAL_CHANNELS), np.inf)
    channel_max = np.full(len(SIGNAL_CHANNELS), -np.inf)
    maximum_clipping_fraction = np.zeros(len(SIGNAL_CHANNELS))
    for row in manifest.itertuples(index=False):
        path = Path(row.signal_path)
        frame = pd.read_csv(path)
        _validate_columns(path, frame.columns.astype(str).tolist())
        if "Time_s" not in frame:
            raise ValueError(f"Missing Time_s column in {path}")
        values = frame.loc[:, SIGNAL_CHANNELS].to_numpy(dtype=np.float64)
        time_s = frame["Time_s"].to_numpy(dtype=np.float64)
        if len(values) == 0:
            raise ValueError(f"Empty signal file: {path}")
        if not np.isfinite(values).all() or not np.isfinite(time_s).all():
            raise ValueError(f"Signal contains non-finite values: {path}")
        expected_count = getattr(row, "sample_count", None)
        if expected_count is not None and not pd.isna(expected_count):
            expected_count = int(expected_count)
        else:
            expected_count = None
        if expected_count is not None and len(values) != expected_count:
            raise ValueError(
                f"Signal sample count mismatch for {row.sample_id}: "
                f"expected {expected_count}, got {len(values)}"
            )
        expected_interval = 1.0 / sample_rate_hz
        intervals = np.diff(time_s)
        if len(intervals) and not np.allclose(
            intervals, expected_interval, rtol=1e-7, atol=1e-12
        ):
            raise ValueError(
                f"Signal sampling interval mismatch for {row.sample_id}"
            )
        actual_duration_s = len(values) / sample_rate_hz
        duration_difference_s = abs(
            actual_duration_s - float(row.duration_s)
        )
        maximum_duration_difference_s = max(
            maximum_duration_difference_s, duration_difference_s
        )
        if duration_difference_s > expected_interval / 2:
            duration_mismatch_count += 1
        std = values.std(axis=0)
        if np.any(std <= 0):
            bad = [SIGNAL_CHANNELS[index] for index in np.flatnonzero(std <= 0)]
            raise ValueError(
                f"Signal contains constant channel for {row.sample_id}: {bad}"
            )
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        clipping_fraction = np.maximum(
            (values == minimum).mean(axis=0),
            (values == maximum).mean(axis=0),
        )
        if np.any(clipping_fraction > clipping_fraction_threshold):
            bad = {
                SIGNAL_CHANNELS[index]: float(clipping_fraction[index])
                for index in np.flatnonzero(
                    clipping_fraction > clipping_fraction_threshold
                )
            }
            raise ValueError(
                f"Signal clipping fraction exceeds threshold for "
                f"{row.sample_id}: {bad}"
            )
        total_samples += len(values)
        channel_min = np.minimum(channel_min, minimum)
        channel_max = np.maximum(channel_max, maximum)
        maximum_clipping_fraction = np.maximum(
            maximum_clipping_fraction, clipping_fraction
        )

    return {
        "segments": int(len(manifest)),
        "total_samples": int(total_samples),
        "duration_mismatch_count": int(duration_mismatch_count),
        "maximum_duration_difference_s": float(maximum_duration_difference_s),
        "channels": list(SIGNAL_CHANNELS),
        "channel_min": channel_min.tolist(),
        "channel_max": channel_max.tolist(),
        "maximum_clipping_fraction": maximum_clipping_fraction.tolist(),
        "sample_rate_hz": int(sample_rate_hz),
    }


def write_signal_artifacts(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    output_dir: str | Path,
    sample_rate_hz: int,
    clipping_fraction_threshold: float = 0.01,
) -> dict:
    _require_manifest_columns(manifest, {"sample_id", "signal_path", "duration_s"})
    missing_fold_columns = sorted({"sample_id", "fold"} - set(folds.columns))
    if missing_fold_columns:
        raise ValueError(f"Folds missing columns: {missing_fold_columns}")

    destination = Path(output_dir)
    fold_dir = destination / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_signal_files(
        manifest,
        sample_rate_hz=sample_rate_hz,
        clipping_fraction_threshold=clipping_fraction_threshold,
    )
    audit_path = destination / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_ids = set(manifest["sample_id"].astype(str))
    fold_frame = folds.assign(sample_id=folds["sample_id"].astype(str))
    fold_paths: dict[int, Path] = {}
    for fold in sorted(pd.to_numeric(fold_frame["fold"]).astype(int).unique()):
        test_ids = set(
            fold_frame.loc[fold_frame["fold"].astype(int) == fold, "sample_id"]
        )
        train_ids = manifest_ids - test_ids
        stats = fit_channel_stats(manifest, train_ids)
        fold_path = fold_dir / f"fold_{fold}_channel_stats.json"
        fold_path.write_text(
            json.dumps(stats.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fold_paths[int(fold)] = fold_path
    return {"audit": audit_path, "folds": fold_paths}
