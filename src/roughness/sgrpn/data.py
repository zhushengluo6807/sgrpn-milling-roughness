from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from roughness.scheme1.folds import validate_outer_folds
from roughness.scheme1.signals import load_signal_csv

from .config import SGRPNConfig


_MANIFEST_COLUMNS = {
    "sample_id",
    "group_id",
    "signal_path",
    "n_rpm",
    "fz_mm_per_tooth",
    "ap_mm",
    "ra_1",
    "ra_2",
    "ra_3",
    "ra_mean",
    "sample_weight",
    "split_count",
    "version",
    "duration_s",
}
_PROCESS_COLUMNS = ("n_rpm", "fz_mm_per_tooth", "ap_mm")
_FINITE_COLUMNS = (
    *_PROCESS_COLUMNS,
    "ra_1",
    "ra_2",
    "ra_3",
    "ra_mean",
    "sample_weight",
    "split_count",
    "duration_s",
)
_WINDOW_COLUMNS = {"segment_id", "group_id", "csv_path", "window_id", "start_sample", "end_sample", "is_tail_aligned"}


@dataclass(frozen=True)
class DataBundle:
    manifest: pd.DataFrame
    folds: pd.DataFrame
    windows: pd.DataFrame
    fold_audit: dict[str, int]
    duration_audit: pd.DataFrame


def recompute_duration(row_count: int, sample_rate_hz: int) -> float:
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if sample_rate_hz != 25600:
        raise ValueError("Phase A sample_rate_hz must be exactly 25600")
    return float(row_count) / 25600.0


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _finite_values(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> np.ndarray:
    try:
        values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} numeric columns must be finite") from error
    finite = np.isfinite(values)
    if not finite.all():
        bad_columns = sorted({columns[index] for index in np.flatnonzero(~finite.all(axis=0))})
        raise ValueError(f"{name} numeric columns must be finite: {bad_columns}")
    return values


def build_process_features(frame: pd.DataFrame) -> np.ndarray:
    _require_columns(frame, set(_PROCESS_COLUMNS), "Process frame")
    n_rpm, feed, depth = _finite_values(frame, _PROCESS_COLUMNS, "Process frame").T
    return np.column_stack(
        (
            n_rpm,
            feed,
            depth,
            n_rpm**2,
            feed**2,
            depth**2,
            n_rpm * feed,
            n_rpm * depth,
            feed * depth,
        )
    )


def validate_group_split(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    train = set(map(str, np.asarray(train_groups).ravel()))
    test = set(map(str, np.asarray(test_groups).ravel()))
    overlap = sorted(train & test)
    if overlap:
        raise ValueError(f"group leakage between train and test: {overlap[:10]}")


def _validate_manifest(manifest: pd.DataFrame) -> None:
    _require_columns(manifest, _MANIFEST_COLUMNS, "Manifest")
    if manifest["sample_id"].isna().any() or manifest["group_id"].isna().any():
        raise ValueError("Manifest sample_id and group_id must not be missing")
    if manifest["sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate manifest sample_id values")
    if manifest["signal_path"].isna().any():
        raise ValueError("Manifest signal_path must not be missing")
    _finite_values(manifest, _FINITE_COLUMNS, "Manifest")
    if (manifest["split_count"].astype(float) <= 0).any():
        raise ValueError("Manifest split_count must be positive")
    if (manifest["sample_weight"].astype(float) <= 0).any():
        raise ValueError("Manifest sample_weight must be positive")
    if (manifest["duration_s"].astype(float) < 0).any():
        raise ValueError("Manifest duration_s must be non-negative")


def _validate_windows(manifest: pd.DataFrame, windows: pd.DataFrame) -> None:
    _require_columns(windows, _WINDOW_COLUMNS, "Window index")
    manifest_groups = manifest.assign(
        sample_id=manifest["sample_id"].astype(str)
    ).set_index("sample_id")["group_id"].astype(str)
    segment_ids = windows["segment_id"].astype(str)
    unknown = sorted(set(segment_ids) - set(manifest_groups.index.astype(str)))
    if unknown:
        raise ValueError(f"Window index contains unknown segment IDs: {unknown[:10]}")
    actual_groups = windows["group_id"].astype(str).to_numpy()
    expected_groups = manifest_groups.reindex(segment_ids).to_numpy()
    if not np.array_equal(actual_groups, expected_groups):
        raise ValueError("Window group_id does not match manifest")


def _build_duration_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in manifest.itertuples(index=False):
        signal_path = Path(row.signal_path)
        row_count = int(len(load_signal_csv(signal_path)))
        recomputed = recompute_duration(row_count, 25600)
        source = float(row.duration_s)
        difference = abs(source - recomputed)
        records.append(
            {
                "sample_id": str(row.sample_id),
                "row_count": row_count,
                "duration_source_s": source,
                "duration_recomputed_s": recomputed,
                "absolute_difference_s": difference,
                "mismatch_over_1ms": difference > 0.001,
            }
        )
    return pd.DataFrame.from_records(records)


def load_data_bundle(config: SGRPNConfig) -> DataBundle:
    if config.sample_rate_hz != 25600:
        raise ValueError("Phase A sample_rate_hz must be exactly 25600")
    manifest = pd.read_csv(config.manifest_path)
    folds = pd.read_csv(config.folds_path)
    windows = pd.read_csv(config.window_index_path)
    _validate_manifest(manifest)
    _validate_windows(manifest, windows)
    fold_audit = validate_outer_folds(manifest, folds)
    duration_audit = _build_duration_audit(manifest)
    return DataBundle(
        manifest=manifest,
        folds=folds,
        windows=windows,
        fold_audit=fold_audit,
        duration_audit=duration_audit,
    )


def outer_indices(bundle: DataBundle, fold: int) -> tuple[np.ndarray, np.ndarray]:
    fold_values = pd.to_numeric(bundle.folds["fold"], errors="raise").astype(int)
    if int(fold) not in set(fold_values):
        raise ValueError(f"Unknown outer fold: {fold}")
    fold_by_sample = pd.Series(
        fold_values.to_numpy(), index=bundle.folds["sample_id"].astype(str)
    )
    assignments = fold_by_sample.reindex(bundle.manifest["sample_id"].astype(str))
    if assignments.isna().any():
        raise ValueError("Missing fold assignments")
    test = np.flatnonzero(assignments.to_numpy(dtype=int) == int(fold))
    train = np.flatnonzero(assignments.to_numpy(dtype=int) != int(fold))
    validate_group_split(
        bundle.manifest.iloc[train]["group_id"].to_numpy(),
        bundle.manifest.iloc[test]["group_id"].to_numpy(),
    )
    return train, test


def write_data_audit(bundle: DataBundle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle.duration_audit.to_csv(destination, index=False)
    return destination
