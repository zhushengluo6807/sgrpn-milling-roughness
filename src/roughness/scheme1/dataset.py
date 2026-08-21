from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .signals import ChannelStats, load_signal_csv, standardize_signal


_MANIFEST_COLUMNS = {
    "sample_id",
    "group_id",
    "signal_path",
    "n_rpm",
    "fz_mm_per_tooth",
    "ap_mm",
    "ra_mean",
    "sample_weight",
}


def _horizontal_representation(
    signal: np.ndarray, mode: str
) -> np.ndarray:
    if mode in {"A", "raw"}:
        return signal
    if mode == "B":
        return signal[:, [1, 0, 2]]
    if mode == "symmetric":
        first = np.sqrt((signal[:, 0] ** 2 + signal[:, 1] ** 2) / 2.0)
        second = (np.abs(signal[:, 0]) + np.abs(signal[:, 1])) / 2.0
        return np.column_stack([first, second, signal[:, 2]])
    raise ValueError("horizontal_mode must be one of A, B, raw, symmetric")


class SegmentBagDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        window_index: pd.DataFrame,
        channel_stats: ChannelStats,
        horizontal_mode: str = "A",
        physics_columns: Sequence[str] = (),
        base_ra_column: str | None = None,
        cache_signals: bool = False,
    ) -> None:
        missing = sorted(_MANIFEST_COLUMNS - set(manifest.columns))
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        required_windows = {
            "segment_id",
            "start_sample",
            "end_sample",
        }
        missing_windows = sorted(required_windows - set(window_index.columns))
        if missing_windows:
            raise ValueError(f"Window index missing columns: {missing_windows}")
        if horizontal_mode not in {"A", "B", "raw", "symmetric"}:
            raise ValueError("Unknown horizontal_mode")
        absent_physics = sorted(set(physics_columns) - set(manifest.columns))
        if absent_physics:
            raise ValueError(f"Missing physics columns: {absent_physics}")
        if base_ra_column is not None and base_ra_column not in manifest:
            raise ValueError(f"Missing base Ra column: {base_ra_column}")

        self._manifest = manifest.copy()
        self._manifest["sample_id"] = self._manifest["sample_id"].astype(str)
        self._manifest = self._manifest.set_index("sample_id", drop=False)
        self._segment_ids = self._manifest.index.tolist()
        self._windows = {
            str(segment_id): group.sort_values("window_id")
            for segment_id, group in window_index.groupby(
                "segment_id", sort=False
            )
        }
        missing_index = sorted(set(self._segment_ids) - set(self._windows))
        if missing_index:
            raise ValueError(f"Segments without windows: {missing_index[:10]}")
        self._stats = channel_stats
        self._horizontal_mode = horizontal_mode
        self._physics_columns = tuple(physics_columns)
        self._base_ra_column = base_ra_column
        self._signal_cache: dict[str, np.ndarray] | None = (
            {} if cache_signals else None
        )
        if self._signal_cache is not None:
            for segment_id in self._segment_ids:
                row = self._manifest.loc[segment_id]
                self._signal_cache[segment_id] = self._load_prepared_signal(
                    row["signal_path"]
                )

    def _load_prepared_signal(self, path) -> np.ndarray:
        signal = load_signal_csv(path)
        signal = standardize_signal(signal, self._stats)
        signal = _horizontal_representation(signal, self._horizontal_mode)
        return signal.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self._segment_ids)

    def __getitem__(self, index: int) -> dict:
        segment_id = self._segment_ids[index]
        row = self._manifest.loc[segment_id]
        signal = (
            self._signal_cache[segment_id]
            if self._signal_cache is not None
            else self._load_prepared_signal(row["signal_path"])
        )
        windows = self._windows[segment_id]
        bag = np.stack(
            [
                signal[int(window.start_sample) : int(window.end_sample)].T
                for window in windows.itertuples(index=False)
            ],
            axis=0,
        )
        physics = np.asarray(
            [row[column] for column in self._physics_columns],
            dtype=np.float32,
        )
        base_ra = (
            float(row[self._base_ra_column])
            if self._base_ra_column is not None
            else float("nan")
        )
        return {
            "signal": torch.as_tensor(bag, dtype=torch.float32),
            "process": torch.tensor(
                [
                    row["n_rpm"],
                    row["fz_mm_per_tooth"],
                    row["ap_mm"],
                ],
                dtype=torch.float32,
            ),
            "physics": torch.as_tensor(physics, dtype=torch.float32),
            "base_ra": torch.tensor(base_ra, dtype=torch.float32),
            "target": torch.tensor(float(row["ra_mean"]), dtype=torch.float32),
            "sample_weight": torch.tensor(
                float(row["sample_weight"]), dtype=torch.float32
            ),
            "segment_id": segment_id,
            "group_id": str(row["group_id"]),
        }


def collate_segment_bags(items: list[dict]) -> dict:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(items)
    max_windows = max(item["signal"].shape[0] for item in items)
    channels = items[0]["signal"].shape[1]
    samples = items[0]["signal"].shape[2]
    signal = torch.zeros(
        (batch_size, max_windows, channels, samples), dtype=torch.float32
    )
    mask = torch.zeros((batch_size, max_windows), dtype=torch.bool)
    for index, item in enumerate(items):
        window_count = item["signal"].shape[0]
        signal[index, :window_count] = item["signal"]
        mask[index, :window_count] = True
    return {
        "signal": signal,
        "window_mask": mask,
        "process": torch.stack([item["process"] for item in items]),
        "physics": torch.stack([item["physics"] for item in items]),
        "base_ra": torch.stack([item["base_ra"] for item in items]),
        "target": torch.stack([item["target"] for item in items]),
        "sample_weight": torch.stack(
            [item["sample_weight"] for item in items]
        ),
        "segment_id": [item["segment_id"] for item in items],
        "group_id": [item["group_id"] for item in items],
    }
