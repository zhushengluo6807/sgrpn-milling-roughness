from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from roughness.scheme1.dataset import SegmentBagDataset, collate_segment_bags
from roughness.scheme1.signals import ChannelStats
from roughness.scheme1.windows import build_window_index


def _write_signal(path: Path, values: np.ndarray) -> None:
    frame = pd.DataFrame(values, columns=["Ch9_g", "Ch10_g", "Ch11_g"])
    frame.insert(0, "Time_s", np.arange(len(frame)) / 10.0)
    frame.to_csv(path, index=False)


def _manifest(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(len(paths))],
            "group_id": [f"g{index}" for index in range(len(paths))],
            "signal_path": paths,
            "n_rpm": [4000.0 + index * 1000 for index in range(len(paths))],
            "fz_mm_per_tooth": [0.03 + index * 0.01 for index in range(len(paths))],
            "ap_mm": [0.5 + index * 0.5 for index in range(len(paths))],
            "ra_mean": [0.8 + index * 0.1 for index in range(len(paths))],
            "sample_weight": [1.0] * len(paths),
        }
    )


def _identity_stats() -> ChannelStats:
    return ChannelStats(
        mean=np.zeros(3),
        std=np.ones(3),
        sample_count=1,
        source_segment_ids=("train",),
    )


def test_dataset_returns_one_target_for_all_windows_of_a_segment(tmp_path):
    path = tmp_path / "signal.csv"
    values = np.arange(21, dtype=float).reshape(7, 3)
    _write_signal(path, values)
    manifest = _manifest([path])
    window_index = build_window_index(manifest, 4, 4)
    dataset = SegmentBagDataset(manifest, window_index, _identity_stats())

    item = dataset[0]

    assert item["signal"].shape == (2, 3, 4)
    torch.testing.assert_close(
        item["signal"][0], torch.tensor(values[:4].T, dtype=torch.float32)
    )
    torch.testing.assert_close(
        item["signal"][1], torch.tensor(values[3:].T, dtype=torch.float32)
    )
    assert item["target"].item() == pytest.approx(0.8)
    assert item["sample_weight"].item() == 1.0
    assert item["segment_id"] == "s0"
    assert item["group_id"] == "g0"


def test_collate_pads_only_window_dimension_and_builds_mask(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_signal(first, np.arange(12, dtype=float).reshape(4, 3))
    _write_signal(second, np.arange(21, dtype=float).reshape(7, 3))
    manifest = _manifest([first, second])
    index = build_window_index(manifest, 4, 4)
    dataset = SegmentBagDataset(manifest, index, _identity_stats())

    batch = collate_segment_bags([dataset[0], dataset[1]])

    assert batch["signal"].shape == (2, 2, 3, 4)
    assert batch["window_mask"].tolist() == [[True, False], [True, True]]
    assert torch.count_nonzero(batch["signal"][0, 1]) == 0
    assert batch["target"].shape == (2,)
    assert batch["process"].shape == (2, 3)
    assert batch["physics"].shape == (2, 0)
    assert batch["base_ra"].shape == (2,)


def test_symmetric_horizontal_representation_is_invariant_to_channel_swap(
    tmp_path,
):
    original = tmp_path / "original.csv"
    swapped = tmp_path / "swapped.csv"
    values = np.array(
        [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0], [7.0, -8.0, 9.0], [2.0, 3.0, 4.0]]
    )
    swapped_values = values[:, [1, 0, 2]]
    _write_signal(original, values)
    _write_signal(swapped, swapped_values)

    first_manifest = _manifest([original])
    second_manifest = _manifest([swapped])
    first = SegmentBagDataset(
        first_manifest,
        build_window_index(first_manifest, 4, 4),
        _identity_stats(),
        horizontal_mode="symmetric",
    )[0]["signal"]
    second = SegmentBagDataset(
        second_manifest,
        build_window_index(second_manifest, 4, 4),
        _identity_stats(),
        horizontal_mode="symmetric",
    )[0]["signal"]

    torch.testing.assert_close(first, second)


def test_cached_dataset_does_not_reread_signal_file(tmp_path):
    path = tmp_path / "cached.csv"
    values = np.arange(12, dtype=float).reshape(4, 3)
    _write_signal(path, values)
    manifest = _manifest([path])
    index = build_window_index(manifest, 4, 4)

    dataset = SegmentBagDataset(
        manifest,
        index,
        _identity_stats(),
        cache_signals=True,
    )
    path.unlink()
    item = dataset[0]

    torch.testing.assert_close(
        item["signal"][0],
        torch.tensor(values.T, dtype=torch.float32),
    )
