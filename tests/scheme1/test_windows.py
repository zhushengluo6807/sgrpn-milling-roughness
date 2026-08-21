from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roughness.scheme1.windows import build_window_index, make_windows


def _write_signal(path: Path, n_samples: int) -> None:
    frame = pd.DataFrame(
        {
            "Time_s": np.arange(n_samples) / 10.0,
            "Ch9_g": np.arange(n_samples, dtype=float),
            "Ch10_g": np.arange(n_samples, dtype=float) + 1,
            "Ch11_g": np.arange(n_samples, dtype=float) + 2,
        }
    )
    frame.to_csv(path, index=False)


def test_make_windows_uses_nonoverlap_and_one_tail_aligned_window():
    assert make_windows(4, window_samples=4, stride_samples=4) == [(0, 4)]
    assert make_windows(8, window_samples=4, stride_samples=4) == [
        (0, 4),
        (4, 8),
    ]
    assert make_windows(7, window_samples=4, stride_samples=4) == [
        (0, 4),
        (3, 7),
    ]


def test_make_windows_rejects_segments_shorter_than_one_window():
    with pytest.raises(ValueError, match="shorter than one window"):
        make_windows(3, window_samples=4, stride_samples=4)


def test_build_window_index_keeps_every_window_inside_its_segment(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_signal(first, 7)
    _write_signal(second, 8)
    manifest = pd.DataFrame(
        {
            "sample_id": ["first", "second"],
            "group_id": ["g1", "g2"],
            "signal_path": [first, second],
        }
    )

    index = build_window_index(
        manifest, window_samples=4, stride_samples=4
    )

    assert list(index.columns) == [
        "segment_id",
        "group_id",
        "csv_path",
        "window_id",
        "start_sample",
        "end_sample",
        "is_tail_aligned",
    ]
    first_rows = index[index["segment_id"] == "first"]
    assert first_rows[["start_sample", "end_sample"]].values.tolist() == [
        [0, 4],
        [3, 7],
    ]
    assert first_rows["is_tail_aligned"].tolist() == [False, True]
    assert (index["start_sample"] >= 0).all()
