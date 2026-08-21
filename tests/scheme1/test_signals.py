import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roughness.scheme1.signals import (
    SIGNAL_CHANNELS,
    audit_signal_files,
    fit_channel_stats,
    load_signal_csv,
    standardize_signal,
    write_signal_artifacts,
)


def _write_signal(path: Path, values: np.ndarray, sample_rate_hz: int = 10) -> None:
    frame = pd.DataFrame(values, columns=list(SIGNAL_CHANNELS))
    frame.insert(0, "Time_s", np.arange(len(frame)) / sample_rate_hz)
    frame.to_csv(path, index=False)


def test_load_signal_preserves_rows_and_fixed_channel_order(tmp_path):
    path = tmp_path / "signal.csv"
    values = np.array(
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]]
    )
    _write_signal(path, values)

    loaded = load_signal_csv(path)

    assert loaded.shape == (3, 3)
    np.testing.assert_allclose(loaded, values)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame(
                {
                    "Time_s": [0.0, 0.1],
                    "Ch9_g": [1.0, np.nan],
                    "Ch10_g": [2.0, 3.0],
                    "Ch11_g": [4.0, 5.0],
                }
            ),
            "non-finite",
        ),
        (
            pd.DataFrame(
                {
                    "Time_s": [0.0, 0.1],
                    "Ch9_g": [1.0, 2.0],
                    "Ch10_g": [2.0, 3.0],
                }
            ),
            "Missing signal columns",
        ),
    ],
)
def test_load_signal_rejects_invalid_files(tmp_path, frame, message):
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_signal_csv(path)


def test_fit_channel_stats_uses_only_requested_training_segments(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    held_out = tmp_path / "held_out.csv"
    _write_signal(first, np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]))
    _write_signal(second, np.array([[4.0, 5.0, 6.0], [6.0, 7.0, 8.0]]))
    _write_signal(held_out, np.full((2, 3), 1000.0))
    manifest = pd.DataFrame(
        {
            "sample_id": ["first", "second", "held_out"],
            "signal_path": [first, second, held_out],
        }
    )

    stats = fit_channel_stats(manifest, {"first", "second"}, chunk_size=1)

    assert stats.source_segment_ids == ("first", "second")
    assert stats.sample_count == 4
    np.testing.assert_allclose(stats.mean, [3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        stats.std, np.std([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]], axis=0)
    )


def test_standardization_uses_training_global_stats_not_window_centering(tmp_path):
    train = tmp_path / "train.csv"
    _write_signal(train, np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]))
    manifest = pd.DataFrame(
        {"sample_id": ["train"], "signal_path": [train]}
    )
    stats = fit_channel_stats(manifest, {"train"})
    shifted_window = np.array([[10.0, 10.0, 10.0], [12.0, 12.0, 12.0]])

    scaled = standardize_signal(shifted_window, stats)

    np.testing.assert_allclose(scaled, [[9.0, 9.0, 9.0], [11.0, 11.0, 11.0]])
    assert not np.allclose(scaled.mean(axis=0), 0.0)


def test_audit_uses_explicit_sample_count_and_only_reports_duration_difference(
    tmp_path,
):
    wrong_count = tmp_path / "wrong_count.csv"
    duration_differs = tmp_path / "duration_differs.csv"
    _write_signal(wrong_count, np.arange(300, dtype=float).reshape(100, 3))
    _write_signal(
        duration_differs, np.arange(300, dtype=float).reshape(100, 3)
    )

    with pytest.raises(ValueError, match="sample count"):
        audit_signal_files(
            pd.DataFrame(
                {
                    "sample_id": ["wrong"],
                    "signal_path": [wrong_count],
                    "duration_s": [10.0],
                    "sample_count": [101],
                }
            ),
            sample_rate_hz=10,
        )

    audit = audit_signal_files(
        pd.DataFrame(
            {
                "sample_id": ["duration_differs"],
                "signal_path": [duration_differs],
                "duration_s": [9.9],
            }
        ),
        sample_rate_hz=10,
    )

    assert audit["duration_mismatch_count"] == 1
    assert audit["maximum_duration_difference_s"] == pytest.approx(0.1)


def test_audit_rejects_constant_channel_and_clipping(tmp_path):
    constant = tmp_path / "constant.csv"
    clipped = tmp_path / "clipped.csv"
    _write_signal(
        constant,
        np.column_stack(
            [np.ones(10), np.arange(10, dtype=float), np.arange(10, dtype=float)]
        ),
    )
    clipped_values = np.column_stack(
        [
            np.array([0.0] * 8 + [1.0, 2.0]),
            np.arange(10, dtype=float),
            np.arange(10, dtype=float) + 1.0,
        ]
    )
    _write_signal(clipped, clipped_values)

    with pytest.raises(ValueError, match="constant channel"):
        audit_signal_files(
            pd.DataFrame(
                {
                    "sample_id": ["constant"],
                    "signal_path": [constant],
                    "duration_s": [1.0],
                }
            ),
            sample_rate_hz=10,
        )
    with pytest.raises(ValueError, match="clipping"):
        audit_signal_files(
            pd.DataFrame(
                {
                    "sample_id": ["clipped"],
                    "signal_path": [clipped],
                    "duration_s": [1.0],
                }
            ),
            sample_rate_hz=10,
            clipping_fraction_threshold=0.5,
        )


def test_audit_rejects_nonuniform_time_axis(tmp_path):
    path = tmp_path / "bad_time.csv"
    values = np.arange(300, dtype=float).reshape(100, 3)
    _write_signal(path, values)
    frame = pd.read_csv(path)
    frame.loc[50:, "Time_s"] += 0.01
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="sampling interval"):
        audit_signal_files(
            pd.DataFrame(
                {
                    "sample_id": ["bad_time"],
                    "signal_path": [path],
                    "duration_s": [10.0],
                }
            ),
            sample_rate_hz=10,
        )


def test_write_signal_artifacts_uses_only_each_fold_training_segments(tmp_path):
    paths = []
    for index, offset in enumerate((0.0, 10.0, 20.0, 30.0)):
        path = tmp_path / f"{index}.csv"
        values = np.arange(300, dtype=float).reshape(100, 3) + offset
        _write_signal(path, values)
        paths.append(path)
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "signal_path": paths,
            "duration_s": [10.0] * 4,
        }
    )
    folds = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "group_id": ["a", "b", "c", "d"],
            "fold": [0, 0, 1, 1],
        }
    )

    paths_written = write_signal_artifacts(
        manifest,
        folds,
        output_dir=tmp_path / "output",
        sample_rate_hz=10,
        clipping_fraction_threshold=0.02,
    )

    assert paths_written["audit"].is_file()
    fold_0 = json.loads(paths_written["folds"][0].read_text(encoding="utf-8"))
    fold_1 = json.loads(paths_written["folds"][1].read_text(encoding="utf-8"))
    assert fold_0["source_segment_ids"] == ["c", "d"]
    assert fold_1["source_segment_ids"] == ["a", "b"]
