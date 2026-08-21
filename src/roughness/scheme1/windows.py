from pathlib import Path

import pandas as pd


WINDOW_INDEX_COLUMNS = [
    "segment_id",
    "group_id",
    "csv_path",
    "window_id",
    "start_sample",
    "end_sample",
    "is_tail_aligned",
]


def make_windows(
    n_samples: int,
    window_samples: int,
    stride_samples: int,
) -> list[tuple[int, int]]:
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples and stride_samples must be positive")
    if n_samples < window_samples:
        raise ValueError(
            f"Segment with {n_samples} samples is shorter than one window "
            f"of {window_samples} samples"
        )
    starts = list(range(0, n_samples - window_samples + 1, stride_samples))
    tail_start = n_samples - window_samples
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return [(start, start + window_samples) for start in starts]


def build_window_index(
    manifest: pd.DataFrame,
    window_samples: int,
    stride_samples: int,
) -> pd.DataFrame:
    required = {"sample_id", "group_id", "signal_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in manifest")

    records = []
    for row in manifest.itertuples(index=False):
        path = Path(row.signal_path)
        sample_count = len(pd.read_csv(path, usecols=["Time_s"]))
        windows = make_windows(sample_count, window_samples, stride_samples)
        regular_starts = set(
            range(0, sample_count - window_samples + 1, stride_samples)
        )
        for window_id, (start, end) in enumerate(windows):
            records.append(
                {
                    "segment_id": str(row.sample_id),
                    "group_id": str(row.group_id),
                    "csv_path": str(path),
                    "window_id": window_id,
                    "start_sample": start,
                    "end_sample": end,
                    "is_tail_aligned": start not in regular_starts,
                }
            )
    return pd.DataFrame.from_records(records, columns=WINDOW_INDEX_COLUMNS)

