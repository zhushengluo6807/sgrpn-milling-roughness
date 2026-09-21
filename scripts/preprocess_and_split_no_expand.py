# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "切削实验" / "截取后实验数据"
OUTPUT_DIR = ROOT / "切削实验" / "预处理并切分后实验数据_不扩展剔除"

FS = 25600
CHANNELS = ["Ch9_g", "Ch10_g", "Ch11_g"]

WINDOW_S = 0.05
WINDOW_N = int(FS * WINDOW_S)
MAD_K = 8.0
EXPAND_S = 0.0

HP_HZ = 20.0
LP_HZ = 10000.0

MIN_ONE_SEG_S = 1.0
MIN_TWO_SEG_S = 2.0
MIN_THREE_SEG_S = 3.0


def order_from_name(path: Path) -> int | None:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[0]) if nums else None


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-12:
        sigma = float(np.std(values))
    if sigma <= 1e-12:
        sigma = 1e-12
    return med, sigma


def detect_invalid_windows(arr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    n_full = len(arr) // WINDOW_N
    if n_full == 0:
        return np.zeros(0, dtype=bool), {
            "rms_threshold": np.nan,
            "p2p_threshold": np.nan,
            "invalid_window_count_raw": 0,
            "invalid_window_count_after_expand": 0,
        }

    block = arr[: n_full * WINDOW_N].reshape(n_full, WINDOW_N, arr.shape[1])
    resultant = np.sqrt(np.sum(block * block, axis=2))
    rms = np.sqrt(np.mean(resultant * resultant, axis=1))
    p2p = np.ptp(block, axis=1).max(axis=1)

    rms_med, rms_sigma = robust_scale(rms)
    p2p_med, p2p_sigma = robust_scale(p2p)
    rms_th = rms_med + MAD_K * rms_sigma
    p2p_th = p2p_med + MAD_K * p2p_sigma

    invalid = (rms > rms_th) | (p2p > p2p_th)
    raw_count = int(invalid.sum())
    return invalid, {
        "rms_threshold": float(rms_th),
        "p2p_threshold": float(p2p_th),
        "invalid_window_count_raw": raw_count,
        "invalid_window_count_after_expand": raw_count,
    }


def remove_invalid_windows(df: pd.DataFrame, invalid_windows: np.ndarray) -> tuple[pd.DataFrame, int]:
    if len(invalid_windows) == 0 or not invalid_windows.any():
        return df.copy(), 0

    keep = np.ones(len(df), dtype=bool)
    for i, bad in enumerate(invalid_windows):
        if bad:
            start = i * WINDOW_N
            end = min(len(df), (i + 1) * WINDOW_N)
            keep[start:end] = False
    removed = int((~keep).sum())
    return df.loc[keep].reset_index(drop=True), removed


def fft_bandpass(arr: np.ndarray) -> np.ndarray:
    if len(arr) < 4:
        return arr.copy()
    freqs = np.fft.rfftfreq(len(arr), d=1 / FS)
    mask = (freqs >= HP_HZ) & (freqs <= LP_HZ)
    out = np.empty_like(arr, dtype=np.float64)
    for c in range(arr.shape[1]):
        spec = np.fft.rfft(arr[:, c])
        spec[~mask] = 0
        out[:, c] = np.fft.irfft(spec, n=len(arr))
    return out


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少通道列: {missing}")

    arr = df[CHANNELS].to_numpy(dtype=np.float64)
    arr = arr - np.median(arr, axis=0)

    invalid_windows, shock_info = detect_invalid_windows(arr)
    cleaned_df, removed_points = remove_invalid_windows(df, invalid_windows)
    arr_clean = cleaned_df[CHANNELS].to_numpy(dtype=np.float64)
    arr_clean = arr_clean - np.median(arr_clean, axis=0)
    arr_filtered = fft_bandpass(arr_clean)

    out = cleaned_df.copy()
    out["Time_s"] = np.arange(len(out)) / FS
    for i, ch in enumerate(CHANNELS):
        out[ch] = arr_filtered[:, i]

    info = {
        **shock_info,
        "removed_points": removed_points,
        "removed_duration_s": removed_points / FS,
        "removed_ratio": removed_points / max(1, len(df)),
        "original_points": len(df),
        "preprocessed_points": len(out),
        "original_duration_s": len(df) / FS,
        "preprocessed_duration_s": len(out) / FS,
    }
    return out, info


def segment_count(duration_s: float) -> int:
    if duration_s >= MIN_THREE_SEG_S:
        return 3
    if duration_s >= MIN_TWO_SEG_S:
        return 2
    if duration_s >= MIN_ONE_SEG_S:
        return 1
    return 0


def split_indices(n_rows: int, n_seg: int) -> list[tuple[int, int]]:
    bounds = [round(i * n_rows / n_seg) for i in range(n_seg + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n_seg)]


def rel_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    preprocessed_dir = OUTPUT_DIR / "preprocessed_full_signals"
    segments_dir = OUTPUT_DIR / "segments"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        INPUT_DIR.rglob("*.csv"),
        key=lambda p: (str(p.parent), order_from_name(p) or 0, p.name),
    )
    summary_rows = []
    segment_rows = []

    for idx, src in enumerate(files, 1):
        rel_parent = src.parent.relative_to(INPUT_DIR)
        version = "v4" if rel_parent.parts and rel_parent.parts[0] == "500" else "v3"
        order = order_from_name(src)
        print(f"[{idx}/{len(files)}] {version} order {order}: {src.name}", flush=True)

        df = pd.read_csv(src)
        pre_df, info = preprocess(df)

        full_parent = preprocessed_dir / rel_parent
        seg_parent = segments_dir / rel_parent
        full_parent.mkdir(parents=True, exist_ok=True)
        seg_parent.mkdir(parents=True, exist_ok=True)

        full_out = full_parent / src.name
        pre_df.to_csv(full_out, index=False, encoding="utf-8-sig")

        n_seg = segment_count(info["preprocessed_duration_s"])
        if n_seg == 0:
            status = "预处理后过短未切分"
        elif n_seg == 1:
            status = "保留单段"
        elif n_seg == 2:
            status = "二等分"
        else:
            status = "三等分"

        summary_rows.append(
            {
                "版本": version,
                "执行顺序": order,
                "来源文件": rel_path(src),
                "预处理完整文件": rel_path(full_out),
                "原始采样点数": info["original_points"],
                "预处理后采样点数": info["preprocessed_points"],
                "原始时长(s)": round(info["original_duration_s"], 6),
                "预处理后时长(s)": round(info["preprocessed_duration_s"], 6),
                "剔除时长(s)": round(info["removed_duration_s"], 6),
                "剔除比例": round(info["removed_ratio"], 6),
                "原始异常窗口数": info["invalid_window_count_raw"],
                "扩展后异常窗口数": info["invalid_window_count_after_expand"],
                "切分段数": n_seg,
                "状态": status,
            }
        )

        if n_seg == 0:
            continue

        for seg_idx, (start, end) in enumerate(split_indices(len(pre_df), n_seg), 1):
            seg_df = pre_df.iloc[start:end].copy()
            out_name = f"{src.stem}_seg{seg_idx}.csv"
            out_path = seg_parent / out_name
            seg_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            segment_rows.append(
                {
                    "版本": version,
                    "执行顺序": order,
                    "片段编号": seg_idx,
                    "片段文件": rel_path(out_path),
                    "预处理完整文件": rel_path(full_out),
                    "起始采样点": start,
                    "结束采样点": end,
                    "采样点数": end - start,
                    "片段时长(s)": round((end - start) / FS, 6),
                    "group_id": f"{version}_{order}",
                }
            )

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "preprocess_split_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(OUTPUT_DIR / "preprocessed_segments_manifest.csv", index=False, encoding="utf-8-sig")

    config = {
        "input_file_count": len(files),
        "output_segment_count": len(segment_rows),
        "preprocess": {
            "fs": FS,
            "dc_removal": "per-channel median removal",
            "shock_detection": f"{WINDOW_S}s RMS/P2P MAD threshold, k={MAD_K}, expand={EXPAND_S}s",
            "filter": f"FFT band-pass {HP_HZ}-{LP_HZ} Hz",
        },
        "split_rule": {
            ">=3s": 3,
            "2-3s": 2,
            "1-2s": 1,
            "<1s": 0,
        },
    }
    (OUTPUT_DIR / "preprocess_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = pd.Series([r["切分段数"] for r in summary_rows]).value_counts().sort_index().to_dict()
    print("\nDONE")
    print(f"输入文件数: {len(files)}")
    print(f"输出片段数: {len(segment_rows)}")
    print(f"切分段数分布: {counts}")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
