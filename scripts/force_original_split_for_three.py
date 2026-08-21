# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"E:\CodeX\机床项目")
OUT_DIR = ROOT / "切削实验" / "预处理并切分后实验数据_不扩展剔除"
SUMMARY = OUT_DIR / "preprocess_split_summary.csv"
MANIFEST = OUT_DIR / "preprocessed_segments_manifest.csv"
FS = 25600

FORCED = {
    ("v4", 102): 2,
    ("v3", 1): 2,
    ("v3", 3): 1,
}


def order_from_name(path: Path) -> int | None:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[0]) if nums else None


def split_indices(n_rows: int, n_seg: int) -> list[tuple[int, int]]:
    bounds = [round(i * n_rows / n_seg) for i in range(n_seg + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n_seg)]


def rel_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def full_signal_path(version: str, order: int) -> Path:
    folder = "500" if version == "v4" else "默认"
    return OUT_DIR / "preprocessed_full_signals" / folder / f"{order}.csv"


def segment_dir(version: str) -> Path:
    folder = "500" if version == "v4" else "默认"
    return OUT_DIR / "segments" / folder


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    manifest = pd.read_csv(MANIFEST)

    forced_keys = set(FORCED)
    keep_mask = ~manifest.apply(lambda r: (r["版本"], int(r["执行顺序"])) in forced_keys, axis=1)
    manifest = manifest.loc[keep_mask].copy()

    new_rows = []
    for (version, order), n_seg in FORCED.items():
        full_path = full_signal_path(version, order)
        df = pd.read_csv(full_path)
        seg_parent = segment_dir(version)
        seg_parent.mkdir(parents=True, exist_ok=True)

        # Remove stale segment files for the three forced groups.
        for old in seg_parent.glob(f"{order}_seg*.csv"):
            old.unlink()

        for seg_idx, (start, end) in enumerate(split_indices(len(df), n_seg), 1):
            seg_df = df.iloc[start:end].copy()
            out_path = seg_parent / f"{order}_seg{seg_idx}.csv"
            seg_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            new_rows.append(
                {
                    "版本": version,
                    "执行顺序": order,
                    "片段编号": seg_idx,
                    "片段文件": rel_path(out_path),
                    "预处理完整文件": rel_path(full_path),
                    "起始采样点": start,
                    "结束采样点": end,
                    "采样点数": end - start,
                    "片段时长(s)": round((end - start) / FS, 6),
                    "group_id": f"{version}_{order}",
                }
            )

        row_mask = (summary["版本"] == version) & (summary["执行顺序"].astype(int) == order)
        status = "保留单段" if n_seg == 1 else "二等分" if n_seg == 2 else "三等分"
        summary.loc[row_mask, "切分段数"] = n_seg
        summary.loc[row_mask, "状态"] = status

    manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
    manifest = manifest.sort_values(["版本", "执行顺序", "片段编号"], ascending=[True, True, True])
    # Keep v4/500 rows first, matching the existing directory traversal order.
    manifest["_version_order"] = manifest["版本"].map({"v4": 0, "v3": 1}).fillna(2)
    manifest = manifest.sort_values(["_version_order", "执行顺序", "片段编号"]).drop(columns=["_version_order"])

    summary["_version_order"] = summary["版本"].map({"v4": 0, "v3": 1}).fillna(2)
    summary = summary.sort_values(["_version_order", "执行顺序"]).drop(columns=["_version_order"])

    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")

    print(f"summary_rows={len(summary)}")
    print(f"manifest_rows={len(manifest)}")


if __name__ == "__main__":
    main()
