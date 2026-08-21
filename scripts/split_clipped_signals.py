# -*- coding: utf-8 -*-
from pathlib import Path
import re
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

FS = 25600
MIN_ONE_SEG_S = 1.0
MIN_TWO_SEG_S = 2.0
MIN_THREE_SEG_S = 3.0


def locate_input_dir() -> Path:
    return next(p for p in Path(".").rglob("截取后实验数据") if p.is_dir())


def order_from_name(path: Path) -> int | None:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[0]) if nums else None


def segment_count(duration_s: float) -> int:
    if duration_s >= MIN_THREE_SEG_S:
        return 3
    if duration_s >= MIN_TWO_SEG_S:
        return 2
    if duration_s >= MIN_ONE_SEG_S:
        return 1
    return 0


def split_indices(n_rows: int, n_seg: int):
    bounds = [round(i * n_rows / n_seg) for i in range(n_seg + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n_seg)]


def main():
    input_dir = locate_input_dir()
    output_dir = input_dir.parent / "切分后三段实验数据"
    summary_rows = []
    segment_rows = []

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob("*.csv"), key=lambda p: (str(p.parent), order_from_name(p) or 0, p.name))
    for idx, src in enumerate(files, 1):
        rel_parent = src.parent.relative_to(input_dir)
        dst_parent = output_dir / rel_parent
        dst_parent.mkdir(parents=True, exist_ok=True)

        order = order_from_name(src)
        version = "v4" if rel_parent.parts and rel_parent.parts[0] == "500" else "v3"
        df = pd.read_csv(src)
        n_rows = len(df)
        duration_s = n_rows / FS
        n_seg = segment_count(duration_s)

        if n_seg == 0:
            status = "过短未切分"
        elif n_seg == 1:
            status = "保留单段"
        elif n_seg == 2:
            status = "二等分"
        else:
            status = "三等分"

        print(f"[{idx}/{len(files)}] {src.name}: {duration_s:.3f}s -> {n_seg} segment(s)", flush=True)

        summary_rows.append(
            {
                "版本": version,
                "执行顺序": order,
                "原文件": str(src),
                "采样点数": n_rows,
                "时长(s)": round(duration_s, 6),
                "切分段数": n_seg,
                "状态": status,
            }
        )

        if n_seg == 0:
            continue

        for seg_idx, (start, end) in enumerate(split_indices(n_rows, n_seg), 1):
            seg = df.iloc[start:end].copy()
            out_name = f"{src.stem}_seg{seg_idx}.csv"
            out_path = dst_parent / out_name
            seg.to_csv(out_path, index=False, encoding="utf-8-sig")
            segment_rows.append(
                {
                    "版本": version,
                    "执行顺序": order,
                    "片段编号": seg_idx,
                    "片段文件": str(out_path),
                    "来源文件": str(src),
                    "起始采样点": start,
                    "结束采样点": end,
                    "采样点数": end - start,
                    "片段时长(s)": round((end - start) / FS, 6),
                    "group_id": f"{version}_{order}",
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(output_dir / "split_segments_manifest.csv", index=False, encoding="utf-8-sig")

    total = len(files)
    out_segments = len(segment_rows)
    counts = pd.Series([r["切分段数"] for r in summary_rows]).value_counts().sort_index().to_dict()
    print("\nDONE")
    print(f"输入文件数: {total}")
    print(f"输出片段数: {out_segments}")
    print(f"切分段数分布: {counts}")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
