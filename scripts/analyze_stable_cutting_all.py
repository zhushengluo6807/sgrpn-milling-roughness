# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FS = 25600
WIN_S = 0.1
WIN_N = int(FS * WIN_S)


def locate_data_root() -> Path:
    for parent in sorted({p.parent for p in Path(".").rglob("*.txt")}, key=lambda p: len(str(p))):
        names = [p.name for p in parent.glob("*.txt")]
        if len(names) == 31 and any(n.startswith("500-") for n in names) and any(n.startswith("1-1") for n in names):
            return parent
    raise RuntimeError("未找到包含 31 个 txt 的实验数据0611文件夹")


def find_xlsx(version: str) -> Path:
    return next(p for p in Path(".").rglob("*.xlsx") if version in p.name)


def load_params(xlsx: Path) -> dict[int, dict[str, float | int | None]]:
    raw = pd.read_excel(xlsx, sheet_name=0, header=2)
    raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()].copy()
    raw.iloc[:, 0] = raw.iloc[:, 0].astype(int)
    params: dict[int, dict[str, float | int | None]] = {}
    for _, row in raw.iterrows():
        order = int(row.iloc[0])
        theory = row.iloc[18]
        if pd.isna(theory):
            continue
        params[order] = {
            "block": int(row.iloc[2]) if not pd.isna(row.iloc[2]) else None,
            "layer": int(row.iloc[4]) if not pd.isna(row.iloc[4]) else None,
            "slot": int(row.iloc[5]) if not pd.isna(row.iloc[5]) else None,
            "rpm": int(row.iloc[6]),
            "fz": float(row.iloc[7]),
            "feed_rev": float(row.iloc[8]),
            "ap": float(row.iloc[9]),
            "theory": float(theory),
        }
    return params


RANGE_RE = re.compile(r"^(?:500-)?(\d+)-(\d+).*?(\d+)\D+(\d+)")
NUM_RE = re.compile(r"^(\d+)\.txt$")


def file_meta(path: Path, data_root: Path) -> dict[str, object]:
    name = path.name
    version = "v4" if name.startswith("500-") else "v3"
    single = NUM_RE.match(name)
    if single and path.parent == data_root:
        order = int(single.group(1))
        return {"version": version, "orders": [order], "label": f"{version}_{order:03d}", "kind": "single"}
    ranged = RANGE_RE.match(name)
    if ranged:
        start = int(ranged.group(3))
        end = int(ranged.group(4))
        return {
            "version": version,
            "orders": list(range(start, end + 1)),
            "label": f"{version}_{start:03d}_{end:03d}",
            "kind": "range",
        }
    raise ValueError(f"无法解析文件名: {name}")


def moving_mean(x: np.ndarray, n: int = 5) -> np.ndarray:
    if len(x) == 0:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def load_rms(path: Path) -> tuple[np.ndarray, float, float]:
    rms_values: list[float] = []
    carry = np.empty((0, 3), dtype=np.float64)
    n_total = 0
    max_abs = 0.0

    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=2,
        header=None,
        usecols=[1, 2, 3],
        chunksize=800000,
        engine="c",
    ):
        arr = chunk.to_numpy(dtype=np.float64)
        arr = arr - np.median(arr, axis=0)
        max_abs = max(max_abs, float(np.max(np.abs(arr))))

        if len(carry):
            arr = np.vstack([carry, arr])

        n_full = (len(arr) // WIN_N) * WIN_N
        if n_full:
            block = arr[:n_full].reshape(-1, WIN_N, 3)
            rms_values.extend(np.sqrt(np.mean(np.sum(block * block, axis=2), axis=1)).tolist())
            n_total += n_full
        carry = arr[n_full:]

    if len(carry):
        rms_values.append(float(np.sqrt(np.mean(np.sum(carry * carry, axis=1)))))
        n_total += len(carry)

    return np.array(rms_values), n_total / FS, max_abs


def segments(mask: np.ndarray, min_dur: float = 1.0, merge_gap: float = 0.5) -> list[tuple[int, int]]:
    raw: list[list[int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        raw.append([i, j])
        i = j

    gap_w = int(round(merge_gap / WIN_S))
    merged: list[list[int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= gap_w:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    min_w = int(round(min_dur / WIN_S))
    return [(start, end) for start, end in merged if end - start >= min_w]


def pick_broad_segments(smoothed: np.ndarray, expected: int) -> tuple[list[tuple[int, int]], str, bool]:
    candidates = []
    for q in [5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50, 55, 60]:
        threshold = np.percentile(smoothed, q)
        segs = segments(smoothed > threshold, min_dur=0.7 if expected == 1 else 1.0, merge_gap=0.5)
        if segs:
            total = sum(end - start for start, end in segs) * WIN_S
            candidates.append((abs(len(segs) - expected), abs(q - 10), -total, q, segs))

    if not candidates:
        return [(0, len(smoothed))], "整段回退", False

    candidates.sort(key=lambda item: item[:3])
    _, _, _, q, segs = candidates[0]
    ok = len(segs) == expected
    return (segs[:expected] if len(segs) > expected else segs), f"能量阈值q{q}", ok


def proportional_segments(smoothed: np.ndarray, orders: list[int], version: str, params: dict[str, dict[int, dict]]) -> list[tuple[int, int]]:
    theories = [float(params[version][order]["theory"]) for order in orders]
    active = segments(smoothed > np.percentile(smoothed, 10), min_dur=1.0, merge_gap=1.0)
    if active:
        start_s = active[0][0] * WIN_S
        end_s = active[-1][1] * WIN_S
    else:
        start_s = 0.0
        end_s = len(smoothed) * WIN_S

    total_theory = sum(theories)
    available_gap = max(0.0, end_s - start_s - total_theory)
    gap = available_gap / (len(orders) + 1)
    result = []
    cur = start_s + gap
    for theory in theories:
        start = int(round(cur / WIN_S))
        end = int(round(min(end_s, cur + theory) / WIN_S))
        result.append((start, max(start + 1, end)))
        cur = end * WIN_S + gap
    return result


def best_window(smoothed: np.ndarray, theory_s: float, allowed: tuple[int, int] | None) -> tuple[float, float, float, float]:
    width = max(1, int(round(theory_s / WIN_S)))
    if width >= len(smoothed):
        mean = float(np.mean(smoothed))
        return 0.0, len(smoothed) * WIN_S, mean, float(np.std(smoothed) / (mean + 1e-12))

    score = np.minimum(smoothed, np.percentile(smoothed, 95))
    if allowed is None:
        lo, hi = 0, len(score) - width
    else:
        lo = max(0, int(allowed[0]))
        hi = min(len(score) - width, int(allowed[1]) - width)

    if hi < lo:
        center = ((allowed[0] + allowed[1]) / 2) if allowed else len(score) / 2
        lo = max(0, min(len(score) - width, int(round(center - width / 2))))
        hi = lo

    cumsum = np.concatenate([[0.0], np.cumsum(score)])
    means = (cumsum[lo + width : hi + width + 1] - cumsum[lo : hi + 1]) / width
    idx = lo + int(np.argmax(means))
    seg = smoothed[idx : idx + width]
    mean = float(np.mean(seg))
    cv = float(np.std(seg) / (mean + 1e-12))
    return idx * WIN_S, (idx + width) * WIN_S, mean, cv


def confidence(cand_dur: float, theory: float, cv: float, max_abs: float, method: str, count_ok: bool) -> tuple[str, str]:
    conf = "高"
    notes = []
    diff = abs(cand_dur - theory)
    if diff > max(1.5, theory * 0.35):
        conf = "低"
        notes.append("候选时长与理论切削时间差异较大")
    elif diff > max(0.8, theory * 0.2):
        conf = "中"
        notes.append("候选时长与理论切削时间略有差异")

    if cv > 0.25:
        conf = "低" if conf != "高" else "中"
        notes.append("候选段内部能量波动较大")
    if max_abs > 1.0:
        conf = "低"
        notes.append("文件存在明显冲击尖峰")
    if not count_ok:
        conf = "低" if conf != "高" else "中"
        notes.append("活动段数量与理论执行顺序数量不完全一致")
    if "比例回退" in method:
        conf = "低"
        notes.append("使用理论时长比例回退拆分")

    if not notes:
        notes.append("候选稳定核心清楚，时长与理论切削时间接近")
    return conf, "；".join(notes)


def make_svg(t: np.ndarray, smoothed: np.ndarray, title: str, bands: list[tuple[str, float, float, str]], outpath: Path) -> None:
    w, h = 1300, 360
    pl, pr, pt, pb = 70, 25, 35, 45
    ymax = max(float(np.percentile(smoothed, 99.5)), 1e-9)
    yplot = np.minimum(smoothed, ymax)

    def sx(value: float) -> float:
        return pl + (w - pl - pr) * value / max(float(t[-1]), 1e-9)

    def sy(value: float) -> float:
        return pt + (h - pt - pb) * (1 - value / ymax)

    pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(t, yplot))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pl}" y="22" font-family="Arial" font-size="16">{html.escape(title)}</text>',
        f'<line x1="{pl}" y1="{h-pb}" x2="{w-pr}" y2="{h-pb}" stroke="#374151"/>',
        f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{h-pb}" stroke="#374151"/>',
    ]
    step = 10 if t[-1] <= 200 else 20
    for sec in range(0, int(t[-1]) + 1, step):
        x = sx(sec)
        lines.append(f'<line x1="{x:.1f}" y1="{h-pb}" x2="{x:.1f}" y2="{h-pb+5}" stroke="#374151"/>')
        lines.append(f'<text x="{x-8:.1f}" y="{h-20}" font-family="Arial" font-size="11">{sec}</text>')

    colors = {"高": "#16a34a", "中": "#f59e0b", "低": "#94a3b8"}
    for label, start, end, conf in bands:
        x1, x2 = sx(start), sx(end)
        lines.append(
            f'<rect x="{x1:.1f}" y="{pt}" width="{max(1, x2-x1):.1f}" height="{h-pt-pb}" '
            f'fill="{colors.get(conf, "#94a3b8")}" opacity="0.16"/>'
        )
        lines.append(f'<text x="{x1+2:.1f}" y="{pt+14}" font-family="Arial" font-size="11">{label}</text>')
    lines.append(f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="1.4"/>')
    lines.append("</svg>")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data_root = locate_data_root()
    cutting_root = data_root.parent
    outdir = cutting_root / "stable_cutting_segments_all_final"
    (outdir / "rms").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    params = {"v3": load_params(find_xlsx("v3")), "v4": load_params(find_xlsx("v4"))}
    files = sorted(data_root.glob("*.txt"), key=lambda p: file_meta(p, data_root)["orders"][0])
    records = []
    file_records = []

    for idx, path in enumerate(files, 1):
        meta = file_meta(path, data_root)
        version = str(meta["version"])
        orders = list(meta["orders"])
        expected = len(orders)
        print(f"[{idx}/{len(files)}] {path.name} -> {version} {orders[0]}-{orders[-1]}", flush=True)

        rms, duration, max_abs = load_rms(path)
        smoothed = moving_mean(rms, 5)
        t = (np.arange(len(rms)) + 0.5) * WIN_S
        label = str(meta["label"])

        pd.DataFrame(
            {"time_s": t, "rms_3axis_g": rms, "rms_smooth_0p5s_g": smoothed}
        ).to_csv(outdir / "rms" / f"{label}_rms_0p1s.csv", index=False)

        broad, method, count_ok = pick_broad_segments(smoothed, expected)
        if not count_ok:
            broad = proportional_segments(smoothed, orders, version, params)
            method = f"{method}+理论时长比例回退"

        bands = []
        for i, order in enumerate(orders):
            pinfo = params[version][order]
            allowed = broad[i] if i < len(broad) else None
            start, end, mean_rms, cv = best_window(smoothed, float(pinfo["theory"]), allowed)
            cand_dur = round(end - start, 2)
            conf, note = confidence(cand_dur, float(pinfo["theory"]), cv, max_abs, method, count_ok)
            bands.append((str(order), start, end, conf))

            records.append(
                {
                    "版本": version,
                    "执行顺序": order,
                    "数据文件": path.name,
                    "文件类型": "单顺序文件" if expected == 1 else "多顺序连续文件",
                    "外层活动段开始(s)": round((allowed[0] * WIN_S if allowed else 0.0), 2),
                    "外层活动段结束(s)": round((allowed[1] * WIN_S if allowed else len(smoothed) * WIN_S), 2),
                    "候选稳定段开始(s)": round(start, 2),
                    "候选稳定段结束(s)": round(end, 2),
                    "候选稳定段时长(s)": cand_dur,
                    "理论切削时间(s)": round(float(pinfo["theory"]), 2),
                    "候选-理论时长差(s)": round(cand_dur - float(pinfo["theory"]), 2),
                    "块号": pinfo["block"],
                    "层": pinfo["layer"],
                    "槽号": pinfo["slot"],
                    "转速(rpm)": pinfo["rpm"],
                    "进给(mm/z)": pinfo["fz"],
                    "每转进给(mm/r)": pinfo["feed_rev"],
                    "切深(mm)": pinfo["ap"],
                    "平均RMS(g)": round(mean_rms, 8),
                    "段内RMS变异系数": round(cv, 4),
                    "文件最大绝对值(g)": round(max_abs, 6),
                    "判断置信度": conf,
                    "分析方法": method,
                    "中文说明": note,
                }
            )

        file_records.append(
            {
                "版本": version,
                "数据文件": path.name,
                "执行顺序范围": f"{orders[0]}-{orders[-1]}",
                "执行顺序数": expected,
                "文件时长(s)": round(duration, 2),
                "最大绝对值(g)": round(max_abs, 6),
                "活动段识别方法": method,
                "活动段数量是否匹配": count_ok,
            }
        )
        make_svg(t, smoothed, f"{path.name} RMS overview", bands, outdir / "plots" / f"{label}_overview.svg")

    all_df = pd.DataFrame(records)
    all_df["版本排序"] = all_df["版本"].map({"v3": 0, "v4": 1})
    all_df = all_df.sort_values(["版本排序", "执行顺序"]).drop(columns=["版本排序"])
    all_df.to_csv(outdir / "all_candidate_stable_segments.csv", index=False, encoding="utf-8-sig")
    all_df[all_df["版本"] == "v3"].to_csv(outdir / "v3_candidate_stable_segments.csv", index=False, encoding="utf-8-sig")
    all_df[all_df["版本"] == "v4"].to_csv(outdir / "v4_candidate_stable_segments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(file_records).to_csv(outdir / "file_analysis_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "数据目录": str(data_root),
        "输出目录": str(outdir),
        "txt文件数": len(files),
        "候选稳定段总数": int(len(all_df)),
        "v3候选稳定段数": int((all_df["版本"] == "v3").sum()),
        "v4候选稳定段数": int((all_df["版本"] == "v4").sum()),
        "置信度统计": all_df["判断置信度"].value_counts().to_dict(),
    }
    (outdir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
