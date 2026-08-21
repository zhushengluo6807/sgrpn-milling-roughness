import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHANNELS = [
    ("Channel 9", "#5477C4"),
    ("Channel 10", "#CC6F47"),
    ("Channel 11", "#71B436"),
]


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def find_data_dir(root: Path) -> Path:
    for path in root.iterdir():
        if path.is_dir() and (path / "1.txt").exists():
            return path
    raise FileNotFoundError("Could not find experiment data directory.")


def read_rows_by_severity(root: Path, severity: str):
    summary = root / "outputs" / "preprocessing" / "entry_exit_detection_summary.csv"
    with summary.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("severity") == severity]


def downsample_minmax(t: np.ndarray, y: np.ndarray, bins: int = 6500):
    if len(t) <= bins * 2:
        return t, y
    edges = np.linspace(0, len(t), bins + 1, dtype=int)
    xs = []
    ys = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        seg = y[start:end]
        min_i = start + int(np.argmin(seg))
        max_i = start + int(np.argmax(seg))
        ordered = [min_i, max_i] if min_i < max_i else [max_i, min_i]
        xs.extend([t[ordered[0]], t[ordered[1]]])
        ys.extend([y[ordered[0]], y[ordered[1]]])
    return np.asarray(xs), np.asarray(ys)


def scale_points(x, y, box, x_min, x_max, y_min, y_max):
    x0, y0, x1, y1 = box
    px = x0 + (x - x_min) / (x_max - x_min) * (x1 - x0)
    py = y1 - (y - y_min) / (y_max - y_min) * (y1 - y0)
    return list(zip(px.astype(float), py.astype(float)))


def draw_axes(draw, box, title, y_min, y_max):
    x0, y0, x1, y1 = box
    title_font = load_font(23, bold=True)
    tick_font = load_font(15)
    label_font = load_font(17)

    draw.text((x0, y0 - 34), title, fill="#1F2430", font=title_font)
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        draw.line((x0, y, x1, y), fill="#E6E8F0", width=1)
    for i in range(7):
        x = x0 + (x1 - x0) * i / 6
        draw.line((x, y0, x, y1), fill="#F0F1F5", width=1)
    draw.line((x0, y0, x0, y1), fill="#D7DBE7", width=2)
    draw.line((x0, y1, x1, y1), fill="#D7DBE7", width=2)
    draw.text((x0 - 44, y0 + (y1 - y0) / 2 - 10), "g", fill="#6F768A", font=label_font)

    for i in range(5):
        value = y_max - (y_max - y_min) * i / 4
        y = y0 + (y1 - y0) * i / 4
        draw.text((x0 - 96, y - 9), f"{value:.3g}", fill="#6F768A", font=tick_font)


def x_at_time(time_s: float, box, max_time: float = 30.0):
    x0, _, x1, _ = box
    return x0 + time_s / max_time * (x1 - x0)


def draw_time_ticks(draw, box, max_time=30.0):
    x0, _, x1, y1 = box
    font = load_font(15)
    for i in range(7):
        value = max_time * i / 6
        x = x0 + (x1 - x0) * i / 6
        draw.text((x - 16, y1 + 6), f"{value:.0f}", fill="#6F768A", font=font)


def mark_vertical(draw, box, time_s, color, label=None, label_y=0):
    if time_s is None:
        return
    x = x_at_time(time_s, box)
    _, y0, _, y1 = box
    draw.line((x, y0, x, y1), fill=color, width=3)
    if label:
        font = load_font(17, bold=True)
        draw.rectangle((x + 6, y0 + 8 + label_y, x + 235, y0 + 36 + label_y), fill="#FCFCFD")
        draw.text((x + 11, y0 + 10 + label_y), label, fill=color, font=font)


def draw_stable_band(draw, box, start_s, end_s):
    if start_s is None or end_s is None:
        return
    x0 = x_at_time(start_s, box)
    x1 = x_at_time(end_s, box)
    _, y0, _, y1 = box
    draw.rectangle((x0, y0, x1, y1), fill="#FFF4C2")


def as_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def plot_one(root: Path, data_dir: Path, row: dict, out_dir: Path, label: str):
    seq = int(row["seq"])
    path = data_dir / f"{seq}.txt"
    arr = np.loadtxt(str(path), skiprows=2)
    t = arr[:, 0]
    signals = arr[:, 1:4]

    width, height = 2600, 1780
    image = Image.new("RGB", (width, height), "#FCFCFD")
    draw = ImageDraw.Draw(image)
    title_font = load_font(40, bold=True)
    subtitle_font = load_font(21)
    note_font = load_font(18)

    entry = as_float(row, "entry_s")
    theory_exit = as_float(row, "theoretical_exit_120mm_s")
    prep_start = as_float(row, "preprocess_start_s")
    prep_end = as_float(row, "preprocess_end_s")
    coverage = as_float(row, "stable_coverage")

    draw.text((92, 50), f"{seq}.txt {label} signal", fill="#1F2430", font=title_font)
    draw.text(
        (92, 104),
        (
            f"n={float(row['rpm']):.0f} rpm, fz={float(row['fz']):.3g} mm/z, "
            f"ap={float(row['ap']):.3g} mm, vf={float(row['vf_mm_min']):.0f} mm/min, "
            f"stable coverage={coverage:.3f}"
        ),
        fill="#6F768A",
        font=subtitle_font,
    )
    draw.text((92, 138), f"Flags: {row['flags']}", fill="#804126", font=note_font)

    boxes = [
        (150, 270, 2500, 600),
        (150, 790, 2500, 1120),
        (150, 1310, 2500, 1640),
    ]

    for i, (name, color) in enumerate(CHANNELS):
        signal = signals[:, i]
        abs_q = float(np.nanpercentile(np.abs(signal), 99.95))
        y_lim = max(abs_q, 1e-3)
        box = boxes[i]
        draw_stable_band(draw, box, prep_start, prep_end)
        draw_axes(draw, box, name, -y_lim, y_lim)
        xs, ys = downsample_minmax(t, signal, bins=7000)
        ys = np.clip(ys, -y_lim, y_lim)
        points = scale_points(xs, ys, box, 0.0, 30.0, -y_lim, y_lim)
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        mark_vertical(draw, box, entry, "#2E4780", "entry" if i == 0 else None, 0)
        mark_vertical(draw, box, theory_exit, "#804126", "theory exit" if i == 0 else None, 34)
        mark_vertical(draw, box, prep_start, "#B8A037")
        mark_vertical(draw, box, prep_end, "#B8A037")
        draw_time_ticks(draw, box)

    legend_y = 1688
    legend_font = load_font(19, bold=True)
    draw.rectangle((150, legend_y - 6, 184, legend_y + 20), fill="#FFF4C2")
    draw.text((195, legend_y - 8), "yellow band = suggested 100 mm stable segment actually captured", fill="#6F768A", font=note_font)
    draw.line((965, legend_y + 6, 1015, legend_y + 6), fill="#2E4780", width=4)
    draw.text((1024, legend_y - 8), "entry", fill="#2E4780", font=legend_font)
    draw.line((1135, legend_y + 6, 1185, legend_y + 6), fill="#804126", width=4)
    draw.text((1194, legend_y - 8), "theory exit", fill="#804126", font=legend_font)
    draw.text((150, 1727), "Amplitude is clipped per channel to the 99.95th percentile for readability; raw signal is unchanged.", fill="#6F768A", font=note_font)

    out_path = out_dir / f"{seq:03d}_{label}_three_channels.png"
    image.save(out_path)
    return out_path


def main():
    root = Path.cwd()
    data_dir = find_data_dir(root)
    severity = sys.argv[1] if len(sys.argv) > 1 else "hard_review"
    label = severity
    rows = read_rows_by_severity(root, severity)
    out_dir = root / "outputs" / "preprocessing" / f"{label}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for row in rows:
        path = plot_one(root, data_dir, row, out_dir, label)
        paths.append(path)
        print(path)

    index = out_dir / f"{label}_plot_index.txt"
    index.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    print(f"INDEX {index}")


if __name__ == "__main__":
    main()
