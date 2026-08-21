from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_data_dir(root: Path) -> Path:
    for path in root.iterdir():
        if path.is_dir() and (path / "1.txt").exists():
            return path
    raise FileNotFoundError("Could not find a data directory containing 1.txt")


def downsample_minmax(t: np.ndarray, y: np.ndarray, bins: int = 7000):
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
        if min_i < max_i:
            xs.extend([t[min_i], t[max_i]])
            ys.extend([y[min_i], y[max_i]])
        else:
            xs.extend([t[max_i], t[min_i]])
            ys.extend([y[max_i], y[min_i]])
    return np.asarray(xs), np.asarray(ys)


def moving_rms(signal: np.ndarray, window: int):
    kernel = np.ones(window, dtype=float) / window
    return np.sqrt(np.convolve(signal * signal, kernel, mode="valid"))


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_axes(draw, box, title, y_label, x_label=None):
    x0, y0, x1, y1 = box
    ink = "#1F2430"
    muted = "#6F768A"
    grid = "#E6E8F0"
    axis = "#D7DBE7"
    title_font = load_font(24, bold=True)
    label_font = load_font(18)
    tick_font = load_font(15)

    draw.text((x0, y0 - 36), title, fill=ink, font=title_font)
    for i in range(6):
        y = y0 + (y1 - y0) * i / 5
        draw.line((x0, y, x1, y), fill=grid, width=1)
    for i in range(7):
        x = x0 + (x1 - x0) * i / 6
        draw.line((x, y0, x, y1), fill="#F0F1F5", width=1)
    draw.line((x0, y0, x0, y1), fill=axis, width=2)
    draw.line((x0, y1, x1, y1), fill=axis, width=2)
    draw.text((x0 - 52, y0 + (y1 - y0) / 2 - 10), y_label, fill=muted, font=label_font)
    if x_label:
        draw.text((x0 + (x1 - x0) / 2 - 34, y1 + 26), x_label, fill=muted, font=label_font)
    return tick_font


def scale_points(x, y, box, x_min, x_max, y_min, y_max):
    x0, y0, x1, y1 = box
    px = x0 + (x - x_min) / (x_max - x_min) * (x1 - x0)
    py = y1 - (y - y_min) / (y_max - y_min) * (y1 - y0)
    return list(zip(px.astype(float), py.astype(float)))


def draw_polyline(draw, points, color, width=2):
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_y_ticks(draw, box, y_min, y_max, font):
    x0, y0, _, y1 = box
    for i in range(6):
        value = y_max - (y_max - y_min) * i / 5
        y = y0 + (y1 - y0) * i / 5
        draw.text((x0 - 88, y - 9), f"{value:.3g}", fill="#6F768A", font=font)


def draw_x_ticks(draw, box, x_min, x_max, font):
    x0, _, x1, y1 = box
    for i in range(7):
        value = x_min + (x_max - x_min) * i / 6
        x = x0 + (x1 - x0) * i / 6
        draw.text((x - 16, y1 + 5), f"{value:.1f}", fill="#6F768A", font=font)


def main():
    root = Path.cwd()
    data_dir = find_data_dir(root)
    txt_path = data_dir / "1.txt"
    out_dir = root / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "1_txt_vibration_overview.png"

    data = np.loadtxt(str(txt_path), skiprows=2)
    t = data[:, 0]
    channels = data[:, 1:4]
    names = ["Channel 9", "Channel 10", "Channel 11"]
    colors = ["#5477C4", "#CC6F47", "#71B436"]

    dt = float(np.median(np.diff(t[:2000])))
    fs = 1.0 / dt
    vector = np.sqrt(np.sum(channels**2, axis=1))
    rms_window = max(1, int(round(0.05 * fs)))
    rms = moving_rms(vector, rms_window)
    rms_t = t[: len(rms)] + (rms_window - 1) * dt / 2

    width, height = 2600, 1700
    image = Image.new("RGB", (width, height), "#FCFCFD")
    draw = ImageDraw.Draw(image)
    title_font = load_font(40, bold=True)
    subtitle_font = load_font(22)
    legend_font = load_font(20)
    tick_font = load_font(15)

    draw.text((95, 55), "1.txt vibration signal overview", fill="#1F2430", font=title_font)
    draw.text(
        (95, 108),
        f"Three acceleration channels, duration {t[-1]:.2f} s, sampling rate {fs/1000:.2f} kHz",
        fill="#6F768A",
        font=subtitle_font,
    )

    full_box = (145, 235, 2500, 705)
    rms_box = (145, 880, 2500, 1170)
    detail_box = (145, 1345, 2500, 1585)

    y_full = float(np.nanpercentile(np.abs(channels), 99.995))
    y_full = max(y_full, 0.01)
    ticks = draw_axes(draw, full_box, "Full 30-second waveform", "g")
    draw_y_ticks(draw, full_box, -y_full, y_full, ticks)
    draw_x_ticks(draw, full_box, 0, float(t[-1]), ticks)
    for i in range(3):
        xs, ys = downsample_minmax(t, channels[:, i], bins=6500)
        points = scale_points(xs, np.clip(ys, -y_full, y_full), full_box, 0, float(t[-1]), -y_full, y_full)
        draw_polyline(draw, points, colors[i], width=2)

    lx = 1890
    for i, name in enumerate(names):
        draw.line((lx + i * 170, 202, lx + 42 + i * 170, 202), fill=colors[i], width=5)
        draw.text((lx + 50 + i * 170, 190), name.replace("Channel ", "Ch "), fill="#1F2430", font=legend_font)

    rms_max = float(np.nanpercentile(rms, 99.8))
    rms_max = max(rms_max, 0.001)
    ticks = draw_axes(draw, rms_box, "Vector RMS envelope, 50 ms window", "RMS")
    draw_y_ticks(draw, rms_box, 0, rms_max, ticks)
    draw_x_ticks(draw, rms_box, 0, float(t[-1]), ticks)
    xs, ys = downsample_minmax(rms_t, rms, bins=5000)
    points = scale_points(xs, np.clip(ys, 0, rms_max), rms_box, 0, float(t[-1]), 0, rms_max)
    draw_polyline(draw, points, "#5477C4", width=3)

    detail_end = min(0.25, float(t[-1]))
    mask = t <= detail_end
    y_detail = float(np.nanpercentile(np.abs(channels[mask]), 99.5))
    y_detail = max(y_detail, 0.001)
    ticks = draw_axes(draw, detail_box, "Local waveform detail: first 0.25 s", "g", "Time")
    draw_y_ticks(draw, detail_box, -y_detail, y_detail, ticks)
    for i in range(3):
        points = scale_points(t[mask], np.clip(channels[mask, i], -y_detail, y_detail), detail_box, 0, detail_end, -y_detail, y_detail)
        draw_polyline(draw, points, colors[i], width=2)
    for i in range(6):
        value = detail_end * i / 5
        x = detail_box[0] + (detail_box[2] - detail_box[0]) * i / 5
        draw.text((x - 16, detail_box[3] + 5), f"{value:.2f}", fill="#6F768A", font=tick_font)

    draw.text(
        (95, 1640),
        "Note: full-waveform amplitudes are clipped to the 99.995th percentile for readability; raw values remain unchanged.",
        fill="#6F768A",
        font=subtitle_font,
    )
    image.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
