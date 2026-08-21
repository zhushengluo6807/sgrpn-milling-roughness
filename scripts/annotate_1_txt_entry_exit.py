from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def x_at_time(time_s: float, box, max_time: float = 30.0):
    x0, _, x1, _ = box
    return x0 + time_s / max_time * (x1 - x0)


def mark(draw, box, time_s, color, label, y_label_offset=0):
    x = x_at_time(time_s, box)
    x0, y0, _, y1 = box
    font = load_font(21, bold=True)
    draw.line((x, y0, x, y1), fill=color, width=4)
    draw.rectangle((x + 8, y0 + 10 + y_label_offset, x + 300, y0 + 46 + y_label_offset), fill="#FCFCFD")
    draw.text((x + 14, y0 + 13 + y_label_offset), label, fill=color, font=font)


def main():
    root = Path.cwd()
    src = root / "outputs" / "plots" / "1_txt_vibration_overview.png"
    dst = root / "outputs" / "plots" / "1_txt_entry_exit_annotated.png"
    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image)

    full_box = (145, 235, 2500, 705)
    rms_box = (145, 880, 2500, 1170)
    detail_box = (145, 1345, 2500, 1585)

    entry = 2.32
    theoretical_exit = 7.12
    observed_exit = 8.65

    for box in [full_box, rms_box]:
        mark(draw, box, entry, "#2E4780", "entry ~2.32 s", 0)
        mark(draw, box, theoretical_exit, "#804126", "120 mm exit ~7.12 s", 42)
        mark(draw, box, observed_exit, "#8A3A6F", "observed pulse ~8.65 s", 84)

    # The detail panel only spans 0-0.25 s, so mark it as not covering entry/exit.
    note_font = load_font(20)
    draw.text(
        (detail_box[0] + 15, detail_box[1] + 12),
        "Entry/exit occur after this 0.25 s detail window",
        fill="#6F768A",
        font=note_font,
    )

    image.save(dst)
    print(dst)


if __name__ == "__main__":
    main()
