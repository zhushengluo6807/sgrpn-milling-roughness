from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser


def validate(latex: str, fontsize: int) -> None:
    """渲染前强制校验公式可被 mathtext 解析。

    matplotlib 在解析失败时会静默回退为「把 LaTeX 源码当普通文字绘制」，
    输出的 PNG 表面上是一张正常图片，实际却是 ``$\\frac{1}{2}$`` 这样的源码文本。
    这里显式解析一次，让问题在构建阶段直接报错，而不是产出一份公式乱码的文档。
    """
    if "\n" in latex or "\r" in latex:
        raise ValueError(f"公式含换行，mathtext 无法解析：{latex[:80]!r}")
    MathTextParser("path").parse(
        "$" + latex + "$", dpi=72, prop=FontProperties(size=fontsize)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fontsize = 14 if args.display else 11
    validate(args.latex, fontsize)
    fig = plt.figure(figsize=(0.1, 0.1), dpi=360)
    fig.patch.set_alpha(0)
    text = fig.text(0, 0, "$" + args.latex + "$", fontsize=fontsize, color="black")
    fig.canvas.draw()
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.08, 1.20)
    fig.set_size_inches(max(bbox.width / fig.dpi, 0.05), max(bbox.height / fig.dpi, 0.05))
    text.set_position((0.02, 0.08))
    fig.savefig(output, dpi=360, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
