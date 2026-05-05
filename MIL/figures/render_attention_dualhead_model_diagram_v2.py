#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/figures")
OUT_PNG = OUT_DIR / "attention_dualhead_model_diagram_v2.png"
OUT_PDF = OUT_DIR / "attention_dualhead_model_diagram_v2.pdf"


def add_round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fc: str = "#ffffff",
    ec: str = "#d6dee8",
    title_size: float = 18,
    subtitle_size: float = 11,
    lw: float = 1.8,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.60,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#0f172a",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.24,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color="#526071",
        )
    return patch


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#44556b", lw: float = 2.4):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=18),
    )


def add_patch_strip(ax, x: float, y: float, w: float, h: float):
    bg = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor="#8fb0d6",
        facecolor="#eef5ff",
    )
    ax.add_patch(bg)
    pad_x = 0.014
    pad_y = 0.02
    cols = 4
    rows = 2
    tile_w = (w - pad_x * (cols + 1)) / cols
    tile_h = (h - 0.08 - pad_y * (rows + 1)) / rows
    colors = ["#b8d2f2", "#cfe2ff", "#d9e8ff", "#bfd8ff"]
    idx = 0
    for r in range(rows):
        for c in range(cols):
            rx = x + pad_x + c * (tile_w + pad_x)
            ry = y + 0.05 + pad_y + (rows - 1 - r) * (tile_h + pad_y)
            rect = Rectangle((rx, ry), tile_w, tile_h, linewidth=0.8, edgecolor="#99b7d9", facecolor=colors[idx % len(colors)])
            ax.add_patch(rect)
            idx += 1
    ax.text(x + w / 2, y + 0.035, "Sampled patch features", ha="center", va="center", fontsize=16.5, fontweight="bold", color="#0f172a")
    return bg


def add_attention_block(ax, x: float, y: float, w: float, h: float):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor="#d6b24a",
        facecolor="#fff3c4",
    )
    ax.add_patch(patch)

    bar_x = x + 0.04
    bar_y = y + h * 0.56
    bar_w = w * 0.50
    n = 8
    gap = bar_w / (n * 1.3)
    heights = [0.03, 0.05, 0.025, 0.08, 0.11, 0.04, 0.07, 0.03]
    for i, hh in enumerate(heights):
        rect = Rectangle(
            (bar_x + i * gap * 1.3, bar_y),
            gap,
            hh,
            linewidth=0,
            facecolor="#d9a700" if i in (4, 6) else "#e7d58c",
        )
        ax.add_patch(rect)

    ax.text(x + w * 0.63, y + h * 0.63, "Gated attention\npooling", ha="center", va="center", fontsize=21, fontweight="bold", color="#0f172a")
    ax.text(x + w * 0.63, y + h * 0.25, "learn patch weights", ha="center", va="center", fontsize=11.2, color="#526071")
    return patch


def add_embedding_bar(ax, x: float, y: float, w: float, h: float):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.03",
        linewidth=1.8,
        edgecolor="#82c4a5",
        facecolor="#dff7eb",
    )
    ax.add_patch(patch)
    inner_y = y + h * 0.52
    for i in range(14):
        alpha = 0.25 + 0.05 * (i % 5)
        rect = Rectangle((x + 0.02 + i * (w - 0.04) / 14, inner_y), (w - 0.06) / 14, h * 0.16, linewidth=0, facecolor="#53a678", alpha=alpha)
        ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.28, "Shared slide embedding", ha="center", va="center", fontsize=17.5, fontweight="bold", color="#0f172a")
    return patch


def add_decoder(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str, fc: str, ec: str):
    return add_round_box(ax, x, y, w, h, title, subtitle, fc=fc, ec=ec, title_size=16.5, subtitle_size=10.5)


def add_fuse_node(ax, x: float, y: float, r: float):
    circ = Circle((x, y), r, facecolor="#eef2ff", edgecolor="#9d8bdd", linewidth=1.8)
    ax.add_patch(circ)
    ax.text(x, y, "Fuse", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#27364a")
    return circ


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor("#f6f8fb")
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.93,
        "Attention Dual-Head MIL for Pan-Cancer SE Prediction",
        fontsize=27,
        fontweight="bold",
        color="#102a43",
    )
    ax.text(
        0.05,
        0.885,
        "Shape-coded view of the inference backbone: sampled patches -> attention pooling -> shared embedding -> pan/specific decoding.",
        fontsize=13.0,
        color="#5b6b7a",
    )

    canvas = FancyBboxPatch(
        (0.04, 0.11),
        0.92,
        0.72,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.3,
        edgecolor="#dde5ee",
        facecolor="white",
    )
    ax.add_patch(canvas)

    add_patch_strip(ax, 0.07, 0.46, 0.16, 0.18)
    add_attention_block(ax, 0.29, 0.43, 0.20, 0.22)
    add_embedding_bar(ax, 0.55, 0.50, 0.22, 0.08)

    add_decoder(ax, 0.81, 0.58, 0.11, 0.08, "Pan head", "shared", fc="#ffe7a3", ec="#d3a927")
    add_decoder(ax, 0.81, 0.42, 0.11, 0.08, "Specific head", "BRCA / LUAD / SKCM", fc="#f8d6e5", ec="#cb8aa9")

    add_fuse_node(ax, 0.79, 0.34, 0.036)
    add_round_box(ax, 0.64, 0.22, 0.21, 0.09, "SE prediction", "union target panel", fc="#e7dcff", ec="#a98ad4", title_size=18.5, subtitle_size=11)
    add_round_box(ax, 0.33, 0.06, 0.34, 0.09, "Masked loss", "1 = valid, 0 = ignore", fc="#fee2e2", ec="#de8d8d", title_size=19, subtitle_size=11.5)

    add_arrow(ax, (0.23, 0.55), (0.29, 0.55))
    add_arrow(ax, (0.49, 0.55), (0.55, 0.55))

    add_arrow(ax, (0.77, 0.56), (0.81, 0.62))
    add_arrow(ax, (0.77, 0.52), (0.81, 0.46))

    add_arrow(ax, (0.865, 0.58), (0.80, 0.375))
    add_arrow(ax, (0.865, 0.42), (0.80, 0.365))
    add_arrow(ax, (0.754, 0.34), (0.745, 0.31))
    add_arrow(ax, (0.70, 0.22), (0.58, 0.15))

    ax.text(0.30, 0.69, "top-attended patches can be visualized", fontsize=11.3, color="#526071", fontweight="bold")
    ax.text(0.30, 0.665, "region-level interpretability", fontsize=10.3, color="#7a8794")
    ax.text(0.95, 0.02, "v2 conference-style architecture", ha="right", fontsize=10.5, color="#7b8794")

    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
