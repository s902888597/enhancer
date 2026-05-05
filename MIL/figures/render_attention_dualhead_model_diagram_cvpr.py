#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/figures")
OUT_PNG = OUT_DIR / "attention_dualhead_model_diagram_cvpr.png"
OUT_PDF = OUT_DIR / "attention_dualhead_model_diagram_cvpr.pdf"


def add_3d_block(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    dx: float,
    dy: float,
    face: str,
    side: str,
    top: str,
    edge: str,
    title: str,
    subtitle: str = "",
    title_size: float = 18,
    subtitle_size: float = 10.5,
):
    front = Polygon(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.8,
        joinstyle="round",
    )
    side_poly = Polygon(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor=side,
        edgecolor=edge,
        linewidth=1.4,
        joinstyle="round",
    )
    top_poly = Polygon(
        [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)],
        closed=True,
        facecolor=top,
        edgecolor=edge,
        linewidth=1.4,
        joinstyle="round",
    )
    ax.add_patch(side_poly)
    ax.add_patch(top_poly)
    ax.add_patch(front)
    ax.text(
        x + w / 2,
        y + h * 0.58,
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
            y + h * 0.23,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color="#516170",
        )
    return front


def add_tensor_bar(ax, x: float, y: float, w: float, h: float, layers: int, base: str, accent: str, edge: str, label: str):
    dx, dy = 0.012, 0.010
    for i in range(layers):
        off = layers - i - 1
        rect = Polygon(
            [
                (x + off * dx, y + off * dy),
                (x + w + off * dx, y + off * dy),
                (x + w + off * dx, y + h + off * dy),
                (x + off * dx, y + h + off * dy),
            ],
            closed=True,
            facecolor=base if i < layers - 1 else accent,
            edgecolor=edge,
            linewidth=1.2,
            alpha=0.95 if i == layers - 1 else 0.55,
        )
        ax.add_patch(rect)
    for j in range(13):
        bx = x + 0.02 + j * (w - 0.05) / 13
        ax.add_patch(Rectangle((bx, y + h * 0.32), 0.013, h * 0.30, facecolor="#87c8a8", edgecolor="none", alpha=0.85))
    ax.text(x + w / 2 + dx * (layers - 1) / 2, y - 0.028, label, ha="center", va="top", fontsize=16.5, fontweight="bold", color="#0f172a")


def add_patch_tokens(ax, x: float, y: float, cols: int = 4, rows: int = 3):
    tile_w = 0.028
    tile_h = 0.024
    gap_x = 0.013
    gap_y = 0.014
    depth_dx = 0.010
    depth_dy = 0.008
    for layer in range(3):
        lx = x + layer * depth_dx
        ly = y + layer * depth_dy
        for r in range(rows):
            for c in range(cols):
                fc = "#dbeafe" if (r + c + layer) % 2 == 0 else "#bcd4f6"
                rect = Rectangle(
                    (lx + c * (tile_w + gap_x), ly + r * (tile_h + gap_y)),
                    tile_w,
                    tile_h,
                    facecolor=fc,
                    edgecolor="#8faed0",
                    linewidth=1.0,
                )
                ax.add_patch(rect)
    width = cols * tile_w + (cols - 1) * gap_x + 2 * depth_dx
    ax.text(x + width / 2, y - 0.035, "Patch tokens", ha="center", va="top", fontsize=16.5, fontweight="bold", color="#0f172a")


def add_attention_weights(ax, x: float, y: float):
    heights = [0.025, 0.05, 0.032, 0.08, 0.11, 0.045, 0.07, 0.03]
    for i, hh in enumerate(heights):
        fc = "#e8d68b" if i not in (3, 4, 6) else "#d6a800"
        ax.add_patch(Rectangle((x + i * 0.018, y), 0.012, hh, facecolor=fc, edgecolor="none", alpha=0.95))
    ax.text(x + 0.06, y + 0.135, "attention", fontsize=10.5, color="#7a8794", ha="center", fontweight="bold")


def add_node(ax, x: float, y: float, r: float, fc: str, ec: str, text: str):
    circ = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.8)
    ax.add_patch(circ)
    ax.text(x, y, text, ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1f2b3a")


def arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = "#4b5d73", lw: float = 2.5):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=18, linewidth=lw, color=color, connectionstyle="arc3,rad=0.0")
    ax.add_patch(arr)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor("#f5f7fb")
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.93,
        "Attention Dual-Head MIL for Pan-Cancer SE Prediction",
        fontsize=28,
        fontweight="bold",
        color="#102a43",
    )
    ax.text(
        0.05,
        0.885,
        "CVPR-style inference diagram: tokens -> attention pooling -> shared embedding -> pan/specific heads -> fused SE prediction.",
        fontsize=13.0,
        color="#5b6b7a",
    )

    add_patch_tokens(ax, 0.08, 0.44)

    add_3d_block(
        ax,
        0.29,
        0.40,
        0.16,
        0.15,
        0.022,
        0.018,
        face="#fff2c2",
        side="#efd88d",
        top="#fff7dc",
        edge="#c9a53b",
        title="Attention\npooling",
        title_size=20,
    )
    add_attention_weights(ax, 0.315, 0.455)

    add_tensor_bar(
        ax,
        0.53,
        0.47,
        0.20,
        0.045,
        layers=4,
        base="#d7f0e2",
        accent="#c3ead7",
        edge="#7cb796",
        label="Shared embedding",
    )

    add_3d_block(
        ax,
        0.81,
        0.56,
        0.09,
        0.07,
        0.016,
        0.012,
        face="#ffe39a",
        side="#e7c15b",
        top="#fff0c4",
        edge="#c79d1f",
        title="Pan head",
        subtitle="shared",
        title_size=15.5,
        subtitle_size=10,
    )
    add_3d_block(
        ax,
        0.81,
        0.40,
        0.09,
        0.07,
        0.016,
        0.012,
        face="#f6d3e3",
        side="#dfaec4",
        top="#fbe5ef",
        edge="#c47f9e",
        title="Specific head",
        subtitle="BRCA / LUAD / SKCM",
        title_size=14.2,
        subtitle_size=9.2,
    )

    add_node(ax, 0.78, 0.30, 0.026, "#eef0ff", "#9b8ad8", "Fuse")

    add_3d_block(
        ax,
        0.62,
        0.18,
        0.17,
        0.08,
        0.018,
        0.014,
        face="#e4d8ff",
        side="#c8b3f5",
        top="#f0eaff",
        edge="#9c81d3",
        title="SE prediction",
        subtitle="union panel",
        title_size=17.5,
        subtitle_size=10.5,
    )

    add_3d_block(
        ax,
        0.30,
        0.06,
        0.22,
        0.055,
        0.018,
        0.012,
        face="#fee0e0",
        side="#f3bcbc",
        top="#fff0f0",
        edge="#d88787",
        title="Masked loss",
        subtitle="valid targets only",
        title_size=16.5,
        subtitle_size=10.2,
    )

    arrow(ax, 0.22, 0.50, 0.29, 0.50)
    arrow(ax, 0.47, 0.50, 0.53, 0.50)
    arrow(ax, 0.74, 0.52, 0.81, 0.595)
    arrow(ax, 0.74, 0.49, 0.81, 0.435)
    arrow(ax, 0.855, 0.56, 0.79, 0.325)
    arrow(ax, 0.855, 0.40, 0.79, 0.305)
    arrow(ax, 0.75, 0.28, 0.705, 0.22)
    arrow(ax, 0.62, 0.18, 0.52, 0.10)

    ax.text(0.33, 0.61, "top-attended patches", fontsize=11.2, color="#66788a", fontweight="bold")
    ax.text(0.95, 0.02, "cvpr-style architecture", ha="right", fontsize=10.5, color="#7b8794")

    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
