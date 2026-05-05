#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/figures")
OUT_PNG = OUT_DIR / "attention_dualhead_model_diagram_paper.png"
OUT_PDF = OUT_DIR / "attention_dualhead_model_diagram_paper.pdf"


def add_prism(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    dx: float,
    dy: float,
    face: str,
    top: str,
    side: str,
    edge: str,
    label: str,
    label_size: float,
):
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
    front_poly = Polygon(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.8,
        joinstyle="round",
    )
    ax.add_patch(side_poly)
    ax.add_patch(top_poly)
    ax.add_patch(front_poly)
    ax.text(
        x + w / 2,
        y - 0.032,
        label,
        ha="center",
        va="top",
        fontsize=label_size,
        fontweight="bold",
        color="#0f172a",
    )


def add_token_stack(ax, x: float, y: float):
    cols = 5
    rows = 3
    tile_w = 0.028
    tile_h = 0.022
    gap_x = 0.009
    gap_y = 0.012
    depth = 3
    depth_dx = 0.008
    depth_dy = 0.006
    colors = ["#dceafe", "#bfd7f7"]
    for layer in range(depth):
        lx = x + layer * depth_dx
        ly = y + layer * depth_dy
        for r in range(rows):
            for c in range(cols):
                rect = Rectangle(
                    (lx + c * (tile_w + gap_x), ly + r * (tile_h + gap_y)),
                    tile_w,
                    tile_h,
                    facecolor=colors[(r + c + layer) % 2],
                    edgecolor="#8fb0d6",
                    linewidth=1.0,
                )
                ax.add_patch(rect)
    width = (cols - 1) * (tile_w + gap_x) + tile_w + depth_dx * (depth - 1)
    ax.text(
        x + width / 2,
        y - 0.038,
        "Patch tokens",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )


def add_attention_module(ax, x: float, y: float):
    add_prism(
        ax,
        x,
        y,
        0.16,
        0.14,
        0.024,
        0.018,
        face="#fff2c7",
        top="#fff8df",
        side="#eed587",
        edge="#c89f27",
        label="Attention pooling",
        label_size=17.5,
    )
    base_x = x + 0.028
    base_y = y + 0.045
    heights = [0.025, 0.055, 0.032, 0.080, 0.110, 0.040, 0.070, 0.028]
    for i, hh in enumerate(heights):
        fc = "#e1b500" if i in (3, 4, 6) else "#ecd98b"
        ax.add_patch(Rectangle((base_x + i * 0.017, base_y), 0.011, hh, facecolor=fc, edgecolor="none", alpha=0.98))


def add_embedding_tensor(ax, x: float, y: float):
    layers = 5
    dx = 0.010
    dy = 0.008
    w = 0.18
    h = 0.045
    for i in range(layers):
        off = layers - i - 1
        rect = Rectangle(
            (x + off * dx, y + off * dy),
            w,
            h,
            facecolor="#dff3e9" if i < layers - 1 else "#c9ead8",
            edgecolor="#7db898",
            linewidth=1.2,
            alpha=0.55 if i < layers - 1 else 0.98,
        )
        ax.add_patch(rect)
    for j in range(14):
        ax.add_patch(
            Rectangle(
                (x + 0.018 + j * 0.0115, y + 0.013),
                0.009,
                0.016,
                facecolor="#7fc6a1",
                edgecolor="none",
                alpha=0.92,
            )
        )
    ax.text(
        x + w / 2 + dx * (layers - 1) / 2,
        y - 0.035,
        "Shared embedding",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
    )


def add_head(ax, x: float, y: float, face: str, top: str, side: str, edge: str, label: str, size: float = 15.5):
    add_prism(ax, x, y, 0.086, 0.062, 0.016, 0.012, face, top, side, edge, label, size)


def add_fuse(ax, x: float, y: float):
    node = FancyBboxPatch(
        (x - 0.026, y - 0.018),
        0.052,
        0.036,
        boxstyle="round,pad=0.004,rounding_size=0.025",
        linewidth=1.8,
        edgecolor="#9987d8",
        facecolor="#eef0ff",
    )
    ax.add_patch(node)
    ax.text(x, y, "Fuse", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#23324a")


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], lw: float = 2.6):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=18,
        linewidth=lw,
        color="#4f6278",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)


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
        "High-level architecture for inference only.",
        fontsize=13.0,
        color="#5b6b7a",
    )

    add_token_stack(ax, 0.08, 0.40)
    add_attention_module(ax, 0.30, 0.37)
    add_embedding_tensor(ax, 0.55, 0.46)

    add_head(
        ax,
        0.83,
        0.56,
        face="#ffe39a",
        top="#fff0c4",
        side="#e6c05d",
        edge="#c79d1f",
        label="Pan head",
    )
    add_head(
        ax,
        0.83,
        0.40,
        face="#f6d3e3",
        top="#fbe6ef",
        side="#dfaec4",
        edge="#c47f9e",
        label="Specific head",
        size=14.2,
    )

    add_fuse(ax, 0.80, 0.30)

    add_prism(
        ax,
        0.63,
        0.18,
        0.16,
        0.075,
        0.018,
        0.014,
        face="#e4d8ff",
        top="#f1ebff",
        side="#c8b3f5",
        edge="#9c81d3",
        label="SE prediction",
        label_size=17.0,
    )
    add_prism(
        ax,
        0.31,
        0.06,
        0.20,
        0.052,
        0.018,
        0.012,
        face="#fee0e0",
        top="#fff0f0",
        side="#f3bcbc",
        edge="#d88787",
        label="Masked loss",
        label_size=16.0,
    )

    ax.text(0.385, 0.565, "top-attended patches", fontsize=10.8, color="#6a7c90", fontweight="bold", ha="center")

    add_arrow(ax, (0.22, 0.46), (0.30, 0.46))
    add_arrow(ax, (0.484, 0.46), (0.55, 0.49))
    add_arrow(ax, (0.76, 0.53), (0.83, 0.59))
    add_arrow(ax, (0.76, 0.48), (0.83, 0.43))
    add_arrow(ax, (0.873, 0.56), (0.80, 0.32))
    add_arrow(ax, (0.873, 0.40), (0.80, 0.305))
    add_arrow(ax, (0.76, 0.28), (0.71, 0.21))
    add_arrow(ax, (0.63, 0.18), (0.51, 0.10))

    ax.text(0.95, 0.02, "paper-style architecture", ha="right", fontsize=10.5, color="#7b8794")

    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
