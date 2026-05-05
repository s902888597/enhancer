#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/figures")
OUT_PNG = OUT_DIR / "attention_dualhead_model_diagram.png"
OUT_PDF = OUT_DIR / "attention_dualhead_model_diagram.pdf"


def add_box(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    facecolor: str = "#ffffff",
    edgecolor: str = "#334155",
    fontsize: float = 18,
    subtitle_size: float = 11.5,
    rounding: float = 0.02,
    lw: float = 2.0,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.60,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
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
            color="#475569",
        )
    return patch


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#475569", lw: float = 2.6):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=20),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor("#fbfaf7")
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.93,
        "Attention-Based Dual-Head MIL for Pan-Cancer SE Prediction",
        fontsize=28,
        fontweight="bold",
        color="#16324a",
    )
    ax.text(
        0.05,
        0.885,
        "Current main architecture: patch features -> gated attention pooling -> shared slide representation -> pan/specific fusion -> masked SE prediction",
        fontsize=13.0,
        color="#586b7a",
    )

    panel = FancyBboxPatch(
        (0.04, 0.11),
        0.92,
        0.73,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.8,
        edgecolor="#cfd8df",
        facecolor="white",
    )
    ax.add_patch(panel)

    input_box = add_box(
        ax,
        (0.07, 0.58),
        0.18,
        0.12,
        "Patch features\nfrom WSI",
        "up to k sampled patch embeddings per slide",
        facecolor="#dbeafe",
        edgecolor="#4f6d8a",
    )
    attn_box = add_box(
        ax,
        (0.31, 0.56),
        0.22,
        0.16,
        "Patch embedding +\ngated attention pooling",
        "attention scores over sampled patches",
        facecolor="#fde68a",
        edgecolor="#967f2c",
    )
    shared_box = add_box(
        ax,
        (0.60, 0.58),
        0.17,
        0.12,
        "Shared slide\nrepresentation",
        "single bag-level vector",
        facecolor="#d1fae5",
        edgecolor="#4c8a6a",
    )

    pan_box = add_box(
        ax,
        (0.20, 0.34),
        0.18,
        0.12,
        "Pan-cancer head",
        "shared across BRCA / LUAD / SKCM",
        facecolor="#fee79a",
        edgecolor="#b38c1d",
    )
    spec_box = add_box(
        ax,
        (0.61, 0.34),
        0.18,
        0.12,
        "Cancer-specific head",
        "BRCA / LUAD / SKCM",
        facecolor="#f9d5e5",
        edgecolor="#a15a7b",
    )
    fuse_box = add_box(
        ax,
        (0.40, 0.31),
        0.16,
        0.10,
        "Fuse pan + specific",
        "sample-level combination",
        facecolor="#d9f99d",
        edgecolor="#6c9a2f",
        fontsize=17,
        subtitle_size=11.0,
    )
    pred_box = add_box(
        ax,
        (0.37, 0.20),
        0.22,
        0.10,
        "Predicted SE activity",
        "union target panel",
        facecolor="#e9d5ff",
        edgecolor="#7e5aa6",
        fontsize=18,
    )
    loss_box = add_box(
        ax,
        (0.34, 0.07),
        0.28,
        0.08,
        "Masked loss on valid targets only",
        "mask = 1 valid, 0 ignore",
        facecolor="#fee2e2",
        edgecolor="#b45353",
        fontsize=16,
        subtitle_size=11.0,
    )

    callout_box = add_box(
        ax,
        (0.30, 0.74),
        0.22,
        0.08,
        "Top-attended patches\ncan be visualized",
        "region-level interpretability",
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
        fontsize=15.5,
        subtitle_size=10.5,
        rounding=0.018,
        lw=1.8,
    )

    add_arrow(ax, (0.25, 0.64), (0.31, 0.64))
    add_arrow(ax, (0.53, 0.64), (0.60, 0.64))

    add_arrow(ax, (0.685, 0.58), (0.285, 0.46))
    add_arrow(ax, (0.685, 0.58), (0.70, 0.46))

    add_arrow(ax, (0.38, 0.40), (0.40, 0.36))
    add_arrow(ax, (0.61, 0.40), (0.56, 0.36))

    add_arrow(ax, (0.48, 0.31), (0.48, 0.30))
    add_arrow(ax, (0.48, 0.20), (0.48, 0.15))
    add_arrow(ax, (0.42, 0.74), (0.42, 0.72))

    ax.text(
        0.82,
        0.61,
        "Attention weights identify\nwhich patches drive the\nslide-level prediction.",
        fontsize=12.5,
        color="#475569",
        va="center",
    )

    ax.text(
        0.95,
        0.02,
        "Attention dual-head MIL architecture",
        ha="right",
        fontsize=10.5,
        color="#7b8794",
    )

    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
