#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/figures")
OUT_PNG = OUT_DIR / "attention_dualhead_model_diagram_conference.png"
OUT_PDF = OUT_DIR / "attention_dualhead_model_diagram_conference.pdf"


def add_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    facecolor: str = "#ffffff",
    edgecolor: str = "#d7dde5",
    title_size: float = 18,
    subtitle_size: float = 11.5,
    shadow: bool = True,
):
    if shadow:
        shadow_patch = FancyBboxPatch(
            (x + 0.006, y - 0.008),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0,
            facecolor="#e8edf3",
            alpha=0.55,
        )
        ax.add_patch(shadow_patch)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
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
        fontsize=title_size,
        fontweight="bold",
        color="#0f172a",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.26,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color="#526071",
        )
    return patch


def add_chip(ax, x: float, y: float, text: str, fc: str = "#eef2f7", ec: str = "#cfd8e3", tc: str = "#334155"):
    patch = FancyBboxPatch(
        (x, y),
        0.10,
        0.036,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.05, y + 0.018, text, ha="center", va="center", fontsize=10.8, color=tc, fontweight="bold")
    return patch


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], lw: float = 2.4, color: str = "#475569"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=18),
    )


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
        fontsize=28,
        fontweight="bold",
        color="#102a43",
    )
    ax.text(
        0.05,
        0.885,
        "Patch-level attention produces an interpretable slide embedding, decoded by both shared and cancer-specific branches.",
        fontsize=13.2,
        color="#5b6b7a",
    )

    canvas = FancyBboxPatch(
        (0.04, 0.12),
        0.92,
        0.70,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.4,
        edgecolor="#dde5ee",
        facecolor="white",
    )
    ax.add_patch(canvas)

    add_chip(ax, 0.08, 0.74, "Input")
    add_chip(ax, 0.32, 0.74, "Aggregation")
    add_chip(ax, 0.57, 0.74, "Prediction")
    add_chip(ax, 0.41, 0.18, "Supervision")

    add_card(
        ax,
        0.07,
        0.49,
        0.16,
        0.15,
        "Sampled patch\nfeatures",
        "k patch embeddings per slide",
        facecolor="#dceafe",
        edgecolor="#8fb0d6",
        title_size=19,
    )
    add_card(
        ax,
        0.29,
        0.47,
        0.19,
        0.19,
        "Gated attention\npooling",
        "learn patch weights and pool into one bag",
        facecolor="#fff1b8",
        edgecolor="#d9b44a",
        title_size=21,
    )
    add_card(
        ax,
        0.54,
        0.50,
        0.15,
        0.13,
        "Shared slide\nembedding",
        "bag-level representation",
        facecolor="#d7f5e7",
        edgecolor="#82c4a5",
        title_size=19,
    )
    add_card(
        ax,
        0.74,
        0.56,
        0.14,
        0.10,
        "Pan head",
        "shared across cancers",
        facecolor="#ffe7a3",
        edgecolor="#d3a927",
        title_size=18,
    )
    add_card(
        ax,
        0.74,
        0.39,
        0.14,
        0.10,
        "Specific head",
        "BRCA / LUAD / SKCM",
        facecolor="#f8d6e5",
        edgecolor="#cb8aa9",
        title_size=18,
    )
    add_card(
        ax,
        0.52,
        0.27,
        0.18,
        0.11,
        "Fused SE prediction",
        "union target panel",
        facecolor="#e7dcff",
        edgecolor="#a98ad4",
        title_size=19,
    )
    add_card(
        ax,
        0.33,
        0.05,
        0.34,
        0.10,
        "Masked loss on valid targets only",
        "sample-wise mask keeps one shared output space across cancers",
        facecolor="#fee2e2",
        edgecolor="#de8d8d",
        title_size=18,
    )

    add_arrow(ax, (0.23, 0.565), (0.29, 0.565))
    add_arrow(ax, (0.48, 0.565), (0.54, 0.565))
    add_arrow(ax, (0.69, 0.565), (0.74, 0.61))
    add_arrow(ax, (0.69, 0.535), (0.74, 0.44))
    add_arrow(ax, (0.74, 0.585), (0.70, 0.325))
    add_arrow(ax, (0.74, 0.44), (0.70, 0.325))
    add_arrow(ax, (0.61, 0.27), (0.50, 0.15))

    ax.text(
        0.29,
        0.69,
        "Top-attended patches provide region-level interpretability",
        fontsize=11.6,
        color="#526071",
        fontweight="bold",
    )
    ax.text(
        0.29,
        0.665,
        "attention maps and top-patch overlays can be exported per sample",
        fontsize=10.5,
        color="#708090",
    )

    ax.text(
        0.91,
        0.10,
        "Current talk version: inference backbone only",
        ha="right",
        fontsize=10.8,
        color="#7b8794",
    )

    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
