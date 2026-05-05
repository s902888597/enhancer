#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch


OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/figures")
OUT_PNG = OUT_DIR / "brca_attention_tissue_interpretation_slide_v4.png"
OUT_PDF = OUT_DIR / "brca_attention_tissue_interpretation_slide_v4.pdf"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"

    classes = ["Tumor-rich", "Immune", "Stroma", "Invasive", "Mixed interface", "Other"]
    top_attention = np.array([67.7, 9.6, 1.7, 2.1, 14.7, 4.2], dtype=float)
    slide_background = np.array([20.8, 13.0, 27.3, 17.0, 10.0, 11.9], dtype=float)
    colors = ["#c75146", "#4c9f70", "#6a8caf", "#dd8b39", "#8a63d2", "#9e9e9e"]

    fig = plt.figure(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor("#fbfaf7")
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.93,
        "Attention Highlights Biologically Meaningful Tissue Regions",
        fontsize=28,
        fontweight="bold",
        color="#16324a",
    )
    ax.text(
        0.05,
        0.885,
        "BRCA test set, top-20 attention patches per sample; cluster 0 merged into tumor-rich, cluster 8 shown as mixed interface",
        fontsize=13.2,
        color="#586b7a",
    )

    panel = FancyBboxPatch(
        (0.05, 0.18),
        0.90,
        0.62,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.8,
        edgecolor="#cfd8df",
        facecolor="white",
    )
    ax.add_patch(panel)

    ax.text(
        0.08,
        0.755,
        "Top-Attention Patches vs Slide-Wide Tissue Background",
        fontsize=20,
        fontweight="bold",
        color="#16324a",
    )

    chart_ax = fig.add_axes([0.09, 0.31, 0.82, 0.40])
    chart_ax.set_facecolor("white")

    x = np.arange(len(classes))
    width = 0.28

    for i, color in enumerate(colors):
        chart_ax.bar(
            i - width / 2,
            slide_background[i],
            width=width,
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
            hatch="///",
        )
        chart_ax.bar(
            i + width / 2,
            top_attention[i],
            width=width,
            facecolor=color,
            edgecolor="#2f3b45",
            linewidth=1.0,
        )
        chart_ax.text(
            i - width / 2,
            slide_background[i] + 1.0,
            f"{slide_background[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.2,
            color="#586b7a",
        )
        chart_ax.text(
            i + width / 2,
            top_attention[i] + 1.0,
            f"{top_attention[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.2,
            color="#1f2d3a",
            fontweight="bold",
        )

    chart_ax.set_xticks(x)
    chart_ax.set_xticklabels(classes, fontsize=12)
    chart_ax.tick_params(axis="x", pad=10)
    chart_ax.set_ylabel("Fraction of patches (%)", fontsize=13)
    chart_ax.set_ylim(0, 74)
    chart_ax.grid(axis="y", linestyle="--", alpha=0.25)
    chart_ax.spines[["top", "right"]].set_visible(False)

    style_handles = [
        Patch(
            facecolor="white",
            edgecolor="#4f5b66",
            hatch="///",
            linewidth=1.6,
            label="Slide-wide background",
        ),
        Patch(
            facecolor="#4f5b66",
            edgecolor="#2f3b45",
            linewidth=1.0,
            label="Top-attention patches",
        ),
    ]
    chart_ax.legend(handles=style_handles, frameon=False, loc="upper right", fontsize=11.5)

    ax.text(
        0.08,
        0.17,
        "After merging cluster 0, tumor-rich regions dominate the attended patches (67.7% vs 20.8%).",
        fontsize=15.5,
        color="#16324a",
        fontweight="bold",
    )
    ax.text(
        0.08,
        0.13,
        "Mixed-interface patches remain a distinct morphology bucket (14.7% vs 10.0%), while residual other clusters are now minor (4.2%).",
        fontsize=15.0,
        color="#16324a",
    )

    ax.text(
        0.08,
        0.075,
        "64 BRCA samples, 1,280 top-attention patches, 0 missing patch matches",
        fontsize=11.5,
        color="#7b8794",
    )
    ax.text(
        0.95,
        0.035,
        "BRCA attention interpretability summary",
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
