#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/figures")
OUT_PNG = OUT_DIR / "attention_dualhead_model_diagram_schematic.png"
OUT_PDF = OUT_DIR / "attention_dualhead_model_diagram_schematic.pdf"


def add_module(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    tc: str = "#111827",
    fontsize: float = 17,
    rounded: float = 0.015,
    lw: float = 2.0,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={rounded}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=tc,
    )
    return patch


def add_arrow(ax, start, end, color="#4b5563", lw=2.5, ls="-", rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=18,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def add_token_input(ax, x: float, y: float):
    add_module(ax, x, y, 0.065, 0.050, "x", fc="white", ec="#2563eb", tc="#1d4ed8", fontsize=18, rounded=0.01, lw=2.2)
    ax.text(x - 0.005, y + 0.070, "Patch tokens", ha="left", va="center", fontsize=15.5, fontweight="bold", color="#111827")
    ax.text(x + 0.073, y + 0.025, "k", ha="left", va="center", fontsize=15, color="#2563eb", fontweight="bold")


def add_fuse_node(ax, x: float, y: float):
    circ = Circle((x, y), 0.020, facecolor="white", edgecolor="#8b5cf6", linewidth=2.0)
    ax.add_patch(circ)
    ax.text(x, y, "+", ha="center", va="center", fontsize=18, fontweight="bold", color="#7c3aed")
    ax.text(x, y - 0.045, "Fuse", ha="center", va="center", fontsize=14, fontweight="bold", color="#111827")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(16, 7.6), dpi=220)
    fig.patch.set_facecolor("white")
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.92,
        "Attention Dual-Head MIL for Pan-Cancer SE Prediction",
        fontsize=28,
        fontweight="bold",
        color="#111827",
    )

    add_token_input(ax, 0.07, 0.53)

    ax.text(0.16, 0.565, r"$z_1, \ldots, z_k$", fontsize=18, color="#1d4ed8", fontweight="bold")

    add_module(
        ax,
        0.23,
        0.49,
        0.16,
        0.10,
        "Attention\npooling",
        fc="#d9cf2a",
        ec="#c2410c",
        tc="#111827",
        fontsize=20,
    )
    add_module(
        ax,
        0.44,
        0.505,
        0.055,
        0.07,
        "Emb",
        fc="white",
        ec="#9ca3af",
        tc="#6b7280",
        fontsize=15,
        rounded=0.008,
        lw=1.8,
    )

    add_module(
        ax,
        0.54,
        0.49,
        0.13,
        0.10,
        "Shared\nembedding",
        fc="#cff5f2",
        ec="#0ea5a4",
        tc="#111827",
        fontsize=19,
    )

    add_module(
        ax,
        0.76,
        0.58,
        0.11,
        0.075,
        "Pan head",
        fc="#fff2c2",
        ec="#ca8a04",
        tc="#111827",
        fontsize=17,
    )
    add_module(
        ax,
        0.76,
        0.39,
        0.13,
        0.085,
        "Specific head",
        fc="#f9d5e5",
        ec="#db2777",
        tc="#111827",
        fontsize=17,
    )

    add_fuse_node(ax, 0.80, 0.29)

    add_module(
        ax,
        0.69,
        0.14,
        0.14,
        0.085,
        "SE prediction",
        fc="#e9d5ff",
        ec="#8b5cf6",
        tc="#111827",
        fontsize=18,
    )

    ax.text(0.865, 0.175, r"$\hat{y}$", fontsize=18, color="#7c3aed", fontweight="bold", ha="left")

    ax.text(0.53, 0.08, r"$\mathcal{L}_{mask}$", fontsize=17, color="#6b7280", fontweight="bold")
    ax.text(0.61, 0.08, "valid targets only", fontsize=13, color="#9ca3af")

    add_arrow(ax, (0.135, 0.555), (0.23, 0.555), color="#ef4444", lw=2.8)
    add_arrow(ax, (0.39, 0.555), (0.44, 0.555), color="#ef4444", lw=2.8)
    add_arrow(ax, (0.495, 0.555), (0.54, 0.555), color="#ef4444", lw=2.8)

    add_arrow(ax, (0.67, 0.555), (0.76, 0.615), color="#2563eb", lw=2.8)
    add_arrow(ax, (0.67, 0.525), (0.76, 0.432), color="#2563eb", lw=2.8)

    add_arrow(ax, (0.815, 0.58), (0.80, 0.31), color="#2563eb", lw=2.6)
    add_arrow(ax, (0.825, 0.39), (0.80, 0.31), color="#2563eb", lw=2.6)
    add_arrow(ax, (0.79, 0.27), (0.76, 0.225), color="#7c3aed", lw=2.8)
    add_arrow(ax, (0.74, 0.14), (0.59, 0.09), color="#9ca3af", lw=2.0, ls="--")

    ax.text(0.27, 0.62, r"$\alpha_1, \ldots, \alpha_k$", fontsize=15, color="#dc2626", fontweight="bold")
    ax.text(0.71, 0.67, r"$\hat{y}_{pan}$", fontsize=15, color="#2563eb", fontweight="bold")
    ax.text(0.71, 0.36, r"$\hat{y}_{spec}$", fontsize=15, color="#2563eb", fontweight="bold")

    ax.text(0.95, 0.03, "schematic architecture", ha="right", fontsize=10.5, color="#9ca3af")

    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
