#!/usr/bin/env python3
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/figures")
OUT_PNG = OUT_DIR / "brca_attention_representative_cases_slide.png"
OUT_PDF = OUT_DIR / "brca_attention_representative_cases_slide.pdf"

ARROW_ROOT = Path(
    "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/explainability/"
    "brca_top1000_attention_rawpatch_ypca5_seed44/arrow_layout_top5"
)


CASES = [
    {
        "title": "Tumor-rich attention",
        "sample": "TCGA-LL-A5YP",
        "r": "0.589",
        "caption": "Top patches concentrate on dense tumor regions.",
        "path": ARROW_ROOT / "TCGA-LL-A5YP_top5_arrows.png",
    },
    {
        "title": "Immune-focused attention",
        "sample": "TCGA-A8-A0A1",
        "r": "0.505",
        "caption": "Top patches highlight lymphocyte-rich areas.",
        "path": ARROW_ROOT / "TCGA-A8-A0A1_top5_arrows.png",
    },
    {
        "title": "Mixed / interface attention",
        "sample": "TCGA-A2-A04T",
        "r": "0.607",
        "caption": "Attention spans heterogeneous tumor-interface regions.",
        "path": ARROW_ROOT / "TCGA-A2-A04T_top5_arrows.png",
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(16, 9), dpi=220)
    fig.patch.set_facecolor("#fbfaf7")
    canvas = plt.axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    canvas.text(
        0.05,
        0.93,
        "Representative BRCA Attention Patterns",
        fontsize=28,
        fontweight="bold",
        color="#16324a",
    )
    canvas.text(
        0.05,
        0.885,
        "Three representative test samples show that attention can focus on tumor-rich, immune-rich, or mixed/interface regions.",
        fontsize=13.0,
        color="#586b7a",
    )

    panel = FancyBboxPatch(
        (0.04, 0.11),
        0.92,
        0.72,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.6,
        edgecolor="#d7dde5",
        facecolor="white",
    )
    canvas.add_patch(panel)

    lefts = [0.055, 0.365, 0.675]
    width = 0.27
    for left, case in zip(lefts, CASES):
        card = FancyBboxPatch(
            (left, 0.16),
            width,
            0.61,
            boxstyle="round,pad=0.010,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#e2e8f0",
            facecolor="#fcfcfb",
        )
        canvas.add_patch(card)

        canvas.text(left + 0.015, 0.735, case["title"], fontsize=17.0, fontweight="bold", color="#16324a")
        canvas.text(left + 0.015, 0.705, f"{case['sample']}   Pearson r = {case['r']}", fontsize=11.6, color="#6b7280")

        ax_img = fig.add_axes([left + 0.012, 0.235, width - 0.024, 0.44])
        ax_img.imshow(mpimg.imread(case["path"]))
        ax_img.axis("off")

        canvas.text(
            left + 0.015,
            0.185,
            case["caption"],
            fontsize=11.7,
            color="#334155",
        )

    canvas.text(
        0.05,
        0.075,
        "Use with the quantitative BRCA tissue-composition slide: together they show that attention is selective and morphologically interpretable.",
        fontsize=11.5,
        color="#7b8794",
    )

    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
