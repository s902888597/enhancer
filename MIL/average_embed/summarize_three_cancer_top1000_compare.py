#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed")
OUT_DIR = ROOT / "three_cancer_top1000_compare"


def test_mean(path: Path) -> float:
    df = pd.read_csv(path)
    row = df.loc[df["split"] == "test"].iloc[0]
    return float(row["pearson_mean"])


def build_rows() -> list[dict]:
    return [
        {
            "Cancer": "BRCA",
            "Mean-MLP + yPCA5": test_mean(ROOT / "single_cancer_top1000_ypca_sweep" / "BRCA_PCAk5_seed44" / "summary.csv"),
            "Transformer + yPCA5": test_mean(ROOT / "brca_top1000_transformer_raw_ypca5_mp800_tta5_lam0p1_seed44" / "summary.csv"),
            "Attention (all patches)": test_mean(ROOT / "brca_top1000_attention_rawpatch_seed44" / "summary.csv"),
            "Attention (300 patches)": test_mean(ROOT / "brca_top1000_attention_rawpatch_k300_seed44" / "summary.csv"),
        },
        {
            "Cancer": "LUAD",
            "Mean-MLP + yPCA5": test_mean(ROOT / "single_cancer_top1000_ypca_sweep" / "LUAD_PCAk5_seed44" / "summary.csv"),
            "Transformer + yPCA5": test_mean(ROOT / "luad_top1000_transformer_raw_ypca5_mp800_tta5_lam0p1_seed44" / "summary.csv"),
            "Attention (all patches)": test_mean(ROOT / "luad_top1000_attention_rawpatch_seed44" / "summary.csv"),
            "Attention (300 patches)": test_mean(ROOT / "luad_top1000_attention_rawpatch_k300_seed44" / "summary.csv"),
        },
        {
            "Cancer": "SKCM",
            "Mean-MLP + yPCA5": test_mean(ROOT / "skcm_top1000_ypca5_mean_mlp_seed44" / "summary.csv"),
            "Transformer + yPCA5": test_mean(ROOT / "skcm_top1000_transformer_ypca5_mp800_tta5_lam0p1_seed44" / "summary.csv"),
            "Attention (all patches)": test_mean(ROOT / "skcm_top1000_attention_rawpatch_seed44" / "summary.csv"),
            "Attention (300 patches)": test_mean(ROOT / "skcm_top1000_attention_rawpatch_k300_seed44" / "summary.csv"),
        },
    ]


def save_csv(rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def save_png(rows: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "three_cancer_top1000_compare.png"
    cols = list(rows[0].keys())
    cell_text = [
        [
            row["Cancer"],
            f'{row["Mean-MLP + yPCA5"]:.4f}',
            f'{row["Transformer + yPCA5"]:.4f}',
            f'{row["Attention (all patches)"]:.4f}',
            f'{row["Attention (300 patches)"]:.4f}',
        ]
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(13, 3.1), dpi=220)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(
        0.0,
        1.07,
        "Top1000 Single-Cancer Comparison",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111827",
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        colLoc="left",
        cellLoc="left",
        colWidths=[0.12, 0.21, 0.22, 0.22, 0.23],
        bbox=[0.0, 0.0, 1.0, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#E5EEF9")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 == 1 else "#F9FAFB")
            if c >= 1:
                cell.set_text_props(weight="bold", color="#0F766E")
            else:
                cell.set_text_props(color="#111827")

    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.18)
    return out_png


def main() -> None:
    rows = build_rows()
    csv_path = save_csv(rows)
    png_path = save_png(rows)
    print(csv_path)
    print(png_path)


if __name__ == "__main__":
    main()
