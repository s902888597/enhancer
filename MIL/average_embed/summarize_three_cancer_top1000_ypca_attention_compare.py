#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed")
OUT_DIR = ROOT / "three_cancer_top1000_ypca_attention_compare"


def test_mean(path: Path) -> float:
    df = pd.read_csv(path)
    row = df.loc[df["split"] == "test"].iloc[0]
    return float(row["pearson_mean"])


def test_pc_mean(path: Path) -> float:
    if path.exists():
        df = pd.read_csv(path)
        row = df.loc[df["split"] == "test"].iloc[0]
        return float(row["pearson_mean"])
    pred_path = path.parent / "test_pred_pca.npy"
    true_path = path.parent / "test_true_pca.npy"
    pred = np.load(pred_path)
    true = np.load(true_path)
    vals = []
    for idx in range(pred.shape[1]):
        x = pred[:, idx]
        y = true[:, idx]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def build_enhancer_rows() -> list[dict]:
    return [
        {
            "Cancer": "BRCA",
            "Mean-MLP + yPCA5": test_mean(ROOT / "single_cancer_top1000_ypca_sweep" / "BRCA_PCAk5_seed44" / "summary.csv"),
            "Attention (all patches) + yPCA5": test_mean(ROOT / "brca_top1000_attention_rawpatch_ypca5_seed44" / "summary.csv"),
            "Attention (300 patches) + yPCA5": test_mean(ROOT / "brca_top1000_attention_rawpatch_k300_ypca5_seed44" / "summary.csv"),
        },
        {
            "Cancer": "LUAD",
            "Mean-MLP + yPCA5": test_mean(ROOT / "single_cancer_top1000_ypca_sweep" / "LUAD_PCAk5_seed44" / "summary.csv"),
            "Attention (all patches) + yPCA5": test_mean(ROOT / "luad_top1000_attention_rawpatch_ypca5_seed44" / "summary.csv"),
            "Attention (300 patches) + yPCA5": test_mean(ROOT / "luad_top1000_attention_rawpatch_k300_ypca5_seed44" / "summary.csv"),
        },
        {
            "Cancer": "SKCM",
            "Mean-MLP + yPCA5": test_mean(ROOT / "skcm_top1000_ypca5_mean_mlp_seed44" / "summary.csv"),
            "Attention (all patches) + yPCA5": test_mean(ROOT / "skcm_top1000_attention_rawpatch_ypca5_seed44" / "summary.csv"),
            "Attention (300 patches) + yPCA5": test_mean(ROOT / "skcm_top1000_attention_rawpatch_k300_ypca5_seed44" / "summary.csv"),
        },
    ]


def build_pc_rows() -> list[dict]:
    return [
        {
            "Cancer": "BRCA",
            "Mean-MLP + yPCA5": test_pc_mean(ROOT / "single_cancer_top1000_ypca_sweep" / "BRCA_PCAk5_seed44" / "predicted_pc_summary.csv"),
            "Attention (all patches) + yPCA5": test_pc_mean(ROOT / "brca_top1000_attention_rawpatch_ypca5_seed44" / "predicted_pc_summary.csv"),
            "Attention (300 patches) + yPCA5": test_pc_mean(ROOT / "brca_top1000_attention_rawpatch_k300_ypca5_seed44" / "predicted_pc_summary.csv"),
        },
        {
            "Cancer": "LUAD",
            "Mean-MLP + yPCA5": test_pc_mean(ROOT / "single_cancer_top1000_ypca_sweep" / "LUAD_PCAk5_seed44" / "predicted_pc_summary.csv"),
            "Attention (all patches) + yPCA5": test_pc_mean(ROOT / "luad_top1000_attention_rawpatch_ypca5_seed44" / "predicted_pc_summary.csv"),
            "Attention (300 patches) + yPCA5": test_pc_mean(ROOT / "luad_top1000_attention_rawpatch_k300_ypca5_seed44" / "predicted_pc_summary.csv"),
        },
        {
            "Cancer": "SKCM",
            "Mean-MLP + yPCA5": test_pc_mean(ROOT / "skcm_top1000_ypca5_mean_mlp_seed44" / "predicted_pc_summary.csv"),
            "Attention (all patches) + yPCA5": test_pc_mean(ROOT / "skcm_top1000_attention_rawpatch_ypca5_seed44" / "predicted_pc_summary.csv"),
            "Attention (300 patches) + yPCA5": test_pc_mean(ROOT / "skcm_top1000_attention_rawpatch_k300_ypca5_seed44" / "predicted_pc_summary.csv"),
        },
    ]


def save_csv(rows: list[dict], out_name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / out_name
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def save_png(rows: list[dict], title: str, out_name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / out_name
    cols = list(rows[0].keys())
    value_cols = cols[1:]
    cell_text = [[row["Cancer"], *[f"{row[c]:.4f}" for c in value_cols]] for row in rows]

    fig, ax = plt.subplots(figsize=(12.6, 3.1), dpi=220)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(
        0.0,
        1.07,
        title,
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
        colWidths=[0.12, 0.29, 0.29, 0.30],
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
    enhancer_rows = build_enhancer_rows()
    pc_rows = build_pc_rows()
    enh_csv = save_csv(enhancer_rows, "enhancer_summary.csv")
    pc_csv = save_csv(pc_rows, "pc_summary.csv")
    enh_png = save_png(
        enhancer_rows,
        "Top1000 Single-Cancer yPCA5 Comparison (Enhancer Space)",
        "three_cancer_top1000_ypca_attention_compare_enhancer.png",
    )
    pc_png = save_png(
        pc_rows,
        "Top1000 Single-Cancer yPCA5 Comparison (PC Space)",
        "three_cancer_top1000_ypca_attention_compare_pc.png",
    )
    print(enh_csv)
    print(pc_csv)
    print(enh_png)
    print(pc_png)


if __name__ == "__main__":
    main()
