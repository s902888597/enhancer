#!/usr/bin/env python3
"""Summarize z-score union top3000 fixed-split runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_ROOT = Path(
    "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/"
    "zscore_union21477_top3000_fixedsplit"
)
CANCERS = ["BRCA", "LUAD", "SKCM"]
METHODS = ["single", "mixed_pan", "mixed_specific", "mixed_fused"]
LABELS = {
    "single": "Single-cancer attention",
    "mixed_pan": "Mixed pan head",
    "mixed_specific": "Mixed cancer-specific head",
    "mixed_fused": "Mixed fused",
}


def load_single(cancer: str) -> float:
    csv_path = RUN_ROOT / "single_attention_k300_mixup" / cancer / "summary.csv"
    df = pd.read_csv(csv_path)
    row = df.loc[df["split"] == "test"]
    if row.empty:
        raise RuntimeError(f"No test row in {csv_path}")
    return float(row.iloc[0]["pearson_mean"])


def load_mixed(cancer: str, head: str) -> float:
    csv_path = RUN_ROOT / "mixed_dualhead_attention_k300_mixup" / "summary_by_split_and_cancer.csv"
    df = pd.read_csv(csv_path)
    group = f"{head}_{cancer}"
    row = df.loc[(df["split"] == "test") & (df["group"] == group)]
    if row.empty:
        raise RuntimeError(f"No test row for {group} in {csv_path}")
    return float(row.iloc[0]["pearson_mean"])


def main() -> None:
    rows = []
    for cancer in CANCERS:
        rows.append({"cancer": cancer, "method": "single", "pearson_mean": load_single(cancer)})
        for head in ["pan", "specific", "fused"]:
            rows.append(
                {
                    "cancer": cancer,
                    "method": f"mixed_{head}",
                    "pearson_mean": load_mixed(cancer, head),
                }
            )
    out_df = pd.DataFrame(rows)
    out_csv = RUN_ROOT / "zscore_top3000_fixedsplit_test_mean_pearson_summary.csv"
    out_df.to_csv(out_csv, index=False)

    pivot = out_df.pivot(index="cancer", columns="method", values="pearson_mean").loc[CANCERS, METHODS]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    colors = ["#6D9DC5", "#E2A06F", "#8BBE8C", "#C07AA4"]
    x = range(len(CANCERS))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for method, offset, color in zip(METHODS, offsets, colors):
        vals = pivot[method].to_numpy()
        bars = ax.bar([i + offset for i in x], vals, width=width, label=LABELS[method], color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Fixed Split Top3000 z-score SE Prediction (Attention, k=300, mixup)", fontsize=14, weight="bold")
    ax.set_ylabel("Test mean Pearson r")
    ax.set_xticks(list(x))
    ax.set_xticklabels(CANCERS)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out_png = RUN_ROOT / "zscore_top3000_fixedsplit_attention_comparison.png"
    fig.savefig(out_png)
    print(out_csv)
    print(out_png)
    print(pivot.round(4))


if __name__ == "__main__":
    main()
