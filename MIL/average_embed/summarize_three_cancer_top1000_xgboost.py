#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/three_cancer_shareddata_top1000_xgboost")
    cancers = ["BRCA", "LUAD", "SKCM"]
    rows = []
    for cancer in cancers:
        summary_path = root / f"{cancer}_xgboost1000_seed44" / "summary.csv"
        df = pd.read_csv(summary_path)
        test_row = df[df["split"] == "test"].iloc[0]
        val_row = df[df["split"] == "validation"].iloc[0]
        rows.append(
            {
                "Cancer": cancer,
                "Validation": float(val_row["pearson_mean"]),
                "Test": float(test_row["pearson_mean"]),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(root / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(out_df["Cancer"], out_df["Test"], color=["#7aa6c2", "#d08770", "#8fbc8f"], alpha=0.9)
    for bar, value in zip(bars, out_df["Test"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Test mean Pearson r")
    ax.set_title("Single-Cancer Top1000 z-score, Mean-Pooled XGBoost", fontsize=17, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(0, out_df["Test"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(root / "three_cancer_shareddata_top1000_xgboost.png", dpi=220)


if __name__ == "__main__":
    main()
