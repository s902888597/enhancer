#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/three_cancer_shareddata_top3000_direct_mean_mlp")
CANCERS = ["BRCA", "LUAD", "SKCM"]


def main() -> None:
    rows = []
    for cancer in CANCERS:
        summary_path = ROOT / f"{cancer}_direct3000_seed44" / "summary.csv"
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
    ROOT.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(ROOT / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = range(len(CANCERS))
    width = 0.34
    ax.bar([i - width / 2 for i in x], out_df["Validation"], width=width, label="Validation", color="#7aa6c2")
    ax.bar([i + width / 2 for i in x], out_df["Test"], width=width, label="Test", color="#d08770")
    ax.set_xticks(list(x))
    ax.set_xticklabels(CANCERS)
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Single-Cancer Direct Top3000 (z-score)")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(out_df["Validation"].max(), out_df["Test"].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(ROOT / "three_cancer_shareddata_top3000_direct_mean_mlp.png", dpi=200)


if __name__ == "__main__":
    main()
