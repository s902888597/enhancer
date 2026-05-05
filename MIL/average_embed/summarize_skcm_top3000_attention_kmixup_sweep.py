#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_shareddata_top3000_attention_kmixup_sweep")
K_VALUES = [300, 500, 800]


def main() -> None:
    rows = []
    for k in K_VALUES:
        summary_path = ROOT / f"skcm_shareddata_top3000_attention_k{k}_mixup_seed44" / "summary.csv"
        df = pd.read_csv(summary_path)
        val_row = df[df["split"] == "validation"].iloc[0]
        test_row = df[df["split"] == "test"].iloc[0]
        rows.append(
            {
                "K": k,
                "Validation": float(val_row["pearson_mean"]),
                "Test": float(test_row["pearson_mean"]),
            }
        )

    out_df = pd.DataFrame(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(ROOT / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(out_df["K"], out_df["Validation"], marker="o", linewidth=2.2, color="#7aa6c2", label="Validation")
    ax.plot(out_df["K"], out_df["Test"], marker="o", linewidth=2.2, color="#d08770", label="Test")
    for _, row in out_df.iterrows():
        ax.text(row["K"], row["Test"] + 0.004, f"{row['Test']:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Max patches (K)")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("SKCM Top3000 Attention + Mixup K Sweep")
    ax.set_xticks(K_VALUES)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(out_df["Validation"].max(), out_df["Test"].max()) * 1.18)
    fig.tight_layout()
    fig.savefig(ROOT / "skcm_shareddata_top3000_attention_kmixup_sweep.png", dpi=220)


if __name__ == "__main__":
    main()
