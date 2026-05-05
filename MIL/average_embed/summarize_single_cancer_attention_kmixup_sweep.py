#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_k_values(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--k-values", required=True, help="Comma-separated list like 300,500,800")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    cancer = args.cancer.upper()
    root = Path(args.root)
    k_values = parse_k_values(args.k_values)

    rows = []
    for k in k_values:
        summary_path = root / f"{cancer}_shareddata_top3000_attention_k{k}_mixup_seed{args.seed}" / "summary.csv"
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

    out_df = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    root.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(root / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(out_df["K"], out_df["Validation"], marker="o", linewidth=2.2, color="#7aa6c2", label="Validation")
    ax.plot(out_df["K"], out_df["Test"], marker="o", linewidth=2.2, color="#d08770", label="Test")
    for _, row in out_df.iterrows():
        ax.text(row["K"], row["Test"] + 0.004, f"{row['Test']:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Max patches (K)")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title(f"{cancer} Top3000 Attention + Mixup K Sweep")
    ax.set_xticks(out_df["K"].tolist())
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(out_df["Validation"].max(), out_df["Test"].max()) * 1.18)
    fig.tight_layout()
    fig.savefig(root / f"{cancer}_shareddata_top3000_attention_kmixup_sweep.png", dpi=220)


if __name__ == "__main__":
    main()
