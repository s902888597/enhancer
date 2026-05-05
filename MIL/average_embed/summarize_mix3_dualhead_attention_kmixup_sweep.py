#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GROUPS = ["Combined", "BRCA", "LUAD", "SKCM"]
GROUP_TO_KEY = {
    "Combined": "fused_ALL",
    "BRCA": "fused_BRCA",
    "LUAD": "fused_LUAD",
    "SKCM": "fused_SKCM",
}


def parse_k_values(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--k-values", required=True)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    root = Path(args.root)
    k_values = parse_k_values(args.k_values)

    rows = []
    for k in k_values:
        summary_path = root / f"mix3_dualhead_attention_union4827_matrix_k{k}_mixup_seed{args.seed}" / "summary_by_split_and_cancer.csv"
        df = pd.read_csv(summary_path)
        for group in GROUPS:
            group_key = GROUP_TO_KEY[group]
            val_row = df[(df["split"] == "validation") & (df["group"] == group_key)].iloc[0]
            test_row = df[(df["split"] == "test") & (df["group"] == group_key)].iloc[0]
            rows.append(
                {
                    "K": k,
                    "Cancer": group,
                    "Validation": float(val_row["pearson_mean"]),
                    "Test": float(test_row["pearson_mean"]),
                }
            )

    out_df = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(root / "summary.csv", index=False)

    colors = {
        "Combined": "#5e81ac",
        "BRCA": "#d08770",
        "LUAD": "#a3be8c",
        "SKCM": "#b48ead",
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for group in GROUPS:
        sub = out_df[out_df["Cancer"] == group].sort_values("K")
        ax.plot(sub["K"], sub["Test"], marker="o", linewidth=2.2, color=colors[group], label=group)
        for _, row in sub.iterrows():
            ax.text(row["K"], row["Test"] + 0.003, f"{row['Test']:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Max patches (K)")
    ax.set_ylabel("Test mean Pearson r")
    ax.set_title("Mixed Dual-Head 4827 Attention + Mixup K Sweep")
    ax.set_xticks(k_values)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_ylim(0, out_df["Test"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(root / "mix3_dualhead_attention_union4827_kmixup_sweep.png", dpi=220)


if __name__ == "__main__":
    main()
