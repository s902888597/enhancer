#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANCERS = ["BRCA", "LUAD", "SKCM"]
METHODS = [
    ("Single-cancer 3000", "single"),
    ("Mixed single-head 4827", "mixed_single"),
    ("Mixed dual-head 4827", "mixed_dual"),
]
COLORS = {
    "Single-cancer 3000": "#7aa6c2",
    "Mixed single-head 4827": "#d08770",
    "Mixed dual-head 4827": "#8fbc8f",
}


def load_single_fold(single_root: Path, fold: int, cancer: str, space: str) -> float:
    if space == "pc":
        summary_path = single_root / f"fold{fold}" / cancer / "predicted_pc_summary.csv"
        df = pd.read_csv(summary_path)
        row = df[df["split"] == "test"].iloc[0]
        return float(row["pearson_mean"])
    summary_path = single_root / f"fold{fold}" / cancer / "summary.csv"
    df = pd.read_csv(summary_path)
    row = df[df["split"] == "test"].iloc[0]
    return float(row["pearson_mean"])


def load_mixed_fold(mixed_root: Path, fold: int, cancer: str, space: str) -> float:
    if space == "pc":
        summary_path = mixed_root / f"fold{fold}" / "predicted_pc_summary_by_split_and_cancer.csv"
        df = pd.read_csv(summary_path)
        row = df[(df["split"] == "test") & (df["group"] == f"fused_{cancer}")].iloc[0]
        return float(row["pearson_mean"])
    summary_path = mixed_root / f"fold{fold}" / "summary_by_split_and_cancer.csv"
    df = pd.read_csv(summary_path)
    row = df[(df["split"] == "test") & (df["group"] == f"fused_{cancer}")].iloc[0]
    return float(row["pearson_mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-root", required=True)
    parser.add_argument("--mixed-single-root", required=True)
    parser.add_argument("--mixed-dual-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--space", choices=["enhancer", "pc"], default="enhancer")
    parser.add_argument("--title-prefix", default="5-Fold Nested CV, Attention Pooling (k=300, mixup)")
    args = parser.parse_args()

    single_root = Path(args.single_root)
    mixed_single_root = Path(args.mixed_single_root)
    mixed_dual_root = Path(args.mixed_dual_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, object]] = []
    for fold in range(args.n_folds):
        for cancer in CANCERS:
            long_rows.append(
                {
                    "fold": fold,
                    "Cancer": cancer,
                    "Method": "Single-cancer 3000",
                    "pearson_mean": load_single_fold(single_root, fold, cancer, args.space),
                }
            )
            long_rows.append(
                {
                    "fold": fold,
                    "Cancer": cancer,
                    "Method": "Mixed single-head 4827",
                    "pearson_mean": load_mixed_fold(mixed_single_root, fold, cancer, args.space),
                }
            )
            long_rows.append(
                {
                    "fold": fold,
                    "Cancer": cancer,
                    "Method": "Mixed dual-head 4827",
                    "pearson_mean": load_mixed_fold(mixed_dual_root, fold, cancer, args.space),
                }
            )

    long_df = pd.DataFrame(long_rows)
    summary = (
        long_df.groupby(["Cancer", "Method"], as_index=False)
        .agg(
            mean_pearson=("pearson_mean", "mean"),
            std_pearson=("pearson_mean", "std"),
            n_folds=("pearson_mean", "size"),
        )
    )
    summary["std_pearson"] = summary["std_pearson"].fillna(0.0)
    long_df.to_csv(out_dir / "long_table.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    x = np.arange(len(CANCERS))
    width = 0.24
    for offset, (method, _) in zip([-width, 0.0, width], METHODS):
        sub = summary[summary["Method"] == method].set_index("Cancer").loc[CANCERS]
        bars = ax.bar(
            x + offset,
            sub["mean_pearson"],
            width=width,
            yerr=sub["std_pearson"],
            capsize=4,
            color=COLORS[method],
            label=method,
            alpha=0.9,
        )
        for bar, value in zip(bars, sub["mean_pearson"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CANCERS, fontsize=12)
    ylabel = "Outer-fold test mean Pearson r"
    if args.space == "pc":
        ylabel = "Outer-fold test mean Pearson r (PC space)"
    ax.set_ylabel(ylabel, fontsize=13)
    title_suffix = "Single-Cancer Top3000 vs Mixed 4827 Single-Head / Dual-Head"
    if args.space == "pc":
        title_suffix = "Single-Cancer Top3000 vs Mixed 4827 Single-Head / Dual-Head (PCA space)"
    ax.set_title(f"{args.title_prefix}:\n{title_suffix}", fontsize=17, weight="bold")
    ax.legend(frameon=False, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ymax = float(summary["mean_pearson"].max() + summary["std_pearson"].max() + 0.06)
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    if args.space == "pc":
        fig.savefig(out_dir / "single3000_vs_mix4827_nestedcv_attention_k300_mixup_pcspace.png", dpi=220)
        fig.savefig(out_dir / "single3000_vs_mix4827_nestedcv_attention_k300_mixup_pcspace_clear_title.png", dpi=220)
    else:
        fig.savefig(out_dir / "single3000_vs_mix4827_nestedcv_attention_k300_mixup.png", dpi=220)
        fig.savefig(out_dir / "single3000_vs_mix4827_nestedcv_attention_k300_mixup_clear_title.png", dpi=220)


if __name__ == "__main__":
    main()
