#!/usr/bin/env python3
"""Summarize eRNA_count>=4 min-max union top3000 fixed-split runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_ROOT = Path(
    "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/"
    "minmax_union21477_erna_ge4_top3000_fixedsplit"
)
CANCERS = ["BRCA", "LUAD", "SKCM"]
METHODS = ["single", "mixed_pan", "mixed_specific", "mixed_fused"]
METHOD_SPECS = {
    "single": {"kind": "single", "subdir": "single_attention_k300_mixup", "summary_name": "summary.csv", "label": "Single-cancer attention"},
    "single_nmf20": {"kind": "single", "subdir": "single_attention_k300_mixup_nmf20", "summary_name": "summary.csv", "label": "Single-cancer attention + NMF20"},
    "mixed_pan": {"kind": "mixed", "subdir": "mixed_dualhead_attention_k300_mixup", "head": "pan", "label": "Mixed pan head"},
    "mixed_specific": {"kind": "mixed", "subdir": "mixed_dualhead_attention_k300_mixup", "head": "specific", "label": "Mixed cancer-specific head"},
    "mixed_fused": {"kind": "mixed", "subdir": "mixed_dualhead_attention_k300_mixup", "head": "fused", "label": "Mixed fused"},
    "mixed_pan_nmf20": {"kind": "mixed", "subdir": "mixed_dualhead_attention_k300_mixup_nmf20", "head": "pan", "label": "Mixed pan head + NMF20"},
    "mixed_specific_nmf20": {"kind": "mixed", "subdir": "mixed_dualhead_attention_k300_mixup_nmf20", "head": "specific", "label": "Mixed cancer-specific head + NMF20"},
    "mixed_fused_nmf20": {"kind": "mixed", "subdir": "mixed_dualhead_attention_k300_mixup_nmf20", "head": "fused", "label": "Mixed fused + NMF20"},
}
LABELS = {k: v["label"] for k, v in METHOD_SPECS.items()}
NMF20_METHODS = ["single_nmf20", "mixed_pan_nmf20", "mixed_specific_nmf20", "mixed_fused_nmf20"]
ALL_METHODS = METHODS + NMF20_METHODS


def load_single(cancer: str, subdir: str, summary_name: str) -> float:
    df = pd.read_csv(RUN_ROOT / subdir / cancer / summary_name)
    return float(df.loc[df["split"] == "test"].iloc[0]["pearson_mean"])


def load_mixed(cancer: str, subdir: str, head: str) -> float:
    df = pd.read_csv(RUN_ROOT / subdir / "summary_by_split_and_cancer.csv")
    return float(df.loc[(df["split"] == "test") & (df["group"] == f"{head}_{cancer}")].iloc[0]["pearson_mean"])


def load_method(cancer: str, method: str) -> float:
    spec = METHOD_SPECS[method]
    if spec["kind"] == "single":
        return load_single(cancer, spec["subdir"], spec["summary_name"])
    return load_mixed(cancer, spec["subdir"], spec["head"])


def main() -> None:
    rows = []
    for cancer in CANCERS:
        for method in ALL_METHODS:
            try:
                pearson = load_method(cancer, method)
            except FileNotFoundError:
                continue
            rows.append({"cancer": cancer, "method": method, "pearson_mean": pearson})
    out_df = pd.DataFrame(rows)
    out_csv = RUN_ROOT / "minmax_erna_ge4_top3000_fixedsplit_test_mean_pearson_summary.csv"
    out_df.to_csv(out_csv, index=False)

    pivot = out_df.pivot(index="cancer", columns="method", values="pearson_mean")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    colors = ["#6D9DC5", "#E2A06F", "#8BBE8C", "#C07AA4"]
    x = range(len(CANCERS))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for method, offset, color in zip(METHODS, offsets, colors):
        if method not in pivot.columns:
            continue
        vals = pivot.loc[CANCERS, method].to_numpy()
        bars = ax.bar([i + offset for i in x], vals, width=width, label=LABELS[method], color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Fixed Split Top3000 min-max SE Prediction, eRNA_count >= 4 (Attention, k=300, mixup)", fontsize=13, weight="bold")
    ax.set_ylabel("Test mean Pearson r")
    ax.set_xticks(list(x))
    ax.set_xticklabels(CANCERS)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out_png = RUN_ROOT / "minmax_erna_ge4_top3000_fixedsplit_attention_comparison.png"
    fig.savefig(out_png)

    if all(method in pivot.columns for method in ALL_METHODS):
        pair_order = [
            ("single", "single_nmf20"),
            ("mixed_pan", "mixed_pan_nmf20"),
            ("mixed_specific", "mixed_specific_nmf20"),
            ("mixed_fused", "mixed_fused_nmf20"),
        ]
        fig2, ax2 = plt.subplots(figsize=(14, 5.8), dpi=180)
        width2 = 0.09
        pair_centers = [0.0, 1.0, 2.0, 3.0]
        cancer_offsets = [-0.22, 0.0, 0.22]
        cancer_colors = {"BRCA": "#6D9DC5", "LUAD": "#E2A06F", "SKCM": "#8BBE8C"}
        legend_done = set()
        for pair_idx, (base_method, nmf_method) in enumerate(pair_order):
            center = pair_centers[pair_idx]
            for cancer_idx, cancer in enumerate(CANCERS):
                offset = cancer_offsets[cancer_idx]
                color = cancer_colors[cancer]
                base_val = float(pivot.loc[cancer, base_method])
                nmf_val = float(pivot.loc[cancer, nmf_method])
                base_x = center + offset - width2 / 2
                nmf_x = center + offset + width2 / 2
                base_label = f"{cancer} baseline" if cancer not in legend_done else None
                nmf_label = f"{cancer} + NMF20" if f"{cancer}_nmf20" not in legend_done else None
                ax2.bar(base_x, base_val, width=width2, color=color, label=base_label)
                ax2.bar(nmf_x, nmf_val, width=width2, color=color, hatch="///", edgecolor="#444444", linewidth=0.6, label=nmf_label)
                ax2.text(base_x, base_val + 0.004, f"{base_val:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
                ax2.text(nmf_x, nmf_val + 0.004, f"{nmf_val:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
                legend_done.add(cancer)
                legend_done.add(f"{cancer}_nmf20")
        ax2.set_xticks(pair_centers)
        ax2.set_xticklabels(
            [
                "Single-cancer\nattention",
                "Mixed\npan head",
                "Mixed\ncancer-specific head",
                "Mixed\nfused",
            ]
        )
        ax2.set_ylabel("Test mean Pearson r")
        ax2.set_title("Fixed Split Top3000 min-max: baseline vs NMF20", fontsize=13, weight="bold")
        ax2.axhline(0, color="#333333", linewidth=0.8)
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(frameon=False, ncol=3)
        fig2.tight_layout()
        compare_png = RUN_ROOT / "minmax_erna_ge4_top3000_fixedsplit_nmf20_vs_baseline.png"
        fig2.savefig(compare_png)

        nmf_rows = []
        for cancer in CANCERS:
            for method in ALL_METHODS:
                nmf_rows.append({"cancer": cancer, "method": method, "pearson_mean": float(pivot.loc[cancer, method])})
        nmf_csv = RUN_ROOT / "minmax_erna_ge4_top3000_fixedsplit_nmf20_test_mean_pearson_summary.csv"
        pd.DataFrame(nmf_rows).to_csv(nmf_csv, index=False)
        print(nmf_csv)
        print(compare_png)
    print(out_csv)
    print(out_png)
    print(pivot.round(4))


if __name__ == "__main__":
    main()
