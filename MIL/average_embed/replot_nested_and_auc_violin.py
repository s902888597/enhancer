#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["Single-cancer Top3000", "Mixed single-head", "Mixed dual-head"]
CANCERS = ["BRCA", "LUAD", "SKCM"]
COLORS = {
    "Single-cancer Top3000": "#7aa6c2",
    "Mixed single-head": "#d08770",
    "Mixed dual-head": "#8fbc8f",
}

METHOD_RENAME = {
    "Single-cancer 3000": "Single-cancer Top3000",
    "Mixed single-head 4827": "Mixed single-head",
    "Mixed dual-head 4827": "Mixed dual-head",
}


def _draw_grouped_violins(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    title_fs: int = 19,
) -> None:
    x = np.arange(len(CANCERS))
    offsets = {
        "Single-cancer Top3000": -0.26,
        "Mixed single-head": 0.0,
        "Mixed dual-head": 0.26,
    }
    width = 0.22

    for method in METHODS:
        color = COLORS[method]
        for i, cancer in enumerate(CANCERS):
            vals = (
                df[(df["Cancer"] == cancer) & (df["Method"] == method)][value_col]
                .dropna()
                .astype(float)
                .to_numpy()
            )
            if vals.size == 0:
                continue
            pos = x[i] + offsets[method]
            parts = ax.violinplot(
                vals,
                positions=[pos],
                widths=width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.55)

            jitter = np.linspace(-0.035, 0.035, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(vals.size, pos) + jitter,
                vals,
                s=28,
                color=color,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            mean_val = float(np.mean(vals))
            ax.hlines(mean_val, pos - width * 0.42, pos + width * 0.42, color="black", linewidth=2.0, zorder=4)
            ax.text(pos, mean_val + 0.004, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(CANCERS, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=title_fs, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="y", labelsize=13)


def _add_method_legend(ax: plt.Axes) -> None:
    handles = [
        plt.Line2D([0], [0], color=COLORS[m], marker="o", linestyle="", markersize=8, label=m)
        for m in METHODS
    ]
    ax.legend(handles=handles, frameon=False, fontsize=13, loc="upper left")


def plot_nested_single_space(long_csv: Path, out_png: Path, title: str, ylabel: str) -> None:
    df = pd.read_csv(long_csv)
    df["Method"] = df["Method"].replace(METHOD_RENAME)
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    _draw_grouped_violins(ax, df, "pearson_mean", title, ylabel)
    _add_method_legend(ax)
    ymax = float(df["pearson_mean"].max() + 0.08)
    ax.set_ylim(min(0.0, float(df["pearson_mean"].min()) - 0.02), ymax)
    fig.text(0.5, 0.01, "Each dot represents one outer fold.", ha="center", va="bottom", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_nested_two_space(enh_csv: Path, pc_csv: Path, out_png: Path, title: str) -> None:
    enh = pd.read_csv(enh_csv).assign(space="Enhancer Space")
    pc = pd.read_csv(pc_csv).assign(space="PCA Space")
    enh["Method"] = enh["Method"].replace(METHOD_RENAME)
    pc["Method"] = pc["Method"].replace(METHOD_RENAME)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.6), sharey=False)
    _draw_grouped_violins(axes[0], enh, "pearson_mean", "Enhancer Space", "Outer-fold test mean Pearson r", title_fs=17)
    _draw_grouped_violins(axes[1], pc, "pearson_mean", "PCA Space", "Outer-fold test mean Pearson r (PC space)", title_fs=17)
    _add_method_legend(axes[1])
    fig.suptitle(title, fontsize=20, weight="bold")
    fig.text(0.5, 0.01, "Each dot represents one outer fold.", ha="center", va="bottom", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_direct_vs_pca_violin(enh_csv: Path, pc_csv: Path, out_png: Path, onepanel: bool) -> None:
    enh = pd.read_csv(enh_csv).assign(Setting="Direct")
    pc = pd.read_csv(pc_csv).assign(Setting="y-PCA5 inverse")
    enh["Method"] = enh["Method"].replace(METHOD_RENAME)
    pc["Method"] = pc["Method"].replace(METHOD_RENAME)
    df = pd.concat([enh, pc], ignore_index=True)

    if onepanel:
        fig, ax = plt.subplots(figsize=(15.0, 6.8))
        settings = ["Direct", "y-PCA5 inverse"]
        base_x = np.arange(len(CANCERS))
        setting_offsets = {"Direct": -0.6, "y-PCA5 inverse": 0.6}
        method_offsets = {
            "Single-cancer Top3000": -0.22,
            "Mixed single-head": 0.0,
            "Mixed dual-head": 0.22,
        }
        width = 0.18
        for setting in settings:
            for method in METHODS:
                color = COLORS[method]
                for i, cancer in enumerate(CANCERS):
                    vals = (
                        df[(df["Setting"] == setting) & (df["Cancer"] == cancer) & (df["Method"] == method)]["pearson_mean"]
                        .dropna()
                        .to_numpy()
                    )
                    if vals.size == 0:
                        continue
                    pos = base_x[i] + setting_offsets[setting] + method_offsets[method]
                    parts = ax.violinplot(vals, positions=[pos], widths=width, showmeans=False, showmedians=False, showextrema=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.50 if setting == "Direct" else 0.80)
                    jitter = np.linspace(-0.03, 0.03, vals.size) if vals.size > 1 else np.array([0.0])
                    marker = "o" if setting == "Direct" else "s"
                    ax.scatter(np.full(vals.size, pos) + jitter, vals, s=32, color=color, marker=marker, edgecolors="white", linewidths=0.7, zorder=3)
                    mean_val = float(np.mean(vals))
                    ax.hlines(mean_val, pos - width * 0.42, pos + width * 0.42, color="black", linewidth=1.8, zorder=4)
        ax.set_xticks(base_x)
        ax.set_xticklabels(CANCERS, fontsize=15)
        ax.set_ylabel("Outer-fold test mean Pearson r", fontsize=16)
        ax.set_title("Nested 5-Fold CV: Direct vs y-PCA5 inverse", fontsize=20, weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="y", labelsize=13)
        handles = [
            plt.Line2D([0], [0], color=COLORS[m], marker="o", linestyle="", markersize=8, label=m)
            for m in METHODS
        ]
        handles += [
            plt.Line2D([0], [0], color="gray", marker="o", linestyle="", markersize=7, label="Direct"),
            plt.Line2D([0], [0], color="gray", marker="s", linestyle="", markersize=7, label="y-PCA5 inverse"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=12, ncol=2, loc="upper left")
        ymax = float(df["pearson_mean"].max() + 0.08)
        ax.set_ylim(min(0.0, float(df["pearson_mean"].min()) - 0.02), ymax)
        fig.text(0.5, 0.01, "Each dot represents one outer fold.", ha="center", va="bottom", fontsize=13)
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.8), sharey=True)
    _draw_grouped_violins(axes[0], enh, "pearson_mean", "Direct Training", "Outer-fold test mean Pearson r", title_fs=17)
    _draw_grouped_violins(axes[1], pc, "pearson_mean", "y-PCA5 Inverse", "Outer-fold test mean Pearson r", title_fs=17)
    _add_method_legend(axes[1])
    fig.suptitle("Nested 5-Fold CV: Direct vs y-PCA5 inverse", fontsize=20, weight="bold")
    fig.text(0.5, 0.01, "Each dot represents one outer fold.", ha="center", va="bottom", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_auc_violin(long_csv: Path, out_png: Path, title: str) -> None:
    df = pd.read_csv(long_csv)
    df["Method"] = df["Method"].replace(METHOD_RENAME)
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    _draw_grouped_violins(ax, df, "roc_auc_mean", title, "5-fold mean ROC AUC")
    _add_method_legend(ax)
    ax.set_ylim(max(0.45, float(df["roc_auc_mean"].min()) - 0.03), min(1.0, float(df["roc_auc_mean"].max()) + 0.05))
    fig.text(0.5, 0.01, "Each dot represents one fold.", ha="center", va="bottom", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed")

    plot_nested_single_space(
        root / "nestedcv_attention_k300_mixup_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_compare/single3000_vs_mix4827_nestedcv_attention_k300_mixup.png",
        "5-Fold Nested CV, Attention Pooling (k=300):\nSingle-Cancer Top3000 vs Mixed Single-Head / Dual-Head",
        "Outer-fold test mean Pearson r",
    )
    plot_nested_single_space(
        root / "nestedcv_attention_k300_mixup_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_compare/single3000_vs_mix4827_nestedcv_attention_k300_mixup_clear_title.png",
        "5-Fold Nested CV, Attention Pooling (k=300):\nSingle-Cancer Top3000 vs Mixed Single-Head / Dual-Head",
        "Outer-fold test mean Pearson r",
    )

    plot_nested_single_space(
        root / "nestedcv_attention_k300_mixup_pca5_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_pca5_compare/single3000_vs_mix4827_nestedcv_attention_k300_mixup_pcspace.png",
        "5-Fold Nested CV, Attention Pooling (k=300):\nSingle-Cancer Top3000 vs Mixed Single-Head / Dual-Head (PCA space)",
        "Outer-fold test mean Pearson r (PC space)",
    )
    plot_nested_single_space(
        root / "nestedcv_attention_k300_mixup_pca5_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_pca5_compare/single3000_vs_mix4827_nestedcv_attention_k300_mixup_pcspace_clear_title.png",
        "5-Fold Nested CV, Attention Pooling (k=300):\nSingle-Cancer Top3000 vs Mixed Single-Head / Dual-Head (PCA space)",
        "Outer-fold test mean Pearson r (PC space)",
    )
    plot_nested_two_space(
        root / "nestedcv_attention_k300_mixup_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_pca5_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_pca5_compare/single3000_vs_mix4827_nestedcv_attention_k300_mixup_both_spaces.png",
        "5-Fold Nested CV, Attention Pooling (k=300): Both Spaces",
    )
    plot_nested_two_space(
        root / "nestedcv_attention_k300_mixup_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_pca5_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_pca5_compare/single3000_vs_mix4827_nestedcv_attention_k300_mixup_both_spaces_clear_title.png",
        "5-Fold Nested CV, Attention Pooling (k=300): Both Spaces",
    )

    plot_direct_vs_pca_violin(
        root / "nestedcv_attention_k300_mixup_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_pca5_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_compare/single3000_vs_mix4827_nestedcv_attention_k300_mixup_direct_vs_pca5_inverse.png",
        onepanel=False,
    )
    plot_direct_vs_pca_violin(
        root / "nestedcv_attention_k300_mixup_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_pca5_compare/long_table.csv",
        root / "nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_compare/single3000_vs_mix4827_nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_clear_title.png",
        onepanel=False,
    )
    for name in [
        "single3000_vs_mix4827_nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_onepanel.png",
        "single3000_vs_mix4827_nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_onepanel_clear_title.png",
        "single3000_vs_mix4827_nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_onepanel_nostripe.png",
        "single3000_vs_mix4827_nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_onepanel_nostripe_clear_title.png",
        "single3000_vs_mix4827_nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_onepanel_spaced.png",
        "single3000_vs_mix4827_nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_onepanel_spaced_clear_title.png",
    ]:
        plot_direct_vs_pca_violin(
            root / "nestedcv_attention_k300_mixup_compare/long_table.csv",
            root / "nestedcv_attention_k300_mixup_pca5_compare/long_table.csv",
            root / f"nestedcv_attention_k300_mixup_direct_vs_pca5_inverse_compare/{name}",
            onepanel=True,
        )

    plot_auc_violin(
        root / "auc_5fold_direct_zgt0/fivefold_zgt0_long_table.csv",
        root / "auc_5fold_direct_zgt0/fivefold_zgt0_auc_compare.png",
        "5-Fold ROC AUC Comparison (z > 0 threshold)",
    )
    plot_auc_violin(
        root / "auc_5fold_direct_zgt0/fivefold_zgt0_long_table.csv",
        root / "auc_5fold_direct_zgt0/fivefold_zgt0_auc_compare_clear_title.png",
        "5-Fold ROC AUC Comparison (z > 0 threshold)",
    )
    plot_auc_violin(
        root / "auc_5fold_direct_median_threshold/fivefold_median_threshold_long_table.csv",
        root / "auc_5fold_direct_median_threshold/fivefold_median_threshold_auc_compare.png",
        "5-Fold ROC AUC Comparison (median threshold)",
    )
    plot_auc_violin(
        root / "auc_5fold_direct_median_threshold/fivefold_median_threshold_long_table.csv",
        root / "auc_5fold_direct_median_threshold/fivefold_median_threshold_auc_compare_clear_title.png",
        "5-Fold ROC AUC Comparison (median threshold)",
    )


if __name__ == "__main__":
    main()
