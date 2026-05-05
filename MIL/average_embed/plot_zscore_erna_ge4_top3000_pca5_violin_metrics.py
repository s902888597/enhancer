#!/usr/bin/env python3
"""Plot PCA5 z-score top3000 metrics in PC space and inverse-transformed SE space."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mzang")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score


CANCERS = ["BRCA", "LUAD", "SKCM"]
METHOD_ORDER = ["Single-cancer", "Mixed single-head", "Mixed dual-head"]
PALETTE = ["#7aa6c2", "#d58b73", "#8fbe8f"]
POINT_PALETTE = ["#1f5673", "#944d38", "#3f7f3f"]


def per_target_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_centered = y_true - y_true.mean(axis=0, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((true_centered**2).sum(axis=0) * (pred_centered**2).sum(axis=0))
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    ok = denom > 1e-12
    out[ok] = (true_centered[:, ok] * pred_centered[:, ok]).sum(axis=0) / denom[ok]
    return out


def per_target_auc(y_train: np.ndarray, y_test: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    thresholds = np.median(y_train, axis=0)
    out = np.full(y_test.shape[1], np.nan, dtype=np.float64)
    for j in range(y_test.shape[1]):
        y_bin = (y_test[:, j] > thresholds[j]).astype(np.int8)
        if y_bin.min() == y_bin.max():
            continue
        out[j] = roc_auc_score(y_bin, y_score[:, j])
    return out


def inverse_pca(scores: np.ndarray, pca_dir: Path) -> np.ndarray:
    comps = np.load(pca_dir / "pca_components.npy")
    mean = np.load(pca_dir / "pca_mean.npy")
    return scores @ comps + mean


def add_metric_rows(
    rows: list[dict],
    space: str,
    cancer: str,
    method: str,
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    pearson = per_target_pearson(y_true, y_pred)
    auc = per_target_auc(y_train, y_true, y_pred)
    target_prefix = "PC" if space == "PCA space" else "SE"
    for idx, value in enumerate(pearson):
        if np.isfinite(value):
            rows.append(
                {
                    "space": space,
                    "metric": "Pearson r",
                    "cancer": cancer,
                    "method": method,
                    "target": f"{target_prefix}{idx + 1}",
                    "value": value,
                }
            )
    for idx, value in enumerate(auc):
        if np.isfinite(value):
            rows.append(
                {
                    "space": space,
                    "metric": "ROC AUC",
                    "cancer": cancer,
                    "method": method,
                    "target": f"{target_prefix}{idx + 1}",
                    "value": value,
                }
            )


def annotate_means(ax, plot_df: pd.DataFrame, metric: str) -> None:
    offsets = np.linspace(-0.26, 0.26, len(METHOD_ORDER))
    means = plot_df.groupby(["cancer", "method"])["value"].mean()
    for cancer_idx, cancer in enumerate(CANCERS):
        for method_idx, method in enumerate(METHOD_ORDER):
            key = (cancer, method)
            if key not in means.index:
                continue
            value = float(means.loc[key])
            y = value + (0.035 if metric == "Pearson r" else 0.025)
            ax.text(
                cancer_idx + offsets[method_idx],
                y,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="#1b1f24",
            )


def draw_one(df: pd.DataFrame, space: str, metric: str, out_png: Path) -> None:
    plot_df = df[(df["space"] == space) & (df["metric"] == metric)].copy()
    sns.set_theme(style="white", context="talk", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(16, 8.5))
    sns.violinplot(
        data=plot_df,
        x="cancer",
        y="value",
        hue="method",
        order=CANCERS,
        hue_order=METHOD_ORDER,
        cut=0,
        inner="box",
        split=False,
        density_norm="width",
        gap=0.08,
        linewidth=1.1,
        palette=PALETTE,
        ax=ax,
    )
    sns.pointplot(
        data=plot_df,
        x="cancer",
        y="value",
        hue="method",
        order=CANCERS,
        hue_order=METHOD_ORDER,
        dodge=0.55,
        join=False,
        errorbar=None,
        markers="D",
        scale=0.55,
        palette=POINT_PALETTE,
        legend=False,
        ax=ax,
    )
    if metric == "ROC AUC":
        ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_ylim(0.0, 1.0)
    else:
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_ylim(0.0, 0.9 if space == "PCA space" else 0.55)
    n_targets = 5 if space == "PCA space" else 3000
    ax.set_title(f"z-score Top3000 yPCA5: test {metric} ({space})", fontsize=25, weight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=19)
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(False)
    sns.despine(ax=ax)
    annotate_means(ax, plot_df, metric)
    ax.text(
        0.01,
        -0.16,
        f"Each violin = {n_targets} targets; diamond = mean. ROC AUC positive label uses train median.",
        transform=ax.transAxes,
        fontsize=14,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[: len(METHOD_ORDER)],
        labels[: len(METHOD_ORDER)],
        title="",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=22,
        handlelength=1.6,
        labelspacing=0.8,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_combined(df: pd.DataFrame, space: str, out_png: Path) -> None:
    sns.set_theme(style="white", context="talk", font_scale=1.15)
    fig, axes = plt.subplots(2, 1, figsize=(17, 14), sharex=True)
    for ax, metric in zip(axes, ["Pearson r", "ROC AUC"]):
        plot_df = df[(df["space"] == space) & (df["metric"] == metric)].copy()
        sns.violinplot(
            data=plot_df,
            x="cancer",
            y="value",
            hue="method",
            order=CANCERS,
            hue_order=METHOD_ORDER,
            cut=0,
            inner="box",
            split=False,
            density_norm="width",
            gap=0.08,
            linewidth=1.0,
            palette=PALETTE,
            ax=ax,
        )
        sns.pointplot(
            data=plot_df,
            x="cancer",
            y="value",
            hue="method",
            order=CANCERS,
            hue_order=METHOD_ORDER,
            dodge=0.55,
            join=False,
            errorbar=None,
            markers="D",
            scale=0.5,
            palette=POINT_PALETTE,
            legend=False,
            ax=ax,
        )
        if metric == "ROC AUC":
            ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8)
            ax.set_ylim(0.0, 1.0)
        else:
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8)
            ax.set_ylim(0.0, 0.9 if space == "PCA space" else 0.55)
        ax.set_ylabel(metric, fontsize=19)
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=17)
        ax.grid(False)
        sns.despine(ax=ax)
        annotate_means(ax, plot_df, metric)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[: len(METHOD_ORDER)],
            labels[: len(METHOD_ORDER)],
            title="",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=22,
            handlelength=1.6,
            labelspacing=0.8,
        )
    n_targets = 5 if space == "PCA space" else 3000
    fig.suptitle(f"z-score Top3000 yPCA5: test metrics ({space})", fontsize=27, weight="bold", y=0.99)
    fig.text(
        0.5,
        0.02,
        f"Each violin = {n_targets} targets; diamond = mean. ROC AUC positive label uses train median.",
        ha="center",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit")
    out_dir = root / "plots_pca5_violin"
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed_ids = [line.strip() for line in (root / "matrix" / "id_test.txt").read_text().splitlines() if line.strip()]
    mixed_train_ids = [line.strip() for line in (root / "matrix" / "id_train.txt").read_text().splitlines() if line.strip()]
    y_train_orig = np.load(root / "matrix" / "y_train.npy")
    y_test_orig = np.load(root / "matrix" / "y_test.npy")

    mixed_single_dir = root / "mixed_singlehead_attention_k300_mixup_pca5"
    mixed_dual_dir = root / "mixed_dualhead_attention_k300_mixup_pca5"
    mixed_single_pred_pca = np.load(mixed_single_dir / "test_fused_pred.npy")
    mixed_dual_pred_pca = np.load(mixed_dual_dir / "test_fused_pred.npy")
    mixed_single_true_pca = np.load(mixed_single_dir / "test_true.npy")
    mixed_dual_true_pca = np.load(mixed_dual_dir / "test_true.npy")
    mixed_single_y_pca_train = np.load(mixed_single_dir / "y_pca_train.npy")
    mixed_dual_y_pca_train = np.load(mixed_dual_dir / "y_pca_train.npy")
    mixed_single_pred_orig = inverse_pca(mixed_single_pred_pca, mixed_single_dir)
    mixed_dual_pred_orig = inverse_pca(mixed_dual_pred_pca, mixed_dual_dir)

    rows: list[dict] = []
    for cancer in CANCERS:
        single_dir = root / "single_attention_k300_mixup_pca5" / cancer
        add_metric_rows(
            rows,
            "PCA space",
            cancer,
            "Single-cancer",
            np.load(single_dir / "y_pca_train.npy"),
            np.load(single_dir / "test_true_pca.npy"),
            np.load(single_dir / "test_pred_pca.npy"),
        )
        add_metric_rows(
            rows,
            "Original SE space",
            cancer,
            "Single-cancer",
            np.load(single_dir / "y_train.npy"),
            np.load(single_dir / "test_true.npy"),
            np.load(single_dir / "test_pred.npy"),
        )

        train_idx = np.array([i for i, sid in enumerate(mixed_train_ids) if sid.startswith(f"{cancer}:")], dtype=np.int64)
        test_idx = np.array([i for i, sid in enumerate(mixed_ids) if sid.startswith(f"{cancer}:")], dtype=np.int64)
        for method, pred_pca, true_pca, y_pca_train, pred_orig in [
            ("Mixed single-head", mixed_single_pred_pca, mixed_single_true_pca, mixed_single_y_pca_train, mixed_single_pred_orig),
            ("Mixed dual-head", mixed_dual_pred_pca, mixed_dual_true_pca, mixed_dual_y_pca_train, mixed_dual_pred_orig),
        ]:
            add_metric_rows(
                rows,
                "PCA space",
                cancer,
                method,
                y_pca_train[train_idx],
                true_pca[test_idx],
                pred_pca[test_idx],
            )
            add_metric_rows(
                rows,
                "Original SE space",
                cancer,
                method,
                y_train_orig[train_idx],
                y_test_orig[test_idx],
                pred_orig[test_idx],
            )

    df = pd.DataFrame(rows)
    long_csv = out_dir / "zscore_erna_ge4_top3000_pca5_test_pearson_auc_per_target_long.csv"
    summary_csv = out_dir / "zscore_erna_ge4_top3000_pca5_test_pearson_auc_summary.csv"
    df.to_csv(long_csv, index=False)
    (
        df.groupby(["space", "metric", "cancer", "method"])["value"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .to_csv(summary_csv, index=False)
    )

    for space, tag in [("PCA space", "pca_space"), ("Original SE space", "original_space")]:
        draw_one(df, space, "Pearson r", out_dir / f"zscore_erna_ge4_top3000_pca5_test_pearson_violin_{tag}.png")
        draw_one(df, space, "ROC AUC", out_dir / f"zscore_erna_ge4_top3000_pca5_test_roc_auc_violin_{tag}.png")
        draw_combined(df, space, out_dir / f"zscore_erna_ge4_top3000_pca5_test_pearson_roc_auc_violin_combined_{tag}.png")

    print(f"wrote {long_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
