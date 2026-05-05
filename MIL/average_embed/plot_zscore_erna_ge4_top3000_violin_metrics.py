#!/usr/bin/env python3
"""Plot latest z-score top3000 fixed-split Pearson and ROC-AUC violins."""

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
METHOD_ORDER = ["Single-cancer", "Mixed single-head", "Mixed dual-head fused"]
PALETTE = ["#7aa6c2", "#d58b73", "#8fbe8f"]
POINT_PALETTE = ["#1f5673", "#944d38", "#3f7f3f"]


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
                rotation=0,
                color="#1b1f24",
            )


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


def add_rows(rows: list[dict], cancer: str, method: str, pearson: np.ndarray, auc: np.ndarray) -> None:
    for idx, value in enumerate(pearson):
        if np.isfinite(value):
            rows.append({"cancer": cancer, "method": method, "metric": "Pearson r", "enhancer_idx": idx, "value": value})
    for idx, value in enumerate(auc):
        if np.isfinite(value):
            rows.append({"cancer": cancer, "method": method, "metric": "ROC AUC", "enhancer_idx": idx, "value": value})


def draw_metric(df: pd.DataFrame, metric: str, out_png: Path) -> None:
    plot_df = df[df["metric"] == metric].copy()
    sns.set_theme(style="white", context="talk", font_scale=1.25)
    fig, ax = plt.subplots(figsize=(16, 8.5))
    sns.violinplot(
        data=plot_df,
        x="cancer",
        y="value",
        hue="method",
        order=CANCERS,
        hue_order=METHOD_ORDER,
        cut=0,
        inner="quart",
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
        note = "AUC positive label: z-score above train median for that enhancer"
    else:
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_ylim(-0.65, 0.85)
        note = "Pearson computed across held-out test samples for each enhancer"
    ax.set_title(f"Latest z-score Top3000 fixed split: test {metric} distribution", fontsize=25, weight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=19)
    ax.tick_params(axis="both", labelsize=17)
    ax.grid(False)
    sns.despine(ax=ax)
    annotate_means(ax, plot_df, metric)
    ax.text(0.01, -0.16, "Each violin = 3000 enhancers; diamond = mean. " + note, transform=ax.transAxes, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[: len(METHOD_ORDER)],
        labels[: len(METHOD_ORDER)],
        title="",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_combined(df: pd.DataFrame, out_png: Path) -> None:
    sns.set_theme(style="white", context="talk", font_scale=1.2)
    fig, axes = plt.subplots(2, 1, figsize=(17, 14), sharex=True)
    for ax, metric in zip(axes, ["Pearson r", "ROC AUC"]):
        plot_df = df[df["metric"] == metric].copy()
        sns.violinplot(
            data=plot_df,
            x="cancer",
            y="value",
            hue="method",
            order=CANCERS,
            hue_order=METHOD_ORDER,
            cut=0,
            inner="quart",
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
            ax.set_ylim(-0.65, 0.85)
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
            fontsize=15,
        )
    fig.suptitle("Latest z-score Top3000 fixed split: test Pearson and ROC AUC", fontsize=27, weight="bold", y=0.99)
    fig.text(
        0.5,
        0.02,
        "Each violin = 3000 enhancers; diamond = mean. ROC AUC positive label uses train median per cancer/enhancer.",
        ha="center",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit")
    out_dir = root / "plots_latest_violin"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    mixed_dir = root / "mixed_dualhead_attention_k300_mixup"
    mixed_single_dir = root / "mixed_singlehead_attention_k300_mixup"
    mixed_ids = [line.strip() for line in (root / "matrix" / "id_test.txt").read_text().splitlines() if line.strip()]
    mixed_train_ids = [line.strip() for line in (root / "matrix" / "id_train.txt").read_text().splitlines() if line.strip()]
    mixed_true = np.load(mixed_dir / "test_true.npy")
    mixed_y_train = np.load(root / "matrix" / "y_train.npy")
    mixed_single_pred = np.load(mixed_single_dir / "test_fused_pred.npy")
    mixed_dual_fused_pred = np.load(mixed_dir / "test_fused_pred.npy")

    for cancer in CANCERS:
        single_dir = root / "single_attention_k300_mixup" / cancer
        y_train = np.load(single_dir / "y_train.npy")
        y_true = np.load(single_dir / "test_true.npy")
        y_pred = np.load(single_dir / "test_pred.npy")
        add_rows(
            rows,
            cancer,
            "Single-cancer",
            per_target_pearson(y_true, y_pred),
            per_target_auc(y_train, y_true, y_pred),
        )

        train_idx = np.array([i for i, sid in enumerate(mixed_train_ids) if sid.startswith(f"{cancer}:")], dtype=np.int64)
        test_idx = np.array([i for i, sid in enumerate(mixed_ids) if sid.startswith(f"{cancer}:")], dtype=np.int64)
        y_train_mixed = mixed_y_train[train_idx]
        y_true_mixed = mixed_true[test_idx]
        for method, y_pred_all in [
            ("Mixed single-head", mixed_single_pred),
            ("Mixed dual-head fused", mixed_dual_fused_pred),
        ]:
            y_pred_mixed = y_pred_all[test_idx]
            add_rows(
                rows,
                cancer,
                method,
                per_target_pearson(y_true_mixed, y_pred_mixed),
                per_target_auc(y_train_mixed, y_true_mixed, y_pred_mixed),
            )

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "zscore_erna_ge4_top3000_test_pearson_auc_per_enhancer_long.csv"
    summary_csv = out_dir / "zscore_erna_ge4_top3000_test_pearson_auc_summary.csv"
    df.to_csv(metrics_csv, index=False)
    (
        df.groupby(["metric", "cancer", "method"])["value"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .to_csv(summary_csv, index=False)
    )

    draw_metric(df, "Pearson r", out_dir / "zscore_erna_ge4_top3000_test_pearson_violin.png")
    draw_metric(df, "ROC AUC", out_dir / "zscore_erna_ge4_top3000_test_roc_auc_violin.png")
    draw_combined(df, out_dir / "zscore_erna_ge4_top3000_test_pearson_roc_auc_violin_combined.png")

    print(f"wrote {metrics_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {out_dir / 'zscore_erna_ge4_top3000_test_pearson_violin.png'}")
    print(f"wrote {out_dir / 'zscore_erna_ge4_top3000_test_roc_auc_violin.png'}")
    print(f"wrote {out_dir / 'zscore_erna_ge4_top3000_test_pearson_roc_auc_violin_combined.png'}")


if __name__ == "__main__":
    main()
