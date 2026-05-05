#!/usr/bin/env python3
"""Summarize and plot PCA K sweep for z-score top3000 fixed split."""

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
K_VALUES = [5, 10, 30, 50]
PALETTE = ["#7aa6c2", "#d58b73", "#8fbe8f"]


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
    return scores @ np.load(pca_dir / "pca_components.npy") + np.load(pca_dir / "pca_mean.npy")


def add_metric_rows(
    rows: list[dict],
    k: int,
    space: str,
    cancer: str,
    method: str,
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    pearson = per_target_pearson(y_true, y_pred)
    auc = per_target_auc(y_train, y_true, y_pred)
    prefix = "PC" if space == "PCA space" else "SE"
    for metric, values in [("Pearson r", pearson), ("ROC AUC", auc)]:
        for idx, value in enumerate(values):
            if np.isfinite(value):
                rows.append(
                    {
                        "k": k,
                        "space": space,
                        "metric": metric,
                        "cancer": cancer,
                        "method": method,
                        "target": f"{prefix}{idx + 1}",
                        "value": value,
                    }
                )


def collect_rows(root: Path) -> pd.DataFrame:
    mixed_ids = [line.strip() for line in (root / "matrix" / "id_test.txt").read_text().splitlines() if line.strip()]
    mixed_train_ids = [line.strip() for line in (root / "matrix" / "id_train.txt").read_text().splitlines() if line.strip()]
    y_train_orig = np.load(root / "matrix" / "y_train.npy")
    y_test_orig = np.load(root / "matrix" / "y_test.npy")

    rows: list[dict] = []
    for k in K_VALUES:
        mixed_single_dir = root / f"mixed_singlehead_attention_k300_mixup_pca{k}"
        mixed_dual_dir = root / f"mixed_dualhead_attention_k300_mixup_pca{k}"
        mixed_single_pred_pca = np.load(mixed_single_dir / "test_fused_pred.npy")
        mixed_dual_pred_pca = np.load(mixed_dual_dir / "test_fused_pred.npy")
        mixed_single_true_pca = np.load(mixed_single_dir / "test_true.npy")
        mixed_dual_true_pca = np.load(mixed_dual_dir / "test_true.npy")
        mixed_single_y_pca_train = np.load(mixed_single_dir / "y_pca_train.npy")
        mixed_dual_y_pca_train = np.load(mixed_dual_dir / "y_pca_train.npy")
        mixed_single_pred_orig = inverse_pca(mixed_single_pred_pca, mixed_single_dir)
        mixed_dual_pred_orig = inverse_pca(mixed_dual_pred_pca, mixed_dual_dir)

        for cancer in CANCERS:
            single_dir = root / f"single_attention_k300_mixup_pca{k}" / cancer
            add_metric_rows(
                rows,
                k,
                "PCA space",
                cancer,
                "Single-cancer",
                np.load(single_dir / "y_pca_train.npy"),
                np.load(single_dir / "test_true_pca.npy"),
                np.load(single_dir / "test_pred_pca.npy"),
            )
            add_metric_rows(
                rows,
                k,
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
                ("Mixed dual-head fused", mixed_dual_pred_pca, mixed_dual_true_pca, mixed_dual_y_pca_train, mixed_dual_pred_orig),
            ]:
                add_metric_rows(
                    rows,
                    k,
                    "PCA space",
                    cancer,
                    method,
                    y_pca_train[train_idx],
                    true_pca[test_idx],
                    pred_pca[test_idx],
                )
                add_metric_rows(
                    rows,
                    k,
                    "Original SE space",
                    cancer,
                    method,
                    y_train_orig[train_idx],
                    y_test_orig[test_idx],
                    pred_orig[test_idx],
                )
    return pd.DataFrame(rows)


def annotate_bars(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2, rotation=90)


def plot_summary_bars(summary: pd.DataFrame, out_dir: Path, space: str, metric: str) -> None:
    df = summary[(summary["space"] == space) & (summary["metric"] == metric)].copy()
    df["k_label"] = "PCA" + df["k"].astype(str)
    sns.set_theme(style="white", context="talk", font_scale=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
    for ax, cancer in zip(axes, CANCERS):
        sub = df[df["cancer"] == cancer]
        sns.barplot(
            data=sub,
            x="k_label",
            y="mean",
            hue="method",
            hue_order=METHOD_ORDER,
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title(cancer, fontsize=18, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric if ax is axes[0] else "")
        ax.grid(False)
        sns.despine(ax=ax)
        annotate_bars(ax)
        if metric == "ROC AUC":
            ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1)
            ax.set_ylim(0.45, 0.8)
        else:
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1)
            ax.set_ylim(0.0, 0.55 if space == "PCA space" else 0.32)
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(f"z-score Top3000 PCA sweep: test {metric} ({space})", fontsize=22, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    tag = "pca_space" if space == "PCA space" else "original_space"
    metric_tag = "pearson" if metric == "Pearson r" else "roc_auc"
    fig.savefig(out_dir / f"zscore_top3000_pca_sweep_{metric_tag}_{tag}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit")
    out_dir = root / "plots_pca_sweep_violin"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = collect_rows(root)
    summary = (
        long_df.groupby(["k", "space", "metric", "cancer", "method"])["value"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    long_df.to_csv(out_dir / "zscore_top3000_pca_sweep_per_target_long.csv", index=False)
    summary.to_csv(out_dir / "zscore_top3000_pca_sweep_summary.csv", index=False)

    for space in ["PCA space", "Original SE space"]:
        for metric in ["Pearson r", "ROC AUC"]:
            plot_summary_bars(summary, out_dir, space, metric)

    print(out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
