#!/usr/bin/env python3
"""Summarize and plot shared-PCA sweep for z-score top3000 fixed split."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mzang")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score


ROOT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit")
CANCERS = ["BRCA", "LUAD", "SKCM"]
K_VALUES = [5, 8, 10, 15]
METHOD_ORDER = ["Single-cancer", "Mixed single-head", "Mixed dual-head"]
PALETTE = ["#7aa6c2", "#d58b73", "#8fbe8f"]


def per_target_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0))
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    ok = denom > 1e-12
    out[ok] = (yt[:, ok] * yp[:, ok]).sum(axis=0) / denom[ok]
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


def add_rows(
    rows: list[dict],
    k: int,
    space: str,
    cancer: str,
    method: str,
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    prefix = "PC" if space == "PCA space" else "SE"
    for metric, values in [
        ("Pearson r", per_target_pearson(y_true, y_pred)),
        ("ROC AUC", per_target_auc(y_train, y_true, y_pred)),
    ]:
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
                        "value": float(value),
                    }
                )


def collect_rows(root: Path) -> pd.DataFrame:
    matrix_dir = root / "matrix"
    train_ids = [x.strip() for x in (matrix_dir / "id_train.txt").read_text().splitlines() if x.strip()]
    test_ids = [x.strip() for x in (matrix_dir / "id_test.txt").read_text().splitlines() if x.strip()]
    y_train_orig = np.load(matrix_dir / "y_train.npy")
    y_test_orig = np.load(matrix_dir / "y_test.npy")
    rows: list[dict] = []

    for k in K_VALUES:
        pca_dir = root / "shared_pca_labels" / f"pca{k}"
        mixed_single_dir = root / f"mixed_singlehead_attention_k300_mixup_sharedpca{k}"
        mixed_dual_dir = root / f"mixed_dualhead_attention_k300_mixup_sharedpca{k}"
        missing = [
            d
            for d in [mixed_single_dir, mixed_dual_dir]
            if not (d / "test_fused_pred.npy").exists()
        ]
        if missing:
            print(f"[skip] PCA{k}: missing mixed outputs: {missing}")
            continue

        mixed_single_pred_pc = np.load(mixed_single_dir / "test_fused_pred.npy")
        mixed_dual_pred_pc = np.load(mixed_dual_dir / "test_fused_pred.npy")
        mixed_true_pc = np.load(mixed_dual_dir / "test_true.npy")
        mixed_train_pc = np.load(mixed_dual_dir / "y_pca_train.npy")
        mixed_single_pred_orig = inverse_pca(mixed_single_pred_pc, pca_dir)
        mixed_dual_pred_orig = inverse_pca(mixed_dual_pred_pc, pca_dir)

        for cancer in CANCERS:
            train_idx = np.array([i for i, sid in enumerate(train_ids) if sid.startswith(f"{cancer}:")], dtype=np.int64)
            test_idx = np.array([i for i, sid in enumerate(test_ids) if sid.startswith(f"{cancer}:")], dtype=np.int64)
            single_dir = root / f"single_attention_k300_mixup_sharedpca{k}" / cancer
            if not (single_dir / "test_pred.npy").exists():
                print(f"[skip] PCA{k} {cancer}: missing single output {single_dir}")
                continue
            single_pred_pc = np.load(single_dir / "test_pred.npy")
            single_true_pc = np.load(single_dir / "test_true.npy")
            single_train_pc = np.load(single_dir / "y_train.npy")
            single_pred_orig = inverse_pca(single_pred_pc, pca_dir)

            add_rows(rows, k, "PCA space", cancer, "Single-cancer", single_train_pc, single_true_pc, single_pred_pc)
            add_rows(rows, k, "Original SE space", cancer, "Single-cancer", y_train_orig[train_idx], y_test_orig[test_idx], single_pred_orig)

            for method, pred_pc, pred_orig in [
                ("Mixed single-head", mixed_single_pred_pc, mixed_single_pred_orig),
                ("Mixed dual-head", mixed_dual_pred_pc, mixed_dual_pred_orig),
            ]:
                add_rows(rows, k, "PCA space", cancer, method, mixed_train_pc[train_idx], mixed_true_pc[test_idx], pred_pc[test_idx])
                add_rows(rows, k, "Original SE space", cancer, method, y_train_orig[train_idx], y_test_orig[test_idx], pred_orig[test_idx])

    return pd.DataFrame(rows)


def draw_combined(df: pd.DataFrame, space: str, out_png: Path) -> None:
    sub = df[df["space"] == space].copy()
    sub["PCA"] = "PCA" + sub["k"].astype(str)
    fig, axes = plt.subplots(2, 3, figsize=(21, 11), sharex=True)
    for row, metric in enumerate(["Pearson r", "ROC AUC"]):
        for col, cancer in enumerate(CANCERS):
            ax = axes[row, col]
            s = sub[(sub["metric"] == metric) & (sub["cancer"] == cancer)]
            sns.violinplot(
                data=s,
                x="PCA",
                y="value",
                hue="method",
                hue_order=METHOD_ORDER,
                order=[f"PCA{k}" for k in K_VALUES],
                palette=PALETTE,
                inner="box",
                density_norm="width",
                cut=0,
                gap=0.08,
                linewidth=1.2,
                ax=ax,
            )
            means = s.groupby(["PCA", "method"], observed=False)["value"].mean()
            sns.pointplot(
                data=s,
                x="PCA",
                y="value",
                hue="method",
                hue_order=METHOD_ORDER,
                order=[f"PCA{k}" for k in K_VALUES],
                palette=PALETTE,
                dodge=0.52,
                errorbar=None,
                markers="o",
                linestyles="none",
                legend=False,
                ax=ax,
            )
            for x_i, k_label in enumerate([f"PCA{k}" for k in K_VALUES]):
                for m_i, method in enumerate(METHOD_ORDER):
                    if (k_label, method) not in means:
                        continue
                    ax.text(
                        x_i + (m_i - 1) * 0.25,
                        means[(k_label, method)] + (0.012 if metric == "Pearson r" else 0.01),
                        f"{means[(k_label, method)]:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        rotation=90,
                    )
            ax.set_title(cancer, fontsize=18, weight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(metric if col == 0 else "", fontsize=16)
            ax.grid(False)
            if metric == "Pearson r":
                ax.set_ylim(0, 0.65 if space == "PCA space" else 0.35)
            else:
                ax.set_ylim(0.45, 0.75)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.tick_params(axis="both", labelsize=13)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="upper center", ncol=3, frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"Shared-PCA z-score Top3000 sweep: test metrics ({space})", fontsize=24, weight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "plots_sharedpca_sweep_violin"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_rows(ROOT)
    if df.empty:
        raise RuntimeError("No completed shared-PCA outputs found.")
    df.to_csv(out_dir / "zscore_top3000_sharedpca_sweep_per_target_long.csv", index=False)
    summary = (
        df.groupby(["k", "space", "metric", "cancer", "method"], observed=False)["value"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary.to_csv(out_dir / "zscore_top3000_sharedpca_sweep_summary.csv", index=False)
    draw_combined(df, "PCA space", out_dir / "zscore_top3000_sharedpca_sweep_pca_space_violin.png")
    draw_combined(df, "Original SE space", out_dir / "zscore_top3000_sharedpca_sweep_original_space_violin.png")
    print(out_dir)


if __name__ == "__main__":
    main()
