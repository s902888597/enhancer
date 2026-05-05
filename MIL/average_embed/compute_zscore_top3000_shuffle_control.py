#!/usr/bin/env python3
"""Enhancer-level shuffle control for z-score top3000 direct models."""

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
METHODS = {
    "Single-cancer": "single",
    "Mixed single-head": "mixed_single",
    "Mixed dual-head fused": "mixed_dual_fused",
}


def per_target_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0))
    out = np.full(y_true.shape[1], np.nan)
    ok = denom > 1e-12
    out[ok] = (yt[:, ok] * yp[:, ok]).sum(axis=0) / denom[ok]
    return out


def per_target_auc(y_train: np.ndarray, y_test: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    thresholds = np.median(y_train, axis=0)
    out = np.full(y_test.shape[1], np.nan)
    for j in range(y_test.shape[1]):
        y_bin = (y_test[:, j] > thresholds[j]).astype(np.int8)
        if y_bin.min() == y_bin.max():
            continue
        out[j] = roc_auc_score(y_bin, y_score[:, j])
    return out


def summarize_once(
    rows: list[dict],
    cancer: str,
    method: str,
    control: str,
    replicate: int,
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    pearson = per_target_pearson(y_true, y_pred)
    auc = per_target_auc(y_train, y_true, y_pred)
    rows.append(
        {
            "cancer": cancer,
            "method": method,
            "control": control,
            "replicate": replicate,
            "pearson_mean": float(np.nanmean(pearson)),
            "pearson_median": float(np.nanmedian(pearson)),
            "auc_mean": float(np.nanmean(auc)),
            "auc_median": float(np.nanmedian(auc)),
            "n_pearson": int(np.isfinite(pearson).sum()),
            "n_auc": int(np.isfinite(auc).sum()),
        }
    )


def plot_control(summary: pd.DataFrame, out_png: Path) -> None:
    long = []
    for _, row in summary.iterrows():
        long.append({**row.to_dict(), "metric": "Mean Pearson r", "value": row["pearson_mean"]})
        long.append({**row.to_dict(), "metric": "Mean ROC AUC", "value": row["auc_mean"]})
    df = pd.DataFrame(long)
    sns.set_theme(style="white", context="talk", font_scale=1.0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for row_idx, metric in enumerate(["Mean Pearson r", "Mean ROC AUC"]):
        for col_idx, cancer in enumerate(CANCERS):
            ax = axes[row_idx, col_idx]
            sub = df[(df["metric"] == metric) & (df["cancer"] == cancer)]
            sns.barplot(
                data=sub,
                x="method",
                y="value",
                hue="control",
                estimator=np.mean,
                errorbar="sd",
                palette=["#6ea3bf", "#d88c77"],
                ax=ax,
            )
            if metric == "Mean ROC AUC":
                ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.0)
                ax.set_ylim(0.45, 0.68)
            else:
                ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
                ax.set_ylim(-0.03, 0.30)
            ax.set_title(cancer, fontsize=17, weight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(metric if col_idx == 0 else "")
            ax.tick_params(axis="x", rotation=25, labelsize=11)
            ax.grid(False)
            sns.despine(ax=ax)
            if col_idx == 2:
                ax.legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            else:
                ax.get_legend().remove()
    fig.suptitle("Enhancer-level shuffle control: direct z-score Top3000", fontsize=22, weight="bold")
    fig.text(
        0.5,
        0.02,
        "Real = correct enhancer identity. Shuffled = test labels and train thresholds randomly permuted across enhancers; 100 permutations.",
        ha="center",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit")
    out_dir = root / "shuffle_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = root / "matrix"
    mixed_test_ids = [x.strip() for x in (matrix / "id_test.txt").read_text().splitlines() if x.strip()]
    mixed_train_ids = [x.strip() for x in (matrix / "id_train.txt").read_text().splitlines() if x.strip()]
    mixed_y_train = np.load(matrix / "y_train.npy")
    mixed_true = np.load(root / "mixed_dualhead_attention_k300_mixup" / "test_true.npy")
    mixed_single_pred = np.load(root / "mixed_singlehead_attention_k300_mixup" / "test_fused_pred.npy")
    mixed_dual_pred = np.load(root / "mixed_dualhead_attention_k300_mixup" / "test_fused_pred.npy")

    rows: list[dict] = []
    rng = np.random.default_rng(44)
    n_perm = 100
    for cancer in CANCERS:
        single_dir = root / "single_attention_k300_mixup" / cancer
        data = {
            "Single-cancer": (
                np.load(single_dir / "y_train.npy"),
                np.load(single_dir / "test_true.npy"),
                np.load(single_dir / "test_pred.npy"),
            )
        }
        train_idx = np.array([i for i, sid in enumerate(mixed_train_ids) if sid.startswith(f"{cancer}:")], dtype=np.int64)
        test_idx = np.array([i for i, sid in enumerate(mixed_test_ids) if sid.startswith(f"{cancer}:")], dtype=np.int64)
        data["Mixed single-head"] = (mixed_y_train[train_idx], mixed_true[test_idx], mixed_single_pred[test_idx])
        data["Mixed dual-head fused"] = (mixed_y_train[train_idx], mixed_true[test_idx], mixed_dual_pred[test_idx])

        for method, (y_train, y_true, y_pred) in data.items():
            summarize_once(rows, cancer, method, "Real", 0, y_train, y_true, y_pred)
            for rep in range(1, n_perm + 1):
                perm = rng.permutation(y_true.shape[1])
                summarize_once(rows, cancer, method, "Enhancer-shuffled", rep, y_train[:, perm], y_true[:, perm], y_pred)

    summary = pd.DataFrame(rows)
    compact = (
        summary.groupby(["cancer", "method", "control"])
        .agg(
            pearson_mean=("pearson_mean", "mean"),
            pearson_sd=("pearson_mean", "std"),
            auc_mean=("auc_mean", "mean"),
            auc_sd=("auc_mean", "std"),
            n_reps=("replicate", "count"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "zscore_top3000_enhancer_shuffle_control_replicates.csv", index=False)
    compact.to_csv(out_dir / "zscore_top3000_enhancer_shuffle_control_summary.csv", index=False)
    plot_control(summary, out_dir / "zscore_top3000_enhancer_shuffle_control.png")
    print(compact.to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()
