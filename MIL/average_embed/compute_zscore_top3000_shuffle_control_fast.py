#!/usr/bin/env python3
"""Fast enhancer-level shuffle control for z-score top3000 direct models."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mzang")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CANCERS = ["BRCA", "LUAD", "SKCM"]
METHOD_ORDER = ["Single-cancer", "Mixed single-head", "Mixed dual-head fused"]


def per_target_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0))
    out = np.full(y_true.shape[1], np.nan)
    ok = denom > 1e-12
    out[ok] = (yt[:, ok] * yp[:, ok]).sum(axis=0) / denom[ok]
    return out


def rank_auc_per_target(y_train: np.ndarray, y_test: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    y_bin = y_test > np.median(y_train, axis=0, keepdims=True)
    n_pos = y_bin.sum(axis=0).astype(np.float64)
    n = y_bin.shape[0]
    n_neg = n - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    order = np.argsort(y_score, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order, np.arange(y_score.shape[1])] = np.arange(1, n + 1, dtype=np.float64)[:, None]
    pos_rank_sum = (ranks * y_bin).sum(axis=0)
    auc = np.full(y_score.shape[1], np.nan)
    auc[valid] = (pos_rank_sum[valid] - n_pos[valid] * (n_pos[valid] + 1.0) / 2.0) / (n_pos[valid] * n_neg[valid])
    return auc


def metric_row(cancer: str, method: str, control: str, replicate: int, y_train, y_true, y_pred) -> dict:
    pearson = per_target_pearson(y_true, y_pred)
    auc = rank_auc_per_target(y_train, y_true, y_pred)
    return {
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


def plot_control(summary: pd.DataFrame, out_png: Path) -> None:
    long = []
    for _, row in summary.iterrows():
        d = row.to_dict()
        long.append({**d, "metric": "Mean Pearson r", "value": d["pearson_mean"]})
        long.append({**d, "metric": "Mean ROC AUC", "value": d["auc_mean"]})
    df = pd.DataFrame(long)
    sns.set_theme(style="white", context="talk", font_scale=1.0)
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), sharex=False)
    for row_idx, cancer in enumerate(CANCERS):
        for col_idx, metric in enumerate(["Mean Pearson r", "Mean ROC AUC"]):
            ax = axes[row_idx, col_idx]
            sub = df[(df["metric"] == metric) & (df["cancer"] == cancer)]
            sns.barplot(
                data=sub,
                x="method",
                y="value",
                hue="control",
                order=METHOD_ORDER,
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
            ax.set_title(f"{cancer} - {metric}", fontsize=17, weight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=25, labelsize=11)
            ax.grid(False)
            sns.despine(ax=ax)
            if row_idx == 0 and col_idx == 1:
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
    out_dir = root / "shuffle_control_fast"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = root / "matrix"
    test_ids = [x.strip() for x in (matrix / "id_test.txt").read_text().splitlines() if x.strip()]
    train_ids = [x.strip() for x in (matrix / "id_train.txt").read_text().splitlines() if x.strip()]
    mixed_y_train = np.load(matrix / "y_train.npy")
    mixed_true = np.load(root / "mixed_dualhead_attention_k300_mixup" / "test_true.npy")
    mixed_single_pred = np.load(root / "mixed_singlehead_attention_k300_mixup" / "test_fused_pred.npy")
    mixed_dual_pred = np.load(root / "mixed_dualhead_attention_k300_mixup" / "test_fused_pred.npy")

    rows = []
    rng = np.random.default_rng(44)
    for cancer in CANCERS:
        single_dir = root / "single_attention_k300_mixup" / cancer
        data = {
            "Single-cancer": (
                np.load(single_dir / "y_train.npy"),
                np.load(single_dir / "test_true.npy"),
                np.load(single_dir / "test_pred.npy"),
            )
        }
        train_idx = np.array([i for i, sid in enumerate(train_ids) if sid.startswith(f"{cancer}:")])
        test_idx = np.array([i for i, sid in enumerate(test_ids) if sid.startswith(f"{cancer}:")])
        data["Mixed single-head"] = (mixed_y_train[train_idx], mixed_true[test_idx], mixed_single_pred[test_idx])
        data["Mixed dual-head fused"] = (mixed_y_train[train_idx], mixed_true[test_idx], mixed_dual_pred[test_idx])
        for method, (y_train, y_true, y_pred) in data.items():
            rows.append(metric_row(cancer, method, "Real", 0, y_train, y_true, y_pred))
            for rep in range(1, 101):
                perm = rng.permutation(y_true.shape[1])
                rows.append(metric_row(cancer, method, "Enhancer-shuffled", rep, y_train[:, perm], y_true[:, perm], y_pred))

    reps = pd.DataFrame(rows)
    compact = (
        reps.groupby(["cancer", "method", "control"])
        .agg(
            pearson_mean=("pearson_mean", "mean"),
            pearson_sd=("pearson_mean", "std"),
            auc_mean=("auc_mean", "mean"),
            auc_sd=("auc_mean", "std"),
            n_reps=("replicate", "count"),
        )
        .reset_index()
    )
    reps.to_csv(out_dir / "zscore_top3000_enhancer_shuffle_control_replicates.csv", index=False)
    compact.to_csv(out_dir / "zscore_top3000_enhancer_shuffle_control_summary.csv", index=False)
    plot_control(reps, out_dir / "zscore_top3000_enhancer_shuffle_control.png")
    print(compact.to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()
