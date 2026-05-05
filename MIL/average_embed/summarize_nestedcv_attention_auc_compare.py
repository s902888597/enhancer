from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


METHOD_LABELS = {
    "single3000": "Single-cancer 3000",
    "mix_single": "Mixed single-head 4827",
    "mix_dual": "Mixed dual-head 4827",
}

THRESHOLDS = {
    "z_gt_0": "z > 0",
    "z_gt_1": "z > 1",
    "top20pct": "top 20%",
}


def positive_mask(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "z_gt_0":
        return values > 0.0
    if mode == "z_gt_1":
        return values > 1.0
    if mode == "top20pct":
        cutoff = np.quantile(values, 0.8)
        return values >= cutoff
    raise ValueError(f"Unknown threshold mode: {mode}")


def per_feature_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_mode: str,
) -> tuple[float, float, int]:
    roc_vals: list[float] = []
    pr_vals: list[float] = []

    for j in range(y_true.shape[1]):
        labels = positive_mask(y_true[:, j], threshold_mode)
        if labels.min() == labels.max():
            continue
        scores = y_score[:, j]
        roc_vals.append(float(roc_auc_score(labels, scores)))
        pr_vals.append(float(average_precision_score(labels, scores)))

    if not roc_vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(roc_vals)), float(np.mean(pr_vals)), len(roc_vals)


def summarize_single_root(root: Path, setting: str) -> pd.DataFrame:
    rows: list[dict] = []
    for fold_dir in sorted(root.glob("fold*")):
        fold = int(fold_dir.name.replace("fold", ""))
        for cancer_dir in sorted([p for p in fold_dir.iterdir() if p.is_dir()]):
            y_true = np.load(cancer_dir / "test_true.npy")
            y_pred = np.load(cancer_dir / "test_pred.npy")
            for threshold_key in THRESHOLDS:
                roc_mean, pr_mean, n_valid = per_feature_auc(
                    y_true=y_true,
                    y_score=y_pred,
                    threshold_mode=threshold_key,
                )
                rows.append(
                    {
                        "setting": setting,
                        "fold": fold,
                        "Cancer": cancer_dir.name,
                        "Method": METHOD_LABELS["single3000"],
                        "threshold_key": threshold_key,
                        "threshold_label": THRESHOLDS[threshold_key],
                        "roc_auc_mean": roc_mean,
                        "pr_auc_mean": pr_mean,
                        "n_valid_enhancers": n_valid,
                    }
                )
    return pd.DataFrame(rows)


def summarize_mix_root(
    root: Path,
    assets_root: Path,
    method_key: str,
    setting: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for fold_dir in sorted(root.glob("fold*")):
        fold = int(fold_dir.name.replace("fold", ""))
        asset_dir = assets_root / f"fold{fold}" / "mix3_union4827_matrix"

        y_true = np.load(fold_dir / "test_true.npy")
        y_pred = np.load(fold_dir / "test_fused_pred.npy")
        mask = np.load(fold_dir / "test_mask.npy") > 0.5
        groups = np.load(asset_dir / "group_test.npy", allow_pickle=True)

        for cancer in ["BRCA", "LUAD", "SKCM"]:
            sample_sel = groups == cancer
            y_true_c = y_true[sample_sel]
            y_pred_c = y_pred[sample_sel]
            mask_c = mask[sample_sel]
            valid_cols = mask_c.any(axis=0)
            y_true_c = y_true_c[:, valid_cols]
            y_pred_c = y_pred_c[:, valid_cols]

            for threshold_key in THRESHOLDS:
                roc_mean, pr_mean, n_valid = per_feature_auc(
                    y_true=y_true_c,
                    y_score=y_pred_c,
                    threshold_mode=threshold_key,
                )
                rows.append(
                    {
                        "setting": setting,
                        "fold": fold,
                        "Cancer": cancer,
                        "Method": METHOD_LABELS[method_key],
                        "threshold_key": threshold_key,
                        "threshold_label": THRESHOLDS[threshold_key],
                        "roc_auc_mean": roc_mean,
                        "pr_auc_mean": pr_mean,
                        "n_valid_enhancers": n_valid,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-root", type=Path, required=True)
    parser.add_argument("--mix-single-root", type=Path, required=True)
    parser.add_argument("--mix-dual-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--setting-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.concat(
        [
            summarize_single_root(args.single_root, args.setting_label),
            summarize_mix_root(
                args.mix_single_root,
                assets_root=args.assets_root,
                method_key="mix_single",
                setting=args.setting_label,
            ),
            summarize_mix_root(
                args.mix_dual_root,
                assets_root=args.assets_root,
                method_key="mix_dual",
                setting=args.setting_label,
            ),
        ],
        ignore_index=True,
    )

    summary_df = (
        long_df.groupby(
            ["setting", "Cancer", "Method", "threshold_key", "threshold_label"],
            as_index=False,
        )
        .agg(
            roc_auc_mean=("roc_auc_mean", "mean"),
            roc_auc_std=("roc_auc_mean", "std"),
            pr_auc_mean=("pr_auc_mean", "mean"),
            pr_auc_std=("pr_auc_mean", "std"),
            mean_valid_enhancers=("n_valid_enhancers", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["threshold_key", "Cancer", "Method"])
    )

    long_df.to_csv(args.out_dir / f"{args.out_prefix}_long_table.csv", index=False)
    summary_df.to_csv(args.out_dir / f"{args.out_prefix}_summary.csv", index=False)


if __name__ == "__main__":
    main()
