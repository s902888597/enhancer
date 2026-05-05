#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from run_mean_regression_pan_cancer import (
    build_case_to_feature_dir,
    build_normalized_identity_case_to_feature_dir,
    load_split_feats,
    pearson_per_feature,
    set_seed,
    summarize_corr,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", choices=["BRCA", "LUAD", "SKCM"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument(
        "--feat-root",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mean-cache-dir", default="")
    parser.add_argument("--label-layout", choices=["samples_as_columns", "samples_as_rows"], default="samples_as_rows")
    parser.add_argument("--sample-id-mode", choices=["tcga_case3", "identity"], default="tcga_case3")
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.mean_cache_dir) if args.mean_cache_dir else None
    feat_root = Path(args.feat_root)
    cancer_root = feat_root / args.cancer
    if not cancer_root.exists():
        raise FileNotFoundError(f"Missing feature root for cancer={args.cancer}: {cancer_root}")
    direct_dirs = sorted([p for p in cancer_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if any(d.name.endswith("_tumor") or d.name.endswith("_normal") for d in direct_dirs):
        case_to_dir = build_normalized_identity_case_to_feature_dir(cancer_root, args.sample_id_mode)
    else:
        case_to_dir = build_case_to_feature_dir(cancer_root)

    train_ids, x_train, y_train, enh_cols = load_split_feats(
        "train",
        Path(args.train_csv),
        case_to_dir,
        args.label_layout,
        args.sample_id_mode,
        enh_ref=None,
        cache_dir=cache_dir,
        cancer=args.cancer,
        pooling=args.pooling,
    )
    val_ids, x_val, y_val, _ = load_split_feats(
        "validation",
        Path(args.val_csv),
        case_to_dir,
        args.label_layout,
        args.sample_id_mode,
        enh_ref=enh_cols,
        cache_dir=cache_dir,
        cancer=args.cancer,
        pooling=args.pooling,
    )
    test_ids, x_test, y_test, _ = load_split_feats(
        "test",
        Path(args.test_csv),
        case_to_dir,
        args.label_layout,
        args.sample_id_mode,
        enh_ref=enh_cols,
        cache_dir=cache_dir,
        cancer=args.cancer,
        pooling=args.pooling,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=args.device,
        multi_strategy="multi_output_tree",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_bin=args.max_bin,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
    else:
        val_pred = model.predict(x_val, iteration_range=(0, best_iteration + 1))
        test_pred = model.predict(x_test, iteration_range=(0, best_iteration + 1))

    val_corr = pearson_per_feature(val_pred, y_val, enh_cols)
    test_corr = pearson_per_feature(test_pred, y_test, enh_cols)
    summarize_corr(val_corr, "validation")
    summarize_corr(test_corr, "test")

    val_corr.to_csv(out_dir / "per_enhancer_correlation_validation.csv", index=False)
    test_corr.to_csv(out_dir / "per_enhancer_correlation_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "validation",
                "n_enhancers": len(enh_cols),
                "pearson_mean": float(val_corr["pearson_r"].mean(skipna=True)),
                "pearson_median": float(val_corr["pearson_r"].median(skipna=True)),
                "gt_0.4": int((val_corr["pearson_r"] > 0.4).sum()),
                "gt_0.5": int((val_corr["pearson_r"] > 0.5).sum()),
                "gt_0.6": int((val_corr["pearson_r"] > 0.6).sum()),
            },
            {
                "split": "test",
                "n_enhancers": len(enh_cols),
                "pearson_mean": float(test_corr["pearson_r"].mean(skipna=True)),
                "pearson_median": float(test_corr["pearson_r"].median(skipna=True)),
                "gt_0.4": int((test_corr["pearson_r"] > 0.4).sum()),
                "gt_0.5": int((test_corr["pearson_r"] > 0.5).sum()),
                "gt_0.6": int((test_corr["pearson_r"] > 0.6).sum()),
            },
        ]
    ).to_csv(out_dir / "summary.csv", index=False)

    np.save(out_dir / "val_pred.npy", val_pred.astype(np.float32))
    np.save(out_dir / "val_true.npy", y_val.astype(np.float32))
    np.save(out_dir / "test_pred.npy", test_pred.astype(np.float32))
    np.save(out_dir / "test_true.npy", y_test.astype(np.float32))
    (out_dir / "ids_train.txt").write_text("\n".join(train_ids) + "\n")
    (out_dir / "ids_validation.txt").write_text("\n".join(val_ids) + "\n")
    (out_dir / "ids_test.txt").write_text("\n".join(test_ids) + "\n")
    (out_dir / "enhancers.txt").write_text("\n".join(enh_cols) + "\n")
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(vars(args) | {"best_iteration": best_iteration}, f, indent=2)
    model.save_model(out_dir / "xgboost_model.json")


if __name__ == "__main__":
    main()
