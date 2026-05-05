#!/usr/bin/env python3
"""
Train one Mean-MLP on mixed BRCA/LUAD/SKCM samples using the shared enhancer set.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from run_mean_regression_pan_cancer import (
    build_case_to_feature_dir,
    build_identity_case_to_feature_dir,
    build_normalized_identity_case_to_feature_dir,
    eval_epoch,
    load_label_df,
    load_split_feats,
    pearson_per_feature,
    set_seed,
    SimpleRegressor,
    summarize_corr,
    train_epoch,
)


class FeatDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def metric_row(corr_df: pd.DataFrame, split: str, group: str, k: int, n_enhancers: int) -> Dict[str, object]:
    s = corr_df["pearson_r"]
    return {
        "split": split,
        "group": group,
        "k": k,
        "n_enhancers": n_enhancers,
        "pearson_mean": float(s.mean(skipna=True)),
        "pearson_median": float(s.median(skipna=True)),
        "gt_0.4": int((s > 0.4).sum()),
        "gt_0.5": int((s > 0.5).sum()),
        "gt_0.6": int((s > 0.6).sum()),
    }


def build_feature_maps(feat_root: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    out: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for split in ["train", "validation", "test"]:
        out[split] = {
            "BRCA": build_case_to_feature_dir(feat_root / split / "BRCA"),
            "LUAD": build_case_to_feature_dir(feat_root / split / "LUAD"),
            "SKCM": build_normalized_identity_case_to_feature_dir(feat_root / split / "SKCM", "tcga_case3"),
        }
    return out


def common_enhancers(train_paths: Dict[str, Path]) -> List[str]:
    sets = []
    for cancer, csv_path in train_paths.items():
        _, label_df = load_label_df(csv_path, "samples_as_columns", "tcga_case3")
        sets.append(set(label_df.columns))
    common = sorted(set.intersection(*sets))
    if not common:
        raise RuntimeError("No common enhancers across BRCA/LUAD/SKCM")
    return common


def load_one_cancer_split(
    split_name: str,
    cancer: str,
    csv_path: Path,
    feature_map: Dict[str, Path],
    enh_ref: List[str],
    cache_dir: Path,
    pooling: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    label_layout = "samples_as_columns"
    sample_id_mode = "tcga_case3"
    ids, x, y, _ = load_split_feats(
        split_name,
        csv_path,
        feature_map,
        label_layout,
        sample_id_mode,
        enh_ref=enh_ref,
        cache_dir=cache_dir,
        cancer=cancer,
        pooling=pooling,
    )
    ids = [f"{cancer}:{sid}" for sid in ids]
    return ids, x, y


def concat_split(
    split_name: str,
    csv_map: Dict[str, Path],
    feat_maps: Dict[str, Dict[str, Path]],
    enh_ref: List[str],
    cache_dir: Path,
    pooling: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    all_ids: List[str] = []
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_groups: List[str] = []
    for cancer in ["BRCA", "LUAD", "SKCM"]:
        ids, x, y = load_one_cancer_split(
            split_name,
            cancer,
            csv_map[cancer],
            feat_maps[cancer],
            enh_ref,
            cache_dir,
            pooling,
        )
        all_ids.extend(ids)
        all_x.append(x)
        all_y.append(y)
        all_groups.extend([cancer] * len(ids))
        print(f"[{split_name}] {cancer}: n={len(ids)}")
    return all_ids, np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0), np.array(all_groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--brca-train-csv", required=True)
    parser.add_argument("--brca-val-csv", required=True)
    parser.add_argument("--brca-test-csv", required=True)
    parser.add_argument("--luad-train-csv", required=True)
    parser.add_argument("--luad-val-csv", required=True)
    parser.add_argument("--luad-test-csv", required=True)
    parser.add_argument("--skcm-train-csv", required=True)
    parser.add_argument("--skcm-val-csv", required=True)
    parser.add_argument("--skcm-test-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--pca-k", type=int, default=0)
    parser.add_argument("--mean-cache-dir", default="")
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_root = Path(args.feat_root)
    cache_dir = Path(args.mean_cache_dir) if args.mean_cache_dir else None

    train_csvs = {
        "BRCA": Path(args.brca_train_csv),
        "LUAD": Path(args.luad_train_csv),
        "SKCM": Path(args.skcm_train_csv),
    }
    val_csvs = {
        "BRCA": Path(args.brca_val_csv),
        "LUAD": Path(args.luad_val_csv),
        "SKCM": Path(args.skcm_val_csv),
    }
    test_csvs = {
        "BRCA": Path(args.brca_test_csv),
        "LUAD": Path(args.luad_test_csv),
        "SKCM": Path(args.skcm_test_csv),
    }

    enh_ref = common_enhancers(train_csvs)
    print(f"Shared enhancers across BRCA/LUAD/SKCM train sets: {len(enh_ref)}")

    feat_maps = build_feature_maps(feat_root)
    train_ids, x_train, y_train, train_groups = concat_split(
        "train", train_csvs, feat_maps["train"], enh_ref, cache_dir, args.pooling
    )
    val_ids, x_val, y_val, val_groups = concat_split(
        "validation", val_csvs, feat_maps["validation"], enh_ref, cache_dir, args.pooling
    )
    test_ids, x_test, y_test, test_groups = concat_split(
        "test", test_csvs, feat_maps["test"], enh_ref, cache_dir, args.pooling
    )

    pca = None
    y_train_fit = y_train
    y_val_fit = y_val
    y_test_fit = y_test
    k = 0
    if args.pca_k > 0:
        k = min(args.pca_k, y_train.shape[0], y_train.shape[1])
        pca = PCA(n_components=k, random_state=args.seed)
        y_train_fit = pca.fit_transform(y_train).astype(np.float32)
        y_val_fit = pca.transform(y_val).astype(np.float32)
        y_test_fit = pca.transform(y_test).astype(np.float32)
        print(f"PCA enabled: k={k}, EVR_sum={float(pca.explained_variance_ratio_.sum()):.4f}")

    train_loader = DataLoader(FeatDataset(x_train, y_train_fit), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatDataset(x_val, y_val_fit), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FeatDataset(x_test, y_test_fit), batch_size=args.batch_size, shuffle=False)

    model = SimpleRegressor(x_train.shape[1], y_train_fit.shape[1], args.hidden_dim, args.dropout).to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, loss_fn, device)
        val_loss, _, _ = eval_epoch(model, val_loader, loss_fn, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_pred_t, _ = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred_t, _ = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    val_pred = val_pred_t.numpy()
    test_pred = test_pred_t.numpy()
    if pca is not None:
        val_pred_orig = pca.inverse_transform(val_pred)
        test_pred_orig = pca.inverse_transform(test_pred)
    else:
        val_pred_orig = val_pred
        test_pred_orig = test_pred

    summary_rows = []
    for split_name, preds, trues, groups in [
        ("validation", val_pred_orig, y_val, val_groups),
        ("test", test_pred_orig, y_test, test_groups),
    ]:
        overall_df = pearson_per_feature(preds, trues, enh_ref)
        summarize_corr(overall_df, f"{split_name}_overall")
        summary_rows.append(metric_row(overall_df, split_name, "ALL", k, len(enh_ref)))
        overall_df.to_csv(out_dir / f"per_enhancer_correlation_{split_name}_all.csv", index=False)
        for cancer in ["BRCA", "LUAD", "SKCM"]:
            mask = groups == cancer
            corr_df = pearson_per_feature(preds[mask], trues[mask], enh_ref)
            summarize_corr(corr_df, f"{split_name}_{cancer}")
            summary_rows.append(metric_row(corr_df, split_name, cancer, k, len(enh_ref)))
            corr_df.to_csv(out_dir / f"per_enhancer_correlation_{split_name}_{cancer}.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_by_split_and_cancer.csv", index=False)

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    np.save(out_dir / "train_ids.npy", np.array(train_ids))
    np.save(out_dir / "val_ids.npy", np.array(val_ids))
    np.save(out_dir / "test_ids.npy", np.array(test_ids))
    np.save(out_dir / "train_groups.npy", train_groups)
    np.save(out_dir / "val_groups.npy", val_groups)
    np.save(out_dir / "test_groups.npy", test_groups)
    np.save(out_dir / "val_pred.npy", val_pred_orig)
    np.save(out_dir / "val_true.npy", y_val)
    np.save(out_dir / "test_pred.npy", test_pred_orig)
    np.save(out_dir / "test_true.npy", y_test)
    pd.Series(enh_ref, name="enhancer").to_csv(out_dir / "shared_enhancers.csv", index=False)

    if pca is not None:
        np.save(out_dir / "pca_components.npy", pca.components_)
        np.save(out_dir / "pca_mean.npy", pca.mean_)
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
