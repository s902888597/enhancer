#!/usr/bin/env python3
"""
Train one Mean-MLP on prebuilt train/validation/test matrices for the shared
enhancers across BRCA/LUAD/SKCM.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from run_mean_regression_pan_cancer import (
    eval_epoch,
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


def metric_row(corr_df: pd.DataFrame, split: str, group: str, k: int, n_enhancers: int):
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


def load_split(matrix_dir: Path, split: str):
    x = np.load(matrix_dir / f"X_{split}.npy")
    y = np.load(matrix_dir / f"y_{split}.npy")
    groups = np.load(matrix_dir / f"group_{split}.npy")
    ids = [line.strip() for line in (matrix_dir / f"id_{split}.txt").read_text().splitlines() if line.strip()]
    return ids, x, y, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--pca-k", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enh_ref = [line.strip() for line in (matrix_dir / "enhancers.txt").read_text().splitlines() if line.strip()]
    train_ids, x_train, y_train, train_groups = load_split(matrix_dir, "train")
    val_ids, x_val, y_val, val_groups = load_split(matrix_dir, "validation")
    test_ids, x_test, y_test, test_groups = load_split(matrix_dir, "test")

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
        np.save(out_dir / "pca_components.npy", pca.components_)
        np.save(out_dir / "pca_mean.npy", pca.mean_)
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)
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

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_split_and_cancer.csv", index=False)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    np.save(out_dir / "val_pred.npy", val_pred_orig)
    np.save(out_dir / "val_true.npy", y_val)
    np.save(out_dir / "test_pred.npy", test_pred_orig)
    np.save(out_dir / "test_true.npy", y_test)
    np.save(out_dir / "train_groups.npy", train_groups)
    np.save(out_dir / "val_groups.npy", val_groups)
    np.save(out_dir / "test_groups.npy", test_groups)
    np.save(out_dir / "train_ids.npy", np.array(train_ids))
    np.save(out_dir / "val_ids.npy", np.array(val_ids))
    np.save(out_dir / "test_ids.npy", np.array(test_ids))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
