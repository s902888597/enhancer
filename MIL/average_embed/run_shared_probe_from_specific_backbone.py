#!/usr/bin/env python3
"""
Probe shared-361 prediction from a frozen backbone trained with specific-only supervision.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from run_mean_regression import set_seed
from run_shared_specific_three_cancer_matrix import SharedSpecificRegressor, pearson_df, read_lines, summarize


class ProbeDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SharedProbe(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.probe = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)
        return self.probe(h)


def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    return total / max(n, 1), np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-group", choices=["all", "BRCA", "LUAD", "SKCM"], default="all")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_enh = read_lines(matrix_dir / "shared_enhancers.txt")
    x_train = np.load(matrix_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(matrix_dir / "y_shared_train.npy").astype(np.float32)
    x_val = np.load(matrix_dir / "X_validation.npy").astype(np.float32)
    y_val = np.load(matrix_dir / "y_shared_validation.npy").astype(np.float32)
    x_test = np.load(matrix_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(matrix_dir / "y_shared_test.npy").astype(np.float32)
    g_train = np.load(matrix_dir / "group_train.npy", allow_pickle=True)
    g_val = np.load(matrix_dir / "group_validation.npy", allow_pickle=True)
    g_test = np.load(matrix_dir / "group_test.npy", allow_pickle=True)

    if args.eval_group != "all":
        train_mask = g_train == args.eval_group
        val_mask = g_val == args.eval_group
        test_mask = g_test == args.eval_group
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]
        x_val = x_val[val_mask]
        y_val = y_val[val_mask]
        x_test = x_test[test_mask]
        y_test = y_test[test_mask]

    train_loader = DataLoader(ProbeDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ProbeDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ProbeDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    base_model = SharedSpecificRegressor(
        input_dim=x_train.shape[1],
        shared_dim=len(shared_enh),
        specific_dim=639,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    base_model.load_state_dict(state, strict=True)
    model = SharedProbe(base_model.backbone, args.hidden_dim, len(shared_enh), args.dropout).to(device)

    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

    val_loss, val_pred, val_true = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred, test_true = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    val_df = pearson_df(val_pred, val_true, shared_enh)
    test_df = pearson_df(test_pred, test_true, shared_enh)
    prefix = args.eval_group if args.eval_group != "all" else "Combined"
    summarize(val_df, f"validation_shared_probe_{prefix}")
    summarize(test_df, f"test_shared_probe_{prefix}")

    val_df.to_csv(out_dir / "per_enhancer_correlation_validation.csv", index=False)
    test_df.to_csv(out_dir / "per_enhancer_correlation_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "validation",
                "group": prefix,
                "mean_pearson": val_df["pearson_r"].mean(skipna=True),
                "median_pearson": val_df["pearson_r"].median(skipna=True),
                "gt_0.4": int((val_df["pearson_r"] > 0.4).sum()),
                "gt_0.5": int((val_df["pearson_r"] > 0.5).sum()),
            },
            {
                "split": "test",
                "group": prefix,
                "mean_pearson": test_df["pearson_r"].mean(skipna=True),
                "median_pearson": test_df["pearson_r"].median(skipna=True),
                "gt_0.4": int((test_df["pearson_r"] > 0.4).sum()),
                "gt_0.5": int((test_df["pearson_r"] > 0.5).sum()),
            },
        ]
    ).to_csv(out_dir / "summary.csv", index=False)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "test_true.npy", test_true)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
