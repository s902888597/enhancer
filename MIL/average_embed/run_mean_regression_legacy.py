#!/usr/bin/env python3
"""
Legacy mean-pool patch embeddings per case and train a simple MLP regressor
to predict super-enhancer expression.

This recreates the simpler pre-pan-cancer version:
- labels are sample-wise CSVs: columns = sample, <enhancer1>, <enhancer2>, ...
- features live under <feat_root>/<split>/<case>/*.npy
- no internal PCA
- no mean cache
- no bad-npy skip logic
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_labels(csv_path: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    ids = df["sample"].astype(str).tolist()
    enh_cols = [c for c in df.columns if c != "sample"]
    labels = df[enh_cols].values.astype(np.float32)
    return ids, labels, enh_cols


def mean_embed_for_case(case_dir: Path) -> np.ndarray:
    files = sorted(case_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No npy files in {case_dir}")
    arrs = [np.load(f).astype(np.float32) for f in files]
    return np.stack(arrs, axis=0).mean(axis=0)


def load_split_feats(
    split: str, csv_path: Path, feat_root: Path
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    ids, labels, enh_cols = load_labels(csv_path)
    feats = []
    kept_ids = []
    kept_labels = []
    missing = []
    for sid, lab in zip(ids, labels):
        case_dir = feat_root / split / sid
        if not case_dir.exists():
            missing.append(sid)
            continue
        try:
            emb = mean_embed_for_case(case_dir)
        except FileNotFoundError:
            missing.append(sid)
            continue
        kept_ids.append(sid)
        kept_labels.append(lab)
        feats.append(emb)

    if missing:
        print(f"[{split}] missing/empty: {len(missing)} (e.g., {missing[:5]})")
    if not feats:
        raise RuntimeError(f"No features loaded for split {split}")
    feats_arr = np.stack(feats, axis=0).astype(np.float32)
    labels_arr = np.stack(kept_labels, axis=0).astype(np.float32)
    return kept_ids, feats_arr, labels_arr, enh_cols


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FeatDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / n


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
            preds.append(pred.cpu())
            trues.append(yb.cpu())
    return total / n, torch.cat(preds), torch.cat(trues)


def pearson_per_feature(preds: np.ndarray, trues: np.ndarray, enh_cols: List[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(enh_cols):
        x = preds[:, i]
        y = trues[:, i]
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            r = np.nan
        else:
            r = np.corrcoef(x, y)[0, 1]
        rows.append({"enhancer": name, "pearson_r": r})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_root = Path(args.feat_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, x_train, y_train, enh_cols = load_split_feats("train", Path(args.train_csv), feat_root)
    val_ids, x_val, y_val, _ = load_split_feats("validation", Path(args.val_csv), feat_root)
    test_ids, x_test, y_test, _ = load_split_feats("test", Path(args.test_csv), feat_root)

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    train_loader = DataLoader(FeatDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FeatDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = SimpleRegressor(input_dim, output_dim, args.hidden_dim, args.dropout).to(device)
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

    if best_state:
        model.load_state_dict(best_state)

    val_loss, _, _ = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred_t, test_true_t = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    test_pred = test_pred_t.numpy()
    test_true = test_true_t.numpy()
    corr_df = pearson_per_feature(test_pred, test_true, enh_cols)

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    np.save(out_dir / "train_ids.npy", np.array(train_ids))
    np.save(out_dir / "val_ids.npy", np.array(val_ids))
    np.save(out_dir / "test_ids.npy", np.array(test_ids))
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "test_true.npy", test_true)
    corr_df.to_csv(out_dir / "per_enhancer_correlation.csv", index=False)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
