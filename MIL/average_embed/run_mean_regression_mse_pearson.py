#!/usr/bin/env python3
"""
Mean-pool case embeddings + MLP regressor with composite loss:
  loss = MSE + lambda_corr * (1 - mean_per_enhancer_corr)
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
    cols = [c for c in df.columns if c != "sample"]
    y = df[cols].values.astype(np.float32)
    return ids, y, cols


def mean_embed_for_case(case_dir: Path) -> np.ndarray:
    files = sorted(case_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No npy in {case_dir}")
    arrs = [np.load(f).astype(np.float32) for f in files]
    return np.stack(arrs, axis=0).mean(axis=0)


def load_split(
    split: str, csv_path: Path, feat_root: Path
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    ids, y, cols = load_labels(csv_path)
    keep_ids, keep_y, x_list = [], [], []
    missing = []
    for sid, yi in zip(ids, y):
        d = feat_root / split / sid
        if not d.exists():
            missing.append(sid)
            continue
        try:
            x = mean_embed_for_case(d)
        except FileNotFoundError:
            missing.append(sid)
            continue
        keep_ids.append(sid)
        keep_y.append(yi)
        x_list.append(x)
    if missing:
        print(f"[{split}] missing/empty={len(missing)}")
    if not x_list:
        raise RuntimeError(f"No data loaded for {split}")
    return (
        keep_ids,
        np.stack(x_list, axis=0).astype(np.float32),
        np.stack(keep_y, axis=0).astype(np.float32),
        cols,
    )


class FeatDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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


def mean_per_output_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred/target: (B, D)
    Computes Pearson correlation per output dim across batch, then mean over D.
    """
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    targ_c = target - target.mean(dim=0, keepdim=True)
    cov = (pred_c * targ_c).mean(dim=0)
    pred_std = torch.sqrt((pred_c.pow(2).mean(dim=0)).clamp_min(eps))
    targ_std = torch.sqrt((targ_c.pow(2).mean(dim=0)).clamp_min(eps))
    corr = cov / (pred_std * targ_std + eps)
    return corr.mean()


def composite_loss(pred: torch.Tensor, target: torch.Tensor, mse_fn, lambda_corr: float) -> torch.Tensor:
    mse = mse_fn(pred, target)
    corr_mean = mean_per_output_corr(pred, target)
    return mse + lambda_corr * (1.0 - corr_mean)


def train_epoch(model, loader, optim, mse_fn, device, lambda_corr: float):
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = composite_loss(pred, yb, mse_fn, lambda_corr)
        loss.backward()
        optim.step()
        tot += loss.item() * xb.size(0)
        n += xb.size(0)
    return tot / max(n, 1)


def eval_epoch(model, loader, mse_fn, device, lambda_corr: float):
    model.eval()
    tot, n = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = composite_loss(pred, yb, mse_fn, lambda_corr)
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
            preds.append(pred.cpu())
            trues.append(yb.cpu())
    return tot / max(n, 1), torch.cat(preds), torch.cat(trues)


def pearson_per_feature(preds: np.ndarray, trues: np.ndarray, cols: List[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(cols):
        x, y = preds[:, i], trues[:, i]
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            r = np.nan
        else:
            r = np.corrcoef(x, y)[0, 1]
        rows.append({"enhancer": name, "pearson_r": r})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--feat-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lambda-corr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=44)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, x_train, y_train, cols = load_split("train", Path(args.train_csv), Path(args.feat_root))
    val_ids, x_val, y_val, _ = load_split("validation", Path(args.val_csv), Path(args.feat_root))
    test_ids, x_test, y_test, _ = load_split("test", Path(args.test_csv), Path(args.feat_root))

    model = SimpleRegressor(x_train.shape[1], y_train.shape[1], args.hidden_dim, args.dropout).to(device)
    mse_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(FeatDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FeatDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optim, mse_fn, device, args.lambda_corr)
        va_loss, _, _ = eval_epoch(model, val_loader, mse_fn, device, args.lambda_corr)
        if va_loss < best_val:
            best_val = va_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, _, _ = eval_epoch(model, val_loader, mse_fn, device, args.lambda_corr)
    test_loss, pred_t, true_t = eval_epoch(model, test_loader, mse_fn, device, args.lambda_corr)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    pred = pred_t.numpy()
    true = true_t.numpy()
    corr_df = pearson_per_feature(pred, true, cols)

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    np.save(out_dir / "train_ids.npy", np.array(train_ids))
    np.save(out_dir / "val_ids.npy", np.array(val_ids))
    np.save(out_dir / "test_ids.npy", np.array(test_ids))
    np.save(out_dir / "test_pred.npy", pred)
    np.save(out_dir / "test_true.npy", true)
    corr_df.to_csv(out_dir / "per_enhancer_correlation.csv", index=False)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

