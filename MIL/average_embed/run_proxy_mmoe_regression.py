#!/usr/bin/env python3
"""
Single-cancer proxy + gate + experts MIL regression.

This is a minimal M4-inspired baseline for one cancer type:
- learnable proxies softly attend to patch embeddings
- shared experts map proxy summaries to bag representations
- one gate chooses expert weights for the current bag
- one regression head predicts the full target vector
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from run_mean_regression import pearson_per_feature, set_seed


def load_labels(csv_path: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    df = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    header = list(df.dtype.names)
    if not header or header[0] != "sample":
        raise RuntimeError(f"Expected first column 'sample' in {csv_path}")
    ids = [row[0] for row in df.tolist()]
    enh_cols = header[1:]
    labels = np.stack([[float(row[i]) for i in range(1, len(header))] for row in df.tolist()], axis=0).astype(np.float32)
    return ids, labels, enh_cols


def load_case_patch_matrix(case_dir: Path) -> np.ndarray:
    files = sorted(case_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No npy files in {case_dir}")
    arrs = []
    for f in files:
        try:
            arrs.append(np.load(f).astype(np.float32))
        except Exception:
            continue
    if not arrs:
        raise FileNotFoundError(f"No readable npy files in {case_dir}")
    return np.stack(arrs, axis=0)


class BagDataset(Dataset):
    def __init__(
        self,
        feat_root: Path,
        csv_path: Path,
        split: str,
        bag_cache_dir: Optional[Path],
        max_patches: int,
        train_mode: bool,
        seed: int,
    ):
        ids, labels, enh_cols = load_labels(csv_path)
        self.items = []
        self.enh_cols = enh_cols
        self.max_patches = max_patches
        self.train_mode = train_mode
        self.rng = np.random.default_rng(seed)
        self.split = split
        self.bag_cache_dir = bag_cache_dir
        if self.bag_cache_dir is not None:
            self.bag_cache_dir.mkdir(parents=True, exist_ok=True)
        missing = 0
        for sid, y in zip(ids, labels):
            case_dir = feat_root / split / sid
            if not case_dir.exists():
                case_dir = feat_root / sid
            if not case_dir.exists():
                missing += 1
                continue
            cache_path = None
            if self.bag_cache_dir is not None:
                cache_path = self.bag_cache_dir / f"{sid}.npy"
            self.items.append((sid, case_dir, cache_path, y))
        if missing:
            print(f"[warn] {csv_path.name}: missing feature dirs={missing}")
        if not self.items:
            raise RuntimeError(f"No usable cases found for {csv_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, case_dir, cache_path, y = self.items[idx]
        if cache_path is not None and cache_path.exists():
            x = np.load(cache_path).astype(np.float32)
        else:
            x = load_case_patch_matrix(case_dir)
            if cache_path is not None:
                np.save(cache_path, x.astype(np.float32))
        if self.max_patches > 0 and x.shape[0] > self.max_patches:
            if self.train_mode:
                keep = self.rng.choice(x.shape[0], size=self.max_patches, replace=False)
            else:
                keep = np.arange(self.max_patches)
            x = x[keep]
        return sid, torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def bag_collate(batch):
    ids = [item[0] for item in batch]
    xs = [item[1] for item in batch]
    ys = torch.stack([item[2] for item in batch], dim=0)
    return ids, xs, ys


class ProxyMILMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_dim: int,
        num_proxies: int,
        num_experts: int,
        expert_dim: int,
        dropout: float,
        gate_temperature: float,
    ):
        super().__init__()
        self.gate_temperature = gate_temperature
        self.patch_proj = nn.Linear(input_dim, model_dim)
        self.proxies = nn.Parameter(torch.randn(num_proxies, model_dim) * 0.02)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, expert_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, expert_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )
        self.proxy_pool = nn.Linear(model_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_experts),
        )
        self.head = nn.Linear(expert_dim, output_dim)

    def forward_one(self, bag_x: torch.Tensor):
        x = self.patch_proj(bag_x)  # [N, H]
        scores = self.proxies @ x.transpose(0, 1)  # [M, N]
        proxy_attn = F.softmax(scores, dim=1)
        proxy_repr = proxy_attn @ x  # [M, H]

        proxy_weights = F.softmax(self.proxy_pool(proxy_repr).squeeze(-1), dim=0)  # [M]
        pooled_proxy = torch.sum(proxy_weights.unsqueeze(-1) * proxy_repr, dim=0)  # [H]

        expert_repr = torch.stack([expert(pooled_proxy) for expert in self.experts], dim=0)  # [K, E]
        gate_logits = self.gate(pooled_proxy) / self.gate_temperature
        gate_weights = F.softmax(gate_logits, dim=0)  # [K]
        mixed = torch.sum(gate_weights.unsqueeze(-1) * expert_repr, dim=0)  # [E]
        pred = self.head(mixed)
        return pred, gate_weights, proxy_weights, proxy_attn

    def forward(self, bags: List[torch.Tensor]):
        preds = []
        gate_weights_all = []
        proxy_weights_all = []
        for bag_x in bags:
            pred, gate_w, proxy_w, _ = self.forward_one(bag_x)
            preds.append(pred)
            gate_weights_all.append(gate_w)
            proxy_weights_all.append(proxy_w)
        return (
            torch.stack(preds, dim=0),
            torch.stack(gate_weights_all, dim=0),
            torch.stack(proxy_weights_all, dim=0),
        )


def train_epoch(model, loader, optim, loss_fn, device, gate_entropy_lambda: float):
    model.train()
    total = 0.0
    n = 0
    for _, bags, yb in loader:
        bags = [bag.to(device) for bag in bags]
        yb = yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred, gate_weights, _ = model(bags)
        loss = loss_fn(pred, yb)
        if gate_entropy_lambda > 0:
            gate_entropy = -(gate_weights.clamp_min(1e-8) * gate_weights.clamp_min(1e-8).log()).sum(dim=1).mean()
            loss = loss - gate_entropy_lambda * gate_entropy
        loss.backward()
        optim.step()
        total += float(loss.item()) * yb.shape[0]
        n += yb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    ids_all = []
    gate_all = []
    proxy_all = []
    for ids, bags, yb in loader:
        bags = [bag.to(device) for bag in bags]
        yb = yb.to(device)
        pred, gate_w, proxy_w = model(bags)
        loss = loss_fn(pred, yb)
        total += float(loss.item()) * yb.shape[0]
        n += yb.shape[0]
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
        gate_all.append(gate_w.cpu().numpy())
        proxy_all.append(proxy_w.cpu().numpy())
        ids_all.extend(ids)
    return (
        total / max(n, 1),
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(gate_all, axis=0),
        np.concatenate(proxy_all, axis=0),
        ids_all,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bag-cache-dir", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-proxies", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--expert-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-entropy-lambda", type=float, default=0.0)
    parser.add_argument("--max-patches-train", type=int, default=512)
    parser.add_argument("--max-patches-eval", type=int, default=512)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--pca-k", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_root = Path(args.feat_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bag_cache_root = Path(args.bag_cache_dir) if args.bag_cache_dir else None

    train_ds = BagDataset(
        feat_root,
        Path(args.train_csv),
        "train",
        None if bag_cache_root is None else bag_cache_root / "train",
        args.max_patches_train,
        True,
        args.seed,
    )
    val_ds = BagDataset(
        feat_root,
        Path(args.val_csv),
        "validation",
        None if bag_cache_root is None else bag_cache_root / "validation",
        args.max_patches_eval,
        False,
        args.seed,
    )
    test_ds = BagDataset(
        feat_root,
        Path(args.test_csv),
        "test",
        None if bag_cache_root is None else bag_cache_root / "test",
        args.max_patches_eval,
        False,
        args.seed,
    )
    enh_cols = train_ds.enh_cols

    y_train = np.stack([item[3] for item in train_ds.items], axis=0).astype(np.float32)
    y_val = np.stack([item[3] for item in val_ds.items], axis=0).astype(np.float32)
    y_test = np.stack([item[3] for item in test_ds.items], axis=0).astype(np.float32)
    pca = None
    y_test_orig = y_test.copy()
    if args.pca_k > 0:
        k = min(args.pca_k, y_train.shape[0], y_train.shape[1])
        pca = PCA(n_components=k, random_state=args.seed)
        y_train_p = pca.fit_transform(y_train).astype(np.float32)
        y_val_p = pca.transform(y_val).astype(np.float32)
        y_test_p = pca.transform(y_test).astype(np.float32)
        for ds, arr in [(train_ds, y_train_p), (val_ds, y_val_p), (test_ds, y_test_p)]:
            for i, item in enumerate(ds.items):
                ds.items[i] = (item[0], item[1], item[2], arr[i])
        np.save(out_dir / "pca_components.npy", pca.components_)
        np.save(out_dir / "pca_mean.npy", pca.mean_)
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)
        print(f"PCA enabled: k={k}, EVR_sum={float(pca.explained_variance_ratio_.sum()):.4f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=bag_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=bag_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=bag_collate)

    sample_bag = train_ds[0][1]
    output_dim = len(enh_cols) if pca is None else int(train_ds.items[0][3].shape[0])
    model = ProxyMILMoE(
        input_dim=sample_bag.shape[1],
        output_dim=output_dim,
        model_dim=args.model_dim,
        num_proxies=args.num_proxies,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        dropout=args.dropout,
        gate_temperature=args.gate_temperature,
    ).to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, loss_fn, device, args.gate_entropy_lambda)
        val_loss, _, _, _, _, _ = eval_epoch(model, val_loader, loss_fn, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_pred, _, val_gate, val_proxy, val_ids = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred, _, test_gate, test_proxy, test_ids = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    if pca is not None:
        val_pred = pca.inverse_transform(val_pred).astype(np.float32)
        test_pred = pca.inverse_transform(test_pred).astype(np.float32)
        test_true = y_test_orig.astype(np.float32)
    else:
        test_true = y_test.astype(np.float32)

    corr_df = pearson_per_feature(test_pred, test_true, enh_cols)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    np.save(out_dir / "val_ids.npy", np.array(val_ids))
    np.save(out_dir / "test_ids.npy", np.array(test_ids))
    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "test_true.npy", test_true)
    np.save(out_dir / "val_gate_weights.npy", val_gate.astype(np.float32))
    np.save(out_dir / "test_gate_weights.npy", test_gate.astype(np.float32))
    np.save(out_dir / "val_proxy_weights.npy", val_proxy.astype(np.float32))
    np.save(out_dir / "test_proxy_weights.npy", test_proxy.astype(np.float32))
    corr_df.to_csv(out_dir / "per_enhancer_correlation.csv", index=False)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
