#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SKCM_FILES = {
    "train": ("SKCM_train_train_X.npy", "SKCM_train_train_ids.txt"),
    "validation": ("SKCM_validation_validation_X.npy", "SKCM_validation_validation_ids.txt"),
    "test": ("SKCM_test_test_X.npy", "SKCM_test_test_ids.txt"),
}


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, x_uni: np.ndarray, x_loki: np.ndarray):
        self.x_uni = torch.tensor(x_uni, dtype=torch.float32)
        self.x_loki = torch.tensor(x_loki, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_uni.shape[0]

    def __getitem__(self, idx: int):
        return self.x_uni[idx], self.x_loki[idx]


class Projector(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


def clip_loss(z_uni: torch.Tensor, z_loki: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (z_uni @ z_loki.T) / temperature
    targets = torch.arange(z_uni.shape[0], device=z_uni.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i + loss_t)


def load_case_features(case_root: Path) -> tuple[np.ndarray, list[str]]:
    x = np.load(case_root / "X.npy").astype(np.float32)
    ids = read_ids(case_root / "ids.txt")
    if x.shape[0] != len(ids):
        raise RuntimeError(f"{case_root}: X/ids mismatch")
    return x, ids


def pair_case(uni_case: Path, loki_case: Path) -> tuple[np.ndarray, np.ndarray, int]:
    x_uni, ids_uni = load_case_features(uni_case)
    x_loki, ids_loki = load_case_features(loki_case)
    loki_index = {name: i for i, name in enumerate(ids_loki)}
    keep_uni = []
    keep_loki = []
    missing = 0
    for i, name in enumerate(ids_uni):
        j = loki_index.get(name)
        if j is None:
            missing += 1
            continue
        keep_uni.append(i)
        keep_loki.append(j)
    if not keep_uni:
        return np.zeros((0, x_uni.shape[1]), dtype=np.float32), np.zeros((0, x_loki.shape[1]), dtype=np.float32), missing
    return x_uni[keep_uni], x_loki[keep_loki], missing


def load_pairs_for_split(split: str, ids: list[str], uni_root: Path, loki_root: Path):
    xs_uni = []
    xs_loki = []
    rows = []
    for sid in ids:
        uni_case = uni_root / sid
        loki_case = loki_root / sid
        if not (uni_case / "X.npy").exists():
            raise RuntimeError(f"{split}: missing UNI case {sid}")
        if not (loki_case / "X.npy").exists():
            raise RuntimeError(f"{split}: missing Loki case {sid}")
        x_uni, x_loki, missing = pair_case(uni_case, loki_case)
        rows.append(
            {
                "split": split,
                "case_id": sid,
                "n_pairs": int(x_uni.shape[0]),
                "n_missing_ids": int(missing),
            }
        )
        if x_uni.shape[0] > 0:
            xs_uni.append(x_uni)
            xs_loki.append(x_loki)
    x_uni_all = np.concatenate(xs_uni, axis=0).astype(np.float32)
    x_loki_all = np.concatenate(xs_loki, axis=0).astype(np.float32)
    return x_uni_all, x_loki_all, pd.DataFrame(rows)


def eval_epoch(model, loader, device: torch.device, temperature: float):
    model.eval()
    total = 0.0
    n = 0
    cosine_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = F.normalize(yb.to(device), p=2, dim=-1)
            pred = model(xb)
            loss = clip_loss(pred, yb, temperature)
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
            cosine_sum += float((pred * yb).sum(dim=1).sum().item())
            count += xb.shape[0]
    return total / max(n, 1), cosine_sum / max(count, 1)


def train_epoch(model, loader, optim, device: torch.device, temperature: float):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = F.normalize(yb.to(device), p=2, dim=-1)
        optim.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = clip_loss(pred, yb, temperature)
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


def project_case_mean(model, uni_case: Path, device: torch.device, batch_size: int) -> np.ndarray:
    x, _ = load_case_features(uni_case)
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).cpu().numpy())
    z = np.concatenate(outs, axis=0).astype(np.float32)
    return z.mean(axis=0).astype(np.float32)


def build_mean_cache(model, base_cache: Path, uni_root: Path, out_cache: Path, device: torch.device, batch_size: int) -> dict:
    out_cache.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, (x_name, ids_name) in SKCM_FILES.items():
        ids = read_ids(base_cache / ids_name)
        feats = []
        for sid in ids:
            feats.append(project_case_mean(model, uni_root / sid, device, batch_size))
        x = np.stack(feats, axis=0).astype(np.float32)
        np.save(out_cache / x_name, x)
        (out_cache / ids_name).write_text("\n".join(ids) + "\n")
        summary[split] = {"n_ids": len(ids), "feature_dim": int(x.shape[1])}
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-cache", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/common361_best3_mean_mlp/cache")
    p.add_argument("--uni-root", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_uni_skcm_packed")
    p.add_argument("--loki-root", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_loki_skcm_packed")
    p.add_argument("--out-dir", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_uni_to_loki_alignment")
    p.add_argument("--aligned-cache-out", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/uni_loki_aligned_skcm_mean_cache")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--early-patience", type=int, default=4)
    args = p.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    base_cache = Path(args.base_cache)
    uni_root = Path(args.uni_root)
    loki_root = Path(args.loki_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_ids = {split: read_ids(base_cache / ids_name) for split, (_, ids_name) in SKCM_FILES.items()}
    x_train_uni, x_train_loki, train_case_df = load_pairs_for_split("train", split_ids["train"], uni_root, loki_root)
    x_val_uni, x_val_loki, val_case_df = load_pairs_for_split("validation", split_ids["validation"], uni_root, loki_root)
    pd.concat([train_case_df, val_case_df], ignore_index=True).to_csv(out_dir / "case_pair_counts.csv", index=False)

    train_loader = DataLoader(PairDataset(x_train_uni, x_train_loki), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(x_val_uni, x_val_loki), batch_size=args.batch_size, shuffle=False)

    model = Projector(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, device, args.temperature)
        val_loss, val_cos = eval_epoch(model, val_loader, device, args.temperature)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mean_cosine": val_cos})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_cos={val_cos:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.early_patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_loss, train_cos = eval_epoch(model, train_loader, device, args.temperature)
    final_val_loss, val_cos = eval_epoch(model, val_loader, device, args.temperature)

    torch.save(model.state_dict(), out_dir / "best_projector.pt")
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train_pairs": int(x_train_uni.shape[0]),
                "validation_pairs": int(x_val_uni.shape[0]),
                "final_train_loss": float(final_train_loss),
                "final_validation_loss": float(final_val_loss),
                "train_mean_cosine": float(train_cos),
                "validation_mean_cosine": float(val_cos),
            },
            indent=2,
        )
    )

    cache_summary = build_mean_cache(
        model=model,
        base_cache=base_cache,
        uni_root=uni_root,
        out_cache=Path(args.aligned_cache_out),
        device=device,
        batch_size=args.batch_size,
    )
    (Path(args.aligned_cache_out) / "cache_summary.json").write_text(json.dumps(cache_summary, indent=2))
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "base_cache": str(base_cache),
                "uni_root": str(uni_root),
                "loki_root": str(loki_root),
                "aligned_cache_out": str(args.aligned_cache_out),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "temperature": args.temperature,
                "seed": args.seed,
                "device": str(device),
            },
            indent=2,
        )
    )
    print(f"Saved aligned cache to {args.aligned_cache_out}", flush=True)


if __name__ == "__main__":
    main()
