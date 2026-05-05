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
from torch.utils.data import DataLoader, Dataset


IMAGE_CACHE = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/loki_brca_mean_cache")
TISSUE_CACHE = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/tissue4_brca_mean_cache")
LABEL_CSV = {
    "train": "/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/BRCA_192train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI/Top1000_SEs_192train_eRNA_zscore_3eRNAremoved.csv",
    "validation": "/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/BRCA_192train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI/Top1000_SEs_63val_eRNA_zscore_3eRNAremoved.csv",
    "test": "/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/BRCA_192train_63val_64test_SE_eRNA_zscore_3eRNAremoved_csv_for_AI/Top1000_SEs_64test_eRNA_zscore_3eRNAremoved.csv",
}
FILES = {
    "train": (
        "BRCA_train_Top1000_SEs_192train_eRNA_zscore_3eRNAremoved_X.npy",
        "BRCA_train_Top1000_SEs_192train_eRNA_zscore_3eRNAremoved_ids.txt",
    ),
    "validation": (
        "BRCA_validation_Top1000_SEs_63val_eRNA_zscore_3eRNAremoved_X.npy",
        "BRCA_validation_Top1000_SEs_63val_eRNA_zscore_3eRNAremoved_ids.txt",
    ),
    "test": (
        "BRCA_test_Top1000_SEs_64test_eRNA_zscore_3eRNAremoved_X.npy",
        "BRCA_test_Top1000_SEs_64test_eRNA_zscore_3eRNAremoved_ids.txt",
    ),
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def canonical_sample_id(sample_id: str) -> str:
    return sample_id[:-6] if sample_id.endswith("_tumor") else sample_id


def subset_by_ids(x: np.ndarray, ids: list[str], keep_ids: list[str]) -> np.ndarray:
    pos = {sid: idx for idx, sid in enumerate(ids)}
    return x[[pos[sid] for sid in keep_ids]].astype(np.float32)


def align_targets_samples_as_columns(csv_path: Path, ids: list[str], enhancers_ref: list[str] | None) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    enhancers = df["SE_ID"].astype(str).tolist()
    if enhancers_ref is not None and enhancers != enhancers_ref:
        raise RuntimeError(f"{csv_path}: enhancer list does not match training split")
    sample_cols = [c for c in df.columns if c not in {"chr", "start", "end", "SE_ID", "eRNA_count"}]
    col_map = {}
    for col in sample_cols:
        canon = canonical_sample_id(col)
        if canon in col_map:
            raise RuntimeError(f"{csv_path}: duplicated canonical sample id {canon}")
        col_map[canon] = col
    keep_ids = [sid for sid in ids if canonical_sample_id(sid) in col_map]
    ordered_cols = [col_map[canonical_sample_id(sid)] for sid in keep_ids]
    y = df.loc[:, ordered_cols].to_numpy(dtype=np.float32).T
    return keep_ids, y, enhancers


class FusionDataset(Dataset):
    def __init__(self, x_img: np.ndarray, x_tissue: np.ndarray, y: np.ndarray):
        self.x_img = torch.tensor(x_img, dtype=torch.float32)
        self.x_tissue = torch.tensor(x_tissue, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_img.shape[0]

    def __getitem__(self, idx: int):
        return self.x_img[idx], self.x_tissue[idx], self.y[idx]


class TissueFiLMRegressor(nn.Module):
    def __init__(self, img_dim: int, tissue_dim: int, output_dim: int, hidden_dim: int, cond_dim: int, dropout: float):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.img_norm = nn.LayerNorm(hidden_dim)
        self.cond = nn.Sequential(
            nn.Linear(tissue_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_img: torch.Tensor, x_tissue: torch.Tensor) -> torch.Tensor:
        h = self.img_norm(self.img_proj(x_img))
        gamma, beta = self.cond(x_tissue).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        return self.head(h)


def train_epoch(model, loader, optim, loss_fn, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb_img, xb_tissue, yb in loader:
        xb_img = xb_img.to(device)
        xb_tissue = xb_tissue.to(device)
        yb = yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(xb_img, xb_tissue)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb_img.shape[0]
        n += xb_img.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    for xb_img, xb_tissue, yb in loader:
        xb_img = xb_img.to(device)
        xb_tissue = xb_tissue.to(device)
        yb = yb.to(device)
        pred = model(xb_img, xb_tissue)
        loss = loss_fn(pred, yb)
        total += float(loss.item()) * xb_img.shape[0]
        n += xb_img.shape[0]
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    return total / max(n, 1), np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def pearson_per_feature(preds: np.ndarray, trues: np.ndarray, enhancers: list[str]) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(enhancers):
        x = preds[:, idx]
        y = trues[:, idx]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"enhancer": name, "pearson_r": r})
    return pd.DataFrame(rows)


def metric_row(split: str, corr_df: pd.DataFrame) -> dict:
    s = corr_df["pearson_r"]
    return {
        "split": split,
        "n_enhancers": int(len(corr_df)),
        "pearson_mean": float(s.mean(skipna=True)),
        "pearson_median": float(s.median(skipna=True)),
        "gt_0.4": int((s > 0.4).sum()),
        "gt_0.5": int((s > 0.5).sum()),
        "gt_0.6": int((s > 0.6).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--cond-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_img_raw = {}
    x_img_raw = {}
    ids_tissue_raw = {}
    x_tissue_raw = {}
    for split in FILES:
        x_name, id_name = FILES[split]
        ids_img_raw[split] = read_ids(IMAGE_CACHE / id_name)
        x_img_raw[split] = np.load(IMAGE_CACHE / x_name).astype(np.float32)
        ids_tissue_raw[split] = read_ids(TISSUE_CACHE / id_name)
        x_tissue_raw[split] = np.load(TISSUE_CACHE / x_name).astype(np.float32)
        if ids_img_raw[split] != ids_tissue_raw[split]:
            raise RuntimeError(f"{split}: image/tissue ids mismatch")

    ids = {}
    x_img = {}
    x_tissue = {}
    y = {}
    enhancers_ref: list[str] | None = None
    for split in FILES:
        keep_ids, y_split, enhancers = align_targets_samples_as_columns(Path(LABEL_CSV[split]), ids_img_raw[split], enhancers_ref)
        if enhancers_ref is None:
            enhancers_ref = enhancers
        ids[split] = keep_ids
        x_img[split] = subset_by_ids(x_img_raw[split], ids_img_raw[split], keep_ids)
        x_tissue[split] = subset_by_ids(x_tissue_raw[split], ids_tissue_raw[split], keep_ids)
        y[split] = y_split
    assert enhancers_ref is not None

    for split in FILES:
        np.save(out_dir / f"X_img_{split}.npy", x_img[split])
        np.save(out_dir / f"X_tissue_{split}.npy", x_tissue[split])
        np.save(out_dir / f"y_{split}.npy", y[split])
        (out_dir / f"ids_{split}.txt").write_text("\n".join(ids[split]) + "\n")
    (out_dir / "enhancers.txt").write_text("\n".join(enhancers_ref) + "\n")

    train_loader = DataLoader(FusionDataset(x_img["train"], x_tissue["train"], y["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FusionDataset(x_img["validation"], x_tissue["validation"], y["validation"]), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FusionDataset(x_img["test"], x_tissue["test"], y["test"]), batch_size=args.batch_size, shuffle=False)

    model = TissueFiLMRegressor(
        img_dim=x_img["train"].shape[1],
        tissue_dim=x_tissue["train"].shape[1],
        output_dim=y["train"].shape[1],
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    train_history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, loss_fn, device)
        val_loss, _, _ = eval_epoch(model, val_loader, loss_fn, device)
        if val_loss < (best_val - args.min_delta):
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        train_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)
        if bad_epochs >= args.early_patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val improvement for {bad_epochs} epochs "
                f"(patience={args.early_patience}, min_delta={args.min_delta}).",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_pred, val_true = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred, test_true = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}", flush=True)

    val_corr = pearson_per_feature(val_pred, val_true, enhancers_ref)
    test_corr = pearson_per_feature(test_pred, test_true, enhancers_ref)
    val_corr.to_csv(out_dir / "per_enhancer_correlation_validation.csv", index=False)
    test_corr.to_csv(out_dir / "per_enhancer_correlation_test.csv", index=False)
    pd.DataFrame([metric_row("validation", val_corr), metric_row("test", test_corr)]).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(train_history).to_csv(out_dir / "train_history.csv", index=False)

    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "test_true.npy", test_true)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    pd.Series(
        {
            "fusion_mode": "tissue4_film",
            "img_cache": str(IMAGE_CACHE),
            "tissue_cache": str(TISSUE_CACHE),
            "hidden_dim": args.hidden_dim,
            "cond_dim": args.cond_dim,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "early_patience": args.early_patience,
            "min_delta": args.min_delta,
            "n_train": int(x_img["train"].shape[0]),
            "n_validation": int(x_img["validation"].shape[0]),
            "n_test": int(x_img["test"].shape[0]),
            "img_dim": int(x_img["train"].shape[1]),
            "tissue_dim": int(x_tissue["train"].shape[1]),
        }
    ).to_json(out_dir / "run_config.json")
    (out_dir / "label_paths.json").write_text(json.dumps(LABEL_CSV, indent=2))
    print(f"Saved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
