#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "sample" not in df.columns:
        raise RuntimeError(f"{csv_path} missing sample column")
    df["sample"] = df["sample"].astype(str)
    return df


def align_targets(csv_path: Path, ids: list[str], enhancers_ref: list[str] | None) -> tuple[np.ndarray, list[str]]:
    df = load_labels(csv_path).set_index("sample")
    missing = [sid for sid in ids if sid not in df.index]
    if missing:
        raise RuntimeError(f"{csv_path}: missing {len(missing)} ids, e.g. {missing[:5]}")
    enhancers = list(df.columns)
    if enhancers_ref is not None and enhancers != enhancers_ref:
        raise RuntimeError(f"{csv_path}: enhancer list does not match training split")
    y = df.loc[ids, enhancers].to_numpy(dtype=np.float32)
    return y, enhancers


class FeatDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(model, loader, optim, loss_fn, device: torch.device) -> float:
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
def eval_epoch(model, loader, loss_fn, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
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


def pc_target_corr_df(pc_values: np.ndarray, y: np.ndarray, enhancers: list[str]) -> pd.DataFrame:
    rows = []
    for pc_idx in range(pc_values.shape[1]):
        x = pc_values[:, pc_idx]
        x_std = float(np.std(x))
        for enh_idx, enhancer in enumerate(enhancers):
            target = y[:, enh_idx]
            y_std = float(np.std(target))
            if x_std < 1e-8 or y_std < 1e-8:
                r = np.nan
            else:
                r = float(np.corrcoef(x, target)[0, 1])
            rows.append(
                {
                    "pc": f"PC{pc_idx + 1}",
                    "pc_index": pc_idx + 1,
                    "enhancer": enhancer,
                    "pearson_r": r,
                    "abs_pearson_r": abs(r) if np.isfinite(r) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def pc_metric_summary(split: str, corr_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        corr_df.groupby(["pc", "pc_index"], as_index=False)
        .agg(
            n_enhancers=("enhancer", "count"),
            mean_r=("pearson_r", "mean"),
            mean_abs_r=("abs_pearson_r", "mean"),
            median_abs_r=("abs_pearson_r", "median"),
            max_abs_r=("abs_pearson_r", "max"),
            gt_0p2=("abs_pearson_r", lambda s: int((s > 0.2).sum())),
            gt_0p3=("abs_pearson_r", lambda s: int((s > 0.3).sum())),
            gt_0p4=("abs_pearson_r", lambda s: int((s > 0.4).sum())),
        )
        .sort_values(["mean_abs_r", "max_abs_r"], ascending=False)
    )
    summary.insert(0, "split", split)
    return summary


def save_pc_top_hits(corr_df: pd.DataFrame, out_path: Path, top_k: int = 20) -> None:
    parts = []
    for pc_idx in sorted(corr_df["pc_index"].unique()):
        sub = corr_df[corr_df["pc_index"] == pc_idx].sort_values("abs_pearson_r", ascending=False).head(top_k).copy()
        parts.append(sub)
    pd.concat(parts, axis=0, ignore_index=True).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/common361_best3_mean_mlp/cache",
    )
    parser.add_argument(
        "--train-csv",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_train_log1p_top1000.csv",
    )
    parser.add_argument(
        "--validation-csv",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_val_log1p_top1000.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_test_log1p_top1000.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/skcm_top1000_ypca5_mean_mlp_seed44",
    )
    parser.add_argument("--pca-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = {
        "train": read_ids(cache_dir / "SKCM_train_train_ids.txt"),
        "validation": read_ids(cache_dir / "SKCM_validation_validation_ids.txt"),
        "test": read_ids(cache_dir / "SKCM_test_test_ids.txt"),
    }
    x = {
        "train": np.load(cache_dir / "SKCM_train_train_X.npy").astype(np.float32),
        "validation": np.load(cache_dir / "SKCM_validation_validation_X.npy").astype(np.float32),
        "test": np.load(cache_dir / "SKCM_test_test_X.npy").astype(np.float32),
    }
    for split in ["train", "validation", "test"]:
        if x[split].shape[0] != len(ids[split]):
            raise RuntimeError(f"{split}: X/ids mismatch")

    y = {}
    enhancers_ref: list[str] | None = None
    split_to_csv = {
        "train": Path(args.train_csv),
        "validation": Path(args.validation_csv),
        "test": Path(args.test_csv),
    }
    for split in ["train", "validation", "test"]:
        y[split], enhancers = align_targets(split_to_csv[split], ids[split], enhancers_ref)
        if enhancers_ref is None:
            enhancers_ref = enhancers
    assert enhancers_ref is not None

    pca = PCA(n_components=args.pca_k, random_state=args.seed)
    y_pca = {
        "train": pca.fit_transform(y["train"]).astype(np.float32),
        "validation": pca.transform(y["validation"]).astype(np.float32),
        "test": pca.transform(y["test"]).astype(np.float32),
    }
    print(f"PCA enabled on y: k={args.pca_k}, EVR_sum={float(pca.explained_variance_ratio_.sum()):.6f}", flush=True)

    for split in ["train", "validation", "test"]:
        np.save(out_dir / f"X_{split}.npy", x[split])
        np.save(out_dir / f"y_{split}.npy", y[split])
        np.save(out_dir / f"y_pca_{split}.npy", y_pca[split])
        (out_dir / f"ids_{split}.txt").write_text("\n".join(ids[split]) + "\n")
    (out_dir / "enhancers.txt").write_text("\n".join(enhancers_ref) + "\n")
    np.save(out_dir / "pca_components.npy", pca.components_.astype(np.float32))
    np.save(out_dir / "pca_mean.npy", pca.mean_.astype(np.float32))
    np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))

    train_loader = DataLoader(FeatDataset(x["train"], y_pca["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatDataset(x["validation"], y_pca["validation"]), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FeatDataset(x["test"], y_pca["test"]), batch_size=args.batch_size, shuffle=False)

    model = SimpleRegressor(x["train"].shape[1], args.pca_k, args.hidden_dim, args.dropout).to(device)
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

    val_loss, val_pred_pca, val_true_pca = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred_pca, test_true_pca = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}", flush=True)

    val_pred = pca.inverse_transform(val_pred_pca).astype(np.float32)
    test_pred = pca.inverse_transform(test_pred_pca).astype(np.float32)
    val_true = y["validation"].astype(np.float32)
    test_true = y["test"].astype(np.float32)

    val_corr = pearson_per_feature(val_pred, val_true, enhancers_ref)
    test_corr = pearson_per_feature(test_pred, test_true, enhancers_ref)
    val_corr.to_csv(out_dir / "per_enhancer_correlation_validation.csv", index=False)
    test_corr.to_csv(out_dir / "per_enhancer_correlation_test.csv", index=False)
    pd.DataFrame([metric_row("validation", val_corr), metric_row("test", test_corr)]).to_csv(
        out_dir / "summary.csv", index=False
    )
    pd.DataFrame(train_history).to_csv(out_dir / "train_history.csv", index=False)

    pc_summaries = []
    for split in ["train", "validation", "test"]:
        pc_corr = pc_target_corr_df(y_pca[split], y[split], enhancers_ref)
        pc_corr.to_csv(out_dir / f"pc_enhancer_correlation_{split}.csv", index=False)
        pc_summary = pc_metric_summary(split, pc_corr)
        pc_summary.to_csv(out_dir / f"pc_summary_{split}.csv", index=False)
        save_pc_top_hits(pc_corr, out_dir / f"pc_top_hits_{split}.csv", top_k=20)
        pc_summaries.append(pc_summary)
    pd.concat(pc_summaries, axis=0, ignore_index=True).to_csv(out_dir / "pc_summary_all_splits.csv", index=False)

    np.save(out_dir / "val_pred_pca.npy", val_pred_pca)
    np.save(out_dir / "val_true_pca.npy", val_true_pca)
    np.save(out_dir / "test_pred_pca.npy", test_pred_pca)
    np.save(out_dir / "test_true_pca.npy", test_true_pca)
    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "test_true.npy", test_true)
    np.save(out_dir / "train_ids.npy", np.array(ids["train"], dtype=object))
    np.save(out_dir / "val_ids.npy", np.array(ids["validation"], dtype=object))
    np.save(out_dir / "test_ids.npy", np.array(ids["test"], dtype=object))
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    pd.Series(
        {
            "feature_mode": "mean_pool_y_pca",
            "pca_k": args.pca_k,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
            "device": str(device),
            "early_patience": args.early_patience,
            "min_delta": args.min_delta,
            "pca_evr_sum": float(pca.explained_variance_ratio_.sum()),
        }
    ).to_json(out_dir / "run_config.json")
    print(f"Saved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
