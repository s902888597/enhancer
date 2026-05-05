#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


CANCER_CACHE = {
    "BRCA": {
        "train": ("BRCA_train_Top1000_SEs_192train_eRNA_zscore_3eRNAremoved_X.npy", "BRCA_train_Top1000_SEs_192train_eRNA_zscore_3eRNAremoved_ids.txt"),
        "validation": ("BRCA_validation_Top1000_SEs_63val_eRNA_zscore_3eRNAremoved_X.npy", "BRCA_validation_Top1000_SEs_63val_eRNA_zscore_3eRNAremoved_ids.txt"),
        "test": ("BRCA_test_Top1000_SEs_64test_eRNA_zscore_3eRNAremoved_X.npy", "BRCA_test_Top1000_SEs_64test_eRNA_zscore_3eRNAremoved_ids.txt"),
    },
    "LUAD": {
        "train": ("LUAD_train_Top1000_SEs_189train_eRNA_zscore_3eRNAremoved_X.npy", "LUAD_train_Top1000_SEs_189train_eRNA_zscore_3eRNAremoved_ids.txt"),
        "validation": ("LUAD_validation_Top1000_SEs_63val_eRNA_zscore_3eRNAremoved_X.npy", "LUAD_validation_Top1000_SEs_63val_eRNA_zscore_3eRNAremoved_ids.txt"),
        "test": ("LUAD_test_Top1000_SEs_64test_eRNA_zscore_3eRNAremoved_X.npy", "LUAD_test_Top1000_SEs_64test_eRNA_zscore_3eRNAremoved_ids.txt"),
    },
    "SKCM": {
        "train": ("SKCM_train_train_X.npy", "SKCM_train_train_ids.txt"),
        "validation": ("SKCM_validation_validation_X.npy", "SKCM_validation_validation_ids.txt"),
        "test": ("SKCM_test_test_X.npy", "SKCM_test_test_ids.txt"),
    },
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


class FeatDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float, positive_output: bool = False):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, output_dim)
        self.positive_output = positive_output
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        if self.positive_output:
            x = self.softplus(x)
        return x


def load_label_csv(path: Path, ids: list[str]) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if "sample" not in df.columns:
        raise RuntimeError(f"{path} missing sample column")
    df["sample"] = df["sample"].astype(str)
    enhancers = [c for c in df.columns if c != "sample"]
    if df["sample"].duplicated().any():
        raise RuntimeError(f"{path} has duplicate sample ids")
    df = df.set_index("sample")
    missing = [sid for sid in ids if sid not in df.index]
    if missing:
        raise RuntimeError(f"{path.name}: missing {len(missing)} ids, first few {missing[:5]}")
    return df.loc[ids, enhancers].to_numpy(dtype=np.float32), enhancers


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


def pearson_per_feature(preds: np.ndarray, trues: np.ndarray, names: list[str], name_col: str) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(names):
        x = preds[:, idx]
        y = trues[:, idx]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({name_col: name, "pearson_r": r})
    return pd.DataFrame(rows)


def metric_row(split: str, corr_df: pd.DataFrame) -> dict[str, object]:
    s = corr_df["pearson_r"]
    return {
        "split": split,
        "n_outputs": int(len(corr_df)),
        "pearson_mean": float(s.mean(skipna=True)),
        "pearson_median": float(s.median(skipna=True)),
        "gt_0.4": int((s > 0.4).sum()),
        "gt_0.5": int((s > 0.5).sum()),
        "gt_0.6": int((s > 0.6).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", required=True, choices=sorted(CANCER_CACHE))
    parser.add_argument(
        "--cache-dir",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/common361_best3_mean_mlp/cache",
    )
    parser.add_argument(
        "--label-dir",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_nonzero_frac_ge_0p20_log2p1",
    )
    parser.add_argument(
        "--label-tag",
        default="nonzero_frac_ge_0p20",
        help="Suffix used in label filenames, e.g. nonzero_frac_ge_0p20",
    )
    parser.add_argument(
        "--label-kind",
        default="log2p1",
        help="Middle token used in label filenames, e.g. log2p1 or raw",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reducer", choices=["none", "nmf"], default="none")
    parser.add_argument("--nmf-k", type=int, default=20)
    parser.add_argument("--nmf-max-iter", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--y-zscore-per-enhancer",
        action="store_true",
        help="For direct regression only, z-score each enhancer using train mean/std and inverse-transform predictions before evaluation.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cache_dir = Path(args.cache_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids: dict[str, list[str]] = {}
    x: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    enhancers_ref: list[str] | None = None
    for split in ("train", "validation", "test"):
        x_name, id_name = CANCER_CACHE[args.cancer][split]
        ids_split = read_ids(cache_dir / id_name)
        x_split = np.load(cache_dir / x_name).astype(np.float32)
        if x_split.shape[0] != len(ids_split):
            raise RuntimeError(f"{args.cancer} {split}: X/ids mismatch")
        y_split, enhancers = load_label_csv(label_dir / f"{args.cancer}_{split}_{args.label_kind}_{args.label_tag}.csv", ids_split)
        if enhancers_ref is None:
            enhancers_ref = enhancers
        elif enhancers != enhancers_ref:
            raise RuntimeError(f"{args.cancer} {split}: enhancer list mismatch")
        ids[split] = ids_split
        x[split] = x_split
        y[split] = y_split
    assert enhancers_ref is not None

    reducer = None
    if args.reducer == "nmf":
        if args.y_zscore_per_enhancer:
            raise RuntimeError("--y-zscore-per-enhancer is only supported with reducer=none")
        reducer = NMF(
            n_components=args.nmf_k,
            init="nndsvda",
            random_state=args.seed,
            max_iter=args.nmf_max_iter,
        )
        y_reduced = {
            "train": reducer.fit_transform(y["train"]).astype(np.float32),
            "validation": reducer.transform(y["validation"]).astype(np.float32),
            "test": reducer.transform(y["test"]).astype(np.float32),
        }
        latent_names = [f"NMF{i+1}" for i in range(args.nmf_k)]
        np.save(out_dir / "nmf_components.npy", reducer.components_.astype(np.float32))
        with (out_dir / "nmf_meta.json").open("w") as f:
            json.dump(
                {
                    "n_components": args.nmf_k,
                    "reconstruction_err_": float(reducer.reconstruction_err_),
                    "n_iter_": int(reducer.n_iter_),
                },
                f,
                indent=2,
            )
        print(
            f"{args.cancer}: NMF enabled on y: k={args.nmf_k}, "
            f"reconstruction_err={float(reducer.reconstruction_err_):.6f}, n_iter={int(reducer.n_iter_)}",
            flush=True,
        )
    else:
        if args.y_zscore_per_enhancer:
            y_mean = y["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
            y_std = y["train"].std(axis=0, dtype=np.float64).astype(np.float32)
            y_std_safe = y_std.copy()
            y_std_safe[y_std_safe < 1e-8] = 1.0
            y_reduced = {
                split: ((y[split] - y_mean) / y_std_safe).astype(np.float32)
                for split in ("train", "validation", "test")
            }
            np.save(out_dir / "y_train_mean.npy", y_mean)
            np.save(out_dir / "y_train_std.npy", y_std_safe)
            print(
                f"{args.cancer}: direct output on {y['train'].shape[1]} enhancers "
                f"with train-only per-enhancer y z-score",
                flush=True,
            )
        else:
            y_reduced = {split: y[split].astype(np.float32) for split in ("train", "validation", "test")}
            print(f"{args.cancer}: direct output on {y['train'].shape[1]} enhancers", flush=True)
        latent_names = enhancers_ref

    for split in ("train", "validation", "test"):
        (out_dir / f"ids_{split}.txt").write_text("\n".join(ids[split]) + "\n")
    (out_dir / "enhancers.txt").write_text("\n".join(enhancers_ref) + "\n")
    if args.reducer == "nmf":
        (out_dir / "latent_components.txt").write_text("\n".join(latent_names) + "\n")

    train_loader = DataLoader(FeatDataset(x["train"], y_reduced["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatDataset(x["validation"], y_reduced["validation"]), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FeatDataset(x["test"], y_reduced["test"]), batch_size=args.batch_size, shuffle=False)

    model = SimpleRegressor(
        x["train"].shape[1],
        y_reduced["train"].shape[1],
        args.hidden_dim,
        args.dropout,
        positive_output=(args.reducer == "nmf"),
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

    val_loss, val_pred_reduced, val_true_reduced = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred_reduced, test_true_reduced = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}", flush=True)

    if reducer is not None:
        val_pred = reducer.inverse_transform(np.maximum(val_pred_reduced, 0.0)).astype(np.float32)
        test_pred = reducer.inverse_transform(np.maximum(test_pred_reduced, 0.0)).astype(np.float32)
    else:
        if args.y_zscore_per_enhancer:
            y_mean = np.load(out_dir / "y_train_mean.npy").astype(np.float32)
            y_std = np.load(out_dir / "y_train_std.npy").astype(np.float32)
            val_pred = (val_pred_reduced * y_std + y_mean).astype(np.float32)
            test_pred = (test_pred_reduced * y_std + y_mean).astype(np.float32)
        else:
            val_pred = val_pred_reduced.astype(np.float32)
            test_pred = test_pred_reduced.astype(np.float32)

    val_true = y["validation"].astype(np.float32)
    test_true = y["test"].astype(np.float32)

    val_corr = pearson_per_feature(val_pred, val_true, enhancers_ref, "enhancer")
    test_corr = pearson_per_feature(test_pred, test_true, enhancers_ref, "enhancer")
    val_corr.to_csv(out_dir / "per_enhancer_correlation_validation.csv", index=False)
    test_corr.to_csv(out_dir / "per_enhancer_correlation_test.csv", index=False)
    pd.DataFrame([metric_row("validation", val_corr), metric_row("test", test_corr)]).to_csv(
        out_dir / "summary.csv", index=False
    )

    if reducer is not None:
        val_latent_corr = pearson_per_feature(val_pred_reduced, val_true_reduced, latent_names, "component")
        test_latent_corr = pearson_per_feature(test_pred_reduced, test_true_reduced, latent_names, "component")
        val_latent_corr.to_csv(out_dir / "per_component_correlation_validation.csv", index=False)
        test_latent_corr.to_csv(out_dir / "per_component_correlation_test.csv", index=False)
        pd.DataFrame([metric_row("validation", val_latent_corr), metric_row("test", test_latent_corr)]).to_csv(
            out_dir / "latent_summary.csv", index=False
        )

    pd.DataFrame(train_history).to_csv(out_dir / "train_history.csv", index=False)
    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "test_true.npy", test_true)
    np.save(out_dir / "val_pred_reduced.npy", val_pred_reduced)
    np.save(out_dir / "val_true_reduced.npy", val_true_reduced)
    np.save(out_dir / "test_pred_reduced.npy", test_pred_reduced)
    np.save(out_dir / "test_true_reduced.npy", test_true_reduced)
    torch.save(model.state_dict(), out_dir / "best_model.pt")

    pd.Series(
        {
            "cancer": args.cancer,
            "reducer": args.reducer,
            "nmf_k": args.nmf_k if args.reducer == "nmf" else 0,
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
            "label_tag": args.label_tag,
            "label_kind": args.label_kind,
            "y_zscore_per_enhancer": bool(args.y_zscore_per_enhancer),
            "n_train": int(x["train"].shape[0]),
            "n_validation": int(x["validation"].shape[0]),
            "n_test": int(x["test"].shape[0]),
            "n_outputs": int(y["train"].shape[1]),
            "n_reduced_outputs": int(y_reduced["train"].shape[1]),
        }
    ).to_json(out_dir / "run_config.json")
    print(f"Saved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
