#!/usr/bin/env python3
"""
Single-cancer top1000 attention-pooling regression.

Pipeline:
- load patch embeddings from either:
  - <feat_root>/<split>/<sample_id>/*.npy
  - <feat_root>/<split>/<cancer>/<sample_id>/*.npy
- gated attention pooling over patches
- optionally fit y-PCA on train and regress in PC space
- export per-sample patch attention weights for validation/test
"""

import argparse
import copy
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from run_mean_regression import pearson_per_feature, set_seed


CANCER_LABEL_FORMAT = {
    "BRCA": "samples_as_columns",
    "LUAD": "samples_as_columns",
    "SKCM": "samples_as_rows",
}


def canonical_sample_id(sample_id: str) -> str:
    sample_id = sample_id[:-6] if sample_id.endswith("_tumor") else sample_id
    if sample_id.startswith("TCGA-") and len(sample_id) >= 12:
        return sample_id[:12]
    return sample_id


def load_labels_samples_as_rows(csv_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if "sample" not in df.columns:
        raise RuntimeError(f"{csv_path} missing sample column")
    ids = df["sample"].astype(str).tolist()
    enh_cols = [c for c in df.columns if c != "sample"]
    labels = df[enh_cols].to_numpy(dtype=np.float32)
    return ids, labels, enh_cols


def load_labels_samples_as_columns(csv_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if "SE_ID" not in df.columns:
        raise RuntimeError(f"{csv_path} missing SE_ID column")
    enh_cols = df["SE_ID"].astype(str).tolist()
    sample_cols = [c for c in df.columns if c not in {"chr", "start", "end", "SE_ID", "eRNA_count"}]
    labels = df.loc[:, sample_cols].to_numpy(dtype=np.float32).T
    return sample_cols, labels, enh_cols


def load_labels(csv_path: Path, cancer: str) -> tuple[list[str], np.ndarray, list[str]]:
    label_format = CANCER_LABEL_FORMAT[cancer]
    if label_format == "samples_as_rows":
        return load_labels_samples_as_rows(csv_path)
    return load_labels_samples_as_columns(csv_path)


def build_case_to_feature_dir(root: Path) -> dict[str, Path]:
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    mapping: dict[str, Path] = {}
    for d in dirs:
        mapping[canonical_sample_id(d.name)] = d
    return mapping


def resolve_split_feature_root(feat_root: Path, split: str, cancer: str) -> Path:
    direct_root = feat_root / split
    cancer_root = feat_root / split / cancer
    if cancer_root.exists():
        return cancer_root
    if direct_root.exists():
        return direct_root
    raise FileNotFoundError(f"No feature root found for split={split}, cancer={cancer} under {feat_root}")


def load_case_tokens_with_names(
    case_dir: Path,
    max_patches: Optional[int],
    rng: Optional[np.random.Generator],
) -> tuple[np.ndarray, list[str]]:
    files = sorted(case_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No npy files in {case_dir}")
    if max_patches is not None and max_patches > 0 and len(files) > max_patches:
        assert rng is not None
        idx = rng.choice(len(files), size=max_patches, replace=False)
        files = [files[i] for i in sorted(idx)]
    arrs: list[np.ndarray] = []
    kept_names: list[str] = []
    bad: list[str] = []
    for f in files:
        try:
            arrs.append(np.load(f).astype(np.float32))
            kept_names.append(f.name)
        except Exception as exc:
            bad.append(f"{f.name} ({type(exc).__name__})")
    if bad:
        print(f"[warn] {case_dir.name}: skipped bad npy files={len(bad)} (e.g., {bad[:3]})")
    if not arrs:
        raise FileNotFoundError(f"No readable npy files in {case_dir}")
    return np.stack(arrs, axis=0), kept_names


class SingleCancerAttentionDataset(Dataset):
    def __init__(
        self,
        cancer: str,
        split: str,
        csv_path: Path,
        feat_root: Path,
        max_patches: Optional[int],
        seed: int,
    ):
        ids_all, labels_all, enh_cols = load_labels(csv_path, cancer)
        case_map = build_case_to_feature_dir(resolve_split_feature_root(feat_root, split, cancer))

        kept_ids: list[str] = []
        kept_labels: list[np.ndarray] = []
        case_dirs: list[Path] = []
        missing: list[str] = []
        for sid, y in zip(ids_all, labels_all):
            case_dir = case_map.get(canonical_sample_id(sid))
            if case_dir is None:
                missing.append(sid)
                continue
            kept_ids.append(sid)
            kept_labels.append(y)
            case_dirs.append(case_dir)

        if missing:
            print(f"[{split}] missing feature dirs: {len(missing)} (e.g., {missing[:5]})")
        if not kept_ids:
            raise RuntimeError(f"No cases kept for split={split}")

        self.cancer = cancer
        self.split = split
        self.ids = kept_ids
        self.y = torch.tensor(np.stack(kept_labels, axis=0).astype(np.float32), dtype=torch.float32)
        self.case_dirs = case_dirs
        self.enh_cols = enh_cols
        self.max_patches = max_patches if max_patches and max_patches > 0 else None
        self.base_seed = seed

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        rng = None
        if self.max_patches is not None:
            rng = np.random.default_rng(self.base_seed + idx)
        tokens, patch_names = load_case_tokens_with_names(self.case_dirs[idx], self.max_patches, rng)
        x = torch.tensor(tokens, dtype=torch.float32)
        return self.ids[idx], x, self.y[idx], patch_names


def collate_batch(batch):
    ids, xs, ys, patch_names = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    k_max = max(lengths)
    feat_dim = xs[0].shape[1]
    batch_size = len(xs)
    tokens = torch.zeros((batch_size, k_max, feat_dim), dtype=torch.float32)
    valid = torch.zeros((batch_size, k_max), dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        tokens[i, :n] = x
        valid[i, :n] = True
    return list(ids), tokens, valid, torch.stack(ys), list(patch_names)


class AttentionPoolRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, embed_dim: int, attn_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(embed_dim, attn_dim)
        self.attention_u = nn.Linear(embed_dim, attn_dim)
        self.attention_w = nn.Linear(attn_dim, 1)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor):
        h_patch = self.patch_embed(x)
        a_v = torch.tanh(self.attention_v(h_patch))
        a_u = torch.sigmoid(self.attention_u(h_patch))
        scores = self.attention_w(a_v * a_u).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=1)
        bag = torch.bmm(attn.unsqueeze(1), h_patch).squeeze(1)
        pred = self.regressor(bag)
        return pred, attn


def train_epoch(model, loader, optim, loss_fn, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for _, xb, vb, yb, _ in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred, _ = model(xb, vb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device: torch.device):
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    ids_all: list[str] = []
    patch_names_all: list[list[str]] = []
    attn_all: list[np.ndarray] = []
    for ids, xb, vb, yb, patch_names in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        pred, attn = model(xb, vb)
        loss = loss_fn(pred, yb)
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
        ids_all.extend(ids)
        patch_names_all.extend(patch_names)
        valid_lengths = vb.sum(dim=1).cpu().numpy().astype(int).tolist()
        attn_np = attn.cpu().numpy()
        for i, length in enumerate(valid_lengths):
            attn_all.append(attn_np[i, :length].astype(np.float32))
    return (
        total / max(n, 1),
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        ids_all,
        patch_names_all,
        attn_all,
    )


def save_attention_tables(
    out_root: Path,
    ids: list[str],
    patch_names_all: list[list[str]],
    attn_all: list[np.ndarray],
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for sample_id, patch_names, attn in zip(ids, patch_names_all, attn_all):
        df = pd.DataFrame({"patch_file": patch_names, "attention_weight": attn})
        df = df.sort_values("attention_weight", ascending=False, kind="stable")
        safe_id = sample_id.replace("/", "_")
        df.to_csv(out_root / f"{safe_id}.csv", index=False)


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


def per_pc_prediction_corr(preds: np.ndarray, trues: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx in range(preds.shape[1]):
        x = preds[:, idx]
        y = trues[:, idx]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"pc": f"PC{idx + 1}", "pc_index": idx + 1, "pearson_r": r, "abs_pearson_r": abs(r) if np.isfinite(r) else np.nan})
    return pd.DataFrame(rows)


def predicted_pc_metric_row(split: str, corr_df: pd.DataFrame) -> dict:
    s = corr_df["pearson_r"]
    return {
        "split": split,
        "n_pcs": int(len(corr_df)),
        "pearson_mean": float(s.mean(skipna=True)),
        "pearson_median": float(s.median(skipna=True)),
        "pearson_min": float(s.min(skipna=True)),
        "pearson_max": float(s.max(skipna=True)),
    }


def infer_input_dim(feat_root: Path, cancer: str) -> int:
    for split in ["train", "validation", "test"]:
        split_root = resolve_split_feature_root(feat_root, split, cancer)
        for npy_path in split_root.glob("*/*.npy"):
            return int(np.load(npy_path).shape[-1])
    raise RuntimeError(f"Could not infer input dim from {feat_root} for cancer={cancer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", choices=sorted(CANCER_LABEL_FORMAT), default="SKCM")
    parser.add_argument("--train-csv", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_train_log1p_top1000.csv")
    parser.add_argument("--val-csv", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_val_log1p_top1000.csv")
    parser.add_argument("--test-csv", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_test_log1p_top1000.csv")
    parser.add_argument("--feat-root", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_attention_top10_npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-patches-train", type=int, default=0)
    parser.add_argument("--max-patches-eval", type=int, default=0)
    parser.add_argument("--pca-k", type=int, default=0, help="If >0, fit PCA on y_train and predict in PC space.")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_root = Path(args.feat_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SingleCancerAttentionDataset(args.cancer, "train", Path(args.train_csv), feat_root, args.max_patches_train, args.seed)
    val_ds = SingleCancerAttentionDataset(args.cancer, "validation", Path(args.val_csv), feat_root, args.max_patches_eval, args.seed)
    test_ds = SingleCancerAttentionDataset(args.cancer, "test", Path(args.test_csv), feat_root, args.max_patches_eval, args.seed)

    enh_cols = train_ds.enh_cols
    if val_ds.enh_cols != enh_cols or test_ds.enh_cols != enh_cols:
        raise RuntimeError("Enhancer columns do not match across splits")

    y = {
        "train": train_ds.y.cpu().numpy().astype(np.float32),
        "validation": val_ds.y.cpu().numpy().astype(np.float32),
        "test": test_ds.y.cpu().numpy().astype(np.float32),
    }
    pca = None
    if args.pca_k > 0:
        if args.pca_k >= len(enh_cols):
            raise RuntimeError(f"--pca-k must be smaller than output dim ({len(enh_cols)}), got {args.pca_k}")
        pca = PCA(n_components=args.pca_k, random_state=args.seed)
        y_fit = {
            "train": pca.fit_transform(y["train"]).astype(np.float32),
            "validation": pca.transform(y["validation"]).astype(np.float32),
            "test": pca.transform(y["test"]).astype(np.float32),
        }
        print(
            f"{args.cancer}: y-PCA enabled: k={args.pca_k}, "
            f"EVR_sum={float(pca.explained_variance_ratio_.sum()):.6f}",
            flush=True,
        )
    else:
        y_fit = {split: y[split].astype(np.float32) for split in ["train", "validation", "test"]}
        print(f"{args.cancer}: y-PCA disabled; predicting {len(enh_cols)} targets directly", flush=True)

    train_ds.y = torch.tensor(y_fit["train"], dtype=torch.float32)
    val_ds.y = torch.tensor(y_fit["validation"], dtype=torch.float32)
    test_ds.y = torch.tensor(y_fit["test"], dtype=torch.float32)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_batch,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    input_dim = infer_input_dim(feat_root, args.cancer)
    model = AttentionPoolRegressor(
        input_dim=input_dim,
        output_dim=int(y_fit["train"].shape[1]),
        embed_dim=args.embed_dim,
        attn_dim=args.attn_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, loss_fn, device)
        val_loss, _, _, _, _, _ = eval_epoch(model, val_loader, loss_fn, device)
        if val_loss < (best_val - args.min_delta):
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if bad_epochs >= args.early_patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val improvement for {bad_epochs} epochs "
                f"(patience={args.early_patience}, min_delta={args.min_delta})."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_pred_raw, val_true_raw, val_ids, val_patch_names, val_attn = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred_raw, test_true_raw, test_ids, test_patch_names, test_attn = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    if pca is not None:
        val_pred = pca.inverse_transform(val_pred_raw).astype(np.float32)
        test_pred = pca.inverse_transform(test_pred_raw).astype(np.float32)
    else:
        val_pred = val_pred_raw.astype(np.float32)
        test_pred = test_pred_raw.astype(np.float32)
    val_true = y["validation"].astype(np.float32)
    test_true = y["test"].astype(np.float32)

    val_corr = pearson_per_feature(val_pred, val_true, enh_cols)
    test_corr = pearson_per_feature(test_pred, test_true, enh_cols)
    summary_df = pd.DataFrame([metric_row("validation", val_corr), metric_row("test", test_corr)])

    np.save(out_dir / "val_pred.npy", val_pred.astype(np.float32))
    np.save(out_dir / "val_true.npy", val_true.astype(np.float32))
    np.save(out_dir / "test_pred.npy", test_pred.astype(np.float32))
    np.save(out_dir / "test_true.npy", test_true.astype(np.float32))
    np.save(out_dir / "val_pred_pca.npy", val_pred_raw.astype(np.float32))
    np.save(out_dir / "val_true_pca.npy", val_true_raw.astype(np.float32))
    np.save(out_dir / "test_pred_pca.npy", test_pred_raw.astype(np.float32))
    np.save(out_dir / "test_true_pca.npy", test_true_raw.astype(np.float32))
    np.save(out_dir / "val_ids.npy", np.array(val_ids, dtype=object))
    np.save(out_dir / "test_ids.npy", np.array(test_ids, dtype=object))
    (out_dir / "ids_train.txt").write_text("\n".join(train_ds.ids) + "\n")
    (out_dir / "ids_validation.txt").write_text("\n".join(val_ids) + "\n")
    (out_dir / "ids_test.txt").write_text("\n".join(test_ids) + "\n")
    (out_dir / "enhancers.txt").write_text("\n".join(enh_cols) + "\n")
    val_corr.to_csv(out_dir / "per_enhancer_correlation_validation.csv", index=False)
    test_corr.to_csv(out_dir / "per_enhancer_correlation_test.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    torch.save(model.state_dict(), out_dir / "best_model.pt")

    np.save(out_dir / "y_train.npy", y["train"])
    np.save(out_dir / "y_validation.npy", y["validation"])
    np.save(out_dir / "y_test.npy", y["test"])
    np.save(out_dir / "y_pca_train.npy", y_fit["train"])
    np.save(out_dir / "y_pca_validation.npy", y_fit["validation"])
    np.save(out_dir / "y_pca_test.npy", y_fit["test"])
    if pca is not None:
        np.save(out_dir / "pca_components.npy", pca.components_.astype(np.float32))
        np.save(out_dir / "pca_mean.npy", pca.mean_.astype(np.float32))
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))
        val_pc_corr = per_pc_prediction_corr(val_pred_raw, val_true_raw)
        test_pc_corr = per_pc_prediction_corr(test_pred_raw, test_true_raw)
        val_pc_corr.to_csv(out_dir / "predicted_pc_correlation_validation.csv", index=False)
        test_pc_corr.to_csv(out_dir / "predicted_pc_correlation_test.csv", index=False)
        pd.DataFrame(
            [
                predicted_pc_metric_row("validation", val_pc_corr),
                predicted_pc_metric_row("test", test_pc_corr),
            ]
        ).to_csv(out_dir / "predicted_pc_summary.csv", index=False)

    np.save(out_dir / "val_attention_weights.npy", np.array(val_attn, dtype=object), allow_pickle=True)
    np.save(out_dir / "test_attention_weights.npy", np.array(test_attn, dtype=object), allow_pickle=True)
    np.save(out_dir / "val_patch_names.npy", np.array(val_patch_names, dtype=object), allow_pickle=True)
    np.save(out_dir / "test_patch_names.npy", np.array(test_patch_names, dtype=object), allow_pickle=True)
    save_attention_tables(out_dir / "val_patch_attention", val_ids, val_patch_names, val_attn)
    save_attention_tables(out_dir / "test_patch_attention", test_ids, test_patch_names, test_attn)

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
