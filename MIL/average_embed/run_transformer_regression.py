#!/usr/bin/env python3
"""
Train WSITransformerRegressor on patch embeddings (no mean pooling).

Inputs:
- Label CSVs in either samples-as-rows or samples-as-columns format
- Patch features under either:
  - <feat_root>/<split>/<case>/*.npy
  - <feat_root>/<split>/<cancer>/<case>/*.npy

Outputs (under --out-dir):
- best_model.pt
- train/val/test ids
- test_pred.npy, test_true.npy
- per_enhancer_correlation.csv (Pearson r on test set)
"""

import argparse
import copy
import json
from pathlib import Path
import re
from typing import List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is on path so `our_code_data.*` imports resolve.
import sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from our_code_data.enhancer_prediction.models import (
    WSITransformerRegressor,
    WSINeighborTransformerRegressor,
)


CANCER_LABEL_FORMAT = {
    "BRCA": "samples_as_columns",
    "LUAD": "samples_as_columns",
    "SKCM": "samples_as_rows",
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_labels_samples_as_rows(csv_path: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if "sample" not in df.columns:
        raise RuntimeError(f"{csv_path} missing sample column")
    ids = df["sample"].astype(str).tolist()
    enh_cols = [c for c in df.columns if c != "sample"]
    labels = df[enh_cols].values.astype(np.float32)
    return ids, labels, enh_cols


def load_labels_samples_as_columns(csv_path: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if "SE_ID" not in df.columns:
        raise RuntimeError(f"{csv_path} missing SE_ID column")
    enh_cols = df["SE_ID"].astype(str).tolist()
    sample_cols = [c for c in df.columns if c not in {"chr", "start", "end", "SE_ID", "eRNA_count"}]
    labels = df.loc[:, sample_cols].to_numpy(dtype=np.float32).T
    return sample_cols, labels, enh_cols


def load_labels(csv_path: Path, cancer: str) -> Tuple[List[str], np.ndarray, List[str]]:
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if "sample" in cols:
        return load_labels_samples_as_rows(csv_path)
    if "SE_ID" in cols:
        return load_labels_samples_as_columns(csv_path)
    if CANCER_LABEL_FORMAT[cancer] == "samples_as_rows":
        return load_labels_samples_as_rows(csv_path)
    return load_labels_samples_as_columns(csv_path)


def canonical_sample_id(sample_id: str) -> str:
    sample_id = sample_id[:-6] if sample_id.endswith("_tumor") else sample_id
    if sample_id.startswith("TCGA-") and len(sample_id) >= 12:
        return sample_id[:12]
    return sample_id


def sample_tss(sample_id: str) -> str:
    sid = canonical_sample_id(sample_id)
    parts = sid.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "UNK"


def build_domain_mapping(train_ids: List[str], min_count: int) -> tuple[dict[str, int], dict[str, int]]:
    counts = Counter(sample_tss(s) for s in train_ids)
    kept = sorted([k for k, v in counts.items() if v >= min_count])
    mapping = {k: i for i, k in enumerate(kept)}
    if len(kept) < len(counts):
        mapping["OTHER"] = len(mapping)
    return mapping, dict(counts)


def map_domain_ids(ids: List[str], mapping: dict[str, int]) -> np.ndarray:
    has_other = "OTHER" in mapping
    out = []
    for sid in ids:
        key = sample_tss(sid)
        if key in mapping:
            out.append(mapping[key])
        elif has_other:
            out.append(mapping["OTHER"])
        else:
            out.append(0)
    return np.asarray(out, dtype=np.int64)


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


COORD_RE = re.compile(r"_x(-?\d+)_y(-?\d+)$")


def parse_patch_xy(path: Path) -> tuple[float, float]:
    m = COORD_RE.search(path.stem)
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))


def load_case_tokens(
    case_dir: Path,
    max_patches: int = None,
    rng: np.random.Generator = None,
    use_coords: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    files = sorted(case_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No npy files in {case_dir}")
    file_coords = [parse_patch_xy(f) for f in files] if use_coords else None
    if max_patches is not None and len(files) > max_patches:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(files), size=max_patches, replace=False)
        files = [files[i] for i in sorted(idx)]
        if file_coords is not None:
            file_coords = [file_coords[i] for i in sorted(idx)]
    arrs = []
    coords = []
    bad = []
    for i, f in enumerate(files):
        try:
            arrs.append(np.load(f).astype(np.float32))
            if file_coords is not None:
                coords.append(file_coords[i])
        except Exception as exc:
            bad.append(f"{f.name} ({type(exc).__name__})")
    if bad:
        print(f"[warn] {case_dir.name}: skipped bad npy files={len(bad)} (e.g., {bad[:3]})")
    if not arrs:
        raise FileNotFoundError(f"No readable npy files in {case_dir}")
    coord_arr = None
    if coords:
        coord_arr = np.asarray(coords, dtype=np.float32)
        coord_arr[:, 0] /= max(float(coord_arr[:, 0].max()), 1.0)
        coord_arr[:, 1] /= max(float(coord_arr[:, 1].max()), 1.0)
    return np.stack(arrs, axis=0), coord_arr


def load_split_tokens(
    cancer: str,
    split: str,
    csv_path: Path,
    feat_root: Path,
    max_patches: int = None,
    rng: np.random.Generator = None,
    use_coords: bool = False,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray | None], np.ndarray, List[str]]:
    ids, labels, enh_cols = load_labels(csv_path, cancer)
    case_map = build_case_to_feature_dir(resolve_split_feature_root(feat_root, split, cancer))
    xs = []
    cs = []
    kept_ids = []
    kept_labels = []
    missing = []
    for sid, lab in zip(ids, labels):
        case_dir = case_map.get(canonical_sample_id(sid))
        if case_dir is None:
            missing.append(sid)
            continue
        try:
            toks, coords = load_case_tokens(case_dir, max_patches=max_patches, rng=rng, use_coords=use_coords)
        except FileNotFoundError:
            missing.append(sid)
            continue
        kept_ids.append(sid)
        kept_labels.append(lab)
        xs.append(toks)
        cs.append(coords)
    if missing:
        print(f"[{split}] missing/empty: {len(missing)} (e.g., {missing[:5]})")
    if not xs:
        raise RuntimeError(f"No tokens loaded for split {split}")
    labels_arr = np.stack(kept_labels, axis=0).astype(np.float32)
    return kept_ids, xs, cs, labels_arr, enh_cols


class TokenDataset(Dataset):
    def __init__(self, xs: List[np.ndarray], coords: List[np.ndarray | None], ys: np.ndarray, domains: np.ndarray | None = None):
        self.xs = xs
        self.coords = coords
        self.ys = torch.tensor(ys, dtype=torch.float32)
        self.domains = None if domains is None else torch.tensor(domains, dtype=torch.long)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = torch.tensor(self.xs[idx], dtype=torch.float32)  # (K,D)
        coord_arr = self.coords[idx]
        if coord_arr is None:
            coords = torch.zeros((x.shape[0], 2), dtype=torch.float32)
        else:
            coords = torch.tensor(coord_arr, dtype=torch.float32)
        y = self.ys[idx]
        domain = torch.tensor(-1, dtype=torch.long) if self.domains is None else self.domains[idx]
        return x, coords, y, domain


def collate_batch(batch):
    """
    Pad variable-length token sequences to max length in batch.
    Returns:
      tokens: (B, K_max, D)
      coords: (B, K_max, 2)
      valid_mask: (B, K_max) bool
      labels: (B, output_dim)
    """
    xs, coords_list, ys, domains = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    k_max = max(lengths)
    d = xs[0].shape[1]
    B = len(xs)
    tokens = torch.zeros((B, k_max, d), dtype=torch.float32)
    coords = torch.zeros((B, k_max, 2), dtype=torch.float32)
    mask = torch.zeros((B, k_max), dtype=torch.bool)
    for i, (x, xy) in enumerate(zip(xs, coords_list)):
        k = x.shape[0]
        tokens[i, :k] = x
        coords[i, :k] = xy
        mask[i, :k] = True
    labels = torch.stack(ys, dim=0)
    domain_labels = torch.stack(domains, dim=0)
    return tokens, coords, mask, labels, domain_labels


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, n_domains: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mean_sample_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson correlation per sample across output dimensions, then mean.
    This keeps the term meaningful even when batch_size=1.
    """
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    targ_c = target - target.mean(dim=1, keepdim=True)
    cov = (pred_c * targ_c).mean(dim=1)
    pred_std = torch.sqrt(pred_c.pow(2).mean(dim=1).clamp_min(eps))
    targ_std = torch.sqrt(targ_c.pow(2).mean(dim=1).clamp_min(eps))
    corr = cov / (pred_std * targ_std + eps)
    return corr.mean()


def composite_loss(pred: torch.Tensor, target: torch.Tensor, mse_fn, lambda_corr: float) -> torch.Tensor:
    mse = mse_fn(pred, target)
    if lambda_corr <= 0:
        return mse
    corr = mean_sample_corr(pred, target)
    return mse + lambda_corr * (1.0 - corr)


def train_epoch(
    model,
    loader,
    optim,
    loss_fn,
    device,
    lambda_corr: float = 0.0,
    domain_head: nn.Module | None = None,
    domain_loss_fn: nn.Module | None = None,
    domain_lambda: float = 0.0,
):
    model.train()
    if domain_head is not None:
        domain_head.train()
    total = 0.0
    n = 0
    for xb, coords, mask, yb, db in loader:
        xb = xb.to(device)
        coords = coords.to(device)
        mask = mask.to(device)
        yb = yb.to(device)
        db = db.to(device)
        optim.zero_grad(set_to_none=True)
        if domain_head is not None and domain_lambda > 0:
            rep = model.encode(xb, valid_mask=mask, coords=coords)
            pred = model.head(rep)
        else:
            pred = model(xb, valid_mask=mask, coords=coords)
        loss = composite_loss(pred, yb, loss_fn, lambda_corr)
        if domain_head is not None and domain_lambda > 0:
            rev_rep = GradientReversal.apply(rep, 1.0)
            domain_logits = domain_head(rev_rep)
            loss = loss + domain_lambda * domain_loss_fn(domain_logits, db)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def eval_epoch(model, loader, loss_fn, device, lambda_corr: float = 0.0):
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    with torch.no_grad():
        for xb, coords, mask, yb, _ in loader:
            xb = xb.to(device)
            coords = coords.to(device)
            mask = mask.to(device)
            yb = yb.to(device)
            pred = model(xb, valid_mask=mask, coords=coords)
            loss = composite_loss(pred, yb, loss_fn, lambda_corr)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
            preds.append(pred.cpu())
            trues.append(yb.cpu())
    return total / max(n, 1), torch.cat(preds), torch.cat(trues)


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


def per_pc_prediction_corr(preds: np.ndarray, trues: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx in range(preds.shape[1]):
        x = preds[:, idx]
        y = trues[:, idx]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append(
            {
                "pc": f"PC{idx + 1}",
                "pc_index": idx + 1,
                "pearson_r": r,
                "abs_pearson_r": abs(r) if np.isfinite(r) else np.nan,
            }
        )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", choices=sorted(CANCER_LABEL_FORMAT), default="SKCM")
    parser.add_argument("--train-csv", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_train_log1p_top1000.csv")
    parser.add_argument("--val-csv", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_val_log1p_top1000.csv")
    parser.add_argument("--test-csv", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/super_enhancer/input/SKCM_test_log1p_top1000.csv")
    parser.add_argument("--feat-root", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_attention_top10_npy")
    parser.add_argument("--out-dir", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1, help="Batches of cases (handles padding internally)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=100, help="Randomly sample up to N patches per case (None = use all)")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--early-patience", type=int, default=5, help="Stop if val loss does not improve for N epochs")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement to reset patience")
    parser.add_argument(
        "--lambda-corr",
        type=float,
        default=0.0,
        help="Use composite loss: MSE + lambda_corr * (1 - sample_corr).",
    )
    parser.add_argument(
        "--test-avg-runs",
        type=int,
        default=1,
        help="Number of test-time resampling runs; predictions are averaged across runs.",
    )
    parser.add_argument("--pca-k", type=int, default=0, help="If >0, fit PCA on y_train and predict in PC space.")
    parser.add_argument("--use-coords", action="store_true", help="Add normalized 2D patch coordinates to token embeddings.")
    parser.add_argument("--neighbor-k", type=int, default=0, help="If >0, use a simple neighbor-constrained attention block with kNN graph.")
    parser.add_argument("--domain-adapt", action="store_true", help="Use a gradient-reversal domain classifier on TSS/site code.")
    parser.add_argument("--domain-lambda", type=float, default=0.1, help="Weight for domain-adversarial loss.")
    parser.add_argument("--domain-min-count", type=int, default=5, help="Minimum train count to keep a TSS as its own domain; rarer sites map to OTHER.")
    args = parser.parse_args()

    print("Args:", vars(args))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_root = Path(args.feat_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    rng = np.random.default_rng(args.seed)
    need_coords = args.use_coords or args.neighbor_k > 0
    train_ids, x_train_list, c_train_list, y_train, enh_cols = load_split_tokens(
        args.cancer,
        "train",
        Path(args.train_csv),
        feat_root,
        max_patches=args.max_patches,
        rng=rng,
        use_coords=need_coords,
    )
    # For val/test use deterministic sampling with a fresh RNG seeded the same way for reproducibility
    val_ids, x_val_list, c_val_list, y_val, _ = load_split_tokens(
        args.cancer,
        "validation",
        Path(args.val_csv),
        feat_root,
        max_patches=args.max_patches,
        rng=np.random.default_rng(args.seed),
        use_coords=need_coords,
    )
    test_ids, x_test_list, c_test_list, y_test, _ = load_split_tokens(
        args.cancer,
        "test",
        Path(args.test_csv),
        feat_root,
        max_patches=args.max_patches,
        rng=np.random.default_rng(args.seed),
        use_coords=need_coords,
    )

    input_dim = x_train_list[0].shape[1]
    output_dim = y_train.shape[1]
    domain_mapping = None
    domain_counts = None
    train_domains = val_domains = test_domains = None
    if args.domain_adapt:
        domain_mapping, domain_counts = build_domain_mapping(train_ids, min_count=args.domain_min_count)
        train_domains = map_domain_ids(train_ids, domain_mapping)
        val_domains = map_domain_ids(val_ids, domain_mapping)
        test_domains = map_domain_ids(test_ids, domain_mapping)
        with (out_dir / "domain_meta.json").open("w") as f:
            json.dump(
                {
                    "domain_type": "tss_code",
                    "min_count": args.domain_min_count,
                    "mapping": domain_mapping,
                    "train_counts": domain_counts,
                },
                f,
                indent=2,
            )
        print(f"domain-adapt enabled: n_domains={len(domain_mapping)}, mapping={domain_mapping}")
    pca = None
    if args.pca_k > 0:
        if args.pca_k >= output_dim:
            raise RuntimeError(f"--pca-k must be smaller than output dim ({output_dim}), got {args.pca_k}")
        pca = PCA(n_components=args.pca_k, random_state=args.seed)
        y_train_fit = pca.fit_transform(y_train).astype(np.float32)
        y_val_fit = pca.transform(y_val).astype(np.float32)
        y_test_fit = pca.transform(y_test).astype(np.float32)
        output_dim = args.pca_k
        print(f"y-PCA enabled: k={args.pca_k}, EVR_sum={float(pca.explained_variance_ratio_.sum()):.6f}")
    else:
        y_train_fit = y_train
        y_val_fit = y_val
        y_test_fit = y_test

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_batch,
    }
    train_loader = DataLoader(TokenDataset(x_train_list, c_train_list, y_train_fit, train_domains), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(TokenDataset(x_val_list, c_val_list, y_val_fit, val_domains), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(TokenDataset(x_test_list, c_test_list, y_test_fit, test_domains), shuffle=False, **loader_kwargs)

    if args.neighbor_k > 0:
        model = WSINeighborTransformerRegressor(
            token_dim=input_dim,
            output_dim=output_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            neighbor_k=args.neighbor_k,
        ).to(device)
    else:
        model = WSITransformerRegressor(
            token_dim=input_dim,
            output_dim=output_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            pooling="cls",
            use_coords=args.use_coords,
        ).to(device)
    domain_head = None
    domain_loss_fn = None
    if args.domain_adapt:
        domain_head = DomainClassifier(input_dim=args.d_model, n_domains=len(domain_mapping)).to(device)
        domain_loss_fn = nn.CrossEntropyLoss()

    loss_fn = nn.MSELoss()
    params = list(model.parameters()) + ([] if domain_head is None else list(domain_head.parameters()))
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_domain_state = None
    best_val = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optim,
            loss_fn,
            device,
            lambda_corr=args.lambda_corr,
            domain_head=domain_head,
            domain_loss_fn=domain_loss_fn,
            domain_lambda=args.domain_lambda if args.domain_adapt else 0.0,
        )
        val_loss, _, _ = eval_epoch(model, val_loader, loss_fn, device, lambda_corr=args.lambda_corr)
        if val_loss < (best_val - args.min_delta):
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_domain_state = None if domain_head is None else copy.deepcopy(domain_head.state_dict())
            patience = 0
        else:
            patience += 1
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if patience >= args.early_patience:
            print(f"Early stopping at epoch {epoch} (val loss did not improve for {args.early_patience} epochs)")
            break

    if best_state:
        model.load_state_dict(best_state)
    if domain_head is not None and best_domain_state is not None:
        domain_head.load_state_dict(best_domain_state)

    val_loss, val_pred_t, val_true_t = eval_epoch(model, val_loader, loss_fn, device, lambda_corr=args.lambda_corr)

    # Optional test-time averaging: resample test patches multiple times and average predictions.
    if args.test_avg_runs <= 1:
        test_loss, test_pred_t, test_true_t = eval_epoch(model, test_loader, loss_fn, device, lambda_corr=args.lambda_corr)
    else:
        test_pred_runs = []
        test_true_ref = None
        test_ids_ref = None
        for i in range(args.test_avg_runs):
            ids_i, x_test_i, c_test_i, y_test_i, _ = load_split_tokens(
                args.cancer,
                "test",
                Path(args.test_csv),
                feat_root,
                max_patches=args.max_patches,
                rng=np.random.default_rng(args.seed + 10000 + i),
                use_coords=need_coords,
            )
            if test_ids_ref is None:
                test_ids_ref = ids_i
                test_true_ref = pca.transform(y_test_i).astype(np.float32) if pca is not None else y_test_i.astype(np.float32)
            elif ids_i != test_ids_ref:
                raise RuntimeError("Test sample order changed across test-time resampling runs.")

            test_loader_i = DataLoader(
                TokenDataset(
                    x_test_i,
                    c_test_i,
                    pca.transform(y_test_i).astype(np.float32) if pca is not None else y_test_i.astype(np.float32),
                    test_domains,
                ),
                shuffle=False,
                **loader_kwargs,
            )
            _, pred_i_t, true_i_t = eval_epoch(model, test_loader_i, loss_fn, device, lambda_corr=args.lambda_corr)
            test_pred_runs.append(pred_i_t.numpy())

            # Consistency check for labels.
            if not np.allclose(true_i_t.numpy(), test_true_ref, atol=1e-6):
                raise RuntimeError("Test labels changed across test-time resampling runs.")

        test_pred_np = np.mean(np.stack(test_pred_runs, axis=0), axis=0)
        test_true_np = test_true_ref.astype(np.float32)
        test_loss = float(np.mean((test_pred_np - test_true_np) ** 2))
        test_pred_t = torch.from_numpy(test_pred_np)
        test_true_t = torch.from_numpy(test_true_np)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    val_pred_raw = val_pred_t.numpy()
    val_true_raw = val_true_t.numpy()
    test_pred_raw = test_pred_t.numpy()
    test_true_raw = test_true_t.numpy()

    if pca is not None:
        val_pred = pca.inverse_transform(val_pred_raw).astype(np.float32)
        test_pred = pca.inverse_transform(test_pred_raw).astype(np.float32)
        val_true = y_val.astype(np.float32)
        test_true = y_test.astype(np.float32)
    else:
        val_pred = val_pred_raw
        val_true = val_true_raw
        test_pred = test_pred_raw
        test_true = test_true_raw

    val_corr_df = pearson_per_feature(val_pred, val_true, enh_cols)
    test_corr_df = pearson_per_feature(test_pred, test_true, enh_cols)
    summary_df = pd.DataFrame([metric_row("validation", val_corr_df), metric_row("test", test_corr_df)])

    # quick stats
    for split, df in [('val', val_corr_df), ('test', test_corr_df)]:
        s = df['pearson_r']
        print(f"{split} pearson: mean={s.mean(skipna=True):.4f} median={s.median(skipna=True):.4f} >0.4={(s>0.4).sum()}")

    torch.save(model.state_dict(), out_dir / "best_model_transformer.pt")
    if domain_head is not None:
        torch.save(domain_head.state_dict(), out_dir / "best_domain_head.pt")
    np.save(out_dir / "train_ids_transformer.npy", np.array(train_ids))
    np.save(out_dir / "val_ids_transformer.npy", np.array(val_ids))
    np.save(out_dir / "test_ids_transformer.npy", np.array(test_ids))
    (out_dir / "ids_train.txt").write_text("\n".join(train_ids) + "\n")
    (out_dir / "ids_validation.txt").write_text("\n".join(val_ids) + "\n")
    (out_dir / "ids_test.txt").write_text("\n".join(test_ids) + "\n")
    (out_dir / "enhancers.txt").write_text("\n".join(enh_cols) + "\n")
    np.save(out_dir / "val_pred_transformer.npy", val_pred)
    np.save(out_dir / "val_true_transformer.npy", val_true)
    np.save(out_dir / "test_pred_transformer.npy", test_pred)
    np.save(out_dir / "test_true_transformer.npy", test_true)
    if pca is not None:
        np.save(out_dir / "y_train.npy", y_train.astype(np.float32))
        np.save(out_dir / "y_validation.npy", y_val.astype(np.float32))
        np.save(out_dir / "y_test.npy", y_test.astype(np.float32))
        np.save(out_dir / "y_pca_train.npy", y_train_fit.astype(np.float32))
        np.save(out_dir / "y_pca_validation.npy", y_val_fit.astype(np.float32))
        np.save(out_dir / "y_pca_test.npy", y_test_fit.astype(np.float32))
        np.save(out_dir / "val_pred_pca.npy", val_pred_raw.astype(np.float32))
        np.save(out_dir / "val_true_pca.npy", val_true_raw.astype(np.float32))
        np.save(out_dir / "test_pred_pca.npy", test_pred_raw.astype(np.float32))
        np.save(out_dir / "test_true_pca.npy", test_true_raw.astype(np.float32))
        np.save(out_dir / "pca_components.npy", pca.components_.astype(np.float32))
        np.save(out_dir / "pca_mean.npy", pca.mean_.astype(np.float32))
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))
        pred_pc_summary_rows = []
        val_pc_corr = per_pc_prediction_corr(val_pred_raw, val_true_raw)
        test_pc_corr = per_pc_prediction_corr(test_pred_raw, test_true_raw)
        val_pc_corr.to_csv(out_dir / "predicted_pc_correlation_validation.csv", index=False)
        test_pc_corr.to_csv(out_dir / "predicted_pc_correlation_test.csv", index=False)
        pred_pc_summary_rows.append(predicted_pc_metric_row("validation", val_pc_corr))
        pred_pc_summary_rows.append(predicted_pc_metric_row("test", test_pc_corr))
        pd.DataFrame(pred_pc_summary_rows).to_csv(out_dir / "predicted_pc_summary.csv", index=False)
    val_corr_df.to_csv(out_dir / "per_enhancer_correlation_transformer_val.csv", index=False)
    val_corr_df.to_csv(out_dir / "per_enhancer_correlation_transformer_validation.csv", index=False)
    test_corr_df.to_csv(out_dir / "per_enhancer_correlation_transformer.csv", index=False)
    test_corr_df.to_csv(out_dir / "per_enhancer_correlation_transformer_test.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
