#!/usr/bin/env python3
"""
Transformer dual-head multi-cancer regression for a union target panel with per-sample masks.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from run_mean_regression_pan_cancer import set_seed, summarize_corr

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from our_code_data.enhancer_prediction.models import WSITransformerRegressor


CANCERS = ["BRCA", "LUAD", "SKCM"]
CANCER_TO_IDX = {c: i for i, c in enumerate(CANCERS)}


def tcga_case3_from_dirname(dirname: str) -> str:
    parts = dirname.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else dirname


def build_case_to_feature_dir(root: Path) -> dict[str, Path]:
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    case_map: dict[str, list[Path]] = {}
    for d in dirs:
        case = tcga_case3_from_dirname(d.name)
        case_map.setdefault(case, []).append(d)
    resolved: dict[str, Path] = {}
    for case, candidates in case_map.items():
        resolved[case] = sorted(candidates, key=lambda p: p.name)[0]
    return resolved


def load_case_tokens(case_dir: Path, max_patches: Optional[int], rng: Optional[np.random.Generator]) -> np.ndarray:
    files = sorted(case_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No npy files in {case_dir}")
    if max_patches is not None and len(files) > max_patches:
        assert rng is not None
        idx = rng.choice(len(files), size=max_patches, replace=False)
        files = [files[i] for i in sorted(idx)]
    arrs = []
    bad = []
    for f in files:
        try:
            arrs.append(np.load(f).astype(np.float32))
        except Exception as exc:
            bad.append(f"{f.name} ({type(exc).__name__})")
    if bad:
        print(f"[warn] {case_dir.name}: skipped bad npy files={len(bad)} (e.g., {bad[:3]})")
    if not arrs:
        raise FileNotFoundError(f"No readable npy files in {case_dir}")
    return np.stack(arrs, axis=0)


def resolve_split_case_maps(feat_root: Path, split: str) -> dict[str, dict[str, Path]]:
    split_root = feat_root / split
    maps: dict[str, dict[str, Path]] = {}
    for cancer in CANCERS:
        cancer_root = split_root / cancer
        fallback_root = feat_root / cancer
        if cancer_root.exists():
            maps[cancer] = build_case_to_feature_dir(cancer_root)
        elif fallback_root.exists():
            maps[cancer] = build_case_to_feature_dir(fallback_root)
        else:
            raise FileNotFoundError(f"No feature directory found for split={split}, cancer={cancer}")
    return maps


class TokenMatrixDataset(Dataset):
    def __init__(
        self,
        split: str,
        feat_root: Path,
        ids: list[str],
        y: np.ndarray,
        mask: np.ndarray,
        groups: np.ndarray,
        max_patches: Optional[int],
        seed: int,
    ):
        self.split = split
        self.ids = ids
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.g = torch.tensor(groups, dtype=torch.long)
        self.max_patches = max_patches
        self.case_maps = resolve_split_case_maps(feat_root, split)
        self.base_seed = seed

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        cancer, case = sample_id.split(":", 1)
        case_dir = self.case_maps[cancer].get(case)
        if case_dir is None:
            raise KeyError(f"Missing feature directory for {sample_id} in split={self.split}")
        rng = None
        if self.max_patches is not None:
            rng = np.random.default_rng(self.base_seed + idx)
        x = torch.tensor(load_case_tokens(case_dir, self.max_patches, rng), dtype=torch.float32)
        return x, self.y[idx], self.mask[idx], self.g[idx]


def collate_batch(batch):
    xs, ys, ms, gs = zip(*batch)
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
    return tokens, valid, torch.stack(ys), torch.stack(ms), torch.stack(gs)


class TransformerDualHeadRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        hidden_dim: int,
        dropout: float,
        pooling: str,
        use_gating: bool,
    ):
        super().__init__()
        self.encoder = WSITransformerRegressor(
            token_dim=input_dim,
            output_dim=1,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pooling=pooling,
            use_coords=False,
        )
        self.backbone = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pan_head = nn.Linear(hidden_dim, output_dim)
        self.spec_heads = nn.ModuleDict({c: nn.Linear(hidden_dim, output_dim) for c in CANCERS})
        self.cancer_head = nn.Linear(hidden_dim, len(CANCERS))
        self.use_gating = use_gating
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, groups: torch.Tensor):
        rep = self.encoder.encode(x, valid_mask=valid_mask, coords=None)
        h = self.backbone(rep)
        pan = self.pan_head(h)
        spec = torch.zeros_like(pan)
        cancer_logits = self.cancer_head(h)
        for idx, cancer in enumerate(CANCERS):
            use = groups == idx
            if use.any():
                spec[use] = self.spec_heads[cancer](h[use])
        gate = self.gate(h) if self.use_gating else None
        return pan, spec, gate, cancer_logits


def load_split_meta(matrix_dir: Path, split: str):
    y = np.load(matrix_dir / f"y_{split}.npy").astype(np.float32)
    mask = np.load(matrix_dir / f"mask_{split}.npy").astype(np.float32)
    groups_str = np.load(matrix_dir / f"group_{split}.npy", allow_pickle=True)
    groups = np.array([CANCER_TO_IDX[g] for g in groups_str], dtype=np.int64)
    ids = [line.strip() for line in (matrix_dir / f"id_{split}.txt").read_text().splitlines() if line.strip()]
    return ids, y, mask, groups


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def fuse_outputs(pan: torch.Tensor, spec: torch.Tensor, gate: Optional[torch.Tensor], alpha: float):
    if gate is None:
        return alpha * spec + (1.0 - alpha) * pan
    return gate * spec + (1.0 - gate) * pan


def train_epoch(model, loader, optim, device, w_spec, w_pan, w_cons, w_cancer, alpha):
    model.train()
    total = 0.0
    n = 0
    ce_loss = nn.CrossEntropyLoss()
    for xb, vb, yb, mb, gb in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        gb = gb.to(device)
        optim.zero_grad(set_to_none=True)
        pan, spec, gate, cancer_logits = model(xb, vb, gb)
        fused = fuse_outputs(pan, spec, gate, alpha)
        loss_pan = masked_mse(pan, yb, mb)
        loss_spec = masked_mse(spec, yb, mb)
        loss_cons = masked_mse(spec, pan, mb)
        loss_fused = masked_mse(fused, yb, mb)
        loss_cancer = ce_loss(cancer_logits, gb)
        loss = loss_fused + w_spec * loss_spec + w_pan * loss_pan + w_cons * loss_cons + w_cancer * loss_cancer
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, alpha):
    model.eval()
    total = 0.0
    n = 0
    pan_preds = []
    spec_preds = []
    fused_preds = []
    trues = []
    masks = []
    groups = []
    cancer_correct = 0
    cancer_total = 0
    for xb, vb, yb, mb, gb in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        gb = gb.to(device)
        pan, spec, gate, cancer_logits = model(xb, vb, gb)
        fused = fuse_outputs(pan, spec, gate, alpha)
        loss = masked_mse(fused, yb, mb)
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
        pred_group = cancer_logits.argmax(dim=1)
        cancer_correct += int((pred_group == gb).sum().item())
        cancer_total += int(gb.numel())
        pan_preds.append(pan.cpu().numpy())
        spec_preds.append(spec.cpu().numpy())
        fused_preds.append(fused.cpu().numpy())
        trues.append(yb.cpu().numpy())
        masks.append(mb.cpu().numpy())
        groups.append(gb.cpu().numpy())
    return (
        total / max(n, 1),
        np.concatenate(pan_preds, axis=0),
        np.concatenate(spec_preds, axis=0),
        np.concatenate(fused_preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(masks, axis=0),
        np.concatenate(groups, axis=0),
        (cancer_correct / max(cancer_total, 1)),
    )


def pearson_per_feature_masked(preds: np.ndarray, trues: np.ndarray, masks: np.ndarray, enhancers: list[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(enhancers):
        use = masks[:, i] > 0.5
        if use.sum() < 2:
            rows.append({"enhancer": name, "pearson_r": np.nan, "n_valid": int(use.sum())})
            continue
        x = preds[use, i]
        y = trues[use, i]
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"enhancer": name, "pearson_r": r, "n_valid": int(use.sum())})
    return pd.DataFrame(rows)


def metric_row(corr_df: pd.DataFrame, split: str, group: str, n_enhancers: int):
    s = corr_df["pearson_r"]
    return {
        "split": split,
        "group": group,
        "n_enhancers": n_enhancers,
        "n_valid_targets_mean": float(corr_df["n_valid"].mean()),
        "pearson_mean": float(s.mean(skipna=True)),
        "pearson_median": float(s.median(skipna=True)),
        "gt_0.4": int((s > 0.4).sum()),
        "gt_0.5": int((s > 0.5).sum()),
        "gt_0.6": int((s > 0.6).sum()),
    }


def summarize_prediction_set(
    split_name: str,
    pred_name: str,
    preds: np.ndarray,
    trues: np.ndarray,
    masks: np.ndarray,
    groups: np.ndarray,
    enh_ref: list[str],
    out_dir: Path,
    summary_rows: list[dict],
):
    overall_df = pearson_per_feature_masked(preds, trues, masks, enh_ref)
    summarize_corr(overall_df, f"{split_name}_{pred_name}_overall")
    summary_rows.append(metric_row(overall_df, split_name, f"{pred_name}_ALL", len(enh_ref)))
    overall_df.to_csv(out_dir / f"per_enhancer_correlation_{split_name}_{pred_name}_all.csv", index=False)
    for idx, cancer in enumerate(CANCERS):
        use = groups == idx
        corr_df = pearson_per_feature_masked(preds[use], trues[use], masks[use], enh_ref)
        summarize_corr(corr_df, f"{split_name}_{pred_name}_{cancer}")
        summary_rows.append(metric_row(corr_df, split_name, f"{pred_name}_{cancer}", len(enh_ref)))
        corr_df.to_csv(out_dir / f"per_enhancer_correlation_{split_name}_{pred_name}_{cancer}.csv", index=False)


def infer_input_dim(feat_root: Path) -> int:
    for split in ["train", "validation", "test"]:
        split_root = feat_root / split
        if not split_root.exists():
            continue
        for npy_path in split_root.glob("*/*/*.npy"):
            return int(np.load(npy_path).shape[-1])
    for npy_path in feat_root.glob("*/*/*.npy"):
        return int(np.load(npy_path).shape[-1])
    raise RuntimeError(f"Could not infer feature dimension from {feat_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--feat-root", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-patches", type=int, default=300)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--loss-weight-specific", type=float, default=1.0)
    parser.add_argument("--loss-weight-pan", type=float, default=1.0)
    parser.add_argument("--loss-weight-consistency", type=float, default=0.1)
    parser.add_argument("--loss-weight-cancer", type=float, default=0.1)
    parser.add_argument("--fusion-alpha", type=float, default=0.5)
    parser.add_argument("--use-gating", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_dir = Path(args.matrix_dir)
    feat_root = Path(args.feat_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enh_ref = [line.strip() for line in (matrix_dir / "enhancers.txt").read_text().splitlines() if line.strip()]
    train_ids, y_train, mask_train, train_groups = load_split_meta(matrix_dir, "train")
    val_ids, y_val, mask_val, val_groups = load_split_meta(matrix_dir, "validation")
    test_ids, y_test, mask_test, test_groups = load_split_meta(matrix_dir, "test")

    train_ds = TokenMatrixDataset("train", feat_root, train_ids, y_train, mask_train, train_groups, args.max_patches, args.seed)
    val_ds = TokenMatrixDataset("validation", feat_root, val_ids, y_val, mask_val, val_groups, args.max_patches, args.seed)
    test_ds = TokenMatrixDataset("test", feat_root, test_ids, y_test, mask_test, test_groups, args.max_patches, args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    input_dim = infer_input_dim(feat_root)
    model = TransformerDualHeadRegressor(
        input_dim=input_dim,
        output_dim=y_train.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        use_gating=args.use_gating,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optim,
            device,
            args.loss_weight_specific,
            args.loss_weight_pan,
            args.loss_weight_consistency,
            args.loss_weight_cancer,
            args.fusion_alpha,
        )
        val_loss, _, _, _, _, _, _, _ = eval_epoch(model, val_loader, device, args.fusion_alpha)
        if val_loss < (best_val - args.min_delta):
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
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

    val_loss, val_pan_pred, val_spec_pred, val_fused_pred, val_true, val_mask, val_groups, val_cancer_acc = eval_epoch(
        model, val_loader, device, args.fusion_alpha
    )
    test_loss, test_pan_pred, test_spec_pred, test_fused_pred, test_true, test_mask, test_groups, test_cancer_acc = eval_epoch(
        model, test_loader, device, args.fusion_alpha
    )
    print(
        f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f} "
        f"val_cancer_acc={val_cancer_acc:.4f} test_cancer_acc={test_cancer_acc:.4f}"
    )

    summary_rows = []
    for split_name, pred_sets, trues, masks, groups in [
        ("validation", {"pan": val_pan_pred, "specific": val_spec_pred, "fused": val_fused_pred}, val_true, val_mask, val_groups),
        ("test", {"pan": test_pan_pred, "specific": test_spec_pred, "fused": test_fused_pred}, test_true, test_mask, test_groups),
    ]:
        for pred_name, preds in pred_sets.items():
            summarize_prediction_set(split_name, pred_name, preds, trues, masks, groups, enh_ref, out_dir, summary_rows)

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_split_and_cancer.csv", index=False)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    pd.Series(
        {
            "feature_mode": "transformer",
            "max_patches": args.max_patches,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "pooling": args.pooling,
            "early_patience": args.early_patience,
            "min_delta": args.min_delta,
            "num_workers": args.num_workers,
            "use_gating": args.use_gating,
            "fusion_alpha": args.fusion_alpha,
            "loss_weight_specific": args.loss_weight_specific,
            "loss_weight_pan": args.loss_weight_pan,
            "loss_weight_consistency": args.loss_weight_consistency,
            "loss_weight_cancer": args.loss_weight_cancer,
            "val_cancer_acc": val_cancer_acc,
            "test_cancer_acc": test_cancer_acc,
        }
    ).to_json(out_dir / "run_config.json")

    np.save(out_dir / "val_pan_pred.npy", val_pan_pred)
    np.save(out_dir / "val_specific_pred.npy", val_spec_pred)
    np.save(out_dir / "val_fused_pred.npy", val_fused_pred)
    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "val_mask.npy", val_mask)
    np.save(out_dir / "test_pan_pred.npy", test_pan_pred)
    np.save(out_dir / "test_specific_pred.npy", test_spec_pred)
    np.save(out_dir / "test_fused_pred.npy", test_fused_pred)
    np.save(out_dir / "test_true.npy", test_true)
    np.save(out_dir / "test_mask.npy", test_mask)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
