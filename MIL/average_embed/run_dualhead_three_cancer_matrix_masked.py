#!/usr/bin/env python3
"""
Dual-head multi-cancer regression for a union target panel with per-sample masks.
"""

import argparse
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from run_mean_regression_pan_cancer import set_seed, summarize_corr

CANCERS = ["BRCA", "LUAD", "SKCM"]
CANCER_TO_IDX = {c: i for i, c in enumerate(CANCERS)}


class MatrixDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray, groups: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.g = torch.tensor(groups, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx], self.g[idx]


class DualHeadRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float, use_gating: bool):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

    def forward(self, x: torch.Tensor, groups: torch.Tensor):
        h = self.backbone(x)
        pan = self.pan_head(h)
        spec = torch.zeros_like(pan)
        cancer_logits = self.cancer_head(h)
        for idx, cancer in enumerate(CANCERS):
            use = groups == idx
            if use.any():
                spec[use] = self.spec_heads[cancer](h[use])
        gate = self.gate(h) if self.use_gating else None
        return pan, spec, gate, cancer_logits


def load_split(matrix_dir: Path, split: str):
    x = np.load(matrix_dir / f"X_{split}.npy").astype(np.float32)
    y = np.load(matrix_dir / f"y_{split}.npy").astype(np.float32)
    mask = np.load(matrix_dir / f"mask_{split}.npy").astype(np.float32)
    groups_str = np.load(matrix_dir / f"group_{split}.npy", allow_pickle=True)
    groups = np.array([CANCER_TO_IDX[g] for g in groups_str], dtype=np.int64)
    ids = [line.strip() for line in (matrix_dir / f"id_{split}.txt").read_text().splitlines() if line.strip()]
    return ids, x, y, mask, groups


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
    for xb, yb, mb, gb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        gb = gb.to(device)
        optim.zero_grad()
        pan, spec, gate, cancer_logits = model(xb, gb)
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
    for xb, yb, mb, gb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        gb = gb.to(device)
        pan, spec, gate, cancer_logits = model(xb, gb)
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
            r = np.nan
            n_valid = int(use.sum())
        else:
            x = preds[use, i]
            y = trues[use, i]
            if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                r = np.nan
            else:
                r = float(np.corrcoef(x, y)[0, 1])
            n_valid = int(use.sum())
        rows.append({"enhancer": name, "pearson_r": r, "n_valid": n_valid})
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enh_ref = [line.strip() for line in (matrix_dir / "enhancers.txt").read_text().splitlines() if line.strip()]
    _, x_train, y_train, mask_train, train_groups = load_split(matrix_dir, "train")
    _, x_val, y_val, mask_val, val_groups = load_split(matrix_dir, "validation")
    _, x_test, y_test, mask_test, test_groups = load_split(matrix_dir, "test")

    train_loader = DataLoader(MatrixDataset(x_train, y_train, mask_train, train_groups), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MatrixDataset(x_val, y_val, mask_val, val_groups), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(MatrixDataset(x_test, y_test, mask_test, test_groups), batch_size=args.batch_size, shuffle=False)

    model = DualHeadRegressor(x_train.shape[1], y_train.shape[1], args.hidden_dim, args.dropout, args.use_gating).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optim, device,
            args.loss_weight_specific, args.loss_weight_pan,
            args.loss_weight_consistency, args.loss_weight_cancer, args.fusion_alpha,
        )
        val_loss, _, _, _, _, _, _, _ = eval_epoch(model, val_loader, device, args.fusion_alpha)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_pan_pred, val_spec_pred, val_fused_pred, val_true, val_mask, val_groups, val_cancer_acc = eval_epoch(
        model, val_loader, device, args.fusion_alpha
    )
    test_loss, test_pan_pred, test_spec_pred, test_fused_pred, test_true, test_mask, test_groups, test_cancer_acc = eval_epoch(
        model, test_loader, device, args.fusion_alpha
    )
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

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
