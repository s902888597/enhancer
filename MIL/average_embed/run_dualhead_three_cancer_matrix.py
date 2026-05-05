#!/usr/bin/env python3
"""
Dual-head multi-cancer regression on prebuilt train/validation/test matrices.

- shared backbone
- one pan-cancer head
- one cancer-specific head per cancer
- either fixed fusion or a learned sample-wise gate
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from run_mean_regression_pan_cancer import pearson_per_feature, set_seed, summarize_corr


CANCERS = ["BRCA", "LUAD", "SKCM"]
CANCER_TO_IDX = {c: i for i, c in enumerate(CANCERS)}


class MatrixDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.g = torch.tensor(groups, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.g[idx]


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
        for idx, cancer in enumerate(CANCERS):
            mask = groups == idx
            if mask.any():
                spec[mask] = self.spec_heads[cancer](h[mask])
        if self.use_gating:
            gate = self.gate(h)
        else:
            gate = None
        return pan, spec, gate


def load_split(matrix_dir: Path, split: str):
    x = np.load(matrix_dir / f"X_{split}.npy")
    y = np.load(matrix_dir / f"y_{split}.npy")
    groups_str = np.load(matrix_dir / f"group_{split}.npy")
    groups = np.array([CANCER_TO_IDX[g] for g in groups_str], dtype=np.int64)
    ids = [line.strip() for line in (matrix_dir / f"id_{split}.txt").read_text().splitlines() if line.strip()]
    return ids, x, y, groups


def metric_row(corr_df: pd.DataFrame, split: str, group: str, k: int, n_enhancers: int):
    s = corr_df["pearson_r"]
    return {
        "split": split,
        "group": group,
        "k": k,
        "n_enhancers": n_enhancers,
        "pearson_mean": float(s.mean(skipna=True)),
        "pearson_median": float(s.median(skipna=True)),
        "gt_0.4": int((s > 0.4).sum()),
        "gt_0.5": int((s > 0.5).sum()),
        "gt_0.6": int((s > 0.6).sum()),
    }


def fuse_outputs(pan: torch.Tensor, spec: torch.Tensor, gate: Optional[torch.Tensor], alpha: float):
    if gate is None:
        return alpha * spec + (1.0 - alpha) * pan
    return gate * spec + (1.0 - gate) * pan


def train_epoch(model, loader, optim, mse_loss, device, w_spec, w_pan, w_cons, alpha):
    model.train()
    total = 0.0
    n = 0
    for xb, yb, gb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        gb = gb.to(device)
        optim.zero_grad()
        pan, spec, gate = model(xb, gb)
        fused = fuse_outputs(pan, spec, gate, alpha)
        loss_pan = mse_loss(pan, yb)
        loss_spec = mse_loss(spec, yb)
        loss_cons = mse_loss(spec, pan)
        loss_fused = mse_loss(fused, yb)
        loss = loss_fused + w_spec * loss_spec + w_pan * loss_pan + w_cons * loss_cons
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, mse_loss, device, alpha):
    model.eval()
    total = 0.0
    n = 0
    pan_preds = []
    spec_preds = []
    preds = []
    trues = []
    groups = []
    for xb, yb, gb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        gb = gb.to(device)
        pan, spec, gate = model(xb, gb)
        fused = fuse_outputs(pan, spec, gate, alpha)
        loss = mse_loss(fused, yb)
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
        pan_preds.append(pan.cpu().numpy())
        spec_preds.append(spec.cpu().numpy())
        preds.append(fused.cpu().numpy())
        trues.append(yb.cpu().numpy())
        groups.append(gb.cpu().numpy())
    return (
        total / max(n, 1),
        np.concatenate(pan_preds, axis=0),
        np.concatenate(spec_preds, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(groups, axis=0),
    )


def summarize_prediction_set(
    split_name: str,
    pred_name: str,
    preds: np.ndarray,
    trues: np.ndarray,
    groups: np.ndarray,
    enh_ref: list[str],
    out_dir: Path,
    summary_rows: list[dict],
    k: int,
):
    overall_df = pearson_per_feature(preds, trues, enh_ref)
    summarize_corr(overall_df, f"{split_name}_{pred_name}_overall")
    summary_rows.append(metric_row(overall_df, split_name, f"{pred_name}_ALL", k, len(enh_ref)))
    overall_df.to_csv(out_dir / f"per_enhancer_correlation_{split_name}_{pred_name}_all.csv", index=False)
    for idx, cancer in enumerate(CANCERS):
        mask = groups == idx
        corr_df = pearson_per_feature(preds[mask], trues[mask], enh_ref)
        summarize_corr(corr_df, f"{split_name}_{pred_name}_{cancer}")
        summary_rows.append(metric_row(corr_df, split_name, f"{pred_name}_{cancer}", k, len(enh_ref)))
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
    parser.add_argument("--pca-k", type=int, default=0)
    parser.add_argument("--loss-weight-specific", type=float, default=1.0)
    parser.add_argument("--loss-weight-pan", type=float, default=1.0)
    parser.add_argument("--loss-weight-consistency", type=float, default=0.1)
    parser.add_argument("--fusion-alpha", type=float, default=0.5)
    parser.add_argument("--use-gating", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enh_ref = [line.strip() for line in (matrix_dir / "enhancers.txt").read_text().splitlines() if line.strip()]
    train_ids, x_train, y_train, train_groups = load_split(matrix_dir, "train")
    val_ids, x_val, y_val, val_groups = load_split(matrix_dir, "validation")
    test_ids, x_test, y_test, test_groups = load_split(matrix_dir, "test")

    pca = None
    y_train_fit = y_train
    y_val_fit = y_val
    y_test_fit = y_test
    k = 0
    if args.pca_k > 0:
        k = min(args.pca_k, y_train.shape[0], y_train.shape[1])
        pca = PCA(n_components=k, random_state=args.seed)
        y_train_fit = pca.fit_transform(y_train).astype(np.float32)
        y_val_fit = pca.transform(y_val).astype(np.float32)
        y_test_fit = pca.transform(y_test).astype(np.float32)
        print(f"PCA enabled: k={k}, EVR_sum={float(pca.explained_variance_ratio_.sum()):.4f}")

    train_loader = DataLoader(MatrixDataset(x_train, y_train_fit, train_groups), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MatrixDataset(x_val, y_val_fit, val_groups), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(MatrixDataset(x_test, y_test_fit, test_groups), batch_size=args.batch_size, shuffle=False)

    model = DualHeadRegressor(
        x_train.shape[1],
        y_train_fit.shape[1],
        args.hidden_dim,
        args.dropout,
        args.use_gating,
    ).to(device)
    mse_loss = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optim,
            mse_loss,
            device,
            args.loss_weight_specific,
            args.loss_weight_pan,
            args.loss_weight_consistency,
            args.fusion_alpha,
        )
        val_loss, _, _, _, _, _ = eval_epoch(model, val_loader, mse_loss, device, args.fusion_alpha)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_pan_pred, val_spec_pred, val_fused_pred, val_true, val_groups = eval_epoch(
        model, val_loader, mse_loss, device, args.fusion_alpha
    )
    test_loss, test_pan_pred, test_spec_pred, test_fused_pred, test_true, test_groups = eval_epoch(
        model, test_loader, mse_loss, device, args.fusion_alpha
    )
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    if pca is not None:
        val_pan_pred_orig = pca.inverse_transform(val_pan_pred)
        val_spec_pred_orig = pca.inverse_transform(val_spec_pred)
        val_fused_pred_orig = pca.inverse_transform(val_fused_pred)
        test_pan_pred_orig = pca.inverse_transform(test_pan_pred)
        test_spec_pred_orig = pca.inverse_transform(test_spec_pred)
        test_fused_pred_orig = pca.inverse_transform(test_fused_pred)
        np.save(out_dir / "pca_components.npy", pca.components_)
        np.save(out_dir / "pca_mean.npy", pca.mean_)
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)
    else:
        val_pan_pred_orig = val_pan_pred
        val_spec_pred_orig = val_spec_pred
        val_fused_pred_orig = val_fused_pred
        test_pan_pred_orig = test_pan_pred
        test_spec_pred_orig = test_spec_pred
        test_fused_pred_orig = test_fused_pred

    summary_rows = []
    for split_name, pred_sets, trues, groups in [
        (
            "validation",
            {
                "pan": val_pan_pred_orig,
                "specific": val_spec_pred_orig,
                "fused": val_fused_pred_orig,
            },
            val_true,
            val_groups,
        ),
        (
            "test",
            {
                "pan": test_pan_pred_orig,
                "specific": test_spec_pred_orig,
                "fused": test_fused_pred_orig,
            },
            test_true,
            test_groups,
        ),
    ]:
        for pred_name, preds in pred_sets.items():
            summarize_prediction_set(
                split_name, pred_name, preds, trues, groups, enh_ref, out_dir, summary_rows, k
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_split_and_cancer.csv", index=False)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    meta = {
        "pca_k": k,
        "use_gating": args.use_gating,
        "fusion_alpha": args.fusion_alpha,
        "loss_weight_specific": args.loss_weight_specific,
        "loss_weight_pan": args.loss_weight_pan,
        "loss_weight_consistency": args.loss_weight_consistency,
    }
    pd.Series(meta).to_json(out_dir / "run_config.json")
    np.save(out_dir / "val_pan_pred.npy", val_pan_pred_orig)
    np.save(out_dir / "val_specific_pred.npy", val_spec_pred_orig)
    np.save(out_dir / "val_fused_pred.npy", val_fused_pred_orig)
    np.save(out_dir / "val_pred.npy", val_fused_pred_orig)
    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "test_pan_pred.npy", test_pan_pred_orig)
    np.save(out_dir / "test_specific_pred.npy", test_spec_pred_orig)
    np.save(out_dir / "test_fused_pred.npy", test_fused_pred_orig)
    np.save(out_dir / "test_pred.npy", test_fused_pred_orig)
    np.save(out_dir / "test_true.npy", test_true)
    np.save(out_dir / "train_groups.npy", train_groups)
    np.save(out_dir / "val_groups.npy", val_groups)
    np.save(out_dir / "test_groups.npy", test_groups)
    np.save(out_dir / "train_ids.npy", np.array(train_ids))
    np.save(out_dir / "val_ids.npy", np.array(val_ids))
    np.save(out_dir / "test_ids.npy", np.array(test_ids))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
