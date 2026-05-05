#!/usr/bin/env python3
"""
Attention-pooling soft-MoE multi-cancer regression for a union target panel with per-sample masks.

Design:
- patch embed -> gated attention pooling -> shared bag representation
- sample-level soft MoE head with K experts
- optional cancer-aware gate (bag embedding + cancer one-hot)
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from run_dualhead_three_cancer_attention_masked import (
    CANCERS,
    TokenMatrixDataset,
    apply_same_cancer_bag_mixup,
    collate_batch,
    infer_input_dim,
    load_split_meta,
    metric_row,
    masked_mse,
    pearson_per_feature_masked,
    per_pc_prediction_corr,
    predicted_pc_metric_row,
)
from run_mean_regression_pan_cancer import set_seed, summarize_corr


class AttentionMoERegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        attn_dim: int,
        hidden_dim: int,
        dropout: float,
        n_experts: int,
        cancer_aware_gate: bool,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.cancer_aware_gate = cancer_aware_gate
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(embed_dim, attn_dim)
        self.attention_u = nn.Linear(embed_dim, attn_dim)
        self.attention_w = nn.Linear(attn_dim, 1)
        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        gate_in_dim = hidden_dim + (len(CANCERS) if cancer_aware_gate else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, max(hidden_dim // 2, 32)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 32), n_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, groups: torch.Tensor):
        h_patch = self.patch_embed(x)
        a_v = torch.tanh(self.attention_v(h_patch))
        a_u = torch.sigmoid(self.attention_u(h_patch))
        scores = self.attention_w(a_v * a_u).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=1)
        bag = torch.bmm(attn.unsqueeze(1), h_patch).squeeze(1)
        h = self.backbone(bag)

        if self.cancer_aware_gate:
            cancer_onehot = torch.nn.functional.one_hot(groups, num_classes=len(CANCERS)).float()
            gate_in = torch.cat([h, cancer_onehot], dim=1)
        else:
            gate_in = h

        gate_logits = self.gate(gate_in)
        gate_probs = torch.softmax(gate_logits, dim=1)
        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=1)
        pred = (gate_probs.unsqueeze(-1) * expert_outs).sum(dim=1)
        return pred, attn, gate_probs


def gate_balance_loss(gate_probs: torch.Tensor) -> torch.Tensor:
    mean_prob = gate_probs.mean(dim=0)
    target = torch.full_like(mean_prob, 1.0 / gate_probs.shape[1])
    return ((mean_prob - target) ** 2).mean()


def train_epoch(model, loader, optim, device, mixup_alpha, mixup_prob, load_balance_weight):
    model.train()
    total = 0.0
    n = 0
    for xb, vb, yb, mb, gb, _, _ in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        gb = gb.to(device)
        xb, vb, yb, mb, gb = apply_same_cancer_bag_mixup(xb, vb, yb, mb, gb, mixup_alpha, mixup_prob)

        optim.zero_grad(set_to_none=True)
        pred, _, gate_probs = model(xb, vb, gb)
        loss_main = masked_mse(pred, yb, mb)
        loss_balance = gate_balance_loss(gate_probs)
        loss = loss_main + load_balance_weight * loss_balance
        loss.backward()
        optim.step()

        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, load_balance_weight):
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    masks = []
    groups = []
    gates = []
    for xb, vb, yb, mb, gb, _, _ in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        gb = gb.to(device)
        pred, _, gate_probs = model(xb, vb, gb)
        loss = masked_mse(pred, yb, mb) + load_balance_weight * gate_balance_loss(gate_probs)
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
        masks.append(mb.cpu().numpy())
        groups.append(gb.cpu().numpy())
        gates.append(gate_probs.cpu().numpy())
    return (
        total / max(n, 1),
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(masks, axis=0),
        np.concatenate(groups, axis=0),
        np.concatenate(gates, axis=0),
    )


def summarize_prediction_set(
    split_name: str,
    preds: np.ndarray,
    trues: np.ndarray,
    masks: np.ndarray,
    groups: np.ndarray,
    enh_ref: list[str],
    out_dir: Path,
    summary_rows: list[dict],
):
    overall_df = pearson_per_feature_masked(preds, trues, masks, enh_ref)
    summarize_corr(overall_df, f"{split_name}_overall")
    summary_rows.append(metric_row(overall_df, split_name, "ALL", len(enh_ref)))
    overall_df.to_csv(out_dir / f"per_enhancer_correlation_{split_name}_all.csv", index=False)
    for idx, cancer in enumerate(CANCERS):
        use = groups == idx
        corr_df = pearson_per_feature_masked(preds[use], trues[use], masks[use], enh_ref)
        summarize_corr(corr_df, f"{split_name}_{cancer}")
        summary_rows.append(metric_row(corr_df, split_name, cancer, len(enh_ref)))
        corr_df.to_csv(out_dir / f"per_enhancer_correlation_{split_name}_{cancer}.csv", index=False)


def summarize_pc_prediction_set(
    split_name: str,
    preds: np.ndarray,
    trues: np.ndarray,
    groups: np.ndarray,
    out_dir: Path,
    summary_rows: list[dict],
):
    overall_df = per_pc_prediction_corr(preds, trues)
    summary_rows.append(predicted_pc_metric_row(split_name, "ALL", overall_df))
    overall_df.to_csv(out_dir / f"predicted_pc_correlation_{split_name}_all.csv", index=False)
    for idx, cancer in enumerate(CANCERS):
        use = groups == idx
        corr_df = per_pc_prediction_corr(preds[use], trues[use])
        summary_rows.append(predicted_pc_metric_row(split_name, cancer, corr_df))
        corr_df.to_csv(out_dir / f"predicted_pc_correlation_{split_name}_{cancer}.csv", index=False)


def summarize_gate_usage(split_name: str, gate_probs: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "split": split_name,
            "group": "ALL",
            **{f"expert_{i:02d}": float(gate_probs[:, i].mean()) for i in range(gate_probs.shape[1])},
        }
    )
    for idx, cancer in enumerate(CANCERS):
        use = groups == idx
        rows.append(
            {
                "split": split_name,
                "group": cancer,
                **{f"expert_{i:02d}": float(gate_probs[use, i].mean()) for i in range(gate_probs.shape[1])},
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--feat-root", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-patches", type=int, default=300)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--pca-k", type=int, default=5)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
    parser.add_argument("--cancer-aware-gate", action="store_true")
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

    pca = None
    if args.pca_k > 0:
        if args.pca_k >= y_train.shape[1]:
            raise RuntimeError(f"--pca-k must be smaller than output dim ({y_train.shape[1]}), got {args.pca_k}")
        pca = PCA(n_components=args.pca_k, random_state=args.seed)
        y_train_fit = pca.fit_transform(y_train).astype(np.float32)
        y_val_fit = pca.transform(y_val).astype(np.float32)
        y_test_fit = pca.transform(y_test).astype(np.float32)
        mask_train_fit = np.ones_like(y_train_fit, dtype=np.float32)
        mask_val_fit = np.ones_like(y_val_fit, dtype=np.float32)
        mask_test_fit = np.ones_like(y_test_fit, dtype=np.float32)
    else:
        y_train_fit, y_val_fit, y_test_fit = y_train, y_val, y_test
        mask_train_fit, mask_val_fit, mask_test_fit = mask_train, mask_val, mask_test

    train_ds = TokenMatrixDataset("train", feat_root, train_ids, y_train_fit, mask_train_fit, train_groups, None, None, args.max_patches, args.seed)
    val_ds = TokenMatrixDataset("validation", feat_root, val_ids, y_val_fit, mask_val_fit, val_groups, None, None, args.max_patches, args.seed)
    test_ds = TokenMatrixDataset("test", feat_root, test_ids, y_test_fit, mask_test_fit, test_groups, None, None, args.max_patches, args.seed)

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
    model = AttentionMoERegressor(
        input_dim=input_dim,
        output_dim=y_train_fit.shape[1],
        embed_dim=args.embed_dim,
        attn_dim=args.attn_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_experts=args.n_experts,
        cancer_aware_gate=args.cancer_aware_gate,
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
            args.mixup_alpha,
            args.mixup_prob,
            args.load_balance_weight,
        )
        val_loss, _, _, _, _, _ = eval_epoch(model, val_loader, device, args.load_balance_weight)
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

    val_loss, val_pred, val_true, val_mask, val_groups_eval, val_gates = eval_epoch(
        model, val_loader, device, args.load_balance_weight
    )
    test_loss, test_pred, test_true, test_mask, test_groups_eval, test_gates = eval_epoch(
        model, test_loader, device, args.load_balance_weight
    )
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    summary_rows = []
    pc_summary_rows = []
    for split_name, preds, trues_fit, masks_fit, groups, trues_orig, masks_orig in [
        ("validation", val_pred, val_true, val_mask, val_groups_eval, y_val, mask_val),
        ("test", test_pred, test_true, test_mask, test_groups_eval, y_test, mask_test),
    ]:
        if pca is not None:
            summarize_pc_prediction_set(split_name, preds, trues_fit, groups, out_dir, pc_summary_rows)
            preds_orig = pca.inverse_transform(preds).astype(np.float32)
            trues_for_summary = trues_orig
            masks_for_summary = masks_orig
        else:
            preds_orig = preds
            trues_for_summary = trues_fit
            masks_for_summary = masks_fit
        summarize_prediction_set(
            split_name,
            preds_orig,
            trues_for_summary,
            masks_for_summary,
            groups,
            enh_ref,
            out_dir,
            summary_rows,
        )

    gate_summary = pd.concat(
        [
            summarize_gate_usage("validation", val_gates, val_groups_eval),
            summarize_gate_usage("test", test_gates, test_groups_eval),
        ],
        ignore_index=True,
    )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_split_and_cancer.csv", index=False)
    if pc_summary_rows:
        pd.DataFrame(pc_summary_rows).to_csv(out_dir / "predicted_pc_summary_by_split_and_cancer.csv", index=False)
    gate_summary.to_csv(out_dir / "gate_usage_summary.csv", index=False)
    torch.save(model.state_dict(), out_dir / "best_model.pt")

    pd.Series(
        {
            "feature_mode": "attention_pooling_moe",
            "max_patches": args.max_patches,
            "embed_dim": args.embed_dim,
            "attn_dim": args.attn_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "n_experts": args.n_experts,
            "cancer_aware_gate": args.cancer_aware_gate,
            "load_balance_weight": args.load_balance_weight,
            "early_patience": args.early_patience,
            "min_delta": args.min_delta,
            "num_workers": args.num_workers,
            "mixup_alpha": args.mixup_alpha,
            "mixup_prob": args.mixup_prob,
            "pca_k": args.pca_k,
            "seed": args.seed,
        }
    ).to_json(out_dir / "run_config.json")

    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "val_mask.npy", val_mask)
    np.save(out_dir / "val_gate_weights.npy", val_gates)
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "test_true.npy", test_true)
    np.save(out_dir / "test_mask.npy", test_mask)
    np.save(out_dir / "test_gate_weights.npy", test_gates)
    np.save(out_dir / "val_ids.npy", np.array(val_ids, dtype=object))
    np.save(out_dir / "test_ids.npy", np.array(test_ids, dtype=object))

    if pca is not None:
        np.save(out_dir / "y_pca_train.npy", y_train_fit)
        np.save(out_dir / "y_pca_validation.npy", y_val_fit)
        np.save(out_dir / "y_pca_test.npy", y_test_fit)
        np.save(out_dir / "pca_components.npy", pca.components_.astype(np.float32))
        np.save(out_dir / "pca_mean.npy", pca.mean_.astype(np.float32))
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
