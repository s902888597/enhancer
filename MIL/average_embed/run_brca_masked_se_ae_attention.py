#!/usr/bin/env python3
"""BRCA masked-SE autoencoder latent target followed by image attention regression."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_mean_regression import pearson_per_feature, set_seed
from run_skcm_top1000_attention_regression import (
    AttentionPoolRegressor,
    SingleCancerAttentionDataset,
    apply_bag_mixup,
    collate_batch,
    infer_input_dim,
    load_labels,
    metric_row,
    save_attention_tables,
)


class MaskedSEAutoencoder(nn.Module):
    def __init__(self, n_targets: int, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_targets * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
        )

    def encode_with_mask(self, y: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([y * observed_mask, observed_mask], dim=1))

    def encode_full(self, y: torch.Tensor) -> torch.Tensor:
        observed_mask = torch.ones_like(y)
        return self.encode_with_mask(y, observed_mask)

    def forward(self, y: torch.Tensor, observed_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_with_mask(y, observed_mask)
        return self.decoder(z), z


def random_observed_mask(shape: torch.Size, mask_ratio: float, device: torch.device) -> torch.Tensor:
    observed = (torch.rand(shape, device=device) > mask_ratio).float()
    # Avoid degenerate all-observed or all-masked rows.
    row_sum = observed.sum(dim=1)
    n_targets = shape[1]
    for i in torch.nonzero(row_sum <= 0, as_tuple=False).flatten():
        observed[i, torch.randint(0, n_targets, (1,), device=device)] = 1.0
    for i in torch.nonzero(row_sum >= n_targets, as_tuple=False).flatten():
        observed[i, torch.randint(0, n_targets, (1,), device=device)] = 0.0
    return observed


def masked_mse(pred: torch.Tensor, target: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    masked = 1.0 - observed_mask
    denom = masked.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * masked).sum() / denom


def train_ae_epoch(
    model: MaskedSEAutoencoder,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for (yb,) in loader:
        yb = yb.to(device)
        observed = random_observed_mask(yb.shape, mask_ratio, device)
        optim.zero_grad(set_to_none=True)
        pred, _ = model(yb, observed)
        loss = masked_mse(pred, yb, observed)
        loss.backward()
        optim.step()
        total += float(loss.item()) * yb.shape[0]
        n += yb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_ae_epoch(
    model: MaskedSEAutoencoder,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for (yb,) in loader:
        yb = yb.to(device)
        observed = random_observed_mask(yb.shape, mask_ratio, device)
        pred, _ = model(yb, observed)
        loss = masked_mse(pred, yb, observed)
        total += float(loss.item()) * yb.shape[0]
        n += yb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def encode_full_dataset(model: MaskedSEAutoencoder, y: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    loader = DataLoader(TensorDataset(torch.tensor(y, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    for (yb,) in loader:
        z = model.encode_full(yb.to(device))
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_masked_ae(
    y_train: np.ndarray,
    y_val: np.ndarray,
    args,
    device: torch.device,
    out_dir: Path,
) -> MaskedSEAutoencoder:
    model = MaskedSEAutoencoder(
        n_targets=y_train.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.ae_hidden_dim,
        dropout=args.ae_dropout,
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.ae_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(y_val, dtype=torch.float32)),
        batch_size=args.ae_batch_size,
        shuffle=False,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.ae_lr, weight_decay=args.ae_weight_decay)
    best_state = None
    best_val = float("inf")
    bad = 0
    history = []
    for epoch in range(1, args.ae_epochs + 1):
        train_loss = train_ae_epoch(model, train_loader, optim, device, args.mask_ratio)
        val_loss = eval_ae_epoch(model, val_loader, device, args.mask_ratio)
        history.append({"epoch": epoch, "train_masked_mse": train_loss, "val_masked_mse": val_loss})
        print(f"AE epoch {epoch}: train_masked_mse={train_loss:.5f} val_masked_mse={val_loss:.5f}", flush=True)
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= args.ae_early_patience:
            print(f"AE early stopping at epoch {epoch}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "masked_se_autoencoder.pt")
    pd.DataFrame(history).to_csv(out_dir / "masked_se_autoencoder_history.csv", index=False)
    return model


def train_image_epoch(
    model: AttentionPoolRegressor,
    decoder: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    latent_dim: int,
    recon_lambda: float,
    mixup_alpha: float,
    mixup_prob: float,
) -> float:
    model.train()
    decoder.eval()
    total = 0.0
    n = 0
    for _, xb, vb, target, _ in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        target = target.to(device)
        xb, vb, target = apply_bag_mixup(xb, vb, target, mixup_alpha, mixup_prob)
        z_true = target[:, :latent_dim]
        y_true = target[:, latent_dim:]
        optim.zero_grad(set_to_none=True)
        z_pred, _ = model(xb, vb)
        y_pred = decoder(z_pred)
        loss = nn.functional.mse_loss(z_pred, z_true) + recon_lambda * nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_image_epoch(
    model: AttentionPoolRegressor,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    latent_dim: int,
    recon_lambda: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[str], list[list[str]], list[np.ndarray]]:
    model.eval()
    decoder.eval()
    total = 0.0
    n = 0
    z_preds = []
    y_preds = []
    y_trues = []
    ids_all: list[str] = []
    patch_names_all: list[list[str]] = []
    attn_all: list[np.ndarray] = []
    for ids, xb, vb, target, patch_names in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        target = target.to(device)
        z_true = target[:, :latent_dim]
        y_true = target[:, latent_dim:]
        z_pred, attn = model(xb, vb)
        y_pred = decoder(z_pred)
        loss = nn.functional.mse_loss(z_pred, z_true) + recon_lambda * nn.functional.mse_loss(y_pred, y_true)
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
        z_preds.append(z_pred.cpu().numpy())
        y_preds.append(y_pred.cpu().numpy())
        y_trues.append(y_true.cpu().numpy())
        ids_all.extend(ids)
        patch_names_all.extend(patch_names)
        valid_lengths = vb.sum(dim=1).cpu().numpy().astype(int).tolist()
        attn_np = attn.cpu().numpy()
        for i, length in enumerate(valid_lengths):
            attn_all.append(attn_np[i, :length].astype(np.float32))
    return (
        total / max(n, 1),
        np.concatenate(z_preds, axis=0).astype(np.float32),
        np.concatenate(y_preds, axis=0).astype(np.float32),
        np.concatenate(y_trues, axis=0).astype(np.float32),
        ids_all,
        patch_names_all,
        attn_all,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--ae-hidden-dim", type=int, default=1024)
    parser.add_argument("--ae-dropout", type=float, default=0.1)
    parser.add_argument("--ae-epochs", type=int, default=300)
    parser.add_argument("--ae-batch-size", type=int, default=32)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-4)
    parser.add_argument("--ae-early-patience", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-patches", type=int, default=300)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--recon-lambda", type=float, default=1.0)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_root = Path(args.feat_root)

    train_ids, y_train, enh_cols = load_labels(Path(args.train_csv), "BRCA")
    val_ids, y_val, enh_val = load_labels(Path(args.val_csv), "BRCA")
    test_ids, y_test, enh_test = load_labels(Path(args.test_csv), "BRCA")
    if enh_val != enh_cols or enh_test != enh_cols:
        raise RuntimeError("Enhancer columns do not match across splits")
    print(f"BRCA labels: train={y_train.shape} val={y_val.shape} test={y_test.shape}", flush=True)

    ae = train_masked_ae(y_train, y_val, args, device, out_dir)
    z_train = encode_full_dataset(ae, y_train, device, args.ae_batch_size)
    z_val = encode_full_dataset(ae, y_val, device, args.ae_batch_size)
    z_test = encode_full_dataset(ae, y_test, device, args.ae_batch_size)
    np.save(out_dir / "z_train.npy", z_train)
    np.save(out_dir / "z_validation.npy", z_val)
    np.save(out_dir / "z_test.npy", z_test)

    train_ds = SingleCancerAttentionDataset("BRCA", "train", Path(args.train_csv), feat_root, args.max_patches, args.seed)
    val_ds = SingleCancerAttentionDataset("BRCA", "validation", Path(args.val_csv), feat_root, args.max_patches, args.seed)
    test_ds = SingleCancerAttentionDataset("BRCA", "test", Path(args.test_csv), feat_root, args.max_patches, args.seed)
    train_ds.y = torch.tensor(np.concatenate([z_train, y_train], axis=1), dtype=torch.float32)
    val_ds.y = torch.tensor(np.concatenate([z_val, y_val], axis=1), dtype=torch.float32)
    test_ds.y = torch.tensor(np.concatenate([z_test, y_test], axis=1), dtype=torch.float32)

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

    input_dim = infer_input_dim(feat_root, "BRCA")
    image_model = AttentionPoolRegressor(
        input_dim=input_dim,
        output_dim=args.latent_dim,
        embed_dim=args.embed_dim,
        attn_dim=args.attn_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        positive_output=False,
    ).to(device)
    decoder = copy.deepcopy(ae.decoder).to(device)
    for p in decoder.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(image_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_image_epoch(
            image_model,
            decoder,
            train_loader,
            optim,
            device,
            args.latent_dim,
            args.recon_lambda,
            args.mixup_alpha,
            args.mixup_prob,
        )
        val_loss, *_ = eval_image_epoch(image_model, decoder, val_loader, device, args.latent_dim, args.recon_lambda)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Image epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}", flush=True)
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(image_model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= args.early_patience:
            print(f"Image early stopping at epoch {epoch}", flush=True)
            break
    if best_state is not None:
        image_model.load_state_dict(best_state)

    val_loss, val_z_pred, val_pred, val_true, val_eval_ids, val_patch_names, val_attn = eval_image_epoch(
        image_model, decoder, val_loader, device, args.latent_dim, args.recon_lambda
    )
    test_loss, test_z_pred, test_pred, test_true, test_eval_ids, test_patch_names, test_attn = eval_image_epoch(
        image_model, decoder, test_loader, device, args.latent_dim, args.recon_lambda
    )
    print(f"Final image: val_loss={val_loss:.5f} test_loss={test_loss:.5f}", flush=True)

    val_corr = pearson_per_feature(val_pred, val_true, enh_cols)
    test_corr = pearson_per_feature(test_pred, test_true, enh_cols)
    summary = pd.DataFrame([metric_row("validation", val_corr), metric_row("test", test_corr)])
    print(summary.to_string(index=False), flush=True)

    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "test_true.npy", test_true)
    np.save(out_dir / "val_z_pred.npy", val_z_pred)
    np.save(out_dir / "test_z_pred.npy", test_z_pred)
    np.save(out_dir / "val_ids.npy", np.array(val_eval_ids, dtype=object))
    np.save(out_dir / "test_ids.npy", np.array(test_eval_ids, dtype=object))
    np.save(out_dir / "val_attention_weights.npy", np.array(val_attn, dtype=object), allow_pickle=True)
    np.save(out_dir / "test_attention_weights.npy", np.array(test_attn, dtype=object), allow_pickle=True)
    np.save(out_dir / "val_patch_names.npy", np.array(val_patch_names, dtype=object), allow_pickle=True)
    np.save(out_dir / "test_patch_names.npy", np.array(test_patch_names, dtype=object), allow_pickle=True)
    save_attention_tables(out_dir / "val_patch_attention", val_eval_ids, val_patch_names, val_attn)
    save_attention_tables(out_dir / "test_patch_attention", test_eval_ids, test_patch_names, test_attn)
    (out_dir / "ids_train.txt").write_text("\n".join(train_ids) + "\n")
    (out_dir / "ids_validation.txt").write_text("\n".join(val_eval_ids) + "\n")
    (out_dir / "ids_test.txt").write_text("\n".join(test_eval_ids) + "\n")
    (out_dir / "enhancers.txt").write_text("\n".join(enh_cols) + "\n")
    val_corr.to_csv(out_dir / "per_enhancer_correlation_validation.csv", index=False)
    test_corr.to_csv(out_dir / "per_enhancer_correlation_test.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "image_train_history.csv", index=False)
    torch.save(image_model.state_dict(), out_dir / "image_attention_latent_model.pt")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"Saved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
