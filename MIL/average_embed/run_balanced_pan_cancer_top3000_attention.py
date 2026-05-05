#!/usr/bin/env python3
"""
Pan-cancer balanced-batch attention pooling with per-cancer top3000 heads.

Goal:
- train a shared image encoder across BRCA/LUAD/SKCM
- keep one cancer-specific regression head per cancer
- compare directly against single-cancer top3000 attention baselines
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, Dataset

from run_mean_regression import pearson_per_feature, set_seed
from run_skcm_top1000_attention_regression import (
    SingleCancerAttentionDataset,
    apply_bag_mixup,
    infer_input_dim,
)


CANCERS = ["BRCA", "LUAD", "SKCM"]
CANCER_TO_IDX = {c: i for i, c in enumerate(CANCERS)}


class MixedCancerTrainDataset(Dataset):
    def __init__(self, cancer_datasets: dict[str, SingleCancerAttentionDataset]):
        self.cancer_datasets = cancer_datasets
        self.cancers = list(CANCERS)
        self.offsets: dict[str, int] = {}
        self.group_indices: dict[str, list[int]] = {}
        offset = 0
        for cancer in self.cancers:
            ds = self.cancer_datasets[cancer]
            self.offsets[cancer] = offset
            idxs = list(range(offset, offset + len(ds)))
            self.group_indices[cancer] = idxs
            offset += len(ds)
        self.total_len = offset

    def __len__(self) -> int:
        return self.total_len

    def _resolve(self, idx: int) -> tuple[str, int]:
        for cancer in self.cancers:
            off = self.offsets[cancer]
            n = len(self.cancer_datasets[cancer])
            if off <= idx < off + n:
                return cancer, idx - off
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        cancer, local_idx = self._resolve(idx)
        sample_id, x, y, patch_names = self.cancer_datasets[cancer][local_idx]
        return cancer, sample_id, x, y, patch_names


class BalancedCancerBatchSampler(BatchSampler):
    def __init__(self, group_indices: dict[str, list[int]], batch_size_per_cancer: int, seed: int):
        self.group_indices = group_indices
        self.batch_size_per_cancer = batch_size_per_cancer
        self.seed = seed
        self.epoch = 0
        self.num_batches = max(
            math.ceil(len(idxs) / batch_size_per_cancer) for idxs in self.group_indices.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        need = self.num_batches * self.batch_size_per_cancer
        pools: dict[str, np.ndarray] = {}
        for cancer, idxs in self.group_indices.items():
            arr = np.array(idxs, dtype=np.int64)
            perm = rng.permutation(arr)
            if len(perm) < need:
                extra = rng.choice(arr, size=need - len(perm), replace=True)
                perm = np.concatenate([perm, extra], axis=0)
            else:
                perm = perm[:need]
            pools[cancer] = perm

        for b in range(self.num_batches):
            batch = []
            lo = b * self.batch_size_per_cancer
            hi = (b + 1) * self.batch_size_per_cancer
            for cancer in CANCERS:
                batch.extend(pools[cancer][lo:hi].tolist())
            rng.shuffle(batch)
            yield batch


def collate_mixed_batch(batch):
    cancers, ids, xs, ys, patch_names = zip(*batch)
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
    groups = torch.tensor([CANCER_TO_IDX[c] for c in cancers], dtype=torch.long)
    return list(cancers), list(ids), tokens, valid, torch.stack(ys), groups, list(patch_names)


class SharedAttentionCancerHeadRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: dict[str, int],
        embed_dim: int,
        attn_dim: int,
        hidden_dim: int,
        dropout: float,
        conditioning_mode: str,
        conditioning_dim: int,
    ):
        super().__init__()
        self.conditioning_mode = conditioning_mode
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(embed_dim, attn_dim)
        self.attention_u = nn.Linear(embed_dim, attn_dim)
        self.attention_w = nn.Linear(attn_dim, 1)
        if conditioning_mode in {"film", "attn_bias"}:
            self.cancer_embedding = nn.Embedding(len(CANCERS), conditioning_dim)
        if conditioning_mode == "film":
            self.cancer_film = nn.Linear(conditioning_dim, 2 * embed_dim)
            nn.init.zeros_(self.cancer_film.weight)
            nn.init.zeros_(self.cancer_film.bias)
        if conditioning_mode == "attn_bias":
            self.attn_v_bias = nn.Linear(conditioning_dim, attn_dim)
            self.attn_u_bias = nn.Linear(conditioning_dim, attn_dim)
            nn.init.zeros_(self.attn_v_bias.weight)
            nn.init.zeros_(self.attn_v_bias.bias)
            nn.init.zeros_(self.attn_u_bias.weight)
            nn.init.zeros_(self.attn_u_bias.bias)
        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {
                cancer: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dims[cancer]),
                )
                for cancer in CANCERS
            }
        )

    def apply_conditioning(self, bag: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        if self.conditioning_mode == "none":
            return bag
        if self.conditioning_mode == "attn_bias":
            return bag
        if self.conditioning_mode == "film":
            cond = self.cancer_embedding(groups)
            gamma, beta = self.cancer_film(cond).chunk(2, dim=-1)
            return bag * (1.0 + gamma) + beta
        raise ValueError(f"Unsupported conditioning_mode={self.conditioning_mode}")

    def encode(self, x: torch.Tensor, valid_mask: torch.Tensor, groups: torch.Tensor):
        h_patch = self.patch_embed(x)
        attn_v = self.attention_v(h_patch)
        attn_u = self.attention_u(h_patch)
        if self.conditioning_mode == "attn_bias":
            cond = self.cancer_embedding(groups)
            attn_v = attn_v + self.attn_v_bias(cond).unsqueeze(1)
            attn_u = attn_u + self.attn_u_bias(cond).unsqueeze(1)
        a_v = torch.tanh(attn_v)
        a_u = torch.sigmoid(attn_u)
        scores = self.attention_w(a_v * a_u).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=1)
        bag = torch.bmm(attn.unsqueeze(1), h_patch).squeeze(1)
        bag = self.apply_conditioning(bag, groups)
        h = self.backbone(bag)
        return h, attn

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, groups: torch.Tensor):
        h, attn = self.encode(x, valid_mask, groups)
        preds = {cancer: self.heads[cancer](h) for cancer in CANCERS}
        return preds, attn


def apply_same_cancer_mixup(
    xb: torch.Tensor,
    vb: torch.Tensor,
    yb: torch.Tensor,
    gb: torch.Tensor,
    mixup_alpha: float,
    mixup_prob: float,
):
    if mixup_alpha <= 0.0 or mixup_prob <= 0.0:
        return xb, vb, yb, gb
    xb_out = xb.clone()
    vb_out = vb.clone()
    yb_out = yb.clone()
    for cancer_idx in range(len(CANCERS)):
        use = torch.nonzero(gb == cancer_idx, as_tuple=False).flatten()
        if use.numel() < 2:
            continue
        mixed_x, mixed_v, mixed_y = apply_bag_mixup(
            xb[use], vb[use], yb[use], mixup_alpha, mixup_prob
        )
        xb_out[use] = mixed_x
        vb_out[use] = mixed_v
        yb_out[use] = mixed_y
    return xb_out, vb_out, yb_out, gb


def train_epoch(
    model,
    loader,
    sampler,
    optim,
    loss_fn,
    device: torch.device,
    mixup_alpha: float,
    mixup_prob: float,
    epoch: int,
) -> float:
    model.train()
    sampler.set_epoch(epoch)
    total = 0.0
    n = 0
    for _, _, xb, vb, yb, gb, _ in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        gb = gb.to(device)
        xb, vb, yb, gb = apply_same_cancer_mixup(xb, vb, yb, gb, mixup_alpha, mixup_prob)
        optim.zero_grad(set_to_none=True)
        preds_all, _ = model(xb, vb, gb)
        losses = []
        for cancer_idx, cancer in enumerate(CANCERS):
            use = gb == cancer_idx
            if use.any():
                losses.append(loss_fn(preds_all[cancer][use], yb[use]))
        loss = torch.stack(losses).mean()
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_one_cancer(model, loader, cancer: str, loss_fn, device: torch.device):
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
        gb = torch.full((xb.shape[0],), CANCER_TO_IDX[cancer], dtype=torch.long, device=device)
        preds_all, attn = model(xb, vb, gb)
        pred = preds_all[cancer]
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


def save_attention_tables(out_root: Path, ids: list[str], patch_names_all: list[list[str]], attn_all: list[np.ndarray]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for sample_id, patch_names, attn in zip(ids, patch_names_all, attn_all):
        df = pd.DataFrame({"patch_file": patch_names, "attention_weight": attn})
        df = df.sort_values("attention_weight", ascending=False, kind="stable")
        safe_id = sample_id.replace("/", "_")
        df.to_csv(out_root / f"{safe_id}.csv", index=False)


def metric_row(split: str, cancer: str, corr_df: pd.DataFrame) -> dict:
    s = corr_df["pearson_r"]
    return {
        "split": split,
        "group": cancer,
        "n_enhancers": int(len(corr_df)),
        "pearson_mean": float(s.mean(skipna=True)),
        "pearson_median": float(s.median(skipna=True)),
        "gt_0.4": int((s > 0.4).sum()),
        "gt_0.5": int((s > 0.5).sum()),
        "gt_0.6": int((s > 0.6).sum()),
    }


def build_macro_row(split: str, rows: list[dict], all_corrs: list[pd.DataFrame]) -> dict:
    concat_df = pd.concat(all_corrs, ignore_index=True)
    base = metric_row(split, "ALL", concat_df)
    return base


def build_eval_loaders(
    feat_root: Path,
    label_root: Path,
    max_patches_eval: int,
    batch_size_eval: int,
    num_workers: int,
    seed: int,
):
    loaders = {"validation": {}, "test": {}}
    datasets = {"validation": {}, "test": {}}
    loader_kwargs = {
        "batch_size": batch_size_eval,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": lambda batch: (
            [item[0] for item in batch],
            torch.stack([item[1] for item in batch]),
            None,
            None,
            None,
        ),
    }
    del loader_kwargs
    for split in ["validation", "test"]:
        for cancer in CANCERS:
            ds = SingleCancerAttentionDataset(
                cancer,
                split,
                label_root / f"{cancer}_{split}_zscore_top3000_shared.csv",
                feat_root,
                max_patches_eval,
                seed,
            )
            datasets[split][cancer] = ds
            loaders[split][cancer] = DataLoader(
                ds,
                batch_size=batch_size_eval,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                collate_fn=lambda batch: (
                    [item[0] for item in batch],
                    torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True),
                    None,
                    torch.stack([item[2] for item in batch]),
                    [item[3] for item in batch],
                ),
            )
    return datasets, loaders


def collate_single_batch(batch):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size-per-cancer", type=int, default=2)
    parser.add_argument("--batch-size-eval", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--conditioning-mode", choices=["none", "film", "attn_bias"], default="none")
    parser.add_argument("--conditioning-dim", type=int, default=16)
    parser.add_argument("--max-patches-train", type=int, default=300)
    parser.add_argument("--max-patches-eval", type=int, default=300)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_root = Path(args.feat_root)
    label_root = Path(args.label_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sets: dict[str, SingleCancerAttentionDataset] = {}
    output_dims: dict[str, int] = {}
    enh_cols_map: dict[str, list[str]] = {}
    for cancer in CANCERS:
        ds = SingleCancerAttentionDataset(
            cancer,
            "train",
            label_root / f"{cancer}_train_zscore_top3000_shared.csv",
            feat_root,
            args.max_patches_train,
            args.seed,
        )
        train_sets[cancer] = ds
        output_dims[cancer] = int(ds.y.shape[1])
        enh_cols_map[cancer] = list(ds.enh_cols)
        print(f"[train] {cancer}: n={len(ds)} targets={output_dims[cancer]}", flush=True)

    eval_sets = {"validation": {}, "test": {}}
    eval_loaders = {"validation": {}, "test": {}}
    for split in ["validation", "test"]:
        for cancer in CANCERS:
            ds = SingleCancerAttentionDataset(
                cancer,
                split,
                label_root / f"{cancer}_{split}_zscore_top3000_shared.csv",
                feat_root,
                args.max_patches_eval,
                args.seed,
            )
            if ds.enh_cols != enh_cols_map[cancer]:
                raise RuntimeError(f"{cancer}: enhancer columns mismatch between train and {split}")
            eval_sets[split][cancer] = ds
            eval_loaders[split][cancer] = DataLoader(
                ds,
                batch_size=args.batch_size_eval,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.num_workers > 0,
                collate_fn=collate_single_batch,
            )
            print(f"[{split}] {cancer}: n={len(ds)}", flush=True)

    mixed_train_ds = MixedCancerTrainDataset(train_sets)
    train_sampler = BalancedCancerBatchSampler(
        mixed_train_ds.group_indices,
        batch_size_per_cancer=args.batch_size_per_cancer,
        seed=args.seed,
    )
    train_loader = DataLoader(
        mixed_train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_mixed_batch,
    )

    input_dim = infer_input_dim(feat_root, "BRCA")
    model = SharedAttentionCancerHeadRegressor(
        input_dim=input_dim,
        output_dims=output_dims,
        embed_dim=args.embed_dim,
        attn_dim=args.attn_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        conditioning_mode=args.conditioning_mode,
        conditioning_dim=args.conditioning_dim,
    ).to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            train_sampler,
            optim,
            loss_fn,
            device,
            args.mixup_alpha,
            args.mixup_prob,
            epoch,
        )
        val_losses = []
        for cancer in CANCERS:
            val_loss, _, _, _, _, _ = eval_one_cancer(
                model, eval_loaders["validation"][cancer], cancer, loss_fn, device
            )
            val_losses.append(val_loss)
        val_macro = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_loss": val_macro})
        if val_macro < (best_val - args.min_delta):
            best_val = val_macro
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_macro_loss={val_macro:.4f}", flush=True)
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

    summary_rows = []
    for split in ["validation", "test"]:
        split_corrs = []
        for cancer in CANCERS:
            split_out = out_dir / split / cancer
            split_out.mkdir(parents=True, exist_ok=True)
            split_loss, pred, true, ids, patch_names, attn = eval_one_cancer(
                model, eval_loaders[split][cancer], cancer, loss_fn, device
            )
            corr_df = pearson_per_feature(pred, true, enh_cols_map[cancer])
            split_corrs.append(corr_df.assign(cancer=cancer))
            summary_rows.append(metric_row(split, cancer, corr_df))
            np.save(split_out / "pred.npy", pred.astype(np.float32))
            np.save(split_out / "true.npy", true.astype(np.float32))
            np.save(split_out / "ids.npy", np.array(ids, dtype=object))
            np.save(split_out / "attention_weights.npy", np.array(attn, dtype=object), allow_pickle=True)
            np.save(split_out / "patch_names.npy", np.array(patch_names, dtype=object), allow_pickle=True)
            corr_df.to_csv(split_out / "per_enhancer_correlation.csv", index=False)
            save_attention_tables(split_out / "patch_attention", ids, patch_names, attn)
            print(
                f"[{split}] {cancer}: loss={split_loss:.4f} pearson_mean={float(corr_df['pearson_r'].mean(skipna=True)):.4f}",
                flush=True,
            )
        summary_rows.append(build_macro_row(split, summary_rows, split_corrs))

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_split_and_cancer.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    (out_dir / "enhancers_by_cancer.json").write_text(
        json.dumps({c: enh_cols_map[c] for c in CANCERS}, indent=2) + "\n"
    )
    pd.Series(
        {
            "feature_mode": "attention_pooling_shared_encoder_cancer_heads",
            "batch_size_per_cancer": args.batch_size_per_cancer,
            "batch_size_total": args.batch_size_per_cancer * len(CANCERS),
            "batch_size_eval": args.batch_size_eval,
            "num_workers": args.num_workers,
            "embed_dim": args.embed_dim,
            "attn_dim": args.attn_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "conditioning_mode": args.conditioning_mode,
            "conditioning_dim": args.conditioning_dim,
            "max_patches_train": args.max_patches_train,
            "max_patches_eval": args.max_patches_eval,
            "mixup_alpha": args.mixup_alpha,
            "mixup_prob": args.mixup_prob,
            "seed": args.seed,
            "early_patience": args.early_patience,
            "min_delta": args.min_delta,
            "feat_root": str(feat_root),
            "label_root": str(label_root),
        }
    ).to_json(out_dir / "run_config.json")

    print(f"Saved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
