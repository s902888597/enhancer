#!/usr/bin/env python3
"""
Attention-pooling dual-head multi-cancer regression for a union target panel with per-sample masks.

Inputs:
- Existing mixed matrix directory for targets/masks/groups/ids
- Patch-level feature root under features_patches_pan_cancer_npy/<split>/<cancer>/<case_dir>/*.npy

Model:
- patch embed -> gated attention pooling -> shared bag representation
- one pan-cancer head
- one cancer-specific head per cancer
- optional sample-wise fusion gate
"""

import argparse
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import NMF, PCA
from torch.utils.data import DataLoader, Dataset

from run_mean_regression_pan_cancer import set_seed, summarize_corr


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
    bad: list[str] = []
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
        top_cancer_root = feat_root / cancer
        if cancer_root.exists():
            case_map = build_case_to_feature_dir(cancer_root)
            if case_map:
                maps[cancer] = case_map
                continue
        if top_cancer_root.exists():
            maps[cancer] = build_case_to_feature_dir(top_cancer_root)
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
        teacher: Optional[np.ndarray],
        teacher_valid: Optional[np.ndarray],
        max_patches: Optional[int],
        seed: int,
    ):
        self.split = split
        self.ids = ids
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.g = torch.tensor(groups, dtype=torch.long)
        if teacher is None:
            teacher = np.zeros((len(ids), 0), dtype=np.float32)
        if teacher_valid is None:
            teacher_valid = np.zeros(len(ids), dtype=np.float32)
        self.teacher = torch.tensor(teacher, dtype=torch.float32)
        self.teacher_valid = torch.tensor(teacher_valid, dtype=torch.float32)
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
        return x, self.y[idx], self.mask[idx], self.g[idx], self.teacher[idx], self.teacher_valid[idx]


def collate_batch(batch):
    xs, ys, ms, gs, ts, tv = zip(*batch)
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
    return tokens, valid, torch.stack(ys), torch.stack(ms), torch.stack(gs), torch.stack(ts), torch.stack(tv)


class AttentionDualHeadRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        attn_dim: int,
        hidden_dim: int,
        dropout: float,
        use_gating: bool,
        teacher_dim: int = 0,
    ):
        super().__init__()
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
        self.pan_head = nn.Linear(hidden_dim, output_dim)
        self.spec_heads = nn.ModuleDict({c: nn.Linear(hidden_dim, output_dim) for c in CANCERS})
        self.use_gating = use_gating
        self.teacher_dim = teacher_dim
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        self.teacher_head = nn.Linear(hidden_dim, teacher_dim) if teacher_dim > 0 else None

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, groups: torch.Tensor):
        h_patch = self.patch_embed(x)
        a_v = torch.tanh(self.attention_v(h_patch))
        a_u = torch.sigmoid(self.attention_u(h_patch))
        scores = self.attention_w(a_v * a_u).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=1)
        bag = torch.bmm(attn.unsqueeze(1), h_patch).squeeze(1)
        h = self.backbone(bag)
        pan = self.pan_head(h)
        spec = torch.zeros_like(pan)
        for idx, cancer in enumerate(CANCERS):
            use = groups == idx
            if use.any():
                spec[use] = self.spec_heads[cancer](h[use])
        gate = self.gate(h) if self.use_gating else None
        teacher = self.teacher_head(h) if self.teacher_head is not None else None
        return pan, spec, gate, attn, teacher


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


def masked_row_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    use = valid > 0.5
    if not use.any():
        return pred.new_tensor(0.0)
    diff = pred[use] - target[use]
    return diff.pow(2).mean()


def attention_entropy(attn: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    attn_safe = attn.clamp_min(1e-8)
    entropy = -(attn_safe * attn_safe.log()) * valid_mask.float()
    entropy = entropy.sum(dim=1)
    norm = torch.log(valid_mask.sum(dim=1).clamp_min(2).float())
    entropy = entropy / norm
    return entropy.mean()


def fuse_outputs(pan: torch.Tensor, spec: torch.Tensor, gate: Optional[torch.Tensor], alpha: float):
    if gate is None:
        return alpha * spec + (1.0 - alpha) * pan
    return gate * spec + (1.0 - gate) * pan


def build_same_cancer_partners(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    partners = torch.arange(groups.shape[0], device=groups.device)
    active = torch.zeros(groups.shape[0], dtype=torch.bool, device=groups.device)
    for cancer_idx in range(len(CANCERS)):
        idx = torch.nonzero(groups == cancer_idx, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        perm = idx[torch.randperm(idx.numel(), device=groups.device)]
        partners[perm] = torch.roll(perm, shifts=-1)
        active[perm] = True
    return partners, active


def sample_mixed_bag(
    xa: torch.Tensor,
    va: torch.Tensor,
    xb: torch.Tensor,
    vb: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_a = torch.nonzero(va, as_tuple=False).flatten()
    idx_b = torch.nonzero(vb, as_tuple=False).flatten()
    if idx_a.numel() == 0 or idx_b.numel() == 0:
        return xa, va

    total_k = max(int(idx_a.numel()), int(idx_b.numel()))
    want_a = int(round(lam * total_k))
    want_a = min(max(want_a, 1), total_k - 1 if total_k > 1 else 1)
    want_b = total_k - want_a
    want_a = min(want_a, int(idx_a.numel()))
    want_b = min(want_b, int(idx_b.numel()))

    chosen_a = idx_a[torch.randperm(idx_a.numel(), device=xa.device)[:want_a]]
    chosen_b = idx_b[torch.randperm(idx_b.numel(), device=xb.device)[:want_b]]
    chosen = torch.cat([xa[chosen_a], xb[chosen_b]], dim=0)
    if chosen.shape[0] < total_k:
        rem_a = idx_a[torch.randperm(idx_a.numel(), device=xa.device)]
        rem_b = idx_b[torch.randperm(idx_b.numel(), device=xb.device)]
        remain = total_k - chosen.shape[0]
        extra = []
        if want_a < idx_a.numel():
            extra.append(xa[rem_a[want_a:want_a + remain]])
        if sum(t.shape[0] for t in extra) < remain and want_b < idx_b.numel():
            need = remain - sum(t.shape[0] for t in extra)
            extra.append(xb[rem_b[want_b:want_b + need]])
        if extra:
            chosen = torch.cat([chosen] + extra, dim=0)
    chosen = chosen[:total_k]

    mixed_x = torch.zeros_like(xa)
    mixed_v = torch.zeros_like(va)
    mixed_x[:chosen.shape[0]] = chosen
    mixed_v[:chosen.shape[0]] = True
    return mixed_x, mixed_v


def apply_same_cancer_bag_mixup(
    xb: torch.Tensor,
    vb: torch.Tensor,
    yb: torch.Tensor,
    mb: torch.Tensor,
    gb: torch.Tensor,
    mixup_alpha: float,
    mixup_prob: float,
):
    if mixup_alpha <= 0.0 or mixup_prob <= 0.0 or xb.shape[0] < 2:
        return xb, vb, yb, mb, gb

    partners, active = build_same_cancer_partners(gb)
    use = active & (torch.rand(xb.shape[0], device=xb.device) < mixup_prob)
    if not use.any():
        return xb, vb, yb, mb, gb

    beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
    lam_all = beta.sample((xb.shape[0],)).to(device=xb.device, dtype=xb.dtype)

    xb_out = xb.clone()
    vb_out = vb.clone()
    yb_out = yb.clone()
    mb_out = mb.clone()

    for i in torch.nonzero(use, as_tuple=False).flatten().tolist():
        j = int(partners[i].item())
        lam = float(lam_all[i].item())
        mixed_x, mixed_v = sample_mixed_bag(xb[i], vb[i], xb[j], vb[j], lam)
        xb_out[i] = mixed_x
        vb_out[i] = mixed_v
        yb_out[i] = lam * yb[i] + (1.0 - lam) * yb[j]
        mb_out[i] = torch.minimum(mb[i], mb[j])

    return xb_out, vb_out, yb_out, mb_out, gb


def train_epoch(
    model,
    loader,
    optim,
    device,
    w_spec,
    w_pan,
    w_cons,
    alpha,
    mixup_alpha,
    mixup_prob,
    teacher_lambda,
    attention_entropy_weight,
):
    model.train()
    total = 0.0
    n = 0
    for xb, vb, yb, mb, gb, tb, tv in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        gb = gb.to(device)
        tb = tb.to(device)
        tv = tv.to(device)
        xb, vb, yb, mb, gb = apply_same_cancer_bag_mixup(xb, vb, yb, mb, gb, mixup_alpha, mixup_prob)
        optim.zero_grad(set_to_none=True)
        pan, spec, gate, attn, teacher_pred = model(xb, vb, gb)
        fused = fuse_outputs(pan, spec, gate, alpha)
        loss_pan = masked_mse(pan, yb, mb)
        loss_spec = masked_mse(spec, yb, mb)
        loss_cons = masked_mse(spec, pan, mb)
        loss_fused = masked_mse(fused, yb, mb)
        loss_teacher = masked_row_mse(teacher_pred, tb, tv) if teacher_pred is not None else xb.new_tensor(0.0)
        loss_entropy = attention_entropy(attn, vb)
        loss = (
            loss_fused
            + w_spec * loss_spec
            + w_pan * loss_pan
            + w_cons * loss_cons
            + teacher_lambda * loss_teacher
            + attention_entropy_weight * loss_entropy
        )
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, alpha, teacher_lambda, attention_entropy_weight):
    model.eval()
    total = 0.0
    n = 0
    pan_preds = []
    spec_preds = []
    fused_preds = []
    trues = []
    masks = []
    groups = []
    for xb, vb, yb, mb, gb, tb, tv in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        gb = gb.to(device)
        tb = tb.to(device)
        tv = tv.to(device)
        pan, spec, gate, attn, teacher_pred = model(xb, vb, gb)
        fused = fuse_outputs(pan, spec, gate, alpha)
        loss = masked_mse(fused, yb, mb)
        if teacher_pred is not None:
            loss = loss + teacher_lambda * masked_row_mse(teacher_pred, tb, tv)
        loss = loss + attention_entropy_weight * attention_entropy(attn, vb)
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
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


def per_pc_prediction_corr(preds: np.ndarray, trues: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(preds.shape[1]):
        x = preds[:, i]
        y = trues[:, i]
        if len(x) < 2 or np.std(x) < 1e-6 or np.std(y) < 1e-6:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"pc": i + 1, "pearson_r": r})
    return pd.DataFrame(rows)


def predicted_pc_metric_row(split: str, group: str, corr_df: pd.DataFrame) -> dict:
    s = corr_df["pearson_r"]
    return {
        "split": split,
        "group": group,
        "n_pcs": int(len(corr_df)),
        "pearson_mean": float(s.mean(skipna=True)),
        "pearson_median": float(s.median(skipna=True)),
        "pearson_min": float(s.min(skipna=True)),
        "pearson_max": float(s.max(skipna=True)),
    }


def summarize_pc_prediction_set(
    split_name: str,
    pred_name: str,
    preds: np.ndarray,
    trues: np.ndarray,
    groups: np.ndarray,
    out_dir: Path,
    summary_rows: list[dict],
):
    overall_df = per_pc_prediction_corr(preds, trues)
    summary_rows.append(predicted_pc_metric_row(split_name, f"{pred_name}_ALL", overall_df))
    overall_df.to_csv(out_dir / f"predicted_pc_correlation_{split_name}_{pred_name}_all.csv", index=False)
    for idx, cancer in enumerate(CANCERS):
        use = groups == idx
        corr_df = per_pc_prediction_corr(preds[use], trues[use])
        summary_rows.append(predicted_pc_metric_row(split_name, f"{pred_name}_{cancer}", corr_df))
        corr_df.to_csv(out_dir / f"predicted_pc_correlation_{split_name}_{pred_name}_{cancer}.csv", index=False)


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


def load_teacher_cache_split(teacher_cache: Path, split: str, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    combined_x = teacher_cache / f"{split}_X.npy"
    combined_ids = teacher_cache / f"{split}_ids.txt"
    if combined_x.exists() and combined_ids.exists():
        teacher_x = np.load(combined_x).astype(np.float32)
        teacher_ids = [line.strip() for line in combined_ids.read_text().splitlines() if line.strip()]
        teacher_map = {sid: teacher_x[i] for i, sid in enumerate(teacher_ids)}
        out = np.zeros((len(ids), teacher_x.shape[1]), dtype=np.float32)
        valid = np.zeros(len(ids), dtype=np.float32)
        for i, sample_id in enumerate(ids):
            vec = teacher_map.get(sample_id)
            if vec is None:
                continue
            out[i] = vec
            valid[i] = 1.0
        return out, valid

    file_map = {
        "train": ("SKCM_train_train_X.npy", "SKCM_train_train_ids.txt"),
        "validation": ("SKCM_validation_validation_X.npy", "SKCM_validation_validation_ids.txt"),
        "test": ("SKCM_test_test_X.npy", "SKCM_test_test_ids.txt"),
    }
    x_name, ids_name = file_map[split]
    teacher_x = np.load(teacher_cache / x_name).astype(np.float32)
    teacher_ids = [line.strip() for line in (teacher_cache / ids_name).read_text().splitlines() if line.strip()]
    teacher_map = {sid: teacher_x[i] for i, sid in enumerate(teacher_ids)}
    out = np.zeros((len(ids), teacher_x.shape[1]), dtype=np.float32)
    valid = np.zeros(len(ids), dtype=np.float32)
    for i, sample_id in enumerate(ids):
        cancer, case = sample_id.split(":", 1)
        if cancer != "SKCM":
            continue
        vec = teacher_map.get(case)
        if vec is None:
            continue
        out[i] = vec
        valid[i] = 1.0
    return out, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--feat-root", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-patches", type=int, default=100)
    parser.add_argument("--early-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--loss-weight-specific", type=float, default=1.0)
    parser.add_argument("--loss-weight-pan", type=float, default=1.0)
    parser.add_argument("--loss-weight-consistency", type=float, default=0.1)
    parser.add_argument("--fusion-alpha", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--use-gating", action="store_true")
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--teacher-lambda", type=float, default=0.0)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.0)
    parser.add_argument("--pca-k", type=int, default=0)
    parser.add_argument("--nmf-k", type=int, default=0)
    parser.add_argument("--nmf-max-iter", type=int, default=1000)
    parser.add_argument("--nmf-clip-nonnegative", action="store_true")
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

    if args.pca_k > 0 and args.nmf_k > 0:
        raise RuntimeError("Use only one reducer: --pca-k or --nmf-k, not both.")

    pca = None
    nmf = None
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
    elif args.nmf_k > 0:
        if args.nmf_k >= y_train.shape[1]:
            raise RuntimeError(f"--nmf-k must be smaller than output dim ({y_train.shape[1]}), got {args.nmf_k}")
        y_train_nmf, y_val_nmf, y_test_nmf = y_train, y_val, y_test
        if args.nmf_clip_nonnegative:
            y_train_nmf = np.clip(y_train_nmf, 0.0, None)
            y_val_nmf = np.clip(y_val_nmf, 0.0, None)
            y_test_nmf = np.clip(y_test_nmf, 0.0, None)
        elif float(y_train.min()) < -1e-8 or float(y_val.min()) < -1e-8 or float(y_test.min()) < -1e-8:
            raise RuntimeError("NMF requires non-negative labels.")
        nmf = NMF(
            n_components=args.nmf_k,
            init="nndsvda",
            random_state=args.seed,
            max_iter=args.nmf_max_iter,
            solver="cd",
            beta_loss="frobenius",
        )
        y_train_fit = nmf.fit_transform(y_train_nmf).astype(np.float32)
        y_val_fit = nmf.transform(y_val_nmf).astype(np.float32)
        y_test_fit = nmf.transform(y_test_nmf).astype(np.float32)
        mask_train_fit = np.ones_like(y_train_fit, dtype=np.float32)
        mask_val_fit = np.ones_like(y_val_fit, dtype=np.float32)
        mask_test_fit = np.ones_like(y_test_fit, dtype=np.float32)
    else:
        y_train_fit, y_val_fit, y_test_fit = y_train, y_val, y_test
        mask_train_fit, mask_val_fit, mask_test_fit = mask_train, mask_val, mask_test

    teacher_dim = 0
    teacher_train = teacher_val = teacher_test = None
    teacher_train_valid = teacher_val_valid = teacher_test_valid = None
    if args.teacher_cache and args.teacher_lambda > 0.0:
        teacher_cache = Path(args.teacher_cache)
        teacher_train, teacher_train_valid = load_teacher_cache_split(teacher_cache, "train", train_ids)
        teacher_val, teacher_val_valid = load_teacher_cache_split(teacher_cache, "validation", val_ids)
        teacher_test, teacher_test_valid = load_teacher_cache_split(teacher_cache, "test", test_ids)
        teacher_dim = int(teacher_train.shape[1])
        use = teacher_train_valid > 0.5
        teacher_mean = teacher_train[use].mean(axis=0, dtype=np.float64).astype(np.float32)
        teacher_std = teacher_train[use].std(axis=0, dtype=np.float64).astype(np.float32)
        teacher_std[teacher_std < 1e-8] = 1.0
        teacher_train = ((teacher_train - teacher_mean) / teacher_std).astype(np.float32)
        teacher_val = ((teacher_val - teacher_mean) / teacher_std).astype(np.float32)
        teacher_test = ((teacher_test - teacher_mean) / teacher_std).astype(np.float32)
        np.save(out_dir / "teacher_train_mean.npy", teacher_mean)
        np.save(out_dir / "teacher_train_std.npy", teacher_std)

    train_ds = TokenMatrixDataset("train", feat_root, train_ids, y_train_fit, mask_train_fit, train_groups, teacher_train, teacher_train_valid, args.max_patches, args.seed)
    val_ds = TokenMatrixDataset("validation", feat_root, val_ids, y_val_fit, mask_val_fit, val_groups, teacher_val, teacher_val_valid, args.max_patches, args.seed)
    test_ds = TokenMatrixDataset("test", feat_root, test_ids, y_test_fit, mask_test_fit, test_groups, teacher_test, teacher_test_valid, args.max_patches, args.seed)

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
    model = AttentionDualHeadRegressor(
        input_dim=input_dim,
        output_dim=y_train_fit.shape[1],
        embed_dim=args.embed_dim,
        attn_dim=args.attn_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_gating=args.use_gating,
        teacher_dim=teacher_dim,
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
            args.fusion_alpha,
            args.mixup_alpha,
            args.mixup_prob,
            args.teacher_lambda,
            args.attention_entropy_weight,
        )
        val_loss, _, _, _, _, _, _ = eval_epoch(
            model, val_loader, device, args.fusion_alpha, args.teacher_lambda, args.attention_entropy_weight
        )
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

    val_loss, val_pan_pred, val_spec_pred, val_fused_pred, val_true, val_mask, val_groups = eval_epoch(
        model, val_loader, device, args.fusion_alpha, args.teacher_lambda, args.attention_entropy_weight
    )
    test_loss, test_pan_pred, test_spec_pred, test_fused_pred, test_true, test_mask, test_groups = eval_epoch(
        model, test_loader, device, args.fusion_alpha, args.teacher_lambda, args.attention_entropy_weight
    )
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    summary_rows = []
    pc_summary_rows = []
    for split_name, pred_sets, trues_fit, masks_fit, groups, trues_orig, masks_orig in [
        (
            "validation",
            {"pan": val_pan_pred, "specific": val_spec_pred, "fused": val_fused_pred},
            val_true,
            val_mask,
            val_groups,
            y_val,
            mask_val,
        ),
        (
            "test",
            {"pan": test_pan_pred, "specific": test_spec_pred, "fused": test_fused_pred},
            test_true,
            test_mask,
            test_groups,
            y_test,
            mask_test,
        ),
    ]:
        for pred_name, preds in pred_sets.items():
            if pca is not None:
                summarize_pc_prediction_set(split_name, pred_name, preds, trues_fit, groups, out_dir, pc_summary_rows)
                preds_orig = pca.inverse_transform(preds).astype(np.float32)
                trues_for_summary = trues_orig
                masks_for_summary = masks_orig
            elif nmf is not None:
                summarize_pc_prediction_set(split_name, pred_name, preds, trues_fit, groups, out_dir, pc_summary_rows)
                preds_orig = nmf.inverse_transform(np.maximum(preds, 0.0)).astype(np.float32)
                trues_for_summary = trues_orig
                masks_for_summary = masks_orig
            else:
                preds_orig = preds
                trues_for_summary = trues_fit
                masks_for_summary = masks_fit
            summarize_prediction_set(
                split_name,
                pred_name,
                preds_orig,
                trues_for_summary,
                masks_for_summary,
                groups,
                enh_ref,
                out_dir,
                summary_rows,
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_split_and_cancer.csv", index=False)
    if pc_summary_rows:
        pd.DataFrame(pc_summary_rows).to_csv(out_dir / "predicted_pc_summary_by_split_and_cancer.csv", index=False)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    pd.Series(
        {
            "feature_mode": "attention_pooling",
            "max_patches": args.max_patches,
            "embed_dim": args.embed_dim,
            "attn_dim": args.attn_dim,
            "hidden_dim": args.hidden_dim,
            "early_patience": args.early_patience,
            "min_delta": args.min_delta,
            "num_workers": args.num_workers,
            "use_gating": args.use_gating,
            "fusion_alpha": args.fusion_alpha,
            "mixup_alpha": args.mixup_alpha,
            "mixup_prob": args.mixup_prob,
            "loss_weight_specific": args.loss_weight_specific,
            "loss_weight_pan": args.loss_weight_pan,
            "loss_weight_consistency": args.loss_weight_consistency,
            "teacher_cache": args.teacher_cache,
            "teacher_lambda": args.teacher_lambda,
            "teacher_dim": teacher_dim,
            "attention_entropy_weight": args.attention_entropy_weight,
            "pca_k": args.pca_k,
            "nmf_k": args.nmf_k,
            "nmf_max_iter": args.nmf_max_iter,
            "nmf_clip_nonnegative": args.nmf_clip_nonnegative,
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
    if pca is not None:
        np.save(out_dir / "y_pca_train.npy", y_train_fit)
        np.save(out_dir / "y_pca_validation.npy", y_val_fit)
        np.save(out_dir / "y_pca_test.npy", y_test_fit)
        np.save(out_dir / "pca_components.npy", pca.components_.astype(np.float32))
        np.save(out_dir / "pca_mean.npy", pca.mean_.astype(np.float32))
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))
    if nmf is not None:
        np.save(out_dir / "y_nmf_train.npy", y_train_fit)
        np.save(out_dir / "y_nmf_validation.npy", y_val_fit)
        np.save(out_dir / "y_nmf_test.npy", y_test_fit)
        np.save(out_dir / "nmf_components.npy", nmf.components_.astype(np.float32))
        np.save(out_dir / "nmf_reconstruction_err.npy", np.array([nmf.reconstruction_err_], dtype=np.float32))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
