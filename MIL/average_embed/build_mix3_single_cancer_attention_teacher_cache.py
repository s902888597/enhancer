#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_mean_regression import set_seed
from run_skcm_top1000_attention_regression import (
    SingleCancerAttentionDataset,
    collate_batch,
    infer_input_dim,
)


DEFAULT_SPECS = {
    "BRCA": {
        "train_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/BRCA_train_zscore_top3000_shared.csv",
        "val_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/BRCA_validation_zscore_top3000_shared.csv",
        "test_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/BRCA_test_zscore_top3000_shared.csv",
        "ckpt": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/single_cancer_shareddata_top3000_attention_kmixup_multi/BRCA/BRCA_shareddata_top3000_attention_k300_mixup_seed44/best_model.pt",
        "max_patches": 300,
    },
    "LUAD": {
        "train_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/LUAD_train_zscore_top3000_shared.csv",
        "val_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/LUAD_validation_zscore_top3000_shared.csv",
        "test_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/LUAD_test_zscore_top3000_shared.csv",
        "ckpt": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/single_cancer_shareddata_top3000_attention_kmixup_multi/LUAD/LUAD_shareddata_top3000_attention_k300_mixup_seed44/best_model.pt",
        "max_patches": 300,
    },
    "SKCM": {
        "train_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/SKCM_train_zscore_top3000_shared.csv",
        "val_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/SKCM_validation_zscore_top3000_shared.csv",
        "test_csv": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows/SKCM_test_zscore_top3000_shared.csv",
        "ckpt": "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/single_cancer_shareddata_top3000_attention_kmixup_multi/SKCM/SKCM_shareddata_top3000_attention_k1600_mixup_seed44/best_model.pt",
        "max_patches": 1600,
    },
}


class LegacyAttentionTeacher(nn.Module):
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

    def encode(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        h_patch = self.patch_embed(x)
        a_v = torch.tanh(self.attention_v(h_patch))
        a_u = torch.sigmoid(self.attention_u(h_patch))
        scores = self.attention_w(a_v * a_u).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=1)
        bag = torch.bmm(attn.unsqueeze(1), h_patch).squeeze(1)
        return self.regressor[:-1](bag)


def build_loader(
    cancer: str,
    split: str,
    csv_path: Path,
    feat_root: Path,
    max_patches: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> tuple[SingleCancerAttentionDataset, DataLoader]:
    ds = SingleCancerAttentionDataset(cancer, split, csv_path, feat_root, max_patches, seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader


def infer_output_dim(ckpt_path: Path) -> int:
    state = torch.load(ckpt_path, map_location="cpu")
    return int(state["regressor.6.weight"].shape[0])


@torch.no_grad()
def encode_loader(model: LegacyAttentionTeacher, loader: DataLoader, device: torch.device, cancer: str) -> tuple[np.ndarray, list[str]]:
    model.eval()
    all_x: list[np.ndarray] = []
    all_ids: list[str] = []
    for ids, xb, vb, _, _ in loader:
        xb = xb.to(device)
        vb = vb.to(device)
        h = model.encode(xb, vb).cpu().numpy().astype(np.float32)
        all_x.append(h)
        all_ids.extend([f"{cancer}:{sid}" for sid in ids])
    x = np.concatenate(all_x, axis=0) if all_x else np.zeros((0, 0), dtype=np.float32)
    return x, all_ids


def build_teacher_for_cancer(cancer: str, spec: dict, feat_root: Path, device: torch.device, batch_size: int, num_workers: int, seed: int):
    ckpt_path = Path(spec["ckpt"])
    input_dim = infer_input_dim(feat_root, cancer)
    output_dim = infer_output_dim(ckpt_path)
    model = LegacyAttentionTeacher(
        input_dim=input_dim,
        output_dim=output_dim,
        embed_dim=256,
        attn_dim=128,
        hidden_dim=512,
        dropout=0.2,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    split_to_encoded = {}
    for split, key in [("train", "train_csv"), ("validation", "val_csv"), ("test", "test_csv")]:
        _, loader = build_loader(
            cancer=cancer,
            split=split,
            csv_path=Path(spec[key]),
            feat_root=feat_root,
            max_patches=int(spec["max_patches"]),
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        split_to_encoded[split] = encode_loader(model, loader, device, cancer)
    return split_to_encoded


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feat-root", default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=44)
    args = p.parse_args()

    set_seed(args.seed)
    feat_root = Path(args.feat_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combined: dict[str, dict[str, list]] = {
        "train": {"x": [], "ids": []},
        "validation": {"x": [], "ids": []},
        "test": {"x": [], "ids": []},
    }
    summary: dict[str, dict] = {"device": str(device), "feature_root": str(feat_root), "teacher_hidden_dim": 512}

    for cancer in ["BRCA", "LUAD", "SKCM"]:
        spec = DEFAULT_SPECS[cancer]
        split_to_encoded = build_teacher_for_cancer(
            cancer=cancer,
            spec=spec,
            feat_root=feat_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        summary[cancer] = {"ckpt": spec["ckpt"], "max_patches": int(spec["max_patches"])}
        for split in ["train", "validation", "test"]:
            x, ids = split_to_encoded[split]
            combined[split]["x"].append(x)
            combined[split]["ids"].extend(ids)
            summary[cancer][f"n_{split}"] = int(len(ids))

    for split in ["train", "validation", "test"]:
        xs = combined[split]["x"]
        x = np.concatenate(xs, axis=0).astype(np.float32)
        ids = combined[split]["ids"]
        np.save(out_dir / f"{split}_X.npy", x)
        (out_dir / f"{split}_ids.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
        summary[split] = {"n_samples": int(len(ids)), "feature_dim": int(x.shape[1])}

    (out_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
