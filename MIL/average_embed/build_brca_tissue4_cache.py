#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_CACHE = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/common361_best3_mean_mlp/cache")
TISSUE_CSV = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/tissue_proxy/brca_loki_k12/composition_wide_userannotated_v1.csv")
OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/tissue4_brca_mean_cache")

FILES = {
    "train": (
        "BRCA_train_Top1000_SEs_192train_eRNA_zscore_3eRNAremoved_X.npy",
        "BRCA_train_Top1000_SEs_192train_eRNA_zscore_3eRNAremoved_ids.txt",
    ),
    "validation": (
        "BRCA_validation_Top1000_SEs_63val_eRNA_zscore_3eRNAremoved_X.npy",
        "BRCA_validation_Top1000_SEs_63val_eRNA_zscore_3eRNAremoved_ids.txt",
    ),
    "test": (
        "BRCA_test_Top1000_SEs_64test_eRNA_zscore_3eRNAremoved_X.npy",
        "BRCA_test_Top1000_SEs_64test_eRNA_zscore_3eRNAremoved_ids.txt",
    ),
}

TISSUE_FEATURES = [
    "stroma_fibrous_frac_after_drop",
    "tumor_high_cell_density_frac_after_drop",
    "tumor_stroma_invasive_mix_frac_after_drop",
    "inflammation_lymphocyte_frac_after_drop",
]


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def canonical_tcga_id(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return str(sample_id)


def main() -> None:
    df = pd.read_csv(TISSUE_CSV)
    df["canonical_id"] = df["case_id"].map(canonical_tcga_id)
    if df["canonical_id"].duplicated().any():
        dup = df.loc[df["canonical_id"].duplicated(), "canonical_id"].tolist()[:5]
        raise RuntimeError(f"duplicated canonical ids, e.g. {dup}")
    for col in TISSUE_FEATURES:
        if col not in df.columns:
            raise RuntimeError(f"missing column {col}")
    df = df.set_index("canonical_id")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_ids = {split: read_ids(BASE_CACHE / ids_name) for split, (_, ids_name) in FILES.items()}
    raw = {}
    missing = {}
    for split, ids in split_ids.items():
        missing[split] = [sid for sid in ids if canonical_tcga_id(sid) not in df.index]
        if missing[split]:
            raise RuntimeError(f"{split}: missing tissue rows, e.g. {missing[split][:5]}")
        raw[split] = df.loc[[canonical_tcga_id(sid) for sid in ids], TISSUE_FEATURES].to_numpy(dtype=np.float32)

    train_mean = raw["train"].mean(axis=0)
    train_std = raw["train"].std(axis=0)
    train_std[train_std < 1e-8] = 1.0
    z = {split: ((raw[split] - train_mean) / train_std).astype(np.float32) for split in FILES}

    summary = {
        "tissue_features": TISSUE_FEATURES,
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "splits": {},
    }
    for split, (x_name, ids_name) in FILES.items():
        np.save(OUT_DIR / x_name, z[split])
        (OUT_DIR / ids_name).write_text("\n".join(split_ids[split]) + "\n")
        summary["splits"][split] = {
            "n_ids": len(split_ids[split]),
            "feature_dim": int(z[split].shape[1]),
        }

    (OUT_DIR / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "tissue_feature_names.txt").write_text("\n".join(TISSUE_FEATURES) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
