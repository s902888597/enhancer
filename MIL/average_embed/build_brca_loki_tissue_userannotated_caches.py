#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_CACHE = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/common361_best3_mean_mlp/cache")
LOKI_CACHE = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/loki_brca_mean_cache")
TISSUE_CSV = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/tissue_proxy/brca_loki_k12/composition_wide_userannotated_v1.csv")
TISSUE_ONLY_OUT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/tissue_brca_userannotated_mean_cache")
CONCAT_OUT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/loki_tissue_brca_userannotated_mean_cache")

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
    "artifact_frac_raw",
    "stroma_fibrous_frac_after_drop",
    "tumor_high_cell_density_frac_after_drop",
    "tumor_stroma_invasive_mix_frac_after_drop",
    "inflammation_lymphocyte_frac_after_drop",
    "unlabeled_after_drop_frac",
]


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def canonical_tcga_id(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return str(sample_id)


def load_tissue_df() -> pd.DataFrame:
    df = pd.read_csv(TISSUE_CSV)
    missing = [c for c in TISSUE_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing tissue columns: {missing}")
    df["canonical_id"] = df["case_id"].map(canonical_tcga_id)
    if df["canonical_id"].duplicated().any():
        dup = df.loc[df["canonical_id"].duplicated(), "canonical_id"].tolist()[:5]
        raise RuntimeError(f"duplicated canonical tissue ids, e.g. {dup}")
    return df.set_index("canonical_id")


def main() -> None:
    tissue_df = load_tissue_df()
    TISSUE_ONLY_OUT.mkdir(parents=True, exist_ok=True)
    CONCAT_OUT.mkdir(parents=True, exist_ok=True)

    split_ids = {split: read_ids(BASE_CACHE / ids_name) for split, (_, ids_name) in FILES.items()}
    missing = {
        split: [sid for sid in ids if canonical_tcga_id(sid) not in tissue_df.index]
        for split, ids in split_ids.items()
    }
    if any(missing.values()):
        raise RuntimeError(f"missing tissue rows: {missing}")

    raw_tissue = {
        split: tissue_df.loc[[canonical_tcga_id(sid) for sid in split_ids[split]], TISSUE_FEATURES].to_numpy(dtype=np.float32)
        for split in FILES
    }
    train_mean = raw_tissue["train"].mean(axis=0)
    train_std = raw_tissue["train"].std(axis=0)
    train_std[train_std < 1e-8] = 1.0
    tissue_z = {
        split: ((raw_tissue[split] - train_mean) / train_std).astype(np.float32)
        for split in FILES
    }

    summary = {
        "tissue_features": TISSUE_FEATURES,
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "splits": {},
    }

    for split, (x_name, ids_name) in FILES.items():
        loki_x = np.load(LOKI_CACHE / x_name).astype(np.float32)
        if loki_x.shape[0] != len(split_ids[split]):
            raise RuntimeError(f"{split}: loki X/ids mismatch")
        concat_x = np.concatenate([loki_x, tissue_z[split]], axis=1).astype(np.float32)

        np.save(TISSUE_ONLY_OUT / x_name, tissue_z[split])
        np.save(CONCAT_OUT / x_name, concat_x)
        (TISSUE_ONLY_OUT / ids_name).write_text("\n".join(split_ids[split]) + "\n")
        (CONCAT_OUT / ids_name).write_text("\n".join(split_ids[split]) + "\n")

        summary["splits"][split] = {
            "n_ids": len(split_ids[split]),
            "tissue_dim": int(tissue_z[split].shape[1]),
            "loki_dim": int(loki_x.shape[1]),
            "concat_dim": int(concat_x.shape[1]),
        }

    (TISSUE_ONLY_OUT / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    (CONCAT_OUT / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    (TISSUE_ONLY_OUT / "tissue_feature_names.txt").write_text("\n".join(TISSUE_FEATURES) + "\n")
    (CONCAT_OUT / "tissue_feature_names.txt").write_text("\n".join(TISSUE_FEATURES) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
