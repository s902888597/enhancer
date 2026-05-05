#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


BASE_CACHE = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/common361_best3_mean_mlp/cache")
LOKI_ROOT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_loki_brca_packed")
OUT_DIR = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/loki_brca_mean_cache")

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


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def mean_embed(case_dir: Path) -> np.ndarray:
    x = np.load(case_dir / "X.npy").astype(np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise RuntimeError(f"bad feature matrix in {case_dir}")
    return x.mean(axis=0).astype(np.float32)


def loki_case_dir(sample_id: str) -> Path:
    return LOKI_ROOT / f"{sample_id}-01Z-00-DX1"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, (x_name, ids_name) in FILES.items():
        ids = read_ids(BASE_CACHE / ids_name)
        feats = []
        missing = []
        for sid in ids:
            case_dir = loki_case_dir(sid)
            if not (case_dir / "X.npy").exists():
                missing.append(sid)
                continue
            feats.append(mean_embed(case_dir))
        if missing:
            raise RuntimeError(f"{split}: missing {len(missing)} Loki cases, e.g. {missing[:5]}")
        x = np.stack(feats, axis=0).astype(np.float32)
        np.save(OUT_DIR / x_name, x)
        (OUT_DIR / ids_name).write_text("\n".join(ids) + "\n")
        summary[split] = {
            "n_ids": len(ids),
            "feature_dim": int(x.shape[1]),
            "example_case_dir": str(loki_case_dir(ids[0])),
        }

    (OUT_DIR / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
