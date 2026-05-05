#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SKCM_FILES = {
    "train": ("SKCM_train_train_X.npy", "SKCM_train_train_ids.txt"),
    "validation": ("SKCM_validation_validation_X.npy", "SKCM_validation_validation_ids.txt"),
    "test": ("SKCM_test_test_X.npy", "SKCM_test_test_ids.txt"),
}


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def mean_embed(case_dir: Path) -> np.ndarray:
    x = np.load(case_dir / "X.npy").astype(np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise RuntimeError(f"bad feature matrix in {case_dir}")
    return x.mean(axis=0).astype(np.float32)


def build_loki_cache(base_cache: Path, loki_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, (_, ids_name) in SKCM_FILES.items():
        ids = read_ids(base_cache / ids_name)
        feats = []
        kept = []
        missing = []
        for sid in ids:
            case_dir = loki_root / sid
            if not (case_dir / "X.npy").exists():
                missing.append(sid)
                continue
            feats.append(mean_embed(case_dir))
            kept.append(sid)
        x = np.stack(feats, axis=0).astype(np.float32)
        np.save(out_dir / SKCM_FILES[split][0], x)
        (out_dir / SKCM_FILES[split][1]).write_text("\n".join(kept) + "\n")
        summary[split] = {
            "n_ids_expected": len(ids),
            "n_ids_written": len(kept),
            "n_missing": len(missing),
            "feature_dim": int(x.shape[1]),
        }
    return summary


def build_concat_cache(base_cache: Path, loki_cache: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, (x_name, ids_name) in SKCM_FILES.items():
        uni_x = np.load(base_cache / x_name).astype(np.float32)
        uni_ids = read_ids(base_cache / ids_name)
        loki_x = np.load(loki_cache / x_name).astype(np.float32)
        loki_ids = read_ids(loki_cache / ids_name)
        if uni_ids != loki_ids:
            raise RuntimeError(f"{split}: UNI/Loki ids mismatch")
        x = np.concatenate([uni_x, loki_x], axis=1).astype(np.float32)
        np.save(out_dir / x_name, x)
        (out_dir / ids_name).write_text("\n".join(uni_ids) + "\n")
        summary[split] = {
            "n_ids": len(uni_ids),
            "feature_dim_uni": int(uni_x.shape[1]),
            "feature_dim_loki": int(loki_x.shape[1]),
            "feature_dim_concat": int(x.shape[1]),
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-cache",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/common361_best3_mean_mlp/cache",
    )
    p.add_argument(
        "--loki-root",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_loki_skcm_packed",
    )
    p.add_argument(
        "--loki-cache-out",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/loki_skcm_mean_cache",
    )
    p.add_argument(
        "--concat-cache-out",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/uni_loki_skcm_mean_cache",
    )
    args = p.parse_args()

    base_cache = Path(args.base_cache)
    loki_root = Path(args.loki_root)
    loki_cache_out = Path(args.loki_cache_out)
    concat_cache_out = Path(args.concat_cache_out)

    loki_summary = build_loki_cache(base_cache, loki_root, loki_cache_out)
    concat_summary = build_concat_cache(base_cache, loki_cache_out, concat_cache_out)

    (loki_cache_out / "cache_summary.json").write_text(json.dumps(loki_summary, indent=2))
    (concat_cache_out / "cache_summary.json").write_text(json.dumps(concat_summary, indent=2))
    print(json.dumps({"loki_cache": loki_summary, "concat_cache": concat_summary}, indent=2))


if __name__ == "__main__":
    main()
