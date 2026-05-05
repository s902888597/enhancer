#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANCERS = ["BRCA", "LUAD", "SKCM"]
SPLITS = ["train", "validation", "test"]


def read_split_ids(label_root: Path, cancer: str, split: str) -> list[str]:
    path = label_root / f"{cancer}_{split}_zscore_top3000_shared.csv"
    df = pd.read_csv(path, usecols=["sample"])
    return df["sample"].astype(str).tolist()


def sample_to_column(sample_id: str) -> str:
    return sample_id if sample_id.endswith("_tumor") else f"{sample_id}_tumor"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/BRCA_LUAD_SKCM_eRNA_sum_per_SE_for_AI/BRCA_LUAD_SKCM_merged_eRNA_sum_per_SE_lg2TPMplus1_5pct_above6.5.csv",
    )
    parser.add_argument(
        "--split-label-root",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_shareddata_top3000_zscore_rows",
    )
    parser.add_argument(
        "--matrix-dir",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/mix3_erna4446_5pct_maxscaled_matrix",
    )
    parser.add_argument(
        "--single-label-root",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/label/three_cancer_erna4446_5pct_maxscaled_rows",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    split_label_root = Path(args.split_label_root)
    matrix_dir = Path(args.matrix_dir)
    single_label_root = Path(args.single_label_root)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    single_label_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    meta_cols = ["chr", "start", "end", "SE_ID", "eRNA_count"]
    missing_meta = [c for c in meta_cols if c not in df.columns]
    if missing_meta:
        raise RuntimeError(f"Missing expected columns in {input_csv}: {missing_meta}")

    enhancers = df["SE_ID"].astype(str).tolist()
    value_df = df.set_index("SE_ID")
    df.loc[:, meta_cols].to_csv(matrix_dir / "enhancer_metadata.csv", index=False)
    df.loc[:, meta_cols].to_csv(single_label_root / "enhancer_metadata.csv", index=False)
    (matrix_dir / "enhancers.txt").write_text("\n".join(enhancers) + "\n")

    split_ids: dict[tuple[str, str], list[str]] = {}
    split_cols: dict[tuple[str, str], list[str]] = {}
    for split in SPLITS:
        for cancer in CANCERS:
            ids = read_split_ids(split_label_root, cancer, split)
            cols = [sample_to_column(sid) for sid in ids]
            missing_cols = [col for col in cols if col not in value_df.columns]
            if missing_cols:
                raise RuntimeError(f"{cancer} {split}: missing sample columns e.g. {missing_cols[:10]}")
            split_ids[(cancer, split)] = ids
            split_cols[(cancer, split)] = cols

    train_cols = []
    for cancer in CANCERS:
        train_cols.extend(split_cols[(cancer, "train")])
    train_raw = value_df.loc[enhancers, train_cols].to_numpy(dtype=np.float32).T
    scale = train_raw.max(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    np.save(matrix_dir / "train_max_per_enhancer.npy", scale)
    np.save(single_label_root / "train_max_per_enhancer.npy", scale)

    summary = {
        "input_csv": str(input_csv),
        "normalization": "per-SE max scaling using mixed train samples only",
        "n_enhancers": int(len(enhancers)),
        "scale_min": float(scale.min()),
        "scale_median": float(np.median(scale)),
        "scale_max": float(scale.max()),
        "matrix_dir": str(matrix_dir),
        "single_label_root": str(single_label_root),
        "splits": {},
    }

    for split in SPLITS:
        y_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        groups: list[str] = []
        mixed_ids: list[str] = []
        split_summary = {}

        for cancer in CANCERS:
            ids = split_ids[(cancer, split)]
            cols = split_cols[(cancer, split)]
            y_raw = value_df.loc[enhancers, cols].to_numpy(dtype=np.float32).T
            y = (y_raw / scale).astype(np.float32)
            out = pd.DataFrame(y, columns=enhancers)
            out.insert(0, "sample", ids)
            out.to_csv(single_label_root / f"{cancer}_{split}_erna4446_5pct_maxscaled.csv", index=False)

            y_rows.append(y)
            mask_rows.append(np.ones_like(y, dtype=np.float32))
            groups.extend([cancer] * len(ids))
            mixed_ids.extend([f"{cancer}:{sid}" for sid in ids])
            split_summary[cancer] = int(len(ids))

        y_all = np.concatenate(y_rows, axis=0).astype(np.float32)
        mask_all = np.concatenate(mask_rows, axis=0).astype(np.float32)
        np.save(matrix_dir / f"y_{split}.npy", y_all)
        np.save(matrix_dir / f"mask_{split}.npy", mask_all)
        np.save(matrix_dir / f"group_{split}.npy", np.array(groups, dtype=object))
        np.save(matrix_dir / f"X_{split}.npy", y_all)
        (matrix_dir / f"id_{split}.txt").write_text("\n".join(mixed_ids) + "\n")
        split_summary["total"] = int(len(mixed_ids))
        split_summary["y_shape"] = list(y_all.shape)
        split_summary["mask_all_ones"] = bool(np.all(mask_all == 1.0))
        split_summary["y_min"] = float(y_all.min())
        split_summary["y_max"] = float(y_all.max())
        summary["splits"][split] = split_summary

    (matrix_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    (single_label_root / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
