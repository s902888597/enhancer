#!/usr/bin/env python3
"""
Build unified train/validation/test X/y/group/id matrices for the shared 361
enhancers across BRCA, LUAD, and SKCM using existing mean-feature caches.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from run_mean_regression_pan_cancer import load_label_df


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_cached_x(cache_dir: Path, split_name: str, cancer: str, stem: str) -> Tuple[List[str], np.ndarray]:
    x_path = cache_dir / f"{cancer}_{split_name}_{stem}_X.npy"
    ids_path = cache_dir / f"{cancer}_{split_name}_{stem}_ids.txt"
    if not x_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Missing cache files for {cancer} {split_name}: {x_path} / {ids_path}")
    ids = read_lines(ids_path)
    x = np.load(x_path)
    return ids, x


def align_rows(label_df, ids: List[str], enhancers: List[str]) -> np.ndarray:
    missing = [sid for sid in ids if sid not in label_df.index]
    if missing:
        raise RuntimeError(f"{len(missing)} ids missing from labels, e.g. {missing[:3]}")
    return label_df.loc[ids, enhancers].to_numpy(dtype=np.float32)


def build_one(
    split_name: str,
    enhancers: List[str],
    cache_dir: Path,
    specs: List[Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    all_x = []
    all_y = []
    all_groups = []
    all_ids: List[str] = []
    for spec in specs:
        ids, x = load_cached_x(cache_dir, split_name, spec["cancer"], spec["cache_stem"])
        _, label_df = load_label_df(Path(spec["csv"]), spec["label_layout"], spec["sample_id_mode"])
        y = align_rows(label_df, ids, enhancers)
        all_x.append(x.astype(np.float32))
        all_y.append(y)
        all_groups.extend([spec["cancer"]] * len(ids))
        all_ids.extend([f'{spec["cancer"]}:{sid}' for sid in ids])
    return (
        np.concatenate(all_x, axis=0),
        np.concatenate(all_y, axis=0),
        np.array(all_groups),
        all_ids,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--enhancers-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--brca-train-csv", required=True)
    parser.add_argument("--brca-val-csv", required=True)
    parser.add_argument("--brca-test-csv", required=True)
    parser.add_argument("--luad-train-csv", required=True)
    parser.add_argument("--luad-val-csv", required=True)
    parser.add_argument("--luad-test-csv", required=True)
    parser.add_argument("--skcm-train-csv", required=True)
    parser.add_argument("--skcm-val-csv", required=True)
    parser.add_argument("--skcm-test-csv", required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enhancers = read_lines(Path(args.enhancers_file))
    if not enhancers:
        raise RuntimeError("Enhancer list is empty")

    split_specs = {
        "train": [
            {
                "cancer": "BRCA",
                "csv": args.brca_train_csv,
                "cache_stem": "Top1000_SEs_192train_eRNA_zscore_3eRNAremoved",
                "label_layout": "samples_as_columns",
                "sample_id_mode": "tcga_case3",
            },
            {
                "cancer": "LUAD",
                "csv": args.luad_train_csv,
                "cache_stem": "Top1000_SEs_189train_eRNA_zscore_3eRNAremoved",
                "label_layout": "samples_as_columns",
                "sample_id_mode": "tcga_case3",
            },
            {
                "cancer": "SKCM",
                "csv": args.skcm_train_csv,
                "cache_stem": "train",
                "label_layout": "samples_as_rows",
                "sample_id_mode": "identity",
            },
        ],
        "validation": [
            {
                "cancer": "BRCA",
                "csv": args.brca_val_csv,
                "cache_stem": "Top1000_SEs_63val_eRNA_zscore_3eRNAremoved",
                "label_layout": "samples_as_columns",
                "sample_id_mode": "tcga_case3",
            },
            {
                "cancer": "LUAD",
                "csv": args.luad_val_csv,
                "cache_stem": "Top1000_SEs_63val_eRNA_zscore_3eRNAremoved",
                "label_layout": "samples_as_columns",
                "sample_id_mode": "tcga_case3",
            },
            {
                "cancer": "SKCM",
                "csv": args.skcm_val_csv,
                "cache_stem": "validation",
                "label_layout": "samples_as_rows",
                "sample_id_mode": "identity",
            },
        ],
        "test": [
            {
                "cancer": "BRCA",
                "csv": args.brca_test_csv,
                "cache_stem": "Top1000_SEs_64test_eRNA_zscore_3eRNAremoved",
                "label_layout": "samples_as_columns",
                "sample_id_mode": "tcga_case3",
            },
            {
                "cancer": "LUAD",
                "csv": args.luad_test_csv,
                "cache_stem": "Top1000_SEs_64test_eRNA_zscore_3eRNAremoved",
                "label_layout": "samples_as_columns",
                "sample_id_mode": "tcga_case3",
            },
            {
                "cancer": "SKCM",
                "csv": args.skcm_test_csv,
                "cache_stem": "test",
                "label_layout": "samples_as_rows",
                "sample_id_mode": "identity",
            },
        ],
    }

    for split_name, specs in split_specs.items():
        x, y, groups, ids = build_one(split_name, enhancers, cache_dir, specs)
        np.save(out_dir / f"X_{split_name}.npy", x)
        np.save(out_dir / f"y_{split_name}.npy", y)
        np.save(out_dir / f"group_{split_name}.npy", groups)
        (out_dir / f"id_{split_name}.txt").write_text("\n".join(ids) + "\n")
        print(
            f"{split_name}: X={x.shape} y={y.shape} groups={groups.shape} "
            f"ids={len(ids)}"
        )

    (out_dir / "enhancers.txt").write_text("\n".join(enhancers) + "\n")
    print(f"saved unified matrices to {out_dir}")


if __name__ == "__main__":
    main()
