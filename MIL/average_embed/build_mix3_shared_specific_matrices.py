#!/usr/bin/env python3
"""
Build unified matrices for mixed-cancer training with:
- shared enhancers: intersection across BRCA/LUAD/SKCM top1000
- cancer-specific enhancers: per-cancer top1000 minus shared set
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from run_mean_regression_pan_cancer import load_label_df


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_lines(path: Path, items: List[str]):
    path.write_text("\n".join(items) + ("\n" if items else ""))


def load_cached_x(cache_dir: Path, split_name: str, cancer: str, stem: str) -> Tuple[List[str], np.ndarray]:
    x_path = cache_dir / f"{cancer}_{split_name}_{stem}_X.npy"
    ids_path = cache_dir / f"{cancer}_{split_name}_{stem}_ids.txt"
    if not x_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Missing cache files for {cancer} {split_name}: {x_path} / {ids_path}")
    ids = read_lines(ids_path)
    x = np.load(x_path).astype(np.float32)
    return ids, x


def align_rows(label_df, ids: List[str], enhancers: List[str]) -> np.ndarray:
    missing = [sid for sid in ids if sid not in label_df.index]
    if missing:
        raise RuntimeError(f"{len(missing)} ids missing from labels, e.g. {missing[:3]}")
    return label_df.loc[ids, enhancers].to_numpy(dtype=np.float32)


def enhancer_lists_from_train(specs: Dict[str, Dict[str, str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    train_cols = {}
    for cancer, spec in specs.items():
        _, label_df = load_label_df(Path(spec["train_csv"]), spec["label_layout"], spec["sample_id_mode"])
        train_cols[cancer] = list(label_df.columns)
    shared = sorted(set(train_cols["BRCA"]) & set(train_cols["LUAD"]) & set(train_cols["SKCM"]))
    specific = {}
    shared_set = set(shared)
    for cancer, cols in train_cols.items():
        specific[cancer] = [c for c in cols if c not in shared_set]
    return shared, specific


def build_one_split(
    split_name: str,
    cache_dir: Path,
    specs: Dict[str, Dict[str, str]],
    shared_enhancers: List[str],
    specific_enhancers: Dict[str, List[str]],
):
    x_all = []
    y_shared_all = []
    y_specific_all = []
    group_all = []
    ids_all = []
    for cancer, spec in specs.items():
        ids, x = load_cached_x(cache_dir, split_name, cancer, spec["cache_stem"][split_name])
        _, label_df = load_label_df(Path(spec[f"{split_name}_csv"]), spec["label_layout"], spec["sample_id_mode"])
        y_shared = align_rows(label_df, ids, shared_enhancers)
        y_specific = align_rows(label_df, ids, specific_enhancers[cancer])
        x_all.append(x)
        y_shared_all.append(y_shared)
        y_specific_all.append(y_specific)
        group_all.extend([cancer] * len(ids))
        ids_all.extend([f"{cancer}:{sid}" for sid in ids])
    return (
        np.concatenate(x_all, axis=0).astype(np.float32),
        np.concatenate(y_shared_all, axis=0).astype(np.float32),
        np.concatenate(y_specific_all, axis=0).astype(np.float32),
        np.array(group_all),
        ids_all,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
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

    specs = {
        "BRCA": {
            "train_csv": args.brca_train_csv,
            "validation_csv": args.brca_val_csv,
            "test_csv": args.brca_test_csv,
            "label_layout": "samples_as_columns",
            "sample_id_mode": "tcga_case3",
            "cache_stem": {
                "train": "Top1000_SEs_192train_eRNA_zscore_3eRNAremoved",
                "validation": "Top1000_SEs_63val_eRNA_zscore_3eRNAremoved",
                "test": "Top1000_SEs_64test_eRNA_zscore_3eRNAremoved",
            },
        },
        "LUAD": {
            "train_csv": args.luad_train_csv,
            "validation_csv": args.luad_val_csv,
            "test_csv": args.luad_test_csv,
            "label_layout": "samples_as_columns",
            "sample_id_mode": "tcga_case3",
            "cache_stem": {
                "train": "Top1000_SEs_189train_eRNA_zscore_3eRNAremoved",
                "validation": "Top1000_SEs_63val_eRNA_zscore_3eRNAremoved",
                "test": "Top1000_SEs_64test_eRNA_zscore_3eRNAremoved",
            },
        },
        "SKCM": {
            "train_csv": args.skcm_train_csv,
            "validation_csv": args.skcm_val_csv,
            "test_csv": args.skcm_test_csv,
            "label_layout": "samples_as_rows",
            "sample_id_mode": "none",
            "cache_stem": {
                "train": "train",
                "validation": "validation",
                "test": "test",
            },
        },
    }

    shared_enhancers, specific_enhancers = enhancer_lists_from_train(specs)
    print(
        f"shared={len(shared_enhancers)} "
        f"brca_specific={len(specific_enhancers['BRCA'])} "
        f"luad_specific={len(specific_enhancers['LUAD'])} "
        f"skcm_specific={len(specific_enhancers['SKCM'])}"
    )

    write_lines(out_dir / "shared_enhancers.txt", shared_enhancers)
    for cancer, enh in specific_enhancers.items():
        write_lines(out_dir / f"{cancer}_specific_enhancers.txt", enh)

    for split_name in ["train", "validation", "test"]:
        x, y_shared, y_specific, groups, ids = build_one_split(
            split_name, cache_dir, specs, shared_enhancers, specific_enhancers
        )
        np.save(out_dir / f"X_{split_name}.npy", x)
        np.save(out_dir / f"y_shared_{split_name}.npy", y_shared)
        np.save(out_dir / f"y_specific_{split_name}.npy", y_specific)
        np.save(out_dir / f"group_{split_name}.npy", groups)
        write_lines(out_dir / f"id_{split_name}.txt", ids)
        print(
            f"{split_name}: X={x.shape} y_shared={y_shared.shape} "
            f"y_specific={y_specific.shape} groups={groups.shape}"
        )

    print(f"Saved matrices to {out_dir}")


if __name__ == "__main__":
    main()
