#!/usr/bin/env python3
"""
Build a new mixed-cancer common-target matrix by reusing X/group/id from an
existing mixed matrix and rebuilding y against a new fixed enhancer panel.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from run_mean_regression_pan_cancer import load_label_df, normalize_sample_id


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_ids_by_cancer(id_path: Path, sample_id_modes: Dict[str, str]) -> Dict[str, List[str]]:
    ids_by_cancer: Dict[str, List[str]] = {"BRCA": [], "LUAD": [], "SKCM": []}
    for item in read_lines(id_path):
        cancer, sample_id = item.split(":", 1)
        ids_by_cancer[cancer].append(normalize_sample_id(sample_id, sample_id_modes[cancer]))
    return ids_by_cancer


def align_rows(label_df, ids: List[str], enhancers: List[str]) -> np.ndarray:
    missing_ids = [sample_id for sample_id in ids if sample_id not in label_df.index]
    if missing_ids:
        raise RuntimeError(f"{len(missing_ids)} ids missing from labels, e.g. {missing_ids[:3]}")

    missing_enh = [enh for enh in enhancers if enh not in label_df.columns]
    if missing_enh:
        raise RuntimeError(f"{len(missing_enh)} enhancers missing from labels, e.g. {missing_enh[:3]}")

    return label_df.loc[ids, enhancers].to_numpy(dtype=np.float32)


def build_y_for_split(
    split: str,
    src_dir: Path,
    enhancers: List[str],
    split_specs: Dict[str, Dict[str, str]],
) -> np.ndarray:
    sample_id_modes = {
        cancer: split_specs[cancer]["sample_id_mode"]
        for cancer in ["BRCA", "LUAD", "SKCM"]
    }
    ids_by_cancer = load_ids_by_cancer(src_dir / f"id_{split}.txt", sample_id_modes)
    y_parts = []
    for cancer in ["BRCA", "LUAD", "SKCM"]:
        spec = split_specs[cancer]
        _, label_df = load_label_df(Path(spec["csv"]), spec["label_layout"], spec["sample_id_mode"])
        y_parts.append(align_rows(label_df, ids_by_cancer[cancer], enhancers))
    return np.concatenate(y_parts, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-matrix-dir", required=True)
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

    src_dir = Path(args.src_matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enhancers = read_lines(Path(args.enhancers_file))
    if not enhancers:
        raise RuntimeError("Enhancer list is empty")

    split_specs = {
        "train": {
            "BRCA": {"csv": args.brca_train_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
            "LUAD": {"csv": args.luad_train_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
            "SKCM": {"csv": args.skcm_train_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
        },
        "validation": {
            "BRCA": {"csv": args.brca_val_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
            "LUAD": {"csv": args.luad_val_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
            "SKCM": {"csv": args.skcm_val_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
        },
        "test": {
            "BRCA": {"csv": args.brca_test_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
            "LUAD": {"csv": args.luad_test_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
            "SKCM": {"csv": args.skcm_test_csv, "label_layout": "samples_as_columns", "sample_id_mode": "tcga_case3"},
        },
    }

    for split in ["train", "validation", "test"]:
        shutil.copy2(src_dir / f"X_{split}.npy", out_dir / f"X_{split}.npy")
        shutil.copy2(src_dir / f"group_{split}.npy", out_dir / f"group_{split}.npy")
        shutil.copy2(src_dir / f"id_{split}.txt", out_dir / f"id_{split}.txt")
        y = build_y_for_split(split, src_dir, enhancers, split_specs[split])
        np.save(out_dir / f"y_{split}.npy", y)
        print(f"{split}: y={y.shape}")

    (out_dir / "enhancers.txt").write_text("\n".join(enhancers) + "\n")
    print(f"Saved rebuilt common matrix to {out_dir}")


if __name__ == "__main__":
    main()
