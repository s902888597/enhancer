#!/usr/bin/env python3
"""Build fixed-split assets from Kenneth's merged-train z-score ANOVA top3000 SE files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


CANCERS = ("BRCA", "LUAD", "SKCM")
META_COLS = ["chr", "start", "end", "SE_ID", "eRNA_count"]
SPLIT_FILES = {
    ("BRCA", "train"): "BRCA_192train_avg_eRNA_zscore_per_SE_top3000.csv",
    ("BRCA", "validation"): "BRCA_63val_avg_eRNA_zscore_per_SE_top3000.csv",
    ("BRCA", "test"): "BRCA_64test_avg_eRNA_zscore_per_SE_top3000.csv",
    ("LUAD", "train"): "LUAD_189train_avg_eRNA_zscore_per_SE_top3000.csv",
    ("LUAD", "validation"): "LUAD_63val_avg_eRNA_zscore_per_SE_top3000.csv",
    ("LUAD", "test"): "LUAD_63test_avg_eRNA_zscore_per_SE_top3000.csv",
    ("SKCM", "train"): "SKCM_189train_avg_eRNA_zscore_per_SE_top3000.csv",
    ("SKCM", "validation"): "SKCM_64val_avg_eRNA_zscore_per_SE_top3000.csv",
    ("SKCM", "test"): "SKCM_64test_avg_eRNA_zscore_per_SE_top3000.csv",
}
IN_ROOT = Path(
    "/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/"
    "BRCA_LUAD_SKCM_subset_MergedTrain_zscoreNorm_for_AI"
)
OUT_ROOT = Path(
    "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/"
    "anova_mergedtrain_zscore_top3000_fixedsplit"
)
FEATURE_CANDIDATES = [
    Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy"),
    Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/siteaware_union4827_v4/features_patches_pan_cancer_npy"),
    Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/siteaware_union4827_v3/features_patches_pan_cancer_npy"),
    Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/siteaware_union4827_v2/features_patches_pan_cancer_npy"),
]


def input_csv(cancer: str, split: str) -> Path:
    return IN_ROOT / SPLIT_FILES[(cancer, split)]


def sample_cols(df: pd.DataFrame) -> list[str]:
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing metadata columns: {missing}")
    return [c for c in df.columns if c not in META_COLS]


def canonical_sample_id(sample_id: str) -> str:
    sample_id = str(sample_id)
    if sample_id.endswith("_tumor"):
        sample_id = sample_id[:-6]
    if sample_id.startswith("TCGA-") and len(sample_id) >= 12:
        return sample_id[:12]
    return sample_id


def canonical_feature_case(dirname: str) -> str:
    dirname = dirname[:-6] if dirname.endswith("_tumor") else dirname
    parts = dirname.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else dirname


def build_feature_map() -> dict[str, dict[str, Path]]:
    maps: dict[str, dict[str, Path]] = {cancer: {} for cancer in CANCERS}
    for root in FEATURE_CANDIDATES:
        if not root.exists():
            continue
        for cancer in CANCERS:
            for parent in [root / cancer, root / "train" / cancer, root / "validation" / cancer, root / "test" / cancer]:
                if not parent.exists():
                    continue
                for d in sorted(parent.iterdir(), key=lambda p: p.name):
                    if d.is_dir() and d.name.startswith("TCGA-"):
                        maps[cancer].setdefault(canonical_feature_case(d.name), d)
    return maps


def link_features(link_root: Path, needed_ids: dict[str, set[str]]) -> dict[str, int]:
    maps = build_feature_map()
    linked: dict[str, int] = {}
    missing: dict[str, list[str]] = {}
    for cancer in CANCERS:
        croot = link_root / cancer
        croot.mkdir(parents=True, exist_ok=True)
        linked[cancer] = 0
        missing[cancer] = []
        for case in sorted(needed_ids[cancer]):
            target = maps[cancer].get(case)
            if target is None:
                missing[cancer].append(case)
                continue
            dest = croot / case
            if dest.is_symlink() or dest.exists():
                if dest.is_symlink() and dest.resolve() == target.resolve():
                    linked[cancer] += 1
                    continue
                raise RuntimeError(f"Existing nonmatching feature link: {dest}")
            os.symlink(target, dest, target_is_directory=True)
            linked[cancer] += 1
    if any(missing.values()):
        raise RuntimeError(f"Missing feature directories: {missing}")
    return linked


def main() -> None:
    label_dir = OUT_ROOT / "labels"
    matrix_dir = OUT_ROOT / "matrix"
    link_root = OUT_ROOT / "features_by_cancer_case_symlinks"
    label_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_csv(input_csv("BRCA", "train"))
    se_ref = ref_df["SE_ID"].astype(str).tolist()
    if len(se_ref) != 3000:
        raise RuntimeError(f"Expected 3000 top SEs, got {len(se_ref)}")
    (matrix_dir / "enhancers.txt").write_text("\n".join(se_ref) + "\n")
    (OUT_ROOT / "top3000_SE_IDs.txt").write_text("\n".join(se_ref) + "\n")
    ref_df[META_COLS].to_csv(OUT_ROOT / "top3000_anova_selection_meta.csv", index=False)

    split_rows: dict[str, list[np.ndarray]] = {s: [] for s in ["train", "validation", "test"]}
    split_groups: dict[str, list[str]] = {s: [] for s in ["train", "validation", "test"]}
    split_ids: dict[str, list[str]] = {s: [] for s in ["train", "validation", "test"]}
    needed_ids: dict[str, set[str]] = {cancer: set() for cancer in CANCERS}
    summary_rows = []

    for cancer in CANCERS:
        for split in ["train", "validation", "test"]:
            df = pd.read_csv(input_csv(cancer, split))
            if df["SE_ID"].astype(str).tolist() != se_ref:
                raise RuntimeError(f"SE_ID row order differs for {cancer} {split}")
            out_csv = label_dir / f"{cancer}_{split}_anova_top3000_zscore.csv"
            df.to_csv(out_csv, index=False)
            cols = sample_cols(df)
            y = df.loc[:, cols].to_numpy(dtype=np.float32).T
            split_rows[split].append(y)
            split_groups[split].extend([cancer] * len(cols))
            cases = [canonical_sample_id(c) for c in cols]
            split_ids[split].extend([f"{cancer}:{case}" for case in cases])
            needed_ids[cancer].update(cases)
            summary_rows.append(
                {
                    "cancer": cancer,
                    "split": split,
                    "input_csv": str(input_csv(cancer, split)),
                    "output_csv": str(out_csv),
                    "samples": len(cols),
                    "enhancers": df.shape[0],
                    "eRNA_count_min": float(df["eRNA_count"].min()),
                    "eRNA_count_median": float(df["eRNA_count"].median()),
                    "eRNA_count_mean": float(df["eRNA_count"].mean()),
                    "min": float(np.nanmin(y)),
                    "max": float(np.nanmax(y)),
                    "mean": float(np.nanmean(y)),
                    "std": float(np.nanstd(y)),
                }
            )

    for split in ["train", "validation", "test"]:
        y = np.concatenate(split_rows[split], axis=0).astype(np.float32)
        np.save(matrix_dir / f"y_{split}.npy", y)
        np.save(matrix_dir / f"mask_{split}.npy", np.ones_like(y, dtype=np.float32))
        np.save(matrix_dir / f"group_{split}.npy", np.array(split_groups[split], dtype=object))
        (matrix_dir / f"id_{split}.txt").write_text("\n".join(split_ids[split]) + "\n")

    feature_links = link_features(link_root, needed_ids)
    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "asset_summary.csv", index=False)
    config = {
        "input_root": str(IN_ROOT),
        "output_root": str(OUT_ROOT),
        "selection": "top3000_by_one_way_welch_anova_F_statistic_from_merged_train",
        "normalization": "merged_train_zscore_from_eRNA_then_average_per_SE",
        "splits": "fixed train/validation/test from uploaded files",
        "matrix_mask": "all ones",
        "cancers": list(CANCERS),
        "n_targets": 3000,
        "feature_links": feature_links,
    }
    (OUT_ROOT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))
    print(pd.DataFrame(summary_rows).loc[:, ["cancer", "split", "samples", "enhancers", "eRNA_count_min", "eRNA_count_median", "min", "max"]])


if __name__ == "__main__":
    main()
