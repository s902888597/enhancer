#!/usr/bin/env python3
"""Build eRNA_count>=4 top3000 z-score fixed-split assets.

Uses uploaded union21477 z-score files. The target panel is common across
BRCA/LUAD/SKCM: filter SEs by eRNA_count >= 4, then select top3000 by variance
across the concatenated train samples from all three cancers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


CANCERS = ("BRCA", "LUAD", "SKCM")
META_COLS = ["chr", "start", "end", "SE_ID", "eRNA_count"]
SPLIT_TO_LABEL = {"train": "train", "validation": "val", "test": "test"}
IN_ROOT = Path(
    "/taiga/illinois/vetmed/cb/kwang222/enhancer/shared_data/"
    "BRCA_LUAD_SKCM_SEs_union_eRNA_zscoreNorm_MinMaxNorm"
)
OUT_ROOT = Path(
    "/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/"
    "zscore_union21477_erna_ge4_top3000_fixedsplit"
)
FEATURE_CANDIDATES = [
    Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy"),
    Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/siteaware_union4827_v4/features_patches_pan_cancer_npy"),
    Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/siteaware_union4827_v3/features_patches_pan_cancer_npy"),
    Path("/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/siteaware_union4827_v2/features_patches_pan_cancer_npy"),
]
TOP_N = 3000
MIN_ERNA_COUNT = 4


def input_csv(cancer: str, split: str) -> Path:
    label = SPLIT_TO_LABEL[split]
    matches = sorted(IN_ROOT.glob(f"{cancer}_*{label}_avg_eRNA_zscore_per_SE.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one file for {cancer} {split}, found {len(matches)}: {matches}")
    return matches[0]


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

    train_dfs: dict[str, pd.DataFrame] = {}
    train_values = []
    se_ref: list[str] | None = None
    meta_ref: pd.DataFrame | None = None

    for cancer in CANCERS:
        df = pd.read_csv(input_csv(cancer, "train"))
        if df.shape[0] == 0:
            raise RuntimeError(f"Empty train CSV for {cancer}")
        se_ids = df["SE_ID"].astype(str).tolist()
        if se_ref is None:
            se_ref = se_ids
            meta_ref = df[META_COLS].copy()
        elif se_ids != se_ref:
            raise RuntimeError(f"SE_ID row order differs for {cancer}")
        train_dfs[cancer] = df

    assert se_ref is not None and meta_ref is not None
    keep_idx = np.flatnonzero(meta_ref["eRNA_count"].to_numpy(dtype=np.float32) >= MIN_ERNA_COUNT)
    if keep_idx.size < TOP_N:
        raise RuntimeError(f"Only {keep_idx.size} SEs pass eRNA_count >= {MIN_ERNA_COUNT}; need {TOP_N}")

    for cancer in CANCERS:
        vals = train_dfs[cancer].iloc[keep_idx].loc[:, sample_cols(train_dfs[cancer])].to_numpy(dtype=np.float32)
        train_values.append(vals)
    all_train = np.concatenate(train_values, axis=1)
    variances = np.nanvar(all_train, axis=1)
    top_rel_idx = np.argsort(variances)[::-1][:TOP_N]
    top_idx = keep_idx[top_rel_idx]
    top_se = [se_ref[i] for i in top_idx]

    top_meta = meta_ref.iloc[top_idx].copy()
    top_meta["train_variance"] = variances[top_rel_idx]
    top_meta.to_csv(OUT_ROOT / "top3000_selection_erna_ge4.csv", index=False)
    (OUT_ROOT / "top3000_SE_IDs.txt").write_text("\n".join(top_se) + "\n")
    (matrix_dir / "enhancers.txt").write_text("\n".join(top_se) + "\n")

    split_rows: dict[str, list[np.ndarray]] = {s: [] for s in SPLIT_TO_LABEL}
    split_groups: dict[str, list[str]] = {s: [] for s in SPLIT_TO_LABEL}
    split_ids: dict[str, list[str]] = {s: [] for s in SPLIT_TO_LABEL}
    needed_ids: dict[str, set[str]] = {cancer: set() for cancer in CANCERS}
    summary_rows = []

    for cancer in CANCERS:
        for split in SPLIT_TO_LABEL:
            if split == "train":
                df = train_dfs[cancer]
            else:
                df = pd.read_csv(input_csv(cancer, split))
                if df["SE_ID"].astype(str).tolist() != se_ref:
                    raise RuntimeError(f"SE_ID row order differs for {cancer} {split}")
            out_df = df.iloc[top_idx].copy()
            out_csv = label_dir / f"{cancer}_{split}_zscore_top3000_shared.csv"
            out_df.to_csv(out_csv, index=False)

            cols = sample_cols(out_df)
            y = out_df.loc[:, cols].to_numpy(dtype=np.float32).T
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
                    "enhancers": out_df.shape[0],
                    "eRNA_count_min": float(out_df["eRNA_count"].min()),
                    "eRNA_count_median": float(out_df["eRNA_count"].median()),
                    "eRNA_count_mean": float(out_df["eRNA_count"].mean()),
                    "min": float(np.nanmin(y)),
                    "max": float(np.nanmax(y)),
                    "mean": float(np.nanmean(y)),
                    "std": float(np.nanstd(y)),
                }
            )

    for split in SPLIT_TO_LABEL:
        y = np.concatenate(split_rows[split], axis=0).astype(np.float32)
        mask = np.ones_like(y, dtype=np.float32)
        groups = np.array(split_groups[split], dtype=object)
        ids = split_ids[split]
        np.save(matrix_dir / f"y_{split}.npy", y)
        np.save(matrix_dir / f"mask_{split}.npy", mask)
        np.save(matrix_dir / f"group_{split}.npy", groups)
        (matrix_dir / f"id_{split}.txt").write_text("\n".join(ids) + "\n")

    feature_links = link_features(link_root, needed_ids)
    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "asset_summary.csv", index=False)
    config = {
        "input_root": str(IN_ROOT),
        "output_root": str(OUT_ROOT),
        "selection": "filter_eRNA_count_ge4_then_top3000_by_variance_across_BRCA_LUAD_SKCM_train_zscore_only",
        "splits": "fixed train/validation/test from uploaded zscore files",
        "matrix_mask": "all ones",
        "cancers": list(CANCERS),
        "min_eRNA_count": MIN_ERNA_COUNT,
        "n_targets": TOP_N,
        "n_source_targets": len(se_ref),
        "n_after_eRNA_filter": int(keep_idx.size),
        "feature_links": feature_links,
    }
    (OUT_ROOT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))
    print(pd.DataFrame(summary_rows).loc[:, ["cancer", "split", "samples", "enhancers", "eRNA_count_min", "eRNA_count_median", "min", "max"]])


if __name__ == "__main__":
    main()
