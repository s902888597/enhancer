#!/usr/bin/env python3
"""Build fixed-split top3000 z-score labels and mixed matrices.

This script uses only train labels to select the top variable SEs across BRCA,
LUAD, and SKCM, then writes:
- per-cancer label CSVs for the single-cancer attention baseline
- all-one-mask matrix files for the mixed dual-head attention model
"""

from __future__ import annotations

import json
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
    "zscore_union21477_top3000_fixedsplit"
)
TOP_N = 3000


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


def main() -> None:
    label_dir = OUT_ROOT / "labels"
    matrix_dir = OUT_ROOT / "matrix"
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
            raise RuntimeError(f"SE_ID row order differs for {cancer}; refusing positional topN selection")
        vals = df.loc[:, sample_cols(df)].to_numpy(dtype=np.float32)
        train_values.append(vals)
        train_dfs[cancer] = df

    all_train = np.concatenate(train_values, axis=1)
    variances = np.nanvar(all_train, axis=1)
    if not np.isfinite(variances).all():
        raise RuntimeError("Non-finite variance found in train labels")

    top_idx = np.argsort(variances)[::-1][:TOP_N]
    assert se_ref is not None and meta_ref is not None
    top_se = [se_ref[i] for i in top_idx]
    top_meta = meta_ref.iloc[top_idx].copy()
    top_meta["train_variance"] = variances[top_idx]
    top_meta.to_csv(OUT_ROOT / "top3000_selection.csv", index=False)
    (OUT_ROOT / "top3000_SE_IDs.txt").write_text("\n".join(top_se) + "\n")
    (matrix_dir / "enhancers.txt").write_text("\n".join(top_se) + "\n")

    split_rows: dict[str, list[np.ndarray]] = {s: [] for s in SPLIT_TO_LABEL}
    split_groups: dict[str, list[str]] = {s: [] for s in SPLIT_TO_LABEL}
    split_ids: dict[str, list[str]] = {s: [] for s in SPLIT_TO_LABEL}
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
            split_ids[split].extend([f"{cancer}:{canonical_sample_id(c)}" for c in cols])
            summary_rows.append(
                {
                    "cancer": cancer,
                    "split": split,
                    "input_csv": str(input_csv(cancer, split)),
                    "output_csv": str(out_csv),
                    "samples": len(cols),
                    "enhancers": out_df.shape[0],
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
        if y.shape[0] != len(groups) or y.shape[0] != len(ids):
            raise RuntimeError(f"Shape mismatch for {split}: y={y.shape}, groups={len(groups)}, ids={len(ids)}")
        np.save(matrix_dir / f"y_{split}.npy", y)
        np.save(matrix_dir / f"mask_{split}.npy", mask)
        np.save(matrix_dir / f"group_{split}.npy", groups)
        (matrix_dir / f"id_{split}.txt").write_text("\n".join(ids) + "\n")

    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "asset_summary.csv", index=False)
    run_config = {
        "input_root": str(IN_ROOT),
        "output_root": str(OUT_ROOT),
        "selection": "top3000_by_variance_across_BRCA_LUAD_SKCM_train_zscore_only",
        "splits": "fixed train/validation/test from uploaded zscore files",
        "matrix_mask": "all ones; no missing target mask needed for shared union SE panel",
        "cancers": list(CANCERS),
        "n_targets": TOP_N,
        "n_source_targets": len(se_ref),
    }
    (OUT_ROOT / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print(json.dumps(run_config, indent=2))
    print(pd.DataFrame(summary_rows).loc[:, ["cancer", "split", "samples", "enhancers", "min", "max"]])


if __name__ == "__main__":
    main()
