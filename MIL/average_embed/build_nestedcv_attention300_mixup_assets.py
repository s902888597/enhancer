#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_mean_regression_pan_cancer import normalize_sample_id


CANCERS = ["BRCA", "LUAD", "SKCM"]
SPLITS = ["train", "validation", "test"]


def load_full_label_df(label_root: Path, cancer: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    enh_cols_ref: list[str] | None = None
    for split in SPLITS:
        csv_path = label_root / f"{cancer}_{split}_zscore_top3000_shared.csv"
        df = pd.read_csv(csv_path)
        if "sample" not in df.columns:
            raise RuntimeError(f"{csv_path} missing sample column")
        enh_cols = [c for c in df.columns if c != "sample"]
        if enh_cols_ref is None:
            enh_cols_ref = enh_cols
        elif enh_cols != enh_cols_ref:
            raise RuntimeError(f"Enhancer columns mismatch for {cancer} across splits")
        part = df.copy()
        if cancer == "SKCM":
            part["sample"] = part["sample"].astype(str).str.strip()
        else:
            part["sample"] = part["sample"].astype(str).map(lambda s: normalize_sample_id(s, "tcga_case3"))
        if part["sample"].duplicated().any():
            dup = part.loc[part["sample"].duplicated(), "sample"].tolist()[:3]
            raise RuntimeError(f"{csv_path} duplicated normalized sample ids, e.g. {dup}")
        part = part.set_index("sample")
        parts.append(part[enh_cols_ref])
    full = pd.concat(parts, axis=0)
    if full.index.duplicated().any():
        dup = full.index[full.index.duplicated()].tolist()[:3]
        raise RuntimeError(f"{cancer} duplicated sample ids after concat, e.g. {dup}")
    return full.astype(np.float32)


def build_outer_inner_splits(ids: list[str], n_outer: int, n_inner: int, seed: int) -> list[dict[str, list[str]]]:
    ids_arr = np.array(ids, dtype=object)
    folds: list[dict[str, list[str]]] = []
    rng_outer = np.random.default_rng(seed)
    outer_perm = rng_outer.permutation(len(ids_arr))
    outer_chunks = np.array_split(outer_perm, n_outer)
    for fold_idx in range(n_outer):
        test_idx = np.sort(outer_chunks[fold_idx])
        trainval_idx = np.sort(np.concatenate([outer_chunks[i] for i in range(n_outer) if i != fold_idx]))
        trainval_ids = ids_arr[trainval_idx]
        test_ids = ids_arr[test_idx]
        rng_inner = np.random.default_rng(seed + 100 + fold_idx)
        inner_perm = rng_inner.permutation(len(trainval_ids))
        inner_chunks = np.array_split(inner_perm, n_inner)
        val_local = np.sort(inner_chunks[0])
        train_local = np.sort(np.concatenate(inner_chunks[1:]))
        train_ids = trainval_ids[train_local].tolist()
        val_ids = trainval_ids[val_local].tolist()
        folds.append(
            {
                "train": train_ids,
                "validation": val_ids,
                "test": test_ids.tolist(),
            }
        )
    return folds


def write_sample_row_csv(df: pd.DataFrame, ids: list[str], out_path: Path) -> None:
    out_df = df.loc[ids].reset_index().rename(columns={"index": "sample"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


def build_mixed_split_arrays(
    label_dfs: dict[str, pd.DataFrame],
    split_ids: dict[str, list[str]],
    enhancers: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    enh_to_idx = {enh: i for i, enh in enumerate(enhancers)}
    y_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    mixed_ids: list[str] = []

    for cancer in CANCERS:
        df = label_dfs[cancer]
        ids = split_ids[cancer]
        valid_cols = [enh for enh in enhancers if enh in df.columns]
        col_idx = [enh_to_idx[enh] for enh in valid_cols]
        y = np.zeros((len(ids), len(enhancers)), dtype=np.float32)
        if valid_cols:
            y[:, col_idx] = df.loc[ids, valid_cols].to_numpy(dtype=np.float32)
        mask_row = np.zeros(len(enhancers), dtype=np.float32)
        mask_row[col_idx] = 1.0
        mask = np.repeat(mask_row[None, :], len(ids), axis=0)
        groups = np.array([cancer] * len(ids), dtype="<U4")

        y_parts.append(y)
        mask_parts.append(mask)
        group_parts.append(groups)
        mixed_ids.extend([f"{cancer}:{sample_id}" for sample_id in ids])

    return (
        np.concatenate(y_parts, axis=0).astype(np.float32),
        np.concatenate(mask_parts, axis=0).astype(np.float32),
        np.concatenate(group_parts, axis=0),
        mixed_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--union-enhancers-file", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--n-outer", type=int, default=5)
    parser.add_argument("--n-inner", type=int, default=5)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    label_root = Path(args.label_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    label_dfs = {cancer: load_full_label_df(label_root, cancer) for cancer in CANCERS}
    fold_ids = {
        cancer: build_outer_inner_splits(label_dfs[cancer].index.tolist(), args.n_outer, args.n_inner, args.seed)
        for cancer in CANCERS
    }
    enhancers = [line.strip() for line in Path(args.union_enhancers_file).read_text().splitlines() if line.strip()]
    if not enhancers:
        raise RuntimeError("Union enhancer list is empty")

    manifest_rows: list[dict[str, object]] = []
    for fold_idx in range(args.n_outer):
        fold_root = out_root / f"fold{fold_idx}"
        single_root = fold_root / "single_cancer_top3000"
        mixed_root = fold_root / "mix3_union4827_matrix"
        mixed_root.mkdir(parents=True, exist_ok=True)

        for cancer in CANCERS:
            for split in SPLITS:
                ids = fold_ids[cancer][fold_idx][split]
                write_sample_row_csv(label_dfs[cancer], ids, single_root / f"{cancer}_{split}.csv")
                manifest_rows.append(
                    {
                        "fold": fold_idx,
                        "family": "single_cancer_top3000",
                        "cancer": cancer,
                        "split": split,
                        "n_samples": len(ids),
                    }
                )

        for split in SPLITS:
            split_ids = {cancer: fold_ids[cancer][fold_idx][split] for cancer in CANCERS}
            y, mask, groups, mixed_ids = build_mixed_split_arrays(label_dfs, split_ids, enhancers)
            np.save(mixed_root / f"y_{split}.npy", y)
            np.save(mixed_root / f"mask_{split}.npy", mask)
            np.save(mixed_root / f"group_{split}.npy", groups)
            (mixed_root / f"id_{split}.txt").write_text("\n".join(mixed_ids) + "\n")
            manifest_rows.append(
                {
                    "fold": fold_idx,
                    "family": "mix3_union4827_matrix",
                    "cancer": "ALL",
                    "split": split,
                    "n_samples": len(mixed_ids),
                    "mask_valid_mean": float(mask.mean()),
                }
            )
        (mixed_root / "enhancers.txt").write_text("\n".join(enhancers) + "\n")

    pd.DataFrame(manifest_rows).to_csv(out_root / "manifest.csv", index=False)
    metadata = {
        "n_outer": args.n_outer,
        "n_inner": args.n_inner,
        "seed": args.seed,
        "cancers": CANCERS,
        "union_enhancers": len(enhancers),
        "sample_counts": {cancer: int(label_dfs[cancer].shape[0]) for cancer in CANCERS},
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
