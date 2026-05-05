#!/usr/bin/env python3
"""
Apply an additional per-SE z-normalization to an existing mixed-cancer
shared/specific matrix.

Normalization is fit:
- separately for shared and specific targets
- separately for each cancer type
- using that cancer type's training samples only

The feature matrices and split metadata are copied through unchanged.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CANCERS = ["BRCA", "LUAD", "SKCM"]
SPLITS = ["train", "validation", "test"]


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_lines(path: Path, items: List[str]):
    path.write_text("\n".join(items) + ("\n" if items else ""))


def fit_standardizer(y_train: np.ndarray):
    mean = y_train.mean(axis=0, dtype=np.float64)
    std = y_train.std(axis=0, dtype=np.float64)
    zero_std = std < 1e-8
    std_safe = std.copy()
    std_safe[zero_std] = 1.0
    return mean.astype(np.float32), std_safe.astype(np.float32), zero_std


def apply_standardizer(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((y.astype(np.float32) - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_enhancers = read_lines(in_dir / "shared_enhancers.txt")
    specific_enhancers = {
        cancer: read_lines(in_dir / f"{cancer}_specific_enhancers.txt")
        for cancer in CANCERS
    }

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for split in SPLITS:
        data[split] = {
            "X": np.load(in_dir / f"X_{split}.npy").astype(np.float32),
            "y_shared": np.load(in_dir / f"y_shared_{split}.npy").astype(np.float32),
            "y_specific": np.load(in_dir / f"y_specific_{split}.npy").astype(np.float32),
            "groups": np.load(in_dir / f"group_{split}.npy", allow_pickle=True),
        }

    if data["train"]["y_shared"].shape[1] != len(shared_enhancers):
        raise ValueError("Shared enhancer count does not match y_shared_train columns")
    for cancer in CANCERS:
        if data["train"]["y_specific"].shape[1] != len(specific_enhancers[cancer]):
            raise ValueError(f"{cancer} specific enhancer count does not match y_specific_train columns")

    stats_rows = []
    for cancer in CANCERS:
        train_mask = data["train"]["groups"] == cancer
        if not np.any(train_mask):
            raise ValueError(f"No training samples found for {cancer}")

        shared_mean, shared_std, shared_zero_std = fit_standardizer(data["train"]["y_shared"][train_mask])
        specific_mean, specific_std, specific_zero_std = fit_standardizer(data["train"]["y_specific"][train_mask])

        np.savez(
            out_dir / f"{cancer}_shared_standardization_stats.npz",
            mean=shared_mean,
            std=shared_std,
            zero_std_mask=shared_zero_std.astype(np.uint8),
            enhancers=np.array(shared_enhancers, dtype=object),
        )
        np.savez(
            out_dir / f"{cancer}_specific_standardization_stats.npz",
            mean=specific_mean,
            std=specific_std,
            zero_std_mask=specific_zero_std.astype(np.uint8),
            enhancers=np.array(specific_enhancers[cancer], dtype=object),
        )

        stats_rows.append(
            {
                "cancer": cancer,
                "n_train_samples": int(train_mask.sum()),
                "n_shared_targets": len(shared_enhancers),
                "n_specific_targets": len(specific_enhancers[cancer]),
                "shared_zero_std_targets": int(shared_zero_std.sum()),
                "specific_zero_std_targets": int(specific_zero_std.sum()),
            }
        )

        for split in SPLITS:
            split_mask = data[split]["groups"] == cancer
            if not np.any(split_mask):
                continue
            data[split]["y_shared"][split_mask] = apply_standardizer(
                data[split]["y_shared"][split_mask], shared_mean, shared_std
            )
            data[split]["y_specific"][split_mask] = apply_standardizer(
                data[split]["y_specific"][split_mask], specific_mean, specific_std
            )

    for split in SPLITS:
        np.save(out_dir / f"X_{split}.npy", data[split]["X"])
        np.save(out_dir / f"y_shared_{split}.npy", data[split]["y_shared"])
        np.save(out_dir / f"y_specific_{split}.npy", data[split]["y_specific"])
        np.save(out_dir / f"group_{split}.npy", data[split]["groups"])
        shutil.copy2(in_dir / f"id_{split}.txt", out_dir / f"id_{split}.txt")

    write_lines(out_dir / "shared_enhancers.txt", shared_enhancers)
    for cancer in CANCERS:
        write_lines(out_dir / f"{cancer}_specific_enhancers.txt", specific_enhancers[cancer])

    pd.DataFrame(stats_rows).to_csv(out_dir / "standardization_summary.csv", index=False)
    print(f"Saved restandardized matrix to {out_dir}")


if __name__ == "__main__":
    main()
