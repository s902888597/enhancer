#!/usr/bin/env python3
"""
Convert label CSVs from samples-as-columns to samples-as-rows.

Input:
  chr,start,end,SE_ID,eRNA_count,<sample1>,<sample2>,...

Output:
  sample,<SE1>,<SE2>,...
"""

import argparse
from pathlib import Path

import pandas as pd

META_COLS = ["chr", "start", "end", "SE_ID", "eRNA_count"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--pattern", default="*.csv")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matched in {src_dir}")

    for path in files:
        df = pd.read_csv(path)
        sample_cols = [c for c in df.columns if c not in META_COLS]
        if not sample_cols:
            continue

        if "SE_ID" in df.columns:
            se_names = df["SE_ID"].astype(str).tolist()
        else:
            se_names = [f"enh_{i}" for i in range(df.shape[0])]

        out = df[sample_cols].T
        out.columns = se_names
        out.insert(0, "sample", sample_cols)
        out_path = dst_dir / path.name
        out.to_csv(out_path, index=False)
        print(f"converted: {path.name} -> {out_path}")

    print(f"done: {len(files)} files")


if __name__ == "__main__":
    main()

