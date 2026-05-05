#!/usr/bin/env python3
"""
Clean pan-cancer label CSVs into stable UTF-8 CSVs.

What it does:
- remove NULL bytes
- robust decode (utf-8/latin1/utf-16 fallback)
- normalize malformed rows (truncate/pad to header length)
- keep only numeric chars for label cells, coerce to float
- fill NaN by row median, then 0
- clip extreme values
- save clean UTF-8 CSV
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = {"chr", "start", "end", "SE_ID", "eRNA_count"}


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "utf-16"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass

    raw = path.read_bytes().replace(b"\x00", b"")
    text = None
    for enc in ("utf-8", "latin1", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            pass
    if text is None:
        text = raw.decode("latin1", errors="ignore")

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    header = rows[0]
    ncol = len(header)
    out = []
    for r in rows[1:]:
        if len(r) > ncol:
            r = r[:ncol]
        elif len(r) < ncol:
            r = r + [""] * (ncol - len(r))
        out.append(r)
    return pd.DataFrame(out, columns=header)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    sample_cols = [c for c in df.columns if c not in META_COLS]
    if sample_cols:
        vals = df[sample_cols].astype(str).apply(
            lambda c: c.str.replace(r"[^0-9eE+\-\.]", "", regex=True)
        )
        vals = vals.apply(pd.to_numeric, errors="coerce")
        row_median = vals.median(axis=1)
        vals = vals.T.fillna(row_median).T.fillna(0.0)
        vals = vals.clip(lower=-1e6, upper=1e6)
        df[sample_cols] = vals
    return df


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
        raise RuntimeError(f"No files matched in {src_dir} with pattern {args.pattern}")

    for f in files:
        df = read_csv_robust(f)
        df = clean_df(df)
        out = dst_dir / f.name
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"cleaned: {f.name} -> {out}")

    print(f"done: {len(files)} files")


if __name__ == "__main__":
    main()

