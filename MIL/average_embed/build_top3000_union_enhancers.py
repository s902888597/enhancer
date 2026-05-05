#!/usr/bin/env python3
"""
Build a fixed enhancer panel from the union of BRCA/LUAD/SKCM Top3000
training SE lists.
"""

import argparse
from pathlib import Path

import pandas as pd


def read_se_ids(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path, usecols=["SE_ID"])
    return set(df["SE_ID"].astype(str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brca-csv", required=True)
    parser.add_argument("--luad-csv", required=True)
    parser.add_argument("--skcm-csv", required=True)
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()

    brca = read_se_ids(Path(args.brca_csv))
    luad = read_se_ids(Path(args.luad_csv))
    skcm = read_se_ids(Path(args.skcm_csv))

    enhancers = sorted(brca | luad | skcm)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(enhancers) + "\n")

    print(f"BRCA Top3000: {len(brca)}")
    print(f"LUAD Top3000: {len(luad)}")
    print(f"SKCM Top3000: {len(skcm)}")
    print(f"Union: {len(enhancers)}")
    print(f"Saved enhancer panel to {out_file}")


if __name__ == "__main__":
    main()
