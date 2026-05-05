#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/MIL/average_embed/single_cancer_top1000_ypca_sweep",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rows = []
    for run_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        cfg_path = run_dir / "run_config.json"
        summary_path = run_dir / "summary.csv"
        pred_pc_path = run_dir / "predicted_pc_summary.csv"
        if not (cfg_path.exists() and summary_path.exists() and pred_pc_path.exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        summary_df = pd.read_csv(summary_path)
        pred_pc_df = pd.read_csv(pred_pc_path)
        row = {
            "run_dir": run_dir.name,
            "cancer": cfg["cancer"],
            "pca_k": cfg["pca_k"],
            "pca_evr_sum": cfg["pca_evr_sum"],
        }
        for _, r in summary_df.iterrows():
            split = r["split"]
            row[f"{split}_enh_mean"] = r["pearson_mean"]
            row[f"{split}_enh_median"] = r["pearson_median"]
            row[f"{split}_enh_gt0.4"] = int(r["gt_0.4"])
        for _, r in pred_pc_df.iterrows():
            split = r["split"]
            row[f"{split}_pc_mean"] = r["pearson_mean"]
            row[f"{split}_pc_median"] = r["pearson_median"]
            row[f"{split}_pc_min"] = r["pearson_min"]
            row[f"{split}_pc_max"] = r["pearson_max"]
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["cancer", "pca_k"])
    out_path = base_dir / "sweep_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(out_path)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
