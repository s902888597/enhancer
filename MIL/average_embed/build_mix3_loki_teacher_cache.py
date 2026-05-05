#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def tcga_case3_from_dirname(dirname: str) -> str:
    parts = dirname.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else dirname


def build_case_map(root: Path, cancer: str) -> dict[str, Path]:
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if cancer == "SKCM":
        return {p.name: p for p in dirs}
    out: dict[str, list[Path]] = {}
    for p in dirs:
        case = tcga_case3_from_dirname(p.name)
        out.setdefault(case, []).append(p)
    return {k: sorted(v, key=lambda p: p.name)[0] for k, v in out.items()}


def mean_embed(case_dir: Path) -> np.ndarray:
    x = np.load(case_dir / "X.npy").astype(np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise RuntimeError(f"bad feature matrix in {case_dir}")
    return x.mean(axis=0).astype(np.float32)


def build_split(split: str, ids: list[str], maps: dict[str, dict[str, Path]]):
    feats = []
    kept = []
    missing = []
    for sample_id in ids:
        cancer, case = sample_id.split(":", 1)
        case_dir = maps[cancer].get(case)
        if case_dir is None or not (case_dir / "X.npy").exists():
            missing.append(sample_id)
            continue
        feats.append(mean_embed(case_dir))
        kept.append(sample_id)
    x = np.stack(feats, axis=0).astype(np.float32)
    return x, kept, missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", required=True)
    p.add_argument("--brca-root", required=True)
    p.add_argument("--luad-root", required=True)
    p.add_argument("--skcm-root", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = {
        "BRCA": build_case_map(Path(args.brca_root), "BRCA"),
        "LUAD": build_case_map(Path(args.luad_root), "LUAD"),
        "SKCM": build_case_map(Path(args.skcm_root), "SKCM"),
    }

    summary = {}
    for split in ["train", "validation", "test"]:
        ids = read_ids(matrix_dir / f"id_{split}.txt")
        x, kept, missing = build_split(split, ids, maps)
        np.save(out_dir / f"{split}_X.npy", x)
        (out_dir / f"{split}_ids.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
        summary[split] = {
            "n_ids_expected": len(ids),
            "n_ids_written": len(kept),
            "n_missing": len(missing),
            "feature_dim": int(x.shape[1]),
            "missing_examples": missing[:20],
        }

    (out_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
