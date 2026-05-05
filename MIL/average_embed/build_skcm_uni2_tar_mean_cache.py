#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import h5py
import numpy as np


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def case3_from_name(name: str) -> str:
    parts = Path(name).stem.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else Path(name).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar-path", required=True)
    parser.add_argument("--ref-cache-dir", required=True)
    parser.add_argument("--out-cache-dir", required=True)
    parser.add_argument("--tmp-dir", default="/tmp")
    args = parser.parse_args()

    tar_path = Path(args.tar_path)
    ref_cache_dir = Path(args.ref_cache_dir)
    out_cache_dir = Path(args.out_cache_dir)
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    split_to_ids = {
        "train": read_ids(ref_cache_dir / "SKCM_train_train_ids.txt"),
        "validation": read_ids(ref_cache_dir / "SKCM_validation_validation_ids.txt"),
        "test": read_ids(ref_cache_dir / "SKCM_test_test_ids.txt"),
    }
    target_ids = set()
    for ids in split_to_ids.values():
        target_ids.update(ids)

    feat_sum: dict[str, np.ndarray] = {}
    feat_count: dict[str, int] = {}
    matched_slides = 0
    scanned_files = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if not member.name.endswith(".h5"):
                continue
            scanned_files += 1
            case_id = case3_from_name(member.name) + "_tumor"
            if case_id not in target_ids:
                continue

            src = tf.extractfile(member)
            if src is None:
                continue

            with tempfile.NamedTemporaryFile(suffix=".h5", dir=tmp_dir, delete=True) as tmp:
                shutil.copyfileobj(src, tmp)
                tmp.flush()
                with h5py.File(tmp.name, "r") as f:
                    x = f["features"][:]
                    if x.ndim == 3:
                        x = x[0]
                    x = np.asarray(x, dtype=np.float32)
                    if x.ndim != 2:
                        raise RuntimeError(f"Unexpected feature shape for {member.name}: {x.shape}")
                    if case_id not in feat_sum:
                        feat_sum[case_id] = np.zeros((x.shape[1],), dtype=np.float64)
                        feat_count[case_id] = 0
                    feat_sum[case_id] += x.sum(axis=0, dtype=np.float64)
                    feat_count[case_id] += int(x.shape[0])
                    matched_slides += 1

            if scanned_files % 50 == 0:
                print(
                    f"[progress] scanned_h5={scanned_files} matched_slides={matched_slides} "
                    f"matched_cases={len(feat_sum)}",
                    flush=True,
                )

    summary = {
        "tar_path": str(tar_path),
        "scanned_h5_files": scanned_files,
        "matched_slides": matched_slides,
        "matched_cases": len(feat_sum),
        "feature_dim": int(next(iter(feat_sum.values())).shape[0]) if feat_sum else 0,
        "splits": {},
    }

    for split, ids in split_to_ids.items():
        kept_ids = [sid for sid in ids if sid in feat_sum and feat_count.get(sid, 0) > 0]
        missing_ids = [sid for sid in ids if sid not in feat_sum or feat_count.get(sid, 0) <= 0]
        x = np.stack([feat_sum[sid] / feat_count[sid] for sid in kept_ids], axis=0).astype(np.float32)
        np.save(out_cache_dir / f"SKCM_{split}_{split}_X.npy", x)
        (out_cache_dir / f"SKCM_{split}_{split}_ids.txt").write_text("\n".join(kept_ids) + ("\n" if kept_ids else ""))
        summary["splits"][split] = {
            "n_ids_requested": len(ids),
            "n_ids_kept": len(kept_ids),
            "n_ids_missing": len(missing_ids),
            "missing_examples": missing_ids[:10],
            "x_shape": list(x.shape),
        }
        print(
            f"[{split}] kept={len(kept_ids)} missing={len(missing_ids)} shape={tuple(x.shape)}",
            flush=True,
        )

    with (out_cache_dir / "cache_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved UNI2 mean cache to {out_cache_dir}", flush=True)


if __name__ == "__main__":
    main()
