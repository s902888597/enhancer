#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pca-k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(src_dir / "X_train.npy").astype(np.float32)
    x_val = np.load(src_dir / "X_validation.npy").astype(np.float32)
    x_test = np.load(src_dir / "X_test.npy").astype(np.float32)

    pca = PCA(n_components=args.pca_k, random_state=args.seed)
    x_train_pca = pca.fit_transform(x_train).astype(np.float32)
    x_val_pca = pca.transform(x_val).astype(np.float32)
    x_test_pca = pca.transform(x_test).astype(np.float32)

    np.save(out_dir / "X_train.npy", x_train_pca)
    np.save(out_dir / "X_validation.npy", x_val_pca)
    np.save(out_dir / "X_test.npy", x_test_pca)

    for name in [
        "y_shared_train.npy",
        "y_shared_validation.npy",
        "y_shared_test.npy",
        "y_specific_train.npy",
        "y_specific_validation.npy",
        "y_specific_test.npy",
        "group_train.npy",
        "group_validation.npy",
        "group_test.npy",
        "shared_enhancers.txt",
        "BRCA_specific_enhancers.txt",
        "LUAD_specific_enhancers.txt",
        "SKCM_specific_enhancers.txt",
        "ids_train.txt",
        "ids_validation.txt",
        "ids_test.txt",
        "id_train.txt",
        "id_validation.txt",
        "id_test.txt",
    ]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    summary = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "pca_k": args.pca_k,
        "seed": args.seed,
        "train_input_dim": int(x_train.shape[1]),
        "output_dim": int(x_train_pca.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    with open(out_dir / "pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Built PCA matrix: k={args.pca_k}, "
        f"EVR_sum={summary['explained_variance_ratio_sum']:.4f}, "
        f"saved to {out_dir}"
    )


if __name__ == "__main__":
    main()
