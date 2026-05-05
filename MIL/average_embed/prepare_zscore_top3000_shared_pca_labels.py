#!/usr/bin/env python3
"""Create shared train-fit PCA label CSVs for single-cancer attention runs."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path("/taiga/illinois/vetmed/cb/kwang222/enhancer")
RUN_ROOT = ROOT / "MIL/average_embed/zscore_union21477_erna_ge4_top3000_fixedsplit"
MATRIX_DIR = RUN_ROOT / "matrix"
OUT_ROOT = RUN_ROOT / "shared_pca_labels"
CANCERS = ("BRCA", "LUAD", "SKCM")
KS = (5, 8, 10, 15)


def read_ids(split: str) -> list[str]:
    return [
        line.strip()
        for line in (MATRIX_DIR / f"id_{split}.txt").read_text().splitlines()
        if line.strip()
    ]


def split_case_ids(ids: list[str], cancer: str) -> list[str]:
    out = []
    prefix = cancer + ":"
    for sample_id in ids:
        if sample_id.startswith(prefix):
            out.append(sample_id.split(":", 1)[1])
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    enhancers = [x.strip() for x in (MATRIX_DIR / "enhancers.txt").read_text().splitlines() if x.strip()]
    y = {split: np.load(MATRIX_DIR / f"y_{split}.npy").astype(np.float32) for split in ("train", "validation", "test")}
    groups = {
        split: np.load(MATRIX_DIR / f"group_{split}.npy", allow_pickle=True).astype(str)
        for split in ("train", "validation", "test")
    }
    ids = {split: read_ids(split) for split in ("train", "validation", "test")}

    for k in KS:
        pca_dir = OUT_ROOT / f"pca{k}"
        pca_dir.mkdir(parents=True, exist_ok=True)
        pca = PCA(n_components=k, random_state=44)
        y_pc = {
            "train": pca.fit_transform(y["train"]).astype(np.float32),
            "validation": pca.transform(y["validation"]).astype(np.float32),
            "test": pca.transform(y["test"]).astype(np.float32),
        }
        np.save(pca_dir / "pca_components.npy", pca.components_.astype(np.float32))
        np.save(pca_dir / "pca_mean.npy", pca.mean_.astype(np.float32))
        np.save(pca_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))
        (pca_dir / "enhancers.txt").write_text("\n".join(enhancers) + "\n")
        pd.Series(
            {
                "pca_k": k,
                "fit": "combined BRCA+LUAD+SKCM train only",
                "n_train_samples": int(y["train"].shape[0]),
                "n_targets_original": int(y["train"].shape[1]),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            }
        ).to_json(pca_dir / "shared_pca_config.json", indent=2)

        pc_cols = [f"PC{i + 1}" for i in range(k)]
        for split in ("train", "validation", "test"):
            np.save(pca_dir / f"y_pca_{split}.npy", y_pc[split])
            for cancer in CANCERS:
                use = groups[split] == cancer
                df = pd.DataFrame(y_pc[split][use], columns=pc_cols)
                df.insert(0, "sample", split_case_ids(ids[split], cancer))
                df.to_csv(pca_dir / f"{cancer}_{split}_sharedpca{k}.csv", index=False)

    print(f"Saved shared PCA labels to {OUT_ROOT}")


if __name__ == "__main__":
    main()
