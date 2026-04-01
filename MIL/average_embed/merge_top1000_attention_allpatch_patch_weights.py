from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

SPECS = {
    "BRCA": ROOT / "brca_top1000_attention_rawpatch_seed44",
    "LUAD": ROOT / "luad_top1000_attention_rawpatch_seed44",
    "SKCM": ROOT / "skcm_top1000_attention_rawpatch_seed44",
}

OUT_DIR = ROOT / "three_cancer_top1000_attention_allpatch_patch_weights"


def load_split(cancer: str, base_dir: Path, split: str) -> pd.DataFrame:
    split_dir = base_dir / f"{split}_patch_attention"
    rows = []
    for csv_path in sorted(split_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["cancer"] = cancer
        df["split"] = split
        df["sample_id"] = csv_path.stem
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["cancer", "split", "sample_id", "patch_file", "attention_weight"])
    out = pd.concat(rows, ignore_index=True)
    return out[["cancer", "split", "sample_id", "patch_file", "attention_weight"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for cancer, base_dir in SPECS.items():
        for split in ["val", "test"]:
            frames.append(load_split(cancer, base_dir, split))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(
        ["cancer", "split", "sample_id", "attention_weight"],
        ascending=[True, True, True, False],
    )
    merged.to_csv(OUT_DIR / "all_patch_attention.csv", index=False)

    top10 = (
        merged.groupby(["cancer", "split", "sample_id"], group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10.to_csv(OUT_DIR / "top10_patch_attention_per_sample.csv", index=False)


if __name__ == "__main__":
    main()
