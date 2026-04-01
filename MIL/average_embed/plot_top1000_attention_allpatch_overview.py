from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
IN_CSV = ROOT / "three_cancer_top1000_attention_allpatch_patch_weights" / "all_patch_attention.csv"
OUT_DIR = ROOT / "three_cancer_top1000_attention_allpatch_patch_weights"


def summarize_by_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows = []
    sample_rows = []
    for (cancer, split, sample_id), g in df.groupby(["cancer", "split", "sample_id"]):
        weights = np.sort(g["attention_weight"].to_numpy())[::-1]
        csum = np.cumsum(weights)
        for rank, weight in enumerate(weights, start=1):
            rank_rows.append(
                {
                    "cancer": cancer,
                    "split": split,
                    "sample_id": sample_id,
                    "rank": rank,
                    "attention_weight": float(weight),
                    "cumulative_attention": float(csum[rank - 1]),
                }
            )
        sample_rows.append(
            {
                "cancer": cancer,
                "split": split,
                "sample_id": sample_id,
                "n_patches": int(len(weights)),
                "top1_attention": float(csum[0]),
                "top5_attention": float(csum[min(4, len(weights) - 1)]),
                "top10_attention": float(csum[min(9, len(weights) - 1)]),
                "top20_attention": float(csum[min(19, len(weights) - 1)]),
                "top50_attention": float(csum[min(49, len(weights) - 1)]),
            }
        )
    return pd.DataFrame(rank_rows), pd.DataFrame(sample_rows)


def make_plot(rank_df: pd.DataFrame, sample_df: pd.DataFrame, out_path: Path) -> None:
    colors = {"BRCA": "#2E6F95", "LUAD": "#4C956C", "SKCM": "#C97C5D"}
    cancers = ["BRCA", "LUAD", "SKCM"]
    split = "test"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    for cancer in cancers:
        tmp = rank_df[(rank_df["cancer"] == cancer) & (rank_df["split"] == split) & (rank_df["rank"] <= 50)]
        mean_curve = tmp.groupby("rank")["attention_weight"].mean()
        ax.plot(
            mean_curve.index,
            mean_curve.values,
            label=cancer,
            color=colors[cancer],
            linewidth=2,
        )
    ax.set_title("Mean Attention by Patch Rank")
    ax.set_xlabel("Patch rank within sample (sorted high to low)")
    ax.set_ylabel("Mean attention weight")
    ax.set_xlim(1, 50)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    x_labels = ["top1_attention", "top5_attention", "top10_attention", "top20_attention", "top50_attention"]
    x_names = ["Top 1", "Top 5", "Top 10", "Top 20", "Top 50"]
    x = np.arange(len(x_labels))
    width = 0.24
    for i, cancer in enumerate(cancers):
        tmp = sample_df[(sample_df["cancer"] == cancer) & (sample_df["split"] == split)]
        means = [tmp[col].mean() for col in x_labels]
        ax.bar(x + (i - 1) * width, means, width=width, color=colors[cancer], label=cancer)
    ax.set_title("Cumulative Attention Captured by Top-k Patches")
    ax.set_xticks(x)
    ax.set_xticklabels(x_names)
    ax.set_ylabel("Mean cumulative attention")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle("Top1000 Attention All-Patch: How Attention Is Distributed Across Patches", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    rank_df, sample_df = summarize_by_sample(df)
    rank_df.to_csv(OUT_DIR / "attention_rank_curve_long.csv", index=False)
    sample_df.to_csv(OUT_DIR / "attention_topk_summary_per_sample.csv", index=False)
    make_plot(rank_df, sample_df, OUT_DIR / "attention_allpatch_overview.png")


if __name__ == "__main__":
    main()
