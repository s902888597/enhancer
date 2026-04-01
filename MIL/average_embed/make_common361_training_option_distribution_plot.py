from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "common361_training_option_distribution"


def load_independent() -> pd.DataFrame:
    path = ROOT / "common361_best3_mean_mlp" / "common361_test_per_enhancer_merged.csv"
    df = pd.read_csv(path)
    rows = []
    for cancer in ["BRCA", "LUAD", "SKCM"]:
        tmp = df[["SE_ID", cancer]].copy()
        tmp.columns = ["enhancer", "pearson_r"]
        tmp["cancer"] = cancer
        tmp["option"] = "Independent"
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def load_joint(subdir: str, option: str) -> pd.DataFrame:
    rows = []
    for cancer in ["BRCA", "LUAD", "SKCM"]:
        path = ROOT / subdir / "noPCA_seed44" / f"per_enhancer_correlation_test_{cancer}.csv"
        tmp = pd.read_csv(path).copy()
        tmp["cancer"] = cancer
        tmp["option"] = option
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["cancer", "option"])["pearson_r"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            gt_0p4=lambda s: (s > 0.4).sum(),
        )
        .reset_index()
    )
    return summary


def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    option_order = ["Independent", "Joint Single-Head", "Joint Dual-Head"]
    cancer_order = ["BRCA", "LUAD", "SKCM"]
    colors = {
        "Independent": "#8FB8DE",
        "Joint Single-Head": "#A8D5BA",
        "Joint Dual-Head": "#F2B880",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, cancer in zip(axes, cancer_order):
        subsets = [
            df[(df["cancer"] == cancer) & (df["option"] == option)]["pearson_r"].dropna().to_numpy()
            for option in option_order
        ]
        parts = ax.violinplot(
            subsets,
            positions=np.arange(1, len(option_order) + 1),
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, option in zip(parts["bodies"], option_order):
            body.set_facecolor(colors[option])
            body.set_edgecolor("#444444")
            body.set_alpha(0.9)
        parts["cmedians"].set_color("#222222")
        parts["cmedians"].set_linewidth(1.5)

        for i, values in enumerate(subsets, start=1):
            q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            ax.vlines(i, q1, q3, color="#222222", linewidth=4, alpha=0.9)
            ax.scatter([i], [med], color="white", edgecolors="#222222", s=28, zorder=3)

        ax.set_title(cancer, fontsize=11, pad=8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Independent", "Single-head", "Dual-head"], rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(-0.25, 0.75)

    axes[0].set_ylabel("Per-enhancer test Pearson r")
    fig.suptitle("Common361 Test Accuracy Distribution Across Training Strategies", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.concat(
        [
            load_independent(),
            load_joint("mix3_mean_mlp", "Joint Single-Head"),
            load_joint("mix3_dualhead_matrix", "Joint Dual-Head"),
        ],
        ignore_index=True,
    )
    df = df[["cancer", "option", "enhancer", "pearson_r"]].sort_values(
        ["cancer", "option", "enhancer"]
    )
    df.to_csv(OUT_DIR / "long_table.csv", index=False)

    summary = build_summary(df)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    make_plot(df, OUT_DIR / "common361_training_option_violin.png")


if __name__ == "__main__":
    main()
