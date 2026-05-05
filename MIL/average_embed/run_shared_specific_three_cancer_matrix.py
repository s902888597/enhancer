#!/usr/bin/env python3
"""
Train a mixed-cancer model with:
- one shared head for shared enhancers
- three cancer-specific heads for cancer-specific enhancers
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from run_mean_regression import set_seed


CANCER_TO_IDX = {"BRCA": 0, "LUAD": 1, "SKCM": 2}


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


class SharedSpecificDataset(Dataset):
    def __init__(self, x, y_shared, y_specific, groups):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_shared = torch.tensor(y_shared, dtype=torch.float32)
        self.y_specific = torch.tensor(y_specific, dtype=torch.float32)
        self.groups = torch.tensor([CANCER_TO_IDX[g] for g in groups], dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y_shared[idx], self.y_specific[idx], self.groups[idx]


class SharedSpecificRegressor(nn.Module):
    def __init__(self, input_dim: int, shared_dim: int, specific_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shared_head = nn.Linear(hidden_dim, shared_dim)
        self.specific_heads = nn.ModuleList([nn.Linear(hidden_dim, specific_dim) for _ in range(3)])

    def forward(self, x, groups):
        h = self.backbone(x)
        pred_shared = self.shared_head(h)
        pred_specific = torch.zeros(x.shape[0], self.specific_heads[0].out_features, device=x.device)
        for idx, head in enumerate(self.specific_heads):
            use = groups == idx
            if use.any():
                pred_specific[use] = head(h[use])
        return pred_shared, pred_specific


def train_epoch(
    model,
    loader,
    optim,
    loss_fn,
    device,
    alpha_shared: float,
    alpha_specific: float,
    alpha_specific_brca: float,
    alpha_specific_luad: float,
    alpha_specific_skcm: float,
):
    model.train()
    total = 0.0
    n = 0
    for xb, ysb, ysp, gb in loader:
        xb = xb.to(device)
        ysb = ysb.to(device)
        ysp = ysp.to(device)
        gb = gb.to(device)
        optim.zero_grad(set_to_none=True)
        pred_shared, pred_specific = model(xb, gb)
        loss_shared = loss_fn(pred_shared, ysb)
        loss = alpha_shared * loss_shared
        for idx, alpha_c in enumerate([alpha_specific_brca, alpha_specific_luad, alpha_specific_skcm]):
            use = gb == idx
            if use.any():
                loss = loss + alpha_specific * alpha_c * loss_fn(pred_specific[use], ysp[use])
        loss.backward()
        optim.step()
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


def eval_epoch(
    model,
    loader,
    loss_fn,
    device,
    alpha_shared: float,
    alpha_specific: float,
    alpha_specific_brca: float,
    alpha_specific_luad: float,
    alpha_specific_skcm: float,
):
    model.eval()
    total = 0.0
    n = 0
    out = {"shared_pred": [], "specific_pred": [], "shared_true": [], "specific_true": [], "groups": []}
    with torch.no_grad():
        for xb, ysb, ysp, gb in loader:
            xb = xb.to(device)
            ysb = ysb.to(device)
            ysp = ysp.to(device)
            gb = gb.to(device)
            pred_shared, pred_specific = model(xb, gb)
            loss_shared = loss_fn(pred_shared, ysb)
            loss = alpha_shared * loss_shared
            for idx, alpha_c in enumerate([alpha_specific_brca, alpha_specific_luad, alpha_specific_skcm]):
                use = gb == idx
                if use.any():
                    loss = loss + alpha_specific * alpha_c * loss_fn(pred_specific[use], ysp[use])
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
            out["shared_pred"].append(pred_shared.cpu().numpy())
            out["specific_pred"].append(pred_specific.cpu().numpy())
            out["shared_true"].append(ysb.cpu().numpy())
            out["specific_true"].append(ysp.cpu().numpy())
            out["groups"].append(gb.cpu().numpy())
    for k in out:
        out[k] = np.concatenate(out[k], axis=0)
    return total / max(n, 1), out


def pearson_df(preds: np.ndarray, trues: np.ndarray, enhancers: List[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(enhancers):
        x = preds[:, i]
        y = trues[:, i]
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            r = np.nan
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"enhancer": name, "pearson_r": r})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str):
    s = df["pearson_r"]
    print(
        f"{label}: mean={s.mean(skipna=True):.4f} "
        f"median={s.median(skipna=True):.4f} >0.4={(s > 0.4).sum()}"
    )


def assemble_full_predictions(out: Dict[str, np.ndarray], shared_enh: List[str], specific_enh: Dict[str, List[str]]):
    groups = out["groups"]
    group_names = np.array(["BRCA", "LUAD", "SKCM"])[groups]
    full_pred = {}
    full_true = {}
    for cancer in ["BRCA", "LUAD", "SKCM"]:
        use = group_names == cancer
        pred = np.concatenate([out["shared_pred"][use], out["specific_pred"][use]], axis=1)
        true = np.concatenate([out["shared_true"][use], out["specific_true"][use]], axis=1)
        enh = shared_enh + specific_enh[cancer]
        full_pred[cancer] = (pred, enh)
        full_true[cancer] = (true, enh)
    return full_pred, full_true


def summarize_head_breakdown(out: Dict[str, np.ndarray], shared_enh: List[str], specific_enh: Dict[str, List[str]], split_name: str, out_dir: Path):
    groups = out["groups"]
    group_names = np.array(["BRCA", "LUAD", "SKCM"])[groups]
    rows = []

    shared_df_all = pearson_df(out["shared_pred"], out["shared_true"], shared_enh)
    rows.append(
        {
            "split": split_name,
            "head": "shared",
            "group": "Combined",
            "n_targets": len(shared_enh),
            "mean_pearson": shared_df_all["pearson_r"].mean(skipna=True),
            "median_pearson": shared_df_all["pearson_r"].median(skipna=True),
            "gt_0.4": int((shared_df_all["pearson_r"] > 0.4).sum()),
            "gt_0.5": int((shared_df_all["pearson_r"] > 0.5).sum()),
        }
    )
    for cancer in ["BRCA", "LUAD", "SKCM"]:
        use = group_names == cancer
        shared_df = pearson_df(out["shared_pred"][use], out["shared_true"][use], shared_enh)
        specific_df = pearson_df(out["specific_pred"][use], out["specific_true"][use], specific_enh[cancer])
        rows.append(
            {
                "split": split_name,
                "head": "shared",
                "group": cancer,
                "n_targets": len(shared_enh),
                "mean_pearson": shared_df["pearson_r"].mean(skipna=True),
                "median_pearson": shared_df["pearson_r"].median(skipna=True),
                "gt_0.4": int((shared_df["pearson_r"] > 0.4).sum()),
                "gt_0.5": int((shared_df["pearson_r"] > 0.5).sum()),
            }
        )
        rows.append(
            {
                "split": split_name,
                "head": "specific",
                "group": cancer,
                "n_targets": len(specific_enh[cancer]),
                "mean_pearson": specific_df["pearson_r"].mean(skipna=True),
                "median_pearson": specific_df["pearson_r"].median(skipna=True),
                "gt_0.4": int((specific_df["pearson_r"] > 0.4).sum()),
                "gt_0.5": int((specific_df["pearson_r"] > 0.5).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{split_name}_head_breakdown.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--group-filter", choices=["all", "BRCA", "LUAD", "SKCM"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha-shared", type=float, default=1.0)
    parser.add_argument("--alpha-specific", type=float, default=1.0)
    parser.add_argument("--alpha-specific-brca", type=float, default=1.0)
    parser.add_argument("--alpha-specific-luad", type=float, default=1.0)
    parser.add_argument("--alpha-specific-skcm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_enh = read_lines(matrix_dir / "shared_enhancers.txt")
    specific_enh = {
        c: read_lines(matrix_dir / f"{c}_specific_enhancers.txt")
        for c in ["BRCA", "LUAD", "SKCM"]
    }

    data = {}
    for split in ["train", "validation", "test"]:
        data[split] = {
            "X": np.load(matrix_dir / f"X_{split}.npy").astype(np.float32),
            "y_shared": np.load(matrix_dir / f"y_shared_{split}.npy").astype(np.float32),
            "y_specific": np.load(matrix_dir / f"y_specific_{split}.npy").astype(np.float32),
            "groups": np.load(matrix_dir / f"group_{split}.npy", allow_pickle=True),
        }
        if args.group_filter != "all":
            mask = data[split]["groups"] == args.group_filter
            data[split]["X"] = data[split]["X"][mask]
            data[split]["y_shared"] = data[split]["y_shared"][mask]
            data[split]["y_specific"] = data[split]["y_specific"][mask]
            data[split]["groups"] = data[split]["groups"][mask]

    train_loader = DataLoader(
        SharedSpecificDataset(
            data["train"]["X"],
            data["train"]["y_shared"],
            data["train"]["y_specific"],
            data["train"]["groups"],
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SharedSpecificDataset(
            data["validation"]["X"],
            data["validation"]["y_shared"],
            data["validation"]["y_specific"],
            data["validation"]["groups"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SharedSpecificDataset(
            data["test"]["X"],
            data["test"]["y_shared"],
            data["test"]["y_specific"],
            data["test"]["groups"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SharedSpecificRegressor(
        input_dim=data["train"]["X"].shape[1],
        shared_dim=len(shared_enh),
        specific_dim=data["train"]["y_specific"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optim,
            loss_fn,
            device,
            args.alpha_shared,
            args.alpha_specific,
            args.alpha_specific_brca,
            args.alpha_specific_luad,
            args.alpha_specific_skcm,
        )
        val_loss, _ = eval_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            args.alpha_shared,
            args.alpha_specific,
            args.alpha_specific_brca,
            args.alpha_specific_luad,
            args.alpha_specific_skcm,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_out = eval_epoch(
        model,
        val_loader,
        loss_fn,
        device,
        args.alpha_shared,
        args.alpha_specific,
        args.alpha_specific_brca,
        args.alpha_specific_luad,
        args.alpha_specific_skcm,
    )
    test_loss, test_out = eval_epoch(
        model,
        test_loader,
        loss_fn,
        device,
        args.alpha_shared,
        args.alpha_specific,
        args.alpha_specific_brca,
        args.alpha_specific_luad,
        args.alpha_specific_skcm,
    )
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    for split_name, out in [("validation", val_out), ("test", test_out)]:
        summarize_head_breakdown(out, shared_enh, specific_enh, split_name, out_dir)
        full_pred, full_true = assemble_full_predictions(out, shared_enh, specific_enh)
        summary_rows = []
        for cancer in ["BRCA", "LUAD", "SKCM"]:
            pred, enh = full_pred[cancer]
            true, _ = full_true[cancer]
            df = pearson_df(pred, true, enh)
            summarize(df, f"{split_name}_{cancer}")
            df.to_csv(out_dir / f"per_enhancer_correlation_{split_name}_{cancer}.csv", index=False)
            summary_rows.append(
                {
                    "split": split_name,
                    "cancer": cancer,
                    "mean_pearson": df["pearson_r"].mean(skipna=True),
                    "median_pearson": df["pearson_r"].median(skipna=True),
                    "gt_0.4": int((df["pearson_r"] > 0.4).sum()),
                    "gt_0.5": int((df["pearson_r"] > 0.5).sum()),
                }
            )
        pd.DataFrame(summary_rows).to_csv(out_dir / f"summary_{split_name}.csv", index=False)

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
