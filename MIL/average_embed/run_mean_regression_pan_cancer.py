#!/usr/bin/env python3
"""
Mean-pool patch embeddings per case and train an MLP regressor.

Supported label layouts:
- samples_as_columns: BRCA/LUAD style
    chr,start,end,SE_ID,eRNA_count,<sample1>,<sample2>,...
- samples_as_rows: SKCM style
    sample,<enh1>,<enh2>,...

Supported feature layouts:
- cancer_dirs: <feat-root>/<CANCER>/<sample_dir>/*.npy
- split_dirs:  <feat-root>/<split>/<case>/*.npy
"""

import argparse
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


META_COLS = {"chr", "start", "end", "SE_ID", "eRNA_count"}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tcga_case3_from_label_col(col: str) -> str:
    # TCGA-50-5066_tumor -> TCGA-50-5066
    s = col.replace("_tumor", "").replace("_normal", "").strip()
    parts = s.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else s


def normalize_sample_id(sample_id: str, sample_id_mode: str) -> str:
    s = str(sample_id).strip()
    if sample_id_mode == "tcga_case3":
        return tcga_case3_from_label_col(s)
    return s


def tcga_case3_from_dirname(dirname: str) -> str:
    # TCGA-50-5066-01Z-00-DX1 -> TCGA-50-5066
    parts = dirname.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else dirname


def sample_type_from_dirname(dirname: str) -> str:
    # TCGA-50-5066-01Z-00-DX1 -> 01
    parts = dirname.split("-")
    if len(parts) >= 4 and len(parts[3]) >= 2:
        return parts[3][:2]
    return ""


def build_case_to_feature_dir(cancer_feat_root: Path) -> Dict[str, Path]:
    dirs = sorted([p for p in cancer_feat_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    case_map: Dict[str, List[Path]] = {}
    for d in dirs:
        case = tcga_case3_from_dirname(d.name)
        case_map.setdefault(case, []).append(d)

    resolved: Dict[str, Path] = {}
    for case, candidates in case_map.items():
        # Prefer sample type 01 when duplicates exist, otherwise first lexicographic.
        c01 = [c for c in candidates if sample_type_from_dirname(c.name) == "01"]
        pick = sorted(c01, key=lambda p: p.name)[0] if c01 else sorted(candidates, key=lambda p: p.name)[0]
        resolved[case] = pick
    return resolved


def load_labels_samples_as_columns(csv_path: Path) -> Tuple[List[str], pd.DataFrame]:
    df = read_csv_with_fallback(csv_path)
    sample_cols = [c for c in df.columns if c not in META_COLS]
    if not sample_cols:
        raise RuntimeError(f"No sample columns found in {csv_path}")

    # enhancer ids for label columns
    if "SE_ID" in df.columns:
        enh_cols = df["SE_ID"].astype(str).fillna("").tolist()
    else:
        enh_cols = [f"enh_{i}" for i in range(df.shape[0])]

    # Some uploaded CSVs contain a few malformed numeric strings (encoding artifacts).
    # Keep only numeric/exponent characters, then coerce to float.
    val_df = df[sample_cols].astype(str).apply(
        lambda c: c.str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    )
    val_df = val_df.apply(pd.to_numeric, errors="coerce")

    nan_count = int(np.isnan(val_df.values).sum())
    if nan_count > 0:
        print(f"[warn] {csv_path.name}: non-numeric cells coerced to NaN = {nan_count}")
        # Fill NaN by enhancer-wise median (row median across samples), then fallback to 0.
        row_median = val_df.median(axis=1)
        val_df = val_df.T.fillna(row_median).T
        val_df = val_df.fillna(0.0)

    # Guard against extreme corrupted values after cleaning.
    val_df = val_df.clip(lower=-1e6, upper=1e6)

    # transpose to samples x enhancers DataFrame
    sample_case_ids = [tcga_case3_from_label_col(c) for c in sample_cols]
    label_df = val_df.T
    label_df.index = sample_case_ids
    label_df.columns = enh_cols

    # Aggregate duplicate case IDs (rare) by mean.
    if label_df.index.duplicated().any():
        dup_n = int(label_df.index.duplicated().sum())
        print(f"[warn] {csv_path.name}: duplicated case ids in label columns = {dup_n}, aggregating by mean")
        label_df = label_df.groupby(level=0).mean()

    return list(label_df.index), label_df


def load_labels_samples_as_rows(csv_path: Path, sample_id_mode: str) -> Tuple[List[str], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "sample" not in df.columns:
        raise RuntimeError(f"'sample' column not found in {csv_path}")
    enh_cols = [c for c in df.columns if c != "sample"]
    if not enh_cols:
        raise RuntimeError(f"No enhancer columns found in {csv_path}")

    label_df = df.copy()
    label_df["sample"] = label_df["sample"].astype(str).map(lambda x: normalize_sample_id(x, sample_id_mode))
    label_df[enh_cols] = label_df[enh_cols].apply(pd.to_numeric, errors="coerce")
    nan_count = int(np.isnan(label_df[enh_cols].values).sum())
    if nan_count > 0:
        print(f"[warn] {csv_path.name}: non-numeric cells coerced to NaN = {nan_count}")
        row_median = label_df[enh_cols].median(axis=1)
        label_df[enh_cols] = label_df[enh_cols].T.fillna(row_median).T.fillna(0.0)
    label_df[enh_cols] = label_df[enh_cols].clip(lower=-1e6, upper=1e6)
    label_df = label_df.set_index("sample")

    if label_df.index.duplicated().any():
        dup_n = int(label_df.index.duplicated().sum())
        print(f"[warn] {csv_path.name}: duplicated sample ids = {dup_n}, aggregating by mean")
        label_df = label_df.groupby(level=0).mean()

    return list(label_df.index), label_df[enh_cols]


def load_label_df(csv_path: Path, label_layout: str, sample_id_mode: str) -> Tuple[List[str], pd.DataFrame]:
    if label_layout == "samples_as_columns":
        return load_labels_samples_as_columns(csv_path)
    if label_layout == "samples_as_rows":
        return load_labels_samples_as_rows(csv_path, sample_id_mode)
    raise ValueError(f"Unsupported label_layout: {label_layout}")


def build_identity_case_to_feature_dir(root: Path) -> Dict[str, Path]:
    return {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    # Try common encodings first.
    for enc in ("utf-8", "latin1", "utf-16"):
        try:
            return pd.read_csv(csv_path, encoding=enc, low_memory=False)
        except Exception:
            pass

    # Robust fallback: strip NULL bytes and parse from decoded text.
    raw = csv_path.read_bytes().replace(b"\x00", b"")
    text = None
    for enc in ("utf-8", "latin1", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            pass
    if text is None:
        text = raw.decode("latin1", errors="ignore")

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty CSV after fallback decode: {csv_path}")
    header = rows[0]
    ncol = len(header)
    data_rows = []
    bad_rows = 0
    for r in rows[1:]:
        if len(r) == ncol:
            data_rows.append(r)
            continue
        bad_rows += 1
        if len(r) > ncol:
            data_rows.append(r[:ncol])
        else:
            data_rows.append(r + [""] * (ncol - len(r)))
    if bad_rows > 0:
        print(f"[warn] {csv_path.name}: normalized malformed rows = {bad_rows}")
    return pd.DataFrame(data_rows, columns=header)


def mean_embed_for_case_dir(case_dir: Path) -> np.ndarray:
    files = sorted(case_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No npy files in {case_dir}")
    arrs = []
    bad = 0
    bad_examples = []
    for f in files:
        try:
            arrs.append(np.load(f).astype(np.float32))
        except Exception as e:
            bad += 1
            if len(bad_examples) < 3:
                bad_examples.append(f"{f.name} ({type(e).__name__})")
    if bad > 0:
        print(
            f"[warn] {case_dir.name}: skipped bad npy files={bad} "
            f"(e.g., {bad_examples})"
        )
    if not arrs:
        raise FileNotFoundError(f"All npy files are unreadable in {case_dir}")
    return np.stack(arrs, axis=0).mean(axis=0)


def load_split_feats(
    split_name: str,
    csv_path: Path,
    case_to_dir: Dict[str, Path],
    label_layout: str,
    sample_id_mode: str,
    enh_ref: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    cancer: str = "",
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    case_ids, label_df = load_label_df(csv_path, label_layout, sample_id_mode)

    if enh_ref is None:
        enh_cols = list(label_df.columns)
    else:
        # Align val/test to train enhancer space.
        cur = set(label_df.columns)
        ref = set(enh_ref)
        miss = ref - cur
        extra = cur - ref
        if miss or extra:
            print(
                f"[{split_name}] enhancer alignment: "
                f"missing_from_split={len(miss)} extra_in_split={len(extra)}"
            )
        label_df = label_df.reindex(columns=enh_ref, fill_value=0.0)
        enh_cols = enh_ref

    labels = label_df.values.astype(np.float32)

    cache_x = None
    cache_ids = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{cancer}_{split_name}_{csv_path.stem}"
        cache_x = cache_dir / f"{tag}_X.npy"
        cache_ids = cache_dir / f"{tag}_ids.txt"
        if cache_x.exists() and cache_ids.exists():
            kept_ids = [s.strip() for s in cache_ids.read_text().splitlines() if s.strip()]
            x = np.load(cache_x).astype(np.float32)
            # align labels in the same kept_ids order
            avail = [sid for sid in kept_ids if sid in label_df.index]
            if len(avail) != len(kept_ids):
                print(
                    f"[{split_name}] cache ids mismatch current labels: "
                    f"cached={len(kept_ids)} matched={len(avail)}, fallback recompute"
                )
            else:
                y = label_df.loc[kept_ids].values.astype(np.float32)
                if x.shape[0] == y.shape[0]:
                    print(f"[{split_name}] loaded mean feature cache: {cache_x}")
                    return kept_ids, x, y, list(enh_cols)
                print(
                    f"[{split_name}] cache shape mismatch X={x.shape} y={y.shape}, fallback recompute"
                )
    feats = []
    kept_ids = []
    kept_labels = []
    missing = []
    for cid, lab in zip(case_ids, labels):
        case_dir = case_to_dir.get(cid)
        if case_dir is None:
            missing.append(cid)
            continue
        try:
            emb = mean_embed_for_case_dir(case_dir)
        except FileNotFoundError:
            missing.append(cid)
            continue
        kept_ids.append(cid)
        kept_labels.append(lab.astype(np.float32))
        feats.append(emb)

    if missing:
        print(f"[{split_name}] missing/empty: {len(missing)} (e.g., {missing[:10]})")
    if not feats:
        raise RuntimeError(f"No features loaded for split {split_name}")
    x = np.stack(feats, axis=0).astype(np.float32)
    y = np.stack(kept_labels, axis=0).astype(np.float32)

    if cache_x is not None and cache_ids is not None:
        np.save(cache_x, x)
        cache_ids.write_text("\n".join(kept_ids) + ("\n" if kept_ids else ""))
        print(f"[{split_name}] saved mean feature cache: {cache_x}")

    return kept_ids, x, y, list(enh_cols)


class FeatDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
            preds.append(pred.cpu())
            trues.append(yb.cpu())
    return total / max(n, 1), torch.cat(preds), torch.cat(trues)


def pearson_per_feature(preds: np.ndarray, trues: np.ndarray, enh_cols: List[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(enh_cols):
        x = preds[:, i]
        y = trues[:, i]
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            r = np.nan
        else:
            r = np.corrcoef(x, y)[0, 1]
        rows.append({"enhancer": name, "pearson_r": r})
    return pd.DataFrame(rows)


def summarize_corr(corr_df: pd.DataFrame, split: str):
    s = corr_df["pearson_r"]
    print(
        f"{split} pearson: mean={s.mean(skipna=True):.4f} "
        f"median={s.median(skipna=True):.4f} >0.4={(s > 0.4).sum()}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", choices=["BRCA", "LUAD", "SKCM"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument(
        "--feat-root",
        default="/taiga/illinois/vetmed/cb/kwang222/enhancer/our_code_data/data/features_patches_pan_cancer_npy",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--pca-k", type=int, default=0, help="0 disables PCA; >0 applies PCA on labels fit on train only")
    parser.add_argument(
        "--label-layout",
        choices=["samples_as_columns", "samples_as_rows"],
        default="samples_as_rows",
    )
    parser.add_argument(
        "--feature-layout",
        choices=["cancer_dirs", "split_dirs"],
        default="cancer_dirs",
    )
    parser.add_argument(
        "--sample-id-mode",
        choices=["tcga_case3", "identity"],
        default="tcga_case3",
    )
    parser.add_argument(
        "--mean-cache-dir",
        default="",
        help="Optional cache dir for split-level mean features; speeds up repeated runs.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.mean_cache_dir) if args.mean_cache_dir else None
    feat_root = Path(args.feat_root)

    if args.feature_layout == "cancer_dirs":
        cancer_feat_root = feat_root / args.cancer
        case_to_dir_train = build_case_to_feature_dir(cancer_feat_root)
        case_to_dir_val = case_to_dir_train
        case_to_dir_test = case_to_dir_train
        print(f"[{args.cancer}] feature_dirs={len(list(cancer_feat_root.glob('*')))} case_map={len(case_to_dir_train)}")
    else:
        train_root = feat_root / "train"
        val_root = feat_root / "validation"
        test_root = feat_root / "test"
        case_to_dir_train = build_identity_case_to_feature_dir(train_root)
        case_to_dir_val = build_identity_case_to_feature_dir(val_root)
        case_to_dir_test = build_identity_case_to_feature_dir(test_root)
        print(
            f"[{args.cancer}] split feature dirs: "
            f"train={len(case_to_dir_train)} validation={len(case_to_dir_val)} test={len(case_to_dir_test)}"
        )

    train_ids, x_train, y_train, enh_cols = load_split_feats(
        "train",
        Path(args.train_csv),
        case_to_dir_train,
        args.label_layout,
        args.sample_id_mode,
        enh_ref=None,
        cache_dir=cache_dir,
        cancer=args.cancer,
    )
    val_ids, x_val, y_val, _ = load_split_feats(
        "validation",
        Path(args.val_csv),
        case_to_dir_val,
        args.label_layout,
        args.sample_id_mode,
        enh_ref=enh_cols,
        cache_dir=cache_dir,
        cancer=args.cancer,
    )
    test_ids, x_test, y_test, _ = load_split_feats(
        "test",
        Path(args.test_csv),
        case_to_dir_test,
        args.label_layout,
        args.sample_id_mode,
        enh_ref=enh_cols,
        cache_dir=cache_dir,
        cancer=args.cancer,
    )

    pca = None
    y_train_fit = y_train
    y_val_fit = y_val
    y_test_fit = y_test
    if args.pca_k > 0:
        k = min(args.pca_k, y_train.shape[0], y_train.shape[1])
        pca = PCA(n_components=k, random_state=args.seed)
        y_train_fit = pca.fit_transform(y_train).astype(np.float32)
        y_val_fit = pca.transform(y_val).astype(np.float32)
        y_test_fit = pca.transform(y_test).astype(np.float32)
        print(f"PCA enabled: k={k}, EVR_sum={float(pca.explained_variance_ratio_.sum()):.4f}")

    input_dim = x_train.shape[1]
    output_dim = y_train_fit.shape[1]
    train_loader = DataLoader(FeatDataset(x_train, y_train_fit), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeatDataset(x_val, y_val_fit), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FeatDataset(x_test, y_test_fit), batch_size=args.batch_size, shuffle=False)

    model = SimpleRegressor(input_dim, output_dim, args.hidden_dim, args.dropout).to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, loss_fn, device)
        val_loss, _, _ = eval_epoch(model, val_loader, loss_fn, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_pred_t, val_true_t = eval_epoch(model, val_loader, loss_fn, device)
    test_loss, test_pred_t, test_true_t = eval_epoch(model, test_loader, loss_fn, device)
    print(f"Final: val_loss={val_loss:.4f} test_loss={test_loss:.4f}")

    val_pred = val_pred_t.numpy()
    test_pred = test_pred_t.numpy()

    if pca is not None:
        val_pred_orig = pca.inverse_transform(val_pred)
        test_pred_orig = pca.inverse_transform(test_pred)
        val_true_orig = y_val
        test_true_orig = y_test
    else:
        val_pred_orig = val_pred
        test_pred_orig = test_pred
        val_true_orig = y_val
        test_true_orig = y_test

    val_corr_df = pearson_per_feature(val_pred_orig, val_true_orig, enh_cols)
    test_corr_df = pearson_per_feature(test_pred_orig, test_true_orig, enh_cols)
    summarize_corr(val_corr_df, "val")
    summarize_corr(test_corr_df, "test")

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    np.save(out_dir / "train_ids.npy", np.array(train_ids))
    np.save(out_dir / "val_ids.npy", np.array(val_ids))
    np.save(out_dir / "test_ids.npy", np.array(test_ids))
    np.save(out_dir / "val_pred.npy", val_pred_orig)
    np.save(out_dir / "val_true.npy", val_true_orig)
    np.save(out_dir / "test_pred.npy", test_pred_orig)
    np.save(out_dir / "test_true.npy", test_true_orig)
    val_corr_df.to_csv(out_dir / "per_enhancer_correlation_val.csv", index=False)
    test_corr_df.to_csv(out_dir / "per_enhancer_correlation_test.csv", index=False)

    if pca is not None:
        np.save(out_dir / "pca_components.npy", pca.components_)
        np.save(out_dir / "pca_mean.npy", pca.mean_)
        np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
