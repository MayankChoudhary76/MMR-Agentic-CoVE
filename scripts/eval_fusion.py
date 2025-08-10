#!/usr/bin/env python3
# scripts/eval_fusion.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from src.data.registry import get_paths
from src.models.fusion import concat_fusion, weighted_sum_fusion


# ---------- path utils ----------
def _resolve_paths(dataset: str) -> Tuple[Path, Path]:
    """
    Resolve raw/processed paths from the registry, being permissive with keys/attrs.
    """
    paths = get_paths(dataset)
    if isinstance(paths, dict):
        raw = paths.get("raw") or paths.get("raw_dir") or paths.get("raw_path")
        proc = (
            paths.get("processed")
            or paths.get("processed_dir")
            or paths.get("proc")
            or paths.get("processed_path")
        )
        if raw and proc:
            return Path(raw), Path(proc)

    if isinstance(paths, (tuple, list)) and len(paths) >= 2:
        return Path(paths[0]), Path(paths[1])

    raw = getattr(paths, "raw", None) or getattr(paths, "raw_dir", None) or getattr(paths, "raw_path", None)
    proc = (
        getattr(paths, "processed", None)
        or getattr(paths, "processed_dir", None)
        or getattr(paths, "processed_path", None)
    )
    if raw and proc:
        return Path(raw), Path(proc)

    # fallback
    return Path("data/raw") / dataset, Path("data/processed") / dataset


# ---------- core helpers ----------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def _leave_one_out(df: pd.DataFrame):
    """
    Split with leave-one-out by timestamp.
    expects df with columns: user_id, item_id, timestamp
    """
    df = df.sort_values(["user_id", "timestamp"])
    last = df.groupby("user_id").tail(1)[["user_id", "item_id"]].rename(columns={"item_id": "target"})
    train = df.merge(last, on="user_id", how="left")
    train = train[train["item_id"] != train["target"]][["user_id", "item_id"]]
    test = last
    return train, test


def _pad_to(a: np.ndarray, dim: int) -> np.ndarray:
    if a.shape[1] == dim:
        return a
    if a.shape[1] < dim:
        pad = np.zeros((a.shape[0], dim - a.shape[1]), dtype=a.dtype)
        return np.concatenate([a, pad], axis=1)
    return a[:, :dim]


def evaluate_topk(user_vecs: np.ndarray,
                  item_vecs: np.ndarray,
                  item_ids: np.ndarray,
                  test: pd.DataFrame,
                  k: int) -> dict:
    S = _cosine_sim(user_vecs, item_vecs)  # [U x I]
    topk_idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    row_indices = np.arange(S.shape[0])[:, None]
    sorted_order = np.argsort(-S[row_indices, topk_idx], axis=1)
    topk_idx = topk_idx[row_indices, sorted_order]

    rec_items = item_ids[topk_idx]  # [U x k]
    test_items = test["target"].to_numpy()

    hits = (rec_items == test_items[:, None]).any(axis=1).astype(np.float32)

    # NDCG@k (binary relevance, single positive per user)
    gains = (rec_items == test_items[:, None]).astype(np.float32)
    ranks = np.argmax(gains, axis=1)
    ndcg = []
    for u, r in enumerate(ranks):
        if gains[u].any():
            ndcg.append(1.0 / np.log2(r + 2))
        else:
            ndcg.append(0.0)

    return {f"hit@{k}": float(hits.mean()),
            f"ndcg@{k}": float(np.mean(ndcg))}


# ---------- main ----------
def main(args):
    _, proc_dir = _resolve_paths(args.dataset)

    # normalized reviews (for split)
    reviews = pd.read_parquet(proc_dir / "reviews.parquet")
    train, test = _leave_one_out(reviews)

    # user/text embeddings
    U_text_df = pd.read_parquet(proc_dir / "user_text_emb.parquet")  # user_id, vector
    I_text_df = pd.read_parquet(proc_dir / "item_text_emb.parquet")  # item_id, vector

    # align users to evaluation subset
    eval_users = train["user_id"].drop_duplicates().to_numpy()
    u_map = {u: i for i, u in enumerate(U_text_df["user_id"])}
    keep_u = [u for u in eval_users if u in u_map]
    u_idx = np.array([u_map[u] for u in keep_u], dtype=np.int64)
    U_text = np.stack(U_text_df.loc[u_idx, "vector"].to_numpy()).astype(np.float32)  # [U x Dt]

    # align items
    item_ids = I_text_df["item_id"].to_numpy()
    Vt = np.stack(I_text_df["vector"].to_numpy()).astype(np.float32)  # [I x Dt]
    Dt = Vt.shape[1]

    # optional image embeddings
    Vi = None; Di = 0
    img_fp = proc_dir / "item_image_emb.parquet"
    if img_fp.exists():
        I_img = pd.read_parquet(img_fp)
        img_map = {iid: vec for iid, vec in zip(I_img["item_id"], I_img["vector"])}
        first = next(iter(img_map.values()), None)
        if first is not None:
            Di = len(first)
            Vi = np.stack([img_map.get(i, np.zeros(Di, dtype=np.float32)) for i in item_ids]).astype(np.float32)

    # optional metadata embeddings
    Vm = None; Dm = 0
    meta_fp = proc_dir / "item_meta_emb.parquet"
    if meta_fp.exists():
        I_meta = pd.read_parquet(meta_fp)
        meta_map = {iid: vec for iid, vec in zip(I_meta["item_id"], I_meta["vector"])}
        first = next(iter(meta_map.values()), None)
        if first is not None:
            Dm = len(first)
            Vm = np.stack([meta_map.get(i, np.zeros(Dm, dtype=np.float32)) for i in item_ids]).astype(np.float32)

    # fuse items; construct matching user vectors
    fusion = args.fusion.lower()
    if fusion == "concat":
        Vf = concat_fusion(Vt, Vi, Vm, weights=(args.w_text, args.w_image, args.w_meta))
        # user side to SAME concat dim: [w_t * U_text | w_i * 0 | w_m * 0]
        if Di == 0 and Dm == 0:
            U_fused = args.w_text * U_text
        else:
            zeros_i = np.zeros((U_text.shape[0], Di), dtype=np.float32) if Di > 0 else np.zeros((U_text.shape[0], 0), dtype=np.float32)
            zeros_m = np.zeros((U_text.shape[0], Dm), dtype=np.float32) if Dm > 0 else np.zeros((U_text.shape[0], 0), dtype=np.float32)
            U_fused = np.concatenate([args.w_text * U_text, args.w_image * zeros_i, args.w_meta * zeros_m], axis=1)

    elif fusion == "weighted":
        Vf = weighted_sum_fusion(Vt, Vi, Vm, weights=(args.w_text, args.w_image, args.w_meta))
        target_dim = Vf.shape[1]
        U_fused = _pad_to(args.w_text * U_text, target_dim)

    else:
        raise ValueError("fusion must be one of: concat, weighted")

    # evaluate on aligned users
    test_users = test["user_id"].to_numpy()
    keep_mask = np.isin(test_users, keep_u)
    test_eval = test.loc[keep_mask].reset_index(drop=True)
    U_eval = U_fused[: len(test_eval)]

    metrics = evaluate_topk(U_eval, Vf, item_ids, test_eval, k=args.k)
    print("📊 Metrics:", json.dumps(metrics, indent=2))

    # ---- logging ----
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "run_name": args.run_name or "",
        "dataset": args.dataset,
        "fusion": fusion,
        "w_text": args.w_text,
        "w_image": args.w_image,
        "w_meta": args.w_meta,
        "k": args.k,
        **metrics,
    }

    # robust CSV append (quoting + explicit header)
    csv_fp = logs_dir / "metrics.csv"
    header_cols = [
        "run_name", "dataset", "fusion",
        "w_text", "w_image", "w_meta",
        "k", f"hit@{args.k}", f"ndcg@{args.k}"
    ]
    df_row = pd.DataFrame([row], columns=[c for c in header_cols if c in row.keys()])
    df_row.to_csv(
        csv_fp,
        mode="a",
        index=False,
        header=not csv_fp.exists(),
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )

    # also dump JSON for reproducibility
    json_fp = logs_dir / f"{args.run_name or (args.dataset + '_fusion_eval')}.json"
    with open(json_fp, "w") as f:
        json.dump(row, f, indent=2)
    print(f"🧾 Appended → {csv_fp}")
    print(f"📝 Saved → {json_fp}")

    # ---- optional plots ----
    if args.plot:
        import matplotlib.pyplot as plt

        dfm = pd.read_csv(csv_fp, engine="python").copy()
        dset = args.dataset
        if "dataset" in dfm.columns:
            dfm = dfm[dfm["dataset"] == dset].copy()

        # ensure str x-axis
        if "run_name" in dfm.columns:
            dfm["run_name"] = dfm["run_name"].fillna("").astype(str)
        else:
            dfm["run_name"] = ""

        plots_dir = logs_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if not dfm.empty:
            # comparison (latest per fusion)
            if "fusion" in dfm.columns:
                latest = dfm.sort_values("run_name").groupby("fusion", dropna=False).tail(1)
                if not latest.empty:
                    fig1 = plt.figure()
                    plt.bar(latest["fusion"].astype(str), latest[f"hit@{args.k}"])
                    plt.title(f"Hit@{args.k} by fusion (dataset={dset})")
                    plt.ylabel(f"Hit@{args.k}")
                    fig1.savefig(plots_dir / f"{dset}_k{args.k}_comparison.png", bbox_inches="tight")
                    plt.close(fig1)

            # trend
            fig2 = plt.figure()
            plt.plot(dfm["run_name"].astype(str), dfm[f"hit@{args.k}"])
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Hit@{args.k} trend (dataset={dset})")
            plt.ylabel(f"Hit@{args.k}")
            fig2.savefig(plots_dir / f"{dset}_k{args.k}_trend.png", bbox_inches="tight")
            plt.close(fig2)
            print(f"📈 Saved comparison plot → {plots_dir / f'{dset}_k{args.k}_comparison.png'}")
            print(f"📈 Saved trend plot → {plots_dir / f'{dset}_k{args.k}_trend.png'}")
        else:
            print("ℹ️ No rows for plotting yet.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fusion", choices=["concat", "weighted"], default="concat")
    ap.add_argument("--w_text", type=float, default=1.0)
    ap.add_argument("--w_image", type=float, default=1.0)
    ap.add_argument("--w_meta", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    main(args)