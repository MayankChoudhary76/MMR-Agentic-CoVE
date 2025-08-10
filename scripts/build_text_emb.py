#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build baseline TEXT embeddings (items & users) and do a quick @K evaluation.

Inputs (from normalize.py):
  data/processed/<dataset>/reviews.parquet
    columns: user_id, item_id, text, rating, timestamp

Outputs:
  data/processed/<dataset>/item_text_emb.parquet
  data/processed/<dataset>/user_text_emb.parquet
  logs/<run_name>_eval.json
  logs/<run_name>_metrics.png (+ per-metric PNGs)
  logs/metrics.csv  (append-only history across runs)

Run:
  PYTHONPATH=$(pwd) python scripts/build_text_emb.py \
    --dataset beauty --k 10 --batch_size 256 --run_name beauty_text_emb
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.text_encoder import TextEncoder
from src.data.registry import get_paths
from src.utils.paths import LOGS_DIR, PROCESSED_DIR


# ---------- helpers to resolve paths (works with any get_paths style) ----------
def _resolve_processed_dir(dataset: str) -> Path:
    """Robustly get the processed dir regardless of get_paths() shape."""
    paths = get_paths(dataset)
    if isinstance(paths, dict) and "processed" in paths:
        return Path(paths["processed"])
    if isinstance(paths, (list, tuple)) and len(paths) >= 2:
        return Path(paths[1])
    return PROCESSED_DIR / dataset


# ------------------------------- data loading -------------------------------- #
def load_reviews(dataset: str) -> pd.DataFrame:
    processed_dir = _resolve_processed_dir(dataset)
    f = processed_dir / "reviews.parquet"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}. Run scripts/normalize.py first.")
    df = pd.read_parquet(f)

    need = ["user_id", "item_id", "text", "timestamp"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{f} missing columns: {missing}")

    # ensure timestamp is datetime
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").fillna(
            pd.to_datetime(df["timestamp"], errors="coerce")
        )
    return df


# --------------------------- item embeddings (text) -------------------------- #
def build_item_embeddings(df: pd.DataFrame, enc: TextEncoder, batch_size: int = 256) -> Tuple[np.ndarray, List[str]]:
    # aggregate all review text per item (timestamp-sorted for a stable concat order)
    items = (
        df.sort_values("timestamp")
          .groupby("item_id", as_index=False)
          .agg(text=("text", lambda s: " ".join(s.astype(str).tolist())))
    )
    texts = items["text"].fillna("").astype(str).tolist()
    item_ids = items["item_id"].tolist()

    vecs = enc.encode(texts, batch_size=batch_size, desc="Batches")
    return vecs, item_ids


# ------------------------------ leave-one-out split -------------------------- #
def make_loo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_id", "timestamp"])
    last = df.groupby("user_id", as_index=False).tail(1)
    train = pd.concat([df, last]).drop_duplicates(keep=False)
    return train, last


# --------------------------- user embeddings (train) ------------------------- #
def _agg_item_vectors_for_users(train: pd.DataFrame, item_vec_df: pd.DataFrame, enc_dim: int) -> pd.DataFrame:
    joined = train.merge(item_vec_df, on="item_id", how="left")

    def _mean_stack(arrs):
        arrs = [a for a in arrs if isinstance(a, np.ndarray)]
        if not arrs:
            return np.zeros(enc_dim, dtype=np.float32)
        return np.mean(np.stack(arrs, axis=0), axis=0)

    user_vecs = (
        joined.groupby("user_id")["vector"]
              .apply(lambda s: _mean_stack(list(s)))
              .reset_index()
    )
    return user_vecs


def build_user_embeddings(train: pd.DataFrame, item_vec: np.ndarray, item_ids: List[str]) -> pd.DataFrame:
    enc_dim = item_vec.shape[1]
    item_vec_df = pd.DataFrame({"item_id": item_ids, "vector": [v for v in item_vec]})
    user_vecs_df = _agg_item_vectors_for_users(train, item_vec_df, enc_dim)
    return user_vecs_df


# -------------------------------- evaluation --------------------------------- #
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def evaluate_topk(
    user_vecs: pd.DataFrame,
    item_vec: np.ndarray,
    item_ids: List[str],
    test: pd.DataFrame,
    k: int = 10
) -> Dict[str, float]:
    # map test true item per user
    true_item = test.set_index("user_id")["item_id"].to_dict()

    # users with vectors and a held-out item
    eval_users = [u for u in user_vecs["user_id"].tolist() if u in true_item]
    if not eval_users:
        return {f"hit@{k}": 0.0, f"ndcg@{k}": 0.0}

    U = np.vstack(user_vecs.set_index("user_id").loc[eval_users, "vector"].to_numpy())
    S = _cosine_sim(U, item_vec)  # [n_users, n_items]
    topk_idx = np.argpartition(-S, kth=min(k, S.shape[1]-1), axis=1)[:, :k]
    item_ids_arr = np.array(item_ids)

    hits = 0
    ndcgs = 0.0
    for i, u in enumerate(eval_users):
        rec_items = item_ids_arr[topk_idx[i]]
        target = true_item[u]
        if target in rec_items:
            hits += 1
            # rank within the full list for NDCG (binary relevance)
            order = np.argsort(-S[i])
            rank = int(np.where(order == np.where(item_ids_arr == target)[0][0])[0][0]) + 1
            ndcgs += 1.0 / math.log2(rank + 1)

    n = max(1, len(eval_users))
    return {f"hit@{k}": hits / n, f"ndcg@{k}": ndcgs / n}


# ----------------------------------- IO ------------------------------------- #
def save_vectors_parquet(path: Path, id_col: str, ids: List[str], vecs: np.ndarray) -> None:
    df = pd.DataFrame({id_col: ids, "vector": [v.astype(np.float32) for v in vecs]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Saved {path}  (dim={vecs.shape[1] if len(vecs) else 0})")


def save_metrics(run_name: str, metrics: Dict[str, float]) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / f"{run_name}_eval.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    return out


def plot_metrics(run_name: str, metrics: Dict[str, float]) -> None:
    # simple bar plots (Hit@K, NDCG@K) + combined
    hit_key = next((k for k in metrics if k.startswith("hit@")), "hit@10")
    ndcg_key = next((k for k in metrics if k.startswith("ndcg@")), "ndcg@10")
    hit = metrics.get(hit_key, 0.0)
    ndcg = metrics.get(ndcg_key, 0.0)

    # Hit@K
    plt.figure(figsize=(9, 5))
    plt.bar([run_name], [hit])
    plt.title(hit_key.upper())
    plt.ylim(0, 1)
    plt.ylabel(hit_key.upper())
    plt.xticks(rotation=20)
    p = LOGS_DIR / f"{run_name}_{hit_key.replace('@','at')}.png"
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()

    # NDCG@K
    plt.figure(figsize=(9, 5))
    plt.bar([run_name], [ndcg])
    plt.title(ndcg_key.upper())
    plt.ylim(0, 1)
    plt.ylabel(ndcg_key.upper())
    plt.xticks(rotation=20)
    p = LOGS_DIR / f"{run_name}_{ndcg_key.replace('@','at')}.png"
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()

    # Combined
    plt.figure(figsize=(10, 6))
    x = np.arange(1)
    plt.bar(x - 0.15, [hit], width=0.3, label=hit_key.upper())
    plt.bar(x + 0.15, [ndcg], width=0.3, label=ndcg_key.upper())
    plt.xticks(x, [run_name], rotation=20)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Top‑K Metrics")
    p = LOGS_DIR / f"{run_name}_metrics.png"
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()


def append_metrics_csv(dataset: str, run_name: str, model: str, k: int, metrics: Dict[str, float]) -> None:
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "dataset": dataset,
        "run_name": run_name,
        "model": model,
        "k": k,
        "hit": metrics.get(f"hit@{k}", 0.0),
        "ndcg": metrics.get(f"ndcg@{k}", 0.0),
        "notes": "",
    }
    csv_path = LOGS_DIR / "metrics.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"🧾 Appended metrics → {csv_path}")


# ----------------------------------- main ----------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build text embeddings + quick eval")
    ap.add_argument("--dataset", required=True, help="dataset key (e.g., beauty)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--run_name", default=None, help="name for logs/plots filenames")
    return ap.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.dataset}_text_emb"

    print("🔹 Building item text embeddings …")
    df = load_reviews(args.dataset)
    enc = TextEncoder(args.model)
    item_vec, item_ids = build_item_embeddings(df, enc, batch_size=args.batch_size)

    processed_dir = _resolve_processed_dir(args.dataset)
    save_vectors_parquet(processed_dir / "item_text_emb.parquet", "item_id", item_ids, item_vec)
    print(f"✅ Saved item vectors → {processed_dir / 'item_text_emb.parquet'}  (dim={item_vec.shape[1]})")

    print("🔹 Leave-one-out split for users …")
    train, test = make_loo(df)

    print("🔹 Building user text embeddings (train interactions only) …")
    user_vecs_df = build_user_embeddings(train, item_vec, item_ids)
    save_vectors_parquet(
        processed_dir / "user_text_emb.parquet",
        "user_id",
        user_vecs_df["user_id"].tolist(),
        np.vstack(user_vecs_df["vector"].to_numpy()),
    )
    print(f"✅ Saved user vectors → {processed_dir / 'user_text_emb.parquet'}")

    print("🔹 Running quick evaluation …")
    metrics = evaluate_topk(user_vecs_df, item_vec, item_ids, test, k=args.k)
    print("📊 Metrics:", json.dumps(metrics, indent=2))

    # Save JSON + simple charts + CSV row
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = save_metrics(run_name, metrics)
    plot_metrics(run_name, metrics)
    append_metrics_csv(args.dataset, run_name, args.model, args.k, metrics)
    print(f"📝 Saved → {out_json}")


if __name__ == "__main__":
    main()