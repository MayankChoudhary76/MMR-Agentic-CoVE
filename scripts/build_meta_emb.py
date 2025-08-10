#!/usr/bin/env python3
# scripts/build_meta_emb.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data.registry import get_paths
from src.utils.paths import ensure_dir


# ---------------------------
# Path resolution (dict/tuple/attrs → fallback)
# ---------------------------
def _resolve_paths(dataset: str):
    paths = get_paths(dataset)
    if isinstance(paths, dict):
        raw = paths.get("raw") or paths.get("raw_dir") or paths.get("raw_path")
        proc = paths.get("processed") or paths.get("processed_dir") or paths.get("proc") or paths.get("processed_path")
        if raw and proc:
            return Path(raw), Path(proc)
    if isinstance(paths, (tuple, list)) and len(paths) >= 2:
        return Path(paths[0]), Path(paths[1])
    raw = getattr(paths, "raw", None) or getattr(paths, "raw_dir", None) or getattr(paths, "raw_path", None)
    proc = getattr(paths, "processed", None) or getattr(paths, "processed_dir", None) or getattr(paths, "processed_path", None)
    if raw and proc:
        return Path(raw), Path(proc)
    return Path("data/raw") / dataset, Path("data/processed") / dataset


# ---------------------------
# Hashing-based lightweight embeddings (reproducible)
# ---------------------------
def _hash_token(token: str, num_buckets: int) -> int:
    return (hash(token) % num_buckets + num_buckets) % num_buckets

def _embed_tokens(tokens: Iterable[str], table: np.ndarray) -> np.ndarray:
    if tokens is None:
        return np.zeros(table.shape[1], dtype=np.float32)
    out = np.zeros(table.shape[1], dtype=np.float32)
    for t in tokens:
        if not t:
            continue
        idx = _hash_token(str(t).lower(), table.shape[0])
        out += table[idx]
    return out

def _embed_scalar_bucket(x: Optional[float], buckets: List[float], table: np.ndarray) -> np.ndarray:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return np.zeros(table.shape[1], dtype=np.float32)
    i = 0
    while i + 1 < len(buckets) and x >= buckets[i + 1]:
        i += 1
    return table[i]

def _l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return (v / n).astype(np.float32)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset key, e.g., beauty")
    # dims can be tuned; keep total ~80 by default
    ap.add_argument("--brand_dim", type=int, default=32)
    ap.add_argument("--cat_dim", type=int, default=32)
    ap.add_argument("--price_dim", type=int, default=16)
    ap.add_argument("--brand_buckets", type=int, default=4096)
    ap.add_argument("--cat_buckets", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    raw_dir, proc_dir = _resolve_paths(args.dataset)
    ensure_dir(proc_dir)

    items_meta_fp = proc_dir / "items_with_meta.parquet"
    if not items_meta_fp.exists():
        raise FileNotFoundError(
            f"{items_meta_fp} not found. Run scripts/join_meta.py --dataset {args.dataset} first."
        )

    items = pd.read_parquet(items_meta_fp)
    # Ensure columns exist with expected types
    if "categories" in items.columns:
        items["categories"] = items["categories"].apply(
            lambda x: x if isinstance(x, list)
            else ([] if x is None or (isinstance(x, float) and np.isnan(x)) else [str(x)])
        )
    else:
        items["categories"] = [[] for _ in range(len(items))]

    if "brand" not in items.columns:
        items["brand"] = None
    if "price" in items.columns:
        items["price"] = pd.to_numeric(items["price"], errors="coerce")
    else:
        items["price"] = np.nan

    # Coverage
    n_items = len(items)
    brand_cov = float(items["brand"].apply(lambda x: isinstance(x, str) and x.strip() != "").mean()) if n_items else 0.0
    cat_cov = float(items["categories"].apply(lambda x: isinstance(x, list) and len(x) > 0).mean()) if n_items else 0.0
    price_cov = float(items["price"].notna().mean()) if n_items else 0.0

    # Embedding tables (fixed seed = reproducible)
    rng = np.random.RandomState(args.seed)
    brand_table = rng.normal(0, 0.02, size=(args.brand_buckets, args.brand_dim)).astype(np.float32)
    cat_table   = rng.normal(0, 0.02, size=(args.cat_buckets, args.cat_dim)).astype(np.float32)

    # Price buckets (log-ish spread; tweak as needed)
    price_bins = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0]
    price_table = rng.normal(0, 0.02, size=(len(price_bins), args.price_dim)).astype(np.float32)

    vectors: List[np.ndarray] = []
    for _, row in items.iterrows():
        # brand (single token if present)
        b_tokens = [row["brand"]] if isinstance(row["brand"], str) and row["brand"].strip() else []
        v_brand = _embed_tokens(b_tokens, brand_table)

        # categories (list[str])
        v_cat = _embed_tokens(row["categories"], cat_table)

        # price → bucket embedding
        price_val = float(row["price"]) if row["price"] is not None and not pd.isna(row["price"]) else None
        v_price = _embed_scalar_bucket(price_val, price_bins, price_table)

        v = np.concatenate([v_brand, v_cat, v_price], axis=0)
        vectors.append(_l2norm(v))

    dim_total = args.brand_dim + args.cat_dim + args.price_dim
    out = pd.DataFrame({"item_id": items["item_id"].tolist(), "vector": vectors})
    out_fp = proc_dir / "item_meta_emb.parquet"
    out.to_parquet(out_fp, index=False)

    print(f"✅ Saved item meta vectors → {out_fp}  (dim={dim_total})")
    print("📝 Coverage:", json.dumps({
        "items": n_items,
        "brand_cov": brand_cov,
        "cat_cov": cat_cov,
        "price_cov": price_cov,
        "dim": dim_total
    }, indent=2))


if __name__ == "__main__":
    main()