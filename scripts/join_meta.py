#!/usr/bin/env python3
# scripts/join_meta.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.registry import get_paths
from src.utils.paths import ensure_dir


# ---------------------------
# Path resolution (dict / tuple / attrs → fallback)
# ---------------------------
def _resolve_paths(dataset: str) -> Tuple[Path, Path]:
    """
    Works with dict keys: raw/raw_dir/processed/processed_dir,
    tuple/list (raw, processed), or attribute-style objects.
    Falls back to data/<raw|processed>/<dataset>.
    Always returns concrete dataset subfolders for both raw and processed.
    """
    paths = get_paths(dataset)

    # dict variants
    if isinstance(paths, dict):
        raw = paths.get("raw") or paths.get("raw_dir") or paths.get("raw_path")
        proc = paths.get("processed") or paths.get("processed_dir") or paths.get("proc") or paths.get("processed_path")
        if raw and proc:
            raw_dir = Path(raw)
            proc_dir = Path(proc)
            if proc_dir.name != dataset:
                proc_dir = proc_dir / dataset
            if raw_dir.name != dataset:
                raw_dir = raw_dir / dataset
            return raw_dir, proc_dir

    # tuple/list (raw, processed)
    if isinstance(paths, (tuple, list)) and len(paths) >= 2:
        raw_dir, proc_dir = Path(paths[0]), Path(paths[1])
        if proc_dir.name != dataset:
            proc_dir = proc_dir / dataset
        if raw_dir.name != dataset:
            raw_dir = raw_dir / dataset
        return raw_dir, proc_dir

    # attribute-style (e.g., SimpleNamespace)
    raw = getattr(paths, "raw", None) or getattr(paths, "raw_dir", None) or getattr(paths, "raw_path", None)
    proc = getattr(paths, "processed", None) or getattr(paths, "processed_dir", None) or getattr(paths, "processed_path", None)
    if raw and proc:
        raw_dir, proc_dir = Path(raw), Path(proc)
        if proc_dir.name != dataset:
            proc_dir = proc_dir / dataset
        if raw_dir.name != dataset:
            raw_dir = raw_dir / dataset
        return raw_dir, proc_dir

    # fallback
    return Path("data/raw") / dataset, Path("data/processed") / dataset


# ---------------------------
# Normalizers / extractors
# ---------------------------
def _to_float_price(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if not s:
            return None
        # keep digits & dot only (handles "$12.34", "USD 12.34", etc.)
        s = re.sub(r"[^0-9.]", "", s)
        return float(s) if s else None
    except Exception:
        return None


def _pick_image(row: dict) -> Optional[str]:
    def _first_url(v):
        if isinstance(v, list):
            for u in v:
                if isinstance(u, str):
                    u=u.strip()
                    if u.startswith(("http://","https://")):
                        return u
        if isinstance(v, str):
            s=v.strip()
            if s.startswith(("http://","https://")):
                return s
        return None

    for key in ("imageURLHighRes", "imageURL"):
        u = _first_url(row.get(key))
        if u:
            return u

    # OPTIONAL: very lenient scrape from big HTML blobs (only if you want it)
    sim = row.get("similar_item")
    if isinstance(sim, str) and "src=" in sim:
        import re
        m = re.search(r'src="\s*(https?://[^"]+)"', sim)
        if m:
            return m.group(1).strip()
    return None

def _norm_brand(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    return s or None


def _flatten_categories(meta_row: dict) -> List[str]:
    """
    Normalize to a list of leaf category strings.
    Priority:
      1) categories (list of lists or list of strings)
      2) category (list or 'A > B > C' string)
      3) salesRank (dict keys)
      4) rank string ("... in X (")
      5) main_cat
    """
    # 1) categories
    cats = meta_row.get("categories", None)
    if isinstance(cats, list):
        # list of lists
        if cats and all(isinstance(c, list) for c in cats):
            leaves = []
            for chain in cats:
                if chain:
                    leaf = str(chain[-1]).strip()
                    if leaf:
                        leaves.append(leaf)
            if leaves:
                # unique, order-preserving
                return [c for c in dict.fromkeys(leaves)]
        # plain list of strings
        if cats and all(isinstance(c, str) for c in cats):
            leaf = str(cats[-1]).strip()
            return [leaf] if leaf else []

    # 2) category
    cat = meta_row.get("category", None)
    if isinstance(cat, list) and cat:
        leaf = str(cat[-1]).strip()
        if leaf:
            return [leaf]
    if isinstance(cat, str) and cat.strip():
        parts = [p.strip() for p in re.split(r"[>/|]", cat) if p.strip()]
        if parts:
            return [parts[-1]]

    # 3) salesRank (dict with top-level category as key)
    sales_rank = meta_row.get("salesRank")
    if isinstance(sales_rank, dict) and len(sales_rank) > 0:
        key = next(iter(sales_rank.keys()))
        key = str(key).strip()
        if key:
            return [key]

    # 4) rank string "... in <Category> ("
    rank = meta_row.get("rank")
    if isinstance(rank, str):
        m = re.search(r"in\s+([^)]+)\s*\(", rank)
        if m:
            cat_name = m.group(1).strip()
            if cat_name:
                return [cat_name]

    # 5) main_cat as last resort
    main_cat = meta_row.get("main_cat")
    if isinstance(main_cat, str) and main_cat.strip():
        return [main_cat.strip()]

    return []


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                # tolerate malformed lines
                pass
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset key, e.g., beauty")
    args = ap.parse_args()

    raw_dir, proc_dir = _resolve_paths(args.dataset)
    ensure_dir(proc_dir)

    reviews_fp = raw_dir / "reviews.json"
    meta_fp = raw_dir / "meta.json"

    if not reviews_fp.exists() or not meta_fp.exists():
        raise FileNotFoundError(f"Missing raw files. Expected {reviews_fp} and {meta_fp}")

    # Use normalized interactions for consistent schema
    reviews_parquet = proc_dir / "reviews.parquet"
    if not reviews_parquet.exists():
        raise FileNotFoundError(f"Missing {reviews_parquet}. Run scripts/normalize.py first.")
    df = pd.read_parquet(reviews_parquet)  # user_id, item_id, text, rating, timestamp

    # Load raw meta JSONL
    meta_rows = _load_jsonl(meta_fp)
    meta_df = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame(columns=["asin"])

    # Map raw meta to compact schema
    def _row_to_meta(r: pd.Series) -> dict:
        d = r.to_dict()
        asin = d.get("asin")
        brand = _norm_brand(d.get("brand"))
        price = _to_float_price(d.get("price"))
        categories = _flatten_categories(d)
        image_url = _pick_image(d)
        # normalize empty strings to None for brand/image_url
        if brand is not None and not brand.strip():
            brand_none = None
        else:
            brand_none = brand
        if isinstance(image_url, str) and not image_url.strip():
            image_url_none = None
        else:
            image_url_none = image_url
        return {
            "item_id": asin,
            "brand": brand_none,
            "price": price,
            "categories": categories,
            "image_url": image_url_none,
        }

    if not meta_df.empty:
        meta_small = meta_df.apply(_row_to_meta, axis=1, result_type="expand")
        meta_small = meta_small[meta_small["item_id"].notna()].drop_duplicates(subset=["item_id"])
    else:
        meta_small = pd.DataFrame(columns=["item_id", "brand", "price", "categories", "image_url"])

    # Join interactions with meta by item_id (asin)
    joined = df.merge(meta_small, on="item_id", how="left")

    # Save per-interaction joined
    out_joined = proc_dir / "joined.parquet"
    joined.to_parquet(out_joined, index=False)

    # Unique items with meta among interacted items
    items_with_meta = joined[["item_id", "brand", "price", "categories", "image_url"]].drop_duplicates("item_id")

    # Clean empty strings to proper missing values for coverage
    def _none_if_empty(x):
        if isinstance(x, str) and not x.strip():
            return None
        return x

    items_with_meta["brand"] = items_with_meta["brand"].apply(_none_if_empty)
    items_with_meta["image_url"] = items_with_meta["image_url"].apply(_none_if_empty)

    out_items = proc_dir / "items_with_meta.parquet"
    items_with_meta.to_parquet(out_items, index=False)

    # Coverage report
    n_items = len(items_with_meta)
    brand_cov = float(items_with_meta["brand"].notna().mean()) if n_items else 0.0
    price_cov = float(items_with_meta["price"].notna().mean()) if n_items else 0.0
    cat_cov = float(items_with_meta["categories"].apply(lambda x: isinstance(x, list) and len(x) > 0).mean()) if n_items else 0.0
    img_cov = float(items_with_meta["image_url"].notna().mean()) if n_items else 0.0

    print(f"✅ Saved: {out_joined} (rows={len(joined):,})")
    print(f"✅ Saved: {out_items} (items={n_items:,})")
    print(f"🏷️ Brand: {brand_cov:.1%} | 💲 Price: {price_cov:.1%} | 📚 Categories: {cat_cov:.1%} | 🖼️ Image URL: {img_cov:.1%}")


if __name__ == "__main__":
    main()