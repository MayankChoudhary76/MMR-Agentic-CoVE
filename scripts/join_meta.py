#!/usr/bin/env python3
# Join normalized reviews with item metadata for a dataset.
# Saves:
#   data/processed/<dataset>/joined.parquet
#   data/processed/<dataset>/items_with_meta.parquet
#
# Run:
#   PYTHONPATH=$(pwd) python scripts/join_meta.py --dataset beauty

import argparse
import json
import re
from pathlib import Path
import pandas as pd

from src.utils.paths import get_dataset_paths, ensure_dir


def _coerce_price(x):
    """Parse price strings like '$12.99' or '12,999' into float."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x)
    s = re.sub(r"[^0-9.,-]", "", s)  # keep digits/./,/-
    s = s.replace(",", "")           # drop thousands sep
    try:
        return float(s) if s else None
    except Exception:
        return None


def _pick_image(row):
    """Prefer high-res; else fallback to low-res; else None."""
    hr = row.get("imageURLHighRes")
    lr = row.get("imageURL")
    if isinstance(hr, list) and len(hr) > 0:
        return hr[0]
    if isinstance(lr, list) and len(lr) > 0:
        return lr[0]
    return None


def _extract_category_leaf(row):
    """
    Amazon metadata sometimes has 'categories' (list of lists),
    sometimes a single 'category' list, sometimes empty.
    We try both; return a lowercase leaf if available.
    """
    cats = row.get("categories")
    if isinstance(cats, list) and len(cats) > 0:
        # categories looks like: [["A","B","C"], ["X","Y"]]
        # take the last path's last element as leaf
        try:
            return str(cats[-1][-1]).strip().lower()
        except Exception:
            pass
    cat = row.get("category")
    if isinstance(cat, list) and len(cat) > 0:
        return str(cat[-1]).strip().lower()
    return None


def load_meta(meta_path: Path) -> pd.DataFrame:
    """Load Amazon meta JSONL into a normalized DataFrame."""
    records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                # skip malformed line
                continue

    m = pd.DataFrame.from_records(records)

    # Ensure expected columns exist
    for col in ["asin", "title", "brand", "main_cat", "price", "imageURL", "imageURLHighRes", "categories", "category"]:
        if col not in m.columns:
            m[col] = None

    # Normalize/derive friendly columns
    m["brand"] = (
        m["brand"].astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": None, "none": None, "": None})
    )
    m["price"] = m["price"].apply(_coerce_price)
    m["image_url"] = m.apply(_pick_image, axis=1)
    m["category_leaf"] = m.apply(_extract_category_leaf, axis=1)

    meta = m.rename(columns={"main_cat": "main_cat_name"})[
        ["asin", "title", "brand", "main_cat_name", "category_leaf", "price", "image_url"]
    ]
    return meta


def main(dataset: str):
    paths = get_dataset_paths(dataset)
    raw_dir = ensure_dir(paths["raw"])
    proc_dir = ensure_dir(paths["processed"])

    reviews_fp = proc_dir / "reviews.parquet"
    meta_fp = raw_dir / "meta.json"

    if not reviews_fp.exists():
        raise FileNotFoundError(f"Missing {reviews_fp}. Run scripts/normalize.py first.")
    if not meta_fp.exists():
        raise FileNotFoundError(f"Missing {meta_fp}. Ensure meta is linked/copied to raw dir.")

    # Load inputs
    reviews = pd.read_parquet(reviews_fp)
    meta = load_meta(meta_fp)

    # Join: reviews.item_id is the ASIN
    merged = reviews.merge(meta, left_on="item_id", right_on="asin", how="left")

    # One row per item (for item-side operations like image embeds)
    items = (
        merged.drop_duplicates("item_id")
              .loc[:, ["item_id", "title", "brand", "main_cat_name", "category_leaf", "price", "image_url"]]
    )

    # Save artifacts
    out_joined = proc_dir / "joined.parquet"
    out_items = proc_dir / "items_with_meta.parquet"
    merged.to_parquet(out_joined, index=False)
    items.to_parquet(out_items, index=False)

    # Simple coverage stats
    img_cov = items["image_url"].notna().mean()
    brand_cov = items["brand"].notna().mean()
    price_cov = items["price"].notna().mean()

    print(f"✅ Saved: {out_joined} (rows={len(merged):,})")
    print(f"✅ Saved: {out_items} (items={len(items):,})")
    print(f"🖼️ Image URL coverage: {img_cov:.1%} | 🏷️ Brand: {brand_cov:.1%} | 💲 Price: {price_cov:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset key (e.g., beauty)")
    args = ap.parse_args()
    main(args.dataset)