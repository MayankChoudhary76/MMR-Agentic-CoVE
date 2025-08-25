# scripts/build_catalog.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "beauty"
PROC = ROOT / "data" / "processed" / "beauty"
PROC.mkdir(parents=True, exist_ok=True)

META_JSON = RAW / "meta.json"
REV_JSON  = RAW / "reviews.json"

ITEMS_OUT = PROC / "items_catalog.parquet"
USERS_OUT = PROC / "user_map.parquet"

_price_re = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)")
def _to_float_price(x: Any) -> Optional[float]:
    if x is None: return None
    s = str(x)
    m = _price_re.search(s)
    if not m: return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _best_image(d: Dict[str, Any]) -> Optional[str]:
    # prefer imageURLHighRes[0], else imageURL[0]
    for key in ("imageURLHighRes", "imageURL"):
        vals = d.get(key)
        if isinstance(vals, list) and vals:
            v = str(vals[0]).strip()
            if v.startswith(("http://", "https://")):
                return v
    return None

def _flatten_categories(d: Dict[str, Any]) -> List[str]:
    cats = d.get("category") or d.get("categories")
    out: List[str] = []
    if isinstance(cats, list):
        for c in cats:
            if isinstance(c, list):
                out.extend([str(x).strip() for x in c if str(x).strip()])
            elif c is not None:
                out.append(str(c).strip())
    elif isinstance(cats, str):
        out.append(cats.strip())
    # dedupe preserve order
    seen=set(); flat=[]
    for c in out:
        if c and c not in seen:
            seen.add(c); flat.append(c)
    return flat

def _category_main(cats: List[str]) -> Optional[str]:
    if not cats: return None
    # pick the first non-empty leaf (e.g., "Beauty & Personal Care")
    return cats[-1] if len(cats) > 0 else None

_rank_re = re.compile(r"([0-9][0-9,]*)")
def _parse_rank(x: Any) -> Optional[int]:
    if x is None: return None
    s = str(x)
    m = _rank_re.search(s)
    if not m: return None
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None

def _iter_json_lines(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def build_items_catalog():
    rows = []
    for d in _iter_json_lines(META_JSON):
        asin = d.get("asin")
        if not asin: continue
        item_id = str(asin).strip()

        title = (d.get("title") or "").strip() or None
        brand = (d.get("brand") or "").strip() or None
        image = _best_image(d)
        price = _to_float_price(d.get("price"))
        rank  = _parse_rank(d.get("rank"))
        cats  = _flatten_categories(d)
        cat_main = _category_main(cats)

        rows.append({
            "item_id": item_id,
            "title": title,
            "brand": brand,
            "categories": cats,
            "category_main": cat_main,
            "image_url": image,
            "rank": rank,
            "price": price,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["item_id"])
    df.to_parquet(ITEMS_OUT, index=False)
    print(f"wrote {ITEMS_OUT} ({len(df)} rows)")

def build_user_map():
    # reviewerID, reviewerName
    seen = {}
    for d in _iter_json_lines(REV_JSON):
        uid = d.get("reviewerID")
        name = d.get("reviewerName")
        if not uid: continue
        uid = str(uid).strip()
        if uid and uid not in seen:
            seen[uid] = (str(name).strip() if name else None)

    rows = [{"user_id": k, "user_name": v} for k,v in seen.items()]
    df = pd.DataFrame(rows)
    df.to_parquet(USERS_OUT, index=False)
    print(f"wrote {USERS_OUT} ({len(df)} rows)")

if __name__ == "__main__":
    build_items_catalog()
    build_user_map()