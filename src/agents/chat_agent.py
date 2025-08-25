# src/agents/chat_agent.py
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.paths import get_processed_path


# ----------------------------- simple config -----------------------------
@dataclass
class ChatAgentConfig:
    # words to ignore when pulling a keyword from the prompt
    stopwords: frozenset = frozenset(
        {
            "under", "below", "less", "than", "beneath",
            "recommend", "something", "for", "me", "i", "need", "want",
            "a", "an", "the", "please", "pls", "ok", "okay",
            "price", "priced", "cost", "costing", "buy", "find", "search",
            "show", "give", "with", "and", "or", "of", "to", "in", "on",
        }
    )
    # price pattern: $12, 12, 12.5
    price_re: re.Pattern = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


# ----------------------------- helpers -----------------------------------
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        # Strip $ and commas if present (common in meta)
        s = s.replace(",", "")
        if s.startswith("$"):
            s = s[1:]
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _fmt_price(v: float) -> str:
    try:
        return f"${float(v):.2f}"
    except Exception:
        return f"${v}"


def _normalize_categories(val) -> List[str]:
    """
    Normalize 'categories' to list[str], handling:
      - None
      - list/tuple/set of str
      - stringified lists like "['A','B']" OR ["['A','B']"]
      - delimited strings "A > B, C; D"
    """
    def _from_string(s: str):
        s = s.strip()
        # Try literal list/tuple: "['A','B']" / '["A","B"]' / "(A,B)"
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            except Exception:
                pass
        # Delimited fallback
        if re.search(r"[>|,/;]+", s):
            return [p.strip() for p in re.split(r"[>|,/;]+", s) if p.strip()]
        return [s] if s else []

    if val is None:
        return []

    # Already a container?
    if isinstance(val, (list, tuple, set)):
        out = []
        for x in val:
            if x is None:
                continue
            if isinstance(x, (list, tuple, set)):
                # flatten nested containers
                for y in x:
                    if y is None:
                        continue
                    if isinstance(y, (list, tuple, set)):
                        out.extend([str(z).strip() for z in y if z is not None and str(z).strip()])
                    elif isinstance(y, str):
                        out.extend(_from_string(y))
                    else:
                        out.append(str(y).strip())
            elif isinstance(x, str):
                out.extend(_from_string(x))
            else:
                out.append(str(x).strip())
        # dedupe + keep order
        seen, dedup = set(), []
        for c in out:
            if c and c not in seen:
                seen.add(c)
                dedup.append(c)
        return dedup

    # Scalar string
    return _from_string(str(val))


# ----------------------------- agent --------------------------------------
class ChatAgent:
    def __init__(self, config: Optional[ChatAgentConfig] = None) -> None:
        self.config = config or ChatAgentConfig()

    # ---- parse last user text ----
    def _parse_price_cap(self, text: str) -> Optional[float]:
        m = self.config.price_re.search(text or "")
        if not m:
            return None
        return _safe_float(m.group(1))

    def _parse_keyword(self, text: str) -> Optional[str]:
        t = (text or "").lower()
        # remove price fragments
        t = self.config.price_re.sub(" ", t)
        # pick first token that isn't a stopword and has letters
        for w in re.findall(r"[a-z][a-z0-9\-]+", t):
            if w in self.config.stopwords:
                continue
            return w
        return None

    # ---- load catalog ----
    def _items_df(self, dataset: str) -> pd.DataFrame:
        """
        Load the product catalog from processed data.
        Prefers items_with_meta.parquet (your structure), falls back to joined.parquet.
        Returns a DataFrame; missing columns are filled with sensible defaults.
        """
        proc = get_processed_path(dataset)
        for fname in ["items_with_meta.parquet", "joined.parquet", "items_meta.parquet", "items.parquet"]:
            fp = proc / fname
            if fp.exists():
                try:
                    df = pd.read_parquet(fp)
                    break
                except Exception:
                    continue
        else:
            # nothing found
            return pd.DataFrame(columns=["item_id", "title", "brand", "price", "categories", "image_url"])

        # Make sure expected columns exist
        for col in ["item_id", "title", "brand", "price", "categories", "image_url"]:
            if col not in df.columns:
                df[col] = None

        # Some pipelines store images under imageURL/imageURLHighRes
        if ("image_url" not in df.columns or df["image_url"].isna().all()):
            for alt in ("imageURLHighRes", "imageURL"):
                if alt in df.columns:
                    # pick first image if it's a list-like
                    def _first_img(v):
                        if isinstance(v, (list, tuple)) and v:
                            return v[0]
                        return v
                    df["image_url"] = df[alt].apply(_first_img)
                    break

        return df

    # --------- main entrypoint expected by API ---------
    def reply(
        self,
        messages: List[Dict[str, str]],
        dataset: Optional[str] = None,
        user_id: Optional[str] = None,  # unused in this simple baseline
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Baseline behavior:
        - Parse last user message → (keyword, price cap)
        - Filter catalog by price<=cap and keyword match in title/brand/categories
        - Rank by lowest price (as a proxy score)
        - Return top-k with normalized fields
        """
        if not dataset:
            dataset = "beauty"

        # last user utterance
        last_user = ""
        for m in reversed(messages or []):
            if (m.get("role") or "").lower() == "user":
                last_user = m.get("content") or ""
                break

        cap = self._parse_price_cap(last_user)
        kw = self._parse_keyword(last_user)

        df = self._items_df(dataset)

        # Column presence map for debugging
        colmap = {
            "item_id": "item_id" if "item_id" in df.columns else None,
            "title": "title" if "title" in df.columns else None,
            "brand": "brand" if "brand" in df.columns else None,
            "price": "price" if "price" in df.columns else None,
            "categories": "categories" if "categories" in df.columns else None,
            "image_url": "image_url" if "image_url" in df.columns else None,
        }

        # ------- filtering -------
        if len(df) == 0:
            sub = df
        else:
            mask = pd.Series(True, index=df.index)

            # price filter
            if cap is not None and colmap["price"]:
                price_num = df[colmap["price"]].apply(_safe_float)
                mask &= pd.to_numeric(price_num, errors="coerce").le(cap)

            # keyword filter (title OR brand OR categories)
            if kw:
                kw_l = kw.lower()
                parts = []
                if colmap["title"]:
                    parts.append(df[colmap["title"]].astype(str).str.lower().str.contains(kw_l, na=False))
                if colmap["brand"]:
                    parts.append(df[colmap["brand"]].astype(str).str.lower().str.contains(kw_l, na=False))
                if colmap["categories"]:
                    parts.append(df[colmap["categories"]].astype(str).str.lower().str.contains(kw_l, na=False))
                if parts:
                    m_any = parts[0]
                    for p in parts[1:]:
                        m_any = m_any | p
                    mask &= m_any

            sub = df[mask].copy()

        # ------- scoring & sorting (cheaper → higher score) -------
        if len(sub) > 0:
            price_num = sub[colmap["price"]].apply(_safe_float) if colmap["price"] else 0.0
            sub["score"] = pd.to_numeric(price_num, errors="coerce").apply(
                lambda p: 1.0 / (p + 1e-6) if pd.notnull(p) and p > 0 else 0.0
            )
            sort_cols = ["score"]
            ascending = [False]
            if colmap["brand"]:
                sort_cols.append(colmap["brand"])
                ascending.append(True)
            if colmap["title"]:
                sort_cols.append(colmap["title"])
                ascending.append(True)
            sub = sub.sort_values(by=sort_cols, ascending=ascending).head(max(1, int(k)))

        # ------- build recs -------
        recs: List[Dict[str, Any]] = []
        for _, r in sub.iterrows():
            recs.append(
                {
                    "item_id": r.get(colmap["item_id"]) if colmap["item_id"] else None,
                    "score": float(r.get("score") or 0.0),
                    "brand": (r.get(colmap["brand"]) if colmap["brand"] else None) or None,
                    "price": _safe_float(r.get(colmap["price"]) if colmap["price"] else None),
                    "categories": _normalize_categories(r.get(colmap["categories"]) if colmap["categories"] else None),
                    "image_url": (r.get(colmap["image_url"]) if colmap["image_url"] else None) or None,
                }
            )

        # Fallback: if filter empty, return cheapest k overall
        if not recs and len(df) > 0:
            df2 = df.copy()
            pnum = df2[colmap["price"]].apply(_safe_float) if colmap["price"] else None
            df2["pnum"] = pd.to_numeric(pnum, errors="coerce")
            df2 = df2.sort_values(by=["pnum"]).head(max(1, int(k)))
            for _, r in df2.iterrows():
                recs.append(
                    {
                        "item_id": r.get(colmap["item_id"]) if colmap["item_id"] else None,
                        "score": 0.0,
                        "brand": (r.get(colmap["brand"]) if colmap["brand"] else None) or None,
                        "price": _safe_float(r.get(colmap["price"]) if colmap["price"] else None),
                        "categories": _normalize_categories(r.get(colmap["categories"]) if colmap["categories"] else None),
                        "image_url": (r.get(colmap["image_url"]) if colmap["image_url"] else None) or None,
                    }
                )

        # reply sentence
        reply_bits = []
        if kw:
            reply_bits.append(f"**{kw}**")
        if cap is not None:
            reply_bits.append(f"≤ {_fmt_price(cap)}")
        reply_str = "I found items " + (" ".join(reply_bits) if reply_bits else "you might like") + f" on **{dataset}**."

        # Helpful debug
        debug = {
            "parsed_keyword": kw,
            "price_cap": cap,
            "matched": len(recs),
            "colmap": colmap,
        }

        return {"reply": reply_str, "recommendations": recs, "debug": debug}