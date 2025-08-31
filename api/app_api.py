# api/app_api.py  (Part 1/5)

from __future__ import annotations

import os
import time
import inspect
import ast
import math
import re
import traceback
from typing import Any, Dict, List, Optional
import json
import numpy as np
from starlette.responses import Response

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path  # NEW

from src.utils.paths import get_processed_path
from src.service.recommender import recommend_for_user, RecommendConfig, FusionWeights
from src.agents.chat_agent import ChatAgent, ChatAgentConfig

# ---------- NEW: light config for logs location ----------
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))

# Instantiate the chat agent used by /chat_recommend
CHAT_AGENT = ChatAgent(ChatAgentConfig())

# =========================
# Introspection (agentz)
# =========================
def _agent_introspection():
    try:
        fn = getattr(ChatAgent, "reply", None)
        code = getattr(fn, "__code__", None)
        file_path = getattr(code, "co_filename", None)
        mtime = None
        if file_path and os.path.exists(file_path):
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(file_path)))
        sig = str(inspect.signature(ChatAgent.reply)) if hasattr(ChatAgent, "reply") else "N/A"
        return {
            "class": str(CHAT_AGENT.__class__),
            "module": ChatAgent.__module__,
            "file": file_path,
            "file_mtime": mtime,
            "reply_signature": sig,
            "has_debug_attr_on_instance": hasattr(CHAT_AGENT, "debug"),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

# ---------- NEW: metrics helper (reads logs/metrics.csv if present) ----------
def _latest_metrics_for(dataset: str, fusion: str, k: int, faiss_name: Optional[str]) -> Dict[str, Any]:
    """
    Heuristic: read logs/metrics.csv and pick the newest row that matches dataset and (faiss_name in model/run_name)
    or at least matches fusion. Returns keys suitable for the UI:
      {"hit@k": <float|str>, "ndcg@k": <float|str>, "memory_mb": <float|str>}
    If nothing found, returns {}.
    """
    csv_fp = LOGS_DIR / "metrics.csv"
    if not csv_fp.exists():
        return {}
    try:
        df = pd.read_csv(csv_fp)
    except Exception:
        return {}

    try:
        if "dataset" in df.columns:
            df = df[df["dataset"].astype(str).str.lower() == str(dataset).lower()]

        # Prefer matching K if available
        if "k" in df.columns:
            with_k = df[df["k"].astype(str) == str(int(k))]
            if not with_k.empty:
                df = with_k

        # newest first if timestamp
        if "timestamp" in df.columns:
            try:
                df = df.sort_values("timestamp", ascending=False)
            except Exception:
                pass

        def _row_matches(row) -> bool:
            text = " ".join(str(row.get(c, "")) for c in ["model", "run_name"])
            if faiss_name:
                return faiss_name in text
            return str(fusion).lower() in text.lower()

        pick = None
        for _, r in df.iterrows():
            if _row_matches(r):
                pick = r
                break
        if pick is None and len(df):
            pick = df.iloc[0]

        if pick is None:
            return {}

        def _safe_float(v):
            try:
                f = float(v)
                if not math.isfinite(f):
                    return None
                return f
            except Exception:
                return None

        return {
            "hit@k": _safe_float(pick.get("hit")),
            "ndcg@k": _safe_float(pick.get("ndcg")),
            "memory_mb": _safe_float(pick.get("memory_mb")),
        }
    except Exception:
        return {}

# =========================
# Helpers (parsing/cleanup)
# =========================
_PRICE_RE = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)")
_STOPWORDS = {"under","below","less","than","max","upto","up","to","recommend","something","for","me","need","budget","cheap","please","soap","shampoos"}

def _parse_price_cap(text: str) -> Optional[float]:
    m = _PRICE_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _parse_keyword(text: str) -> Optional[str]:
    t = (text or "").lower()
    t = _PRICE_RE.sub(" ", t)
    for w in re.findall(r"[a-z][a-z0-9\-]+", t):
        if w in _STOPWORDS:
            continue
        return w
    return None

def _parse_listlike_string(s: str) -> List[str]:
    """Parse strings like "['A','B']" or '["A"]' into ['A','B']; otherwise a best-effort list."""
    if not isinstance(s, str):
        return []
    t = s.strip()
    if (t.startswith("[") and t.endswith("]")) or (t.startswith("(") and t.endswith(")")):
        try:
            val = ast.literal_eval(t)
            if isinstance(val, (list, tuple, set)):
                return [str(x).strip() for x in val if x is not None and str(x).strip()]
        except Exception:
            pass
    if re.search(r"[>|,/;]+", t):
        return [p.strip() for p in re.split(r"[>|,/;]+", t) if p.strip()]
    return [t] if t else []

def _normalize_categories_in_place(items):
    """
    Force each item's 'categories' into a clean List[str].
    Supports None, stringified lists, nested containers, etc.
    """
    def _as_list_from_string(s: str) -> List[str]:
        s = (s or "").strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            except Exception:
                pass
        return [s]

    for r in items or []:
        cats = r.get("categories")
        out: List[str] = []
        if cats is None:
            out = []
        elif isinstance(cats, str):
            out = _as_list_from_string(cats)
        elif isinstance(cats, (list, tuple, set)):
            tmp: List[str] = []
            for c in cats:
                if c is None:
                    continue
                if isinstance(c, str):
                    tmp.extend(_as_list_from_string(c))
                elif isinstance(c, (list, tuple, set)):
                    for y in c:
                        if y is None:
                            continue
                        if isinstance(y, str):
                            tmp.extend(_as_list_from_string(y))
                        else:
                            ys = str(y).strip()
                            if ys:
                                tmp.append(ys)
                else:
                    s = str(c).strip()
                    if s:
                        tmp.append(s)
            seen = set()
            out = []
            for x in tmp:
                if x and x not in seen:
                    seen.add(x)
                    out.append(x)
        else:
            s = str(cats).strip()
            out = [s] if s else []
        r["categories"] = out

def _first_image_url_from_row(row: pd.Series) -> Optional[str]:
    """
    Return a single best image URL from several possible columns or formats:
      - 'image_url' scalar string or list
      - 'imageURL' / 'imageURLHighRes' (AMZ style) with lists or stringified lists
    """
    candidates: List[Any] = []
    for col in ["image_url", "imageURLHighRes", "imageURL"]:
        if col in row.index:
            candidates.append(row[col])

    urls: List[str] = []
    for v in candidates:
        if v is None:
            continue
        if isinstance(v, str):
            vv = v.strip()
            if (vv.startswith("[") and vv.endswith("]")) or (vv.startswith("(") and vv.endswith(")")):
                try:
                    lst = ast.literal_eval(vv)
                    if isinstance(lst, (list, tuple, set)):
                        urls.extend([str(x).strip() for x in lst if x])
                except Exception:
                    if vv:
                        urls.append(vv)
            else:
                urls.append(vv)
        elif isinstance(v, (list, tuple, set)):
            urls.extend([str(x).strip() for x in v if x])
        else:
            s = str(v).strip()
            if s:
                urls.append(s)

    for u in urls:
        if u.lower().startswith("http"):
            return u
    return urls[0] if urls else None

def _parse_rank_num(s: Any) -> Optional[int]:
    """Extract numeric rank from strings like '2,938,573 in Beauty & Personal Care ('."""
    if s is None or (isinstance(s, float) and not math.isfinite(s)):
        return None
    try:
        if isinstance(s, (int, float)):
            return int(s)
        txt = str(s)
        m = re.search(r"([\d,]+)", txt)
        if not m:
            return None
        return int(m.group(1).replace(",", ""))
    except Exception:
        return None

def _to_jsonable(obj: Any):
    """Convert numpy/pandas and other non-JSON-serializable objects to plain Python types."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    if np is not None:
        if isinstance(obj, getattr(np, "integer", ())):
            return int(obj)
        if isinstance(obj, getattr(np, "floating", ())):
            f = float(obj)
            return None if not math.isfinite(f) else f
        if isinstance(obj, getattr(np, "bool_", ())):
            return bool(obj)

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    if isinstance(obj, pd.Series):
        return {str(k): _to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        return [_to_jsonable(r) for r in obj.to_dict(orient="records")]

    if hasattr(obj, "_asdict"):
        return {str(k): _to_jsonable(v) for k, v in obj._asdict().items()}

    return str(obj)

# =========================
# Catalog enrichment (API)
# =========================
def _load_catalog_like(dataset: str) -> pd.DataFrame:
    """
    Load an item catalog table for enrichment.
    Preference:
      1) items_catalog.parquet (enriched)
      2) items_with_meta.parquet
      3) joined.parquet (dedup on item_id)
    Ensures presence of: item_id, title, brand, price, categories, image_url, rank.
    """
    proc = get_processed_path(dataset)
    cands = [
        proc / "items_catalog.parquet",
        proc / "items_with_meta.parquet",
        proc / "joined.parquet",
    ]
    df = pd.DataFrame()
    for fp in cands:
        if fp.exists():
            try:
                df = pd.read_parquet(fp)
                break
            except Exception:
                pass

    if df.empty:
        return pd.DataFrame(columns=["item_id","title","brand","price","categories","image_url","rank"])

    # If we loaded joined.parquet, dedup rows to unique item_id
    if "item_id" in df.columns and df["item_id"].duplicated().any():
        df = df.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"])

    # Guarantee columns exist
    for c in ["item_id","title","brand","price","categories","image_url","imageURL","imageURLHighRes","rank","rank_num"]:
        if c not in df.columns:
            df[c] = None

    # Normalize derived columns
    df["item_id"] = df["item_id"].astype(str)

    # Best-effort image_url column
    img_urls: List[Optional[str]] = []
    for row in df.itertuples(index=False):
        r = pd.Series(row._asdict() if hasattr(row, "_asdict") else row._asdict())
        img_urls.append(_first_image_url_from_row(r))
    df["image_url_best"] = img_urls

    # Best-effort numeric rank
    if "rank_num" in df.columns:
        need = df["rank_num"].isna()
        if "rank" in df.columns and need.any():
            df.loc[need, "rank_num"] = df.loc[need, "rank"].map(_parse_rank_num)
    else:
        df["rank_num"] = df["rank"].map(_parse_rank_num)

    return df[["item_id","title","brand","price","categories","image_url_best","rank","rank_num"]].rename(
        columns={"image_url_best":"image_url"}
    )


def _enrich_with_catalog(dataset: str, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not recs:
        return recs
    try:
        proc = get_processed_path(dataset)

        # Load sources and keep extra image columns if present
        sources: List[pd.DataFrame] = []
        for name in ["items_catalog.parquet", "items_with_meta.parquet", "joined.parquet"]:
            fp = proc / name
            if fp.exists():
                try:
                    df = pd.read_parquet(fp)
                    keep = [c for c in [
                        "item_id","title","brand","price","categories","image_url","rank","rank_num",
                        "imageURLHighRes","imageURL"  # extra image columns from raw meta
                    ] if c in df.columns]
                    if "item_id" in keep:
                        slim = df[keep].copy()
                        slim["item_id"] = slim["item_id"].astype(str)
                        sources.append(slim.set_index("item_id", drop=False))
                except Exception:
                    pass
        if not sources:
            return recs

        import ast, math, re

        def _pick_non_empty(*vals):
            for v in vals:
                if v is None:
                    continue
                if isinstance(v, float) and not math.isfinite(v):
                    continue
                s = v.strip() if isinstance(v, str) else v
                if s == "" or s == "nan":
                    continue
                return v
            return None

        def _pick_price(*vals):
            for v in vals:
                try:
                    if v in (None, "", "nan"):
                        continue
                    f = float(v)
                    if math.isfinite(f):
                        return f
                except Exception:
                    continue
            return None

        def _norm_categories(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple, set)):
                return [str(x).strip() for x in v if x is not None and str(x).strip()]
            if isinstance(v, str):
                s = v.strip()
                if not s or s == "[]":
                    return []
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple, set)):
                        return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
                except Exception:
                    return [s]
            return []

        def _pick_categories(*vals):
            for v in vals:
                cats = _norm_categories(v)
                if cats:
                    return cats
            return []

        def _first_url_from_list(v):
            if isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, str) and u.strip():
                        return u.strip()
            return None

        def _pick_image_url(cand_image_url, cand_highres, cand_image):
            # priority: explicit image_url (string), then imageURLHighRes[0], then imageURL[0]
            if isinstance(cand_image_url, str) and cand_image_url.strip():
                return cand_image_url.strip()
            u = _first_url_from_list(cand_highres)
            if u:
                return u
            u = _first_url_from_list(cand_image)
            if u:
                return u
            if isinstance(cand_image_url, list):
                u = _first_url_from_list(cand_image_url)
                if u:
                    return u
            return None

        def _pick_rank(*vals):
            for v in vals:
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    continue
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, str):
                    m = re.search(r"[\d,]+", v)
                    if m:
                        try:
                            return int(m.group(0).replace(",", ""))
                        except Exception:
                            pass
            return None

        def _lookup(iid: str, col: str):
            for src in sources:
                if iid in src.index and col in src.columns:
                    return src.at[iid, col]
            return None

        out = []
        for r in recs:
            iid = str(r.get("item_id", ""))
            if not iid:
                out.append(r); continue

            title = _pick_non_empty(r.get("title"), _lookup(iid, "title"))
            brand = _pick_non_empty(r.get("brand"), _lookup(iid, "brand"))
            price = _pick_price(r.get("price"), _lookup(iid, "price"))
            cats  = _pick_categories(r.get("categories"), _lookup(iid, "categories"))
            img   = _pick_image_url(
                _lookup(iid, "image_url"),
                _lookup(iid, "imageURLHighRes"),
                _lookup(iid, "imageURL"),
            )
            rank  = _pick_rank(r.get("rank"), _lookup(iid, "rank_num"), _lookup(iid, "rank"))

            if not cats and dataset.lower() == "beauty":
                cats = ["Beauty & Personal Care"]

            rr = {**r}
            if title is not None: rr["title"] = title
            if brand is not None: rr["brand"] = brand
            rr["price"] = price
            rr["categories"] = cats
            rr["image_url"] = img
            rr["rank"] = rank
            out.append(rr)
        return out
    except Exception:
        return recs
    
# =========================
# FastAPI app
# =========================
app = FastAPI(title="MMR-Agentic-CoVE API", version="1.0.5")  # bumped

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import HTTPException

def _discover_faiss_names_api(dataset: str) -> List[str]:
    proc = get_processed_path(dataset)
    idx_dir = proc / "index"
    if not idx_dir.exists():
        return []
    names: List[str] = []
    for p in sorted(idx_dir.glob("items_*.faiss")):
        # only keep this dataset's indices: items_<dataset>_*.faiss
        if not p.stem.startswith(f"items_{dataset}_"):
            continue
        # expose the name AFTER 'items_'
        names.append(p.stem[len("items_"):])  # e.g. beauty_concat
    return names

@app.get("/faiss")
def list_faiss(dataset: str = Query(..., description="Dataset name")):
    try:
        names = _discover_faiss_names_api(dataset)
        return {"dataset": dataset, "indexes": names}
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return JSONResponse(status_code=500, content={"detail": f"/faiss failed: {e}", "traceback": tb})

@app.get("/defaults")
def get_defaults(dataset: str = Query(..., description="Dataset name")):
    """
    Return defaults.json contents (if present) so the UI can auto-fill
    weights, k, and a suggested FAISS name.
    """
    try:
        proc = get_processed_path(dataset)
        fp = proc / "index" / "defaults.json"
        if not fp.exists():
            return {"dataset": dataset, "defaults": {}}
        try:
            payload = json.loads(fp.read_text())
        except Exception:
            payload = {}
        return {"dataset": dataset, "defaults": payload}
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})

# =========================
# Schemas
# =========================
class RecommendIn(BaseModel):
    dataset: str
    user_id: str
    k: int = 10
    fusion: str = Field(default="weighted", pattern="^(concat|weighted)$")
    # If any of these are None, the service will fall back to defaults.json (or the internal fallback).
    w_text: Optional[float] = None
    w_image: Optional[float] = None
    w_meta: Optional[float] = None
    use_faiss: bool = False
    faiss_name: Optional[str] = None
    exclude_seen: bool = True
    alpha: Optional[float] = None  # legacy/no-op but accepted
    # Optional passthrough for future CoVE handling (UI may send it; safe to ignore)
    cove: Optional[str] = None     # NEW (optional, ignored by service unless you wire it)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    messages: List[ChatMessage]
    dataset: Optional[str] = None
    user_id: Optional[str] = None
    k: int = 5
    use_faiss: bool = False
    faiss_name: Optional[str] = None

# =========================
# JSON helpers
# =========================
def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

# =========================
# Endpoints (info)
# =========================

@app.get("/users")
def list_users(dataset: str = Query(..., description="Dataset name, e.g., 'beauty'")):
    """
    Return available user_ids (and optional display names if user_map.parquet exists).
    """
    try:
        proc = get_processed_path(dataset)
        fp_ids = proc / "user_text_emb.parquet"
        if not fp_ids.exists():
            return JSONResponse(
                status_code=400,
                content={"detail": f"Unknown dataset '{dataset}' or missing '{fp_ids.name}' in {proc}."},
            )

        # Load ids
        df_ids = pd.read_parquet(fp_ids, columns=["user_id"])
        users = sorted(df_ids["user_id"].astype(str).unique().tolist())

        # Optional names
        names: Dict[str, str] = {}
        try:
            umap_fp = proc / "user_map.parquet"
            if umap_fp.exists():
                umap = pd.read_parquet(umap_fp)
                if {"user_id", "user_name"} <= set(umap.columns):
                    umap["user_id"] = umap["user_id"].astype(str)
                    umap = umap.dropna(subset=["user_id"]).drop_duplicates("user_id")
                    names = dict(zip(umap["user_id"], umap["user_name"].fillna("").astype(str)))
        except Exception:
            names = {}

        return {"dataset": dataset, "count": len(users), "users": users, "names": names}

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return JSONResponse(status_code=500, content={"detail": f"/users failed: {e}", "traceback": tb})

@app.get("/agentz")
def agentz():
    return _agent_introspection()

    
# api/app_api.py  (Part 4/5)

@app.post("/recommend")
def make_recommend(body: RecommendIn):
    """
    Core recommendation endpoint.
    - Validates dataset files exist
    - Optionally validates FAISS index if use_faiss=true
    - Calls service.recommender.recommend_for_user
    - Enriches with catalog info
    - Normalizes JSON (numpy/pandas safe)
    - NEW: adds 'metrics' block (hit@k, ndcg@k, memory_mb) if found
    """
    try:
        # --- Preflight dataset/file check (mirrors /users) ---
        proc = get_processed_path(body.dataset)
        user_fp = proc / "user_text_emb.parquet"
        if not user_fp.exists():
            return JSONResponse(
                status_code=400,
                content={"detail": f"Unknown dataset '{body.dataset}' or missing '{user_fp.name}' in {proc}."},
            )

        # --- Build service config ---
        cfg = RecommendConfig(
            dataset=body.dataset,
            user_id=str(body.user_id),
            k=int(body.k),
            fusion=body.fusion,
            weights=FusionWeights(text=body.w_text, image=body.w_image, meta=body.w_meta),
            alpha=body.alpha,                # legacy; ignored by service
            use_faiss=body.use_faiss,
            faiss_name=body.faiss_name,
            exclude_seen=body.exclude_seen,
        )

        # --- Optional FAISS check (if explicit name given) ---
        if cfg.use_faiss and cfg.faiss_name:
            index_path = proc / "index" / f"items_{cfg.faiss_name}.faiss"
            if not index_path.exists():
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"FAISS index not found: {index_path}. Build it or set use_faiss=false."},
                )

        # --- Call recommender service ---
        out = recommend_for_user(cfg)

        # Normalize list key
        recs = out.get("results")
        if recs is None:
            recs = out.get("recommendations", [])
        recs = list(recs or [])[: int(cfg.k)]

        # Enrich & normalize
        recs = _enrich_with_catalog(body.dataset, recs)
        _normalize_categories_in_place(recs)

        # Final coercions
        for r in recs:
            # rank
            rn = r.get("rank_num")
            if rn is not None:
                try: r["rank"] = int(rn)
                except Exception: r["rank"] = None
            else:
                rv = r.get("rank")
                if isinstance(rv, str):
                    m = re.search(r"[\d,]+", rv); r["rank"] = int(m.group(0).replace(",", "")) if m else None
                elif isinstance(rv, (int, float)):
                    try: r["rank"] = int(rv)
                    except Exception: r["rank"] = None
                else:
                    r["rank"] = None

            # price
            v = r.get("price")
            try:
                rv = float(v) if v not in (None, "", "nan") else None
                r["price"] = rv if (rv is None or math.isfinite(rv)) else None
            except Exception:
                r["price"] = None

            # score
            v = r.get("score")
            try:
                rv = float(v) if v not in (None, "", "nan") else None
                r["score"] = rv if (rv is None or math.isfinite(rv)) else None
            except Exception:
                r["score"] = None

            # image_url
            v = r.get("image_url")
            if isinstance(v, list):
                r["image_url"] = next((u for u in v if isinstance(u, str) and u.strip()), None)
            elif isinstance(v, str):
                r["image_url"] = v.strip() or None
            else:
                r["image_url"] = None

            # guard
            cats = r.get("categories")
            if isinstance(cats, list) and len(cats) == 1 and isinstance(cats[0], str) and cats[0].strip() == "[]":
                r["categories"] = []

        # Put normalized list back
        out["results"] = _to_jsonable(recs)
        out["recommendations"] = _to_jsonable(recs)

        # ---------- NEW: attach metrics if we can find them ----------
        try:
            metrics = _latest_metrics_for(
                dataset=body.dataset,
                fusion=body.fusion,
                k=int(body.k),
                faiss_name=body.faiss_name,
            )
            if metrics:
                out["metrics"] = metrics
        except Exception:
            # swallow — metrics are optional
            pass

        return JSONResponse(content=_to_jsonable(out))

    except FileNotFoundError:
        return JSONResponse(status_code=400, content={"detail": f"Dataset '{body.dataset}' not found or incomplete."})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": f"/recommend failed: {e}"})
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return JSONResponse(status_code=500, content={"detail": f"/recommend failed: {e}", "traceback": tb})
    
# api/app_api.py  (Part 5/5)

@app.post("/chat_recommend")
def chat_recommend(body: ChatIn):
    # Tolerant parse of messages
    msgs: List[Dict[str, str]] = []
    for m in body.messages:
        if isinstance(m, dict):
            msgs.append({"role": m.get("role"), "content": m.get("content")})
        else:
            d = m.model_dump() if hasattr(m, "model_dump") else m.dict()
            msgs.append({"role": d.get("role"), "content": d.get("content")})

    try:
        out: Dict[str, Any] = {"reply": "", "recommendations": []}
        recs: List[Dict[str, Any]] = []

        # 1) Ask the agent
        if hasattr(CHAT_AGENT, "reply"):
            candidate_kwargs = {
                "messages": msgs,
                "dataset": body.dataset,
                "user_id": body.user_id,
                "k": body.k,
                "use_faiss": body.use_faiss,
                "faiss_name": body.faiss_name,
            }
            sig = inspect.signature(CHAT_AGENT.reply)
            allowed = set(sig.parameters.keys())
            safe_kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed}
            agent_out = CHAT_AGENT.reply(**safe_kwargs)
            if isinstance(agent_out, dict):
                out.update(agent_out)
                recs = agent_out.get("recommendations") or []
            else:
                out["reply"] = str(agent_out) if agent_out is not None else ""

        recs = [dict(r) if not isinstance(r, dict) else r for r in (recs or [])]

        # 2) Fallback
        if not recs:
            cfg = RecommendConfig(
                dataset=body.dataset or "beauty",
                user_id=str(body.user_id or ""),
                k=int(body.k or 5),
                fusion="weighted",
                weights=FusionWeights(text=1.0, image=0.2, meta=0.2),
                alpha=None,
                use_faiss=False,
                faiss_name=None,
                exclude_seen=True,
            )
            try:
                reco_out = recommend_for_user(cfg)
                recs = reco_out.get("results") or reco_out.get("recommendations") or []
                recs = [dict(r) if not isinstance(r, dict) else r for r in recs]
                if not out.get("reply"):
                    out["reply"] = "Here are some items you might like."
            except Exception:
                pass

        # 3) Enrich + normalize (like /recommend)
        ds = body.dataset or "beauty"
        recs = _enrich_with_catalog(ds, recs)
        _normalize_categories_in_place(recs)

        for r in recs:
            # price
            v = r.get("price")
            try:
                rv = float(v) if v not in (None, "", "nan") else None
                r["price"] = rv if (rv is None or math.isfinite(rv)) else None
            except Exception:
                r["price"] = None
            # score
            v = r.get("score")
            try:
                rv = float(v) if v not in (None, "", "nan") else None
                r["score"] = rv if (rv is None or math.isfinite(rv)) else None
            except Exception:
                r["score"] = None
            # rank
            rn = r.get("rank_num")
            if rn is not None:
                try: r["rank"] = int(rn)
                except Exception: r["rank"] = None
            else:
                rv = r.get("rank")
                if isinstance(rv, str):
                    m = re.search(r"[\d,]+", rv); r["rank"] = int(m.group(0).replace(",", "")) if m else None
                elif isinstance(rv, (int, float)):
                    try: r["rank"] = int(rv)
                    except Exception: r["rank"] = None
                else:
                    r["rank"] = None
            # image_url (string)
            v = r.get("image_url")
            if isinstance(v, list):
                r["image_url"] = next((u for u in v if isinstance(u, str) and u.strip()), None)
            elif isinstance(v, str):
                r["image_url"] = v.strip() or None
            else:
                r["image_url"] = None

        # 4) Lightweight chat constraints (budget/keyword) — unchanged
        last = (msgs[-1]["content"] if msgs else "") or ""
        cap = _parse_price_cap(last)
        kw  = _parse_keyword(last)
        if cap is not None:
            recs = [r for r in recs if (r.get("price") is not None and r["price"] <= cap)]
        if kw:
            lowkw = kw.lower()
            def _matches(item: Dict[str, Any]) -> bool:
                fields = [str(item.get("brand") or ""), str(item.get("item_id") or "")]
                fields.extend(item.get("categories") or [])
                return lowkw in " ".join(fields).lower()
            filtered = [r for r in recs if _matches(r)]
            if filtered:
                recs = filtered

        out["recommendations"] = recs
        out["results"] = recs

        # ---------- NEW: attach metrics (optional best-effort) ----------
        try:
            metrics = _latest_metrics_for(
                dataset=ds,
                fusion="weighted",   # chat uses weighted defaults for now
                k=int(body.k or 5),
                faiss_name=body.faiss_name,
            )
            if metrics:
                out["metrics"] = metrics
        except Exception:
            pass

        return JSONResponse(content=_to_jsonable(out))

    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return JSONResponse(status_code=400, content={"detail": f"/chat_recommend failed: {e}", "traceback": tb})

# =========================
# Health & root
# =========================
@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "MMR-Agentic-CoVE API", "version": getattr(app, "version", None) or "unknown"}

@app.get("/")
def root():
    return {"ok": True, "service": "MMR-Agentic-CoVE API"}