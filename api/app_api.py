# api/app_api.py
from __future__ import annotations

import os
import time
import inspect
import ast
import math
import re
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.paths import get_processed_path
from src.service.recommender import recommend_for_user, RecommendConfig, FusionWeights
from src.agents.chat_agent import ChatAgent, ChatAgentConfig

# Instantiate the chat agent used by /chat_recommend
CHAT_AGENT = ChatAgent(ChatAgentConfig())

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
    
# ---------------- Helpers ----------------
def _parse_listlike_string(s: str) -> List[str]:
    """
    Parse strings like "['A','B']" or '["A"]' into ['A','B']; otherwise produce a best-effort list.
    """
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
    # Fallback: split on common delimiters or return scalar
    if re.search(r"[>|,/;]+", t):
        return [p.strip() for p in re.split(r"[>|,/;]+", t) if p.strip()]
    return [t] if t else []

def _enrich_with_catalog(dataset: str, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill missing fields (title, rank, image_url, price, categories) from items_catalog.parquet."""
    try:
        proc = get_processed_path(dataset)
        cat_fp = proc / "items_catalog.parquet"
        if not cat_fp.exists() or not recs:
            return recs
        df = pd.read_parquet(cat_fp)
        df = df[["item_id","title","rank","image_url","price","categories"]].copy()
        cat = {str(r.item_id): r for r in df.itertuples(index=False)}
        out = []
        for r in recs:
            iid = str(r.get("item_id",""))
            add = cat.get(iid)
            if add:
                # fill if missing
                r.setdefault("title", getattr(add,"title"))
                r.setdefault("rank",  None if pd.isna(getattr(add,"rank")) else int(getattr(add,"rank")))
                r.setdefault("image_url", getattr(add,"image_url"))
                if r.get("price") in (None,"",0) and pd.notna(getattr(add,"price")):
                    r["price"] = float(getattr(add,"price"))
                if not r.get("categories"):
                    r["categories"] = getattr(add,"categories")
            out.append(r)
        return out
    except Exception:
        return recs
    
# --- simple parsers for budget + keyword ---
_PRICE_RE = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)")

def _parse_price_cap(text: str) -> Optional[float]:
    m = _PRICE_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

_STOPWORDS = {"under","below","less","than","max","upto","up","to","recommend","something","for","me","need","budget","cheap","please","soap","shampoos"}
def _parse_keyword(text: str) -> Optional[str]:
    t = (text or "").lower()
    t = _PRICE_RE.sub(" ", t)
    for w in re.findall(r"[a-z][a-z0-9\-]+", t):
        if w in _STOPWORDS:
            continue
        return w
    return None

def _normalize_categories_in_place(items):
    """
    Force each item's 'categories' into a clean List[str].
    Handles:
      - None
      - "['A','B']"  (stringified list)
      - ["['A','B']"] (list holding a stringified list)
      - list/tuple/set of strings and/or nested containers
    """
    def _as_list_from_string(s: str) -> List[str]:
        s = (s or "").strip()
        if not s:
            return []
        # Try to parse a literal list/tuple first
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            except Exception:
                pass
        # Fallback: treat as a single label
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
            # dedupe while preserving order
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

def _to_jsonable(obj: Any):
    """
    Convert numpy/pandas and other non‑JSON‑serializable objects to plain Python types.
    Replace NaN/Inf with None to satisfy strict JSON.
    """
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    # primitives
    if obj is None or isinstance(obj, (str, bool)):
        return obj

    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    # numpy scalars
    if np is not None:
        if isinstance(obj, getattr(np, "integer", ())):
            return int(obj)
        if isinstance(obj, getattr(np, "floating", ())):
            f = float(obj)
            return None if not math.isfinite(f) else f
        if isinstance(obj, getattr(np, "bool_", ())):
            return bool(obj)

    # containers
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # pandas
    if isinstance(obj, pd.Series):
        return {str(k): _to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        return [_to_jsonable(r) for r in obj.to_dict(orient="records")]

    # namedtuple / objects with _asdict
    if hasattr(obj, "_asdict"):
        return {str(k): _to_jsonable(v) for k, v in obj._asdict().items()}

    # last resort
    return str(obj)


# ---------------- FastAPI app ----------------
app = FastAPI(title="MMR-Agentic-CoVE API", version="1.0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Schemas ----------------
class RecommendIn(BaseModel):
    dataset: str
    user_id: str
    k: int = 10
    fusion: str = Field(default="concat", pattern="^(concat|weighted)$")
    w_text: float = 1.0
    w_image: float = 1.0
    w_meta: float = 0.0
    use_faiss: bool = False
    faiss_name: Optional[str] = None
    exclude_seen: bool = True
    alpha: Optional[float] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: List[ChatMessage]
    dataset: Optional[str] = None
    user_id: Optional[str] = None
    k: int = 5
    # These are present in schema for future compatibility, but the agent
    # may or may not accept them; we filter before calling reply().
    use_faiss: bool = False
    faiss_name: Optional[str] = None


# ---------------- Endpoints ----------------
@app.get("/users")
def list_users(dataset: str = Query(..., description="Dataset name, e.g., 'beauty'")):
    try:
        proc = get_processed_path(dataset)
        fp_ids = proc / "user_text_emb.parquet"
        if not fp_ids.exists():
            return JSONResponse(
                status_code=400,
                content={"detail": f"Unknown dataset '{dataset}' or missing '{fp_ids.name}' in {proc}."},
            )

        # load ids
        df_ids = pd.read_parquet(fp_ids, columns=["user_id"])
        users = sorted(df_ids["user_id"].astype(str).unique().tolist())

        # optional names
        names_fp = proc / "user_map.parquet"
        names = {}
        if names_fp.exists():
            df_names = pd.read_parquet(names_fp)
            if "user_id" in df_names.columns and "user_name" in df_names.columns:
                for r in df_names.itertuples(index=False):
                    uid = str(getattr(r, "user_id"))
                    nm  = getattr(r, "user_name")
                    if uid and uid not in names and pd.notna(nm):
                        names[uid] = str(nm)
        # Try to include optional user name map if present (built by build_catalog.py)
        names = {}
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

@app.post("/recommend")
def make_recommend(body: RecommendIn):
    try:
        # Preflight dataset/file check (mirrors /users)
        proc = get_processed_path(body.dataset)
        user_fp = proc / "user_text_emb.parquet"
        if not user_fp.exists():
            return JSONResponse(
                status_code=400,
                content={
                    "detail": (
                        f"Unknown dataset '{body.dataset}' or missing required file "
                        f"'{user_fp.name}' in {proc}."
                    )
                },
            )

        cfg = RecommendConfig(
            dataset=body.dataset,
            user_id=str(body.user_id),
            k=int(body.k),
            fusion=body.fusion,
            weights=FusionWeights(text=body.w_text, image=body.w_image, meta=body.w_meta),
            alpha=body.alpha,
            use_faiss=body.use_faiss,
            faiss_name=body.faiss_name,
            exclude_seen=body.exclude_seen,
        )

        # Pre-check FAISS index to produce clean 400s
        if cfg.use_faiss:
            if not cfg.faiss_name:
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": (
                            "FAISS is enabled but 'faiss_name' is missing. "
                            "Provide faiss_name or set use_faiss=false."
                        )
                    },
                )
            index_path = proc / "index" / f"items_{cfg.faiss_name}.faiss"
            if not index_path.exists():
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": f"FAISS index not found: {index_path}. "
                                  f"Build it or set use_faiss=false."
                    },
                )

        out = recommend_for_user(cfg)
        # Robust cap so we never exceed available results
        out_recs = (out.get("recommendations") or [])[: int(cfg.k)]
        out_recs = _enrich_with_catalog(body.dataset, out_recs)
        out["recommendations"] = _to_jsonable(out_recs)
        return JSONResponse(content=out)

    except FileNotFoundError:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Dataset '{body.dataset}' not found or incomplete."},
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"detail": f"/recommend failed: {e}"},
        )
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return JSONResponse(
            status_code=500,
            content={"detail": f"/recommend failed: {e}", "traceback": tb},
        )

@app.post("/chat_recommend")
def chat_recommend(body: ChatIn):
    # Tolerant parse of messages (works with Pydantic v1/v2 and plain dicts)
    msgs: List[Dict[str, str]] = []
    for m in body.messages:
        if isinstance(m, dict):
            msgs.append({"role": m.get("role"), "content": m.get("content")})
        else:
            d = m.model_dump() if hasattr(m, "model_dump") else m.dict()
            msgs.append({"role": d.get("role"), "content": d.get("content")})

    try:
        # === 1) Ask the agent (if present) ===
        out: Dict[str, Any] = {"reply": "", "recommendations": []}
        recs: List[Dict[str, Any]] = []

        if hasattr(CHAT_AGENT, "reply"):
            candidate_kwargs = {
                "messages": msgs,
                "dataset": body.dataset,
                "user_id": body.user_id,
                "k": body.k,
                # don't pass use_faiss/faiss_name unless agent supports them
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

        # ensure list[dict]
        recs = [dict(r) if not isinstance(r, dict) else r for r in (recs or [])]

        # === 2) If agent returned nothing, fallback to recommender ===
        if not recs:
            # make a safe RecommendConfig; default to concat/no-faiss
            cfg = RecommendConfig(
                dataset=body.dataset or "beauty",
                user_id=str(body.user_id or ""),
                k=int(body.k or 5),
                fusion="concat",
                weights=FusionWeights(text=1.0, image=1.0, meta=0.4),
                alpha=None,
                use_faiss=False,
                faiss_name=None,
                exclude_seen=True,
            )
            try:
                reco_out = recommend_for_user(cfg)
                recs = reco_out.get("recommendations") or []
                recs = [dict(r) if not isinstance(r, dict) else r for r in recs]
                if not out.get("reply"):
                    out["reply"] = "Here are some items you might like."
            except Exception:
                recs = recs  # keep empty on failure

        # === 3) Normalize fields (categories, score, price) ===
        _normalize_categories_in_place(recs)

        for r in recs:
            # coerce price
            v = r.get("price")
            try:
                rv = float(v) if v not in (None, "", "nan") else None
                if isinstance(rv, float) and not math.isfinite(rv):
                    rv = None
                r["price"] = rv
            except Exception:
                r["price"] = None
            # coerce score
            v = r.get("score")
            try:
                rv = float(v) if v not in (None, "", "nan") else None
                if isinstance(rv, float) and not math.isfinite(rv):
                    rv = None
                r["score"] = rv
            except Exception:
                r["score"] = None

        # === 4) Apply lightweight chat constraints (budget + keyword) ===
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
                hay = " ".join(fields).lower()
                return lowkw in hay
            filtered = [r for r in recs if _matches(r)]
            # only narrow if we actually get something
            if filtered:
                recs = filtered
        # --- FINAL categories fix (right before returning) ---
        for r in recs:
            cats = r.get("categories")
            # If it's exactly one string like "['A','B']" -> parse it
            if isinstance(cats, list) and len(cats) == 1 and isinstance(cats[0], str):
                try:
                    lit = cats[0].strip()
                    if (lit.startswith("[") and lit.endswith("]")) or (lit.startswith("(") and lit.endswith(")")):
                        parsed = ast.literal_eval(lit)
                        if isinstance(parsed, (list, tuple, set)):
                            r["categories"] = [str(x).strip() for x in parsed if x is not None and str(x).strip()]
                except Exception:
                    # leave as-is if it can't be parsed
                    pass
        # === 5) Return ===
        out.setdefault("reply", "")
        out["recommendations"] = recs
        return JSONResponse(content=_to_jsonable(out))

    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return JSONResponse(
            status_code=400,
            content={"detail": f"/chat_recommend failed: {e}", "traceback": tb},
        )
    
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "MMR-Agentic-CoVE API",
        "version": getattr(app, "version", None) or "unknown",
    }


@app.get("/")
def root():
    return {"ok": True, "service": "MMR-Agentic-CoVE API"}