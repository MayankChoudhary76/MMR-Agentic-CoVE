# ui/app_gradio.py
from __future__ import annotations

import os
import math
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# Backend configuration
# --------------------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/") or "http://127.0.0.1:8000"
RECOMMEND_ENDPOINT      = f"{API_BASE_URL}/recommend"
CHAT_RECOMMEND_ENDPOINT = f"{API_BASE_URL}/chat_recommend"
USERS_ENDPOINT          = f"{API_BASE_URL}/users"
HEALTH_ENDPOINT         = f"{API_BASE_URL}/healthz"

# --------------------------------------------------------------------------------------
# Local helpers (project paths & discovery)
# --------------------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

def _processed_path(dataset: str) -> Path:
    return PROCESSED_DIR / (dataset or "").lower().strip()

def _discover_users_local(dataset: str) -> List[str]:
    fp = _processed_path(dataset) / "user_text_emb.parquet"
    if not fp.exists():
        return []
    try:
        df = pd.read_parquet(fp, columns=["user_id"])
        return sorted(df["user_id"].astype(str).unique().tolist())
    except Exception:
        return []

def _discover_faiss_names(dataset: str) -> List[str]:
    idx_dir = _processed_path(dataset) / "index"
    if not idx_dir.exists():
        return []
    names = []
    for p in sorted(idx_dir.glob("items_*.faiss")):
        stem = p.stem  # items_<name>
        if stem.startswith("items_"):
            names.append(stem[len("items_"):])
    return names

# --------------------------------------------------------------------------------------
# Light metadata caches (for richer UI)
# --------------------------------------------------------------------------------------
_ITEM_META: Dict[str, pd.DataFrame] = {}      # dataset -> DataFrame indexed by item_id
_USER_NAMES: Dict[str, Dict[str, str]] = {}   # dataset -> {user_id: display_name}

_TITLE_TYPE_PATTERNS = [
    ("shampoo", "Shampoo"),
    ("conditioner", "Conditioner"),
    ("soap", "Soap"),
    ("body wash", "Body Wash"),
    ("face wash", "Face Wash"),
    ("lotion", "Lotion"),
    ("cream", "Cream"),
    ("serum", "Serum"),
    ("hair oil", "Hair Oil"),
    ("beard oil", "Beard Oil"),
    ("gel", "Gel"),
]

def _load_item_meta(dataset: str) -> pd.DataFrame:
    """
    Load per-item metadata for display.
    Prefer the enriched catalog we built (items_catalog.parquet),
    then fall back to earlier joins if needed.
    """
    if dataset in _ITEM_META:
        return _ITEM_META[dataset]

    proc = _processed_path(dataset)
    candidates = [
        proc / "items_catalog.parquet",      # <-- new enriched catalog (title, rank_num, etc.)
        proc / "items_with_meta.parquet",
        proc / "joined.parquet",
    ]

    df = pd.DataFrame()
    for fp in candidates:
        if fp.exists():
            try:
                df = pd.read_parquet(fp)
                break
            except Exception:
                pass

    # Normalize expected cols
    for c in ["item_id", "title", "brand", "price", "categories", "image_url", "rank", "rank_num", "rank_cat"]:
        if c not in df.columns:
            df[c] = None

    # Parse rank if we only have a string
    def _parse_rank_str(s):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return None, None
        s = str(s)
        m = re.search(r"([\d,]+)\s+in\s+(.+?)(?:\(|$)", s)
        if m:
            n = int(m.group(1).replace(",", ""))
            cat = m.group(2).strip()
            return n, cat
        m2 = re.search(r"[\d,]+", s)
        if m2:
            return int(m2.group(0).replace(",", "")), None
        return None, None

    if "rank_num" in df.columns:
        # fill rank_num from rank string where missing
        need = df["rank_num"].isna()
        if "rank" in df.columns and need.any():
            parsed = df.loc[need, "rank"].apply(_parse_rank_str)
            df.loc[need, "rank_num"] = [x[0] for x in parsed]
            df.loc[need, "rank_cat"] = [x[1] for x in parsed]
    else:
        parsed = df["rank"].apply(_parse_rank_str)
        df["rank_num"] = [x[0] for x in parsed]
        df["rank_cat"] = [x[1] for x in parsed]

    # Index by item_id for quick lookups
    df["item_id"] = df["item_id"].astype(str)
    _ITEM_META[dataset] = df.set_index("item_id", drop=False)
    return _ITEM_META[dataset]

def _load_user_names(dataset: str) -> Dict[str, str]:
    if dataset in _USER_NAMES:
        return _USER_NAMES[dataset]
    proc = _processed_path(dataset)
    # Prefer user_map.parquet (built by build_catalog.py); fall back to reviews.parquet
    mapping: Dict[str, str] = {}
    umap_fp = proc / "user_map.parquet"
    if umap_fp.exists():
        try:
            umap = pd.read_parquet(umap_fp)
            if {"user_id", "user_name"} <= set(umap.columns):
                umap["user_id"] = umap["user_id"].astype(str)
                umap = umap.dropna(subset=["user_id"]).drop_duplicates("user_id")
                mapping = dict(zip(umap["user_id"], umap["user_name"].fillna("").astype(str)))
        except Exception:
            mapping = {}

    if not mapping:
        rev_fp = proc / "reviews.parquet"
        if rev_fp.exists():
            try:
                df = pd.read_parquet(rev_fp, columns=["reviewerID", "reviewerName"]).dropna()
                df["reviewerID"] = df["reviewerID"].astype(str)
                df["reviewerName"] = df["reviewerName"].astype(str)
                mapping = df.drop_duplicates("reviewerID").set_index("reviewerID")["reviewerName"].to_dict()
            except Exception:
                mapping = {}

    _USER_NAMES[dataset] = mapping
    return mapping

def _infer_type_from_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    t = title.lower()
    for needle, label in _TITLE_TYPE_PATTERNS:
        if needle in t:
            return label
    return None

# --------------------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------------------
def _fetch_json(url: str, method: str = "GET", payload: Optional[Dict] = None, timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    try:
        if method.upper() == "GET":
            r = requests.get(url, params=(payload or {}), timeout=timeout)
        else:
            r = requests.post(url, json=(payload or {}), timeout=timeout)
        ct = (r.headers.get("content-type") or "")
        if "application/json" in ct:
            return r.status_code, r.json()
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"detail": r.text}
    except Exception as e:
        return 0, {"detail": f"Request failed: {e}"}

# --------------------------------------------------------------------------------------
# Data shaping helpers
# --------------------------------------------------------------------------------------
def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default

def _fmt_price(p: Any) -> str:
    p = _safe_float(p)
    return f"${p:.2f}" if isinstance(p, float) else "-"

def _sort_recs_desc(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(recs or [], key=lambda r: _safe_float(r.get("score"), 0.0) or 0.0, reverse=True)

def _augment_with_meta(recs: List[Dict[str, Any]], dataset: str) -> List[Dict[str, Any]]:
    """Merge title/type/rank and any missing brand/price/image from processed meta caches."""
    meta = _load_item_meta(dataset)
    out = []
    for r in recs:
        iid = str(r.get("item_id", ""))
        row = meta.loc[iid] if iid in meta.index else None

        # Base fields
        title = (None if row is None else row.get("title"))
        brand = r.get("brand") or (None if row is None else row.get("brand"))
        price = r.get("price")
        if price is None and row is not None:
            price = row.get("price")
        image_url = r.get("image_url") or (None if row is None else row.get("image_url"))

        # Type from title
        typ = _infer_type_from_title(title)

        # Rank
        rank_num = None if row is None else row.get("rank_num")
        if rank_num is None and row is not None:
            # last resort: parse a raw 'rank' string if present
            s = row.get("rank")
            if s:
                m = re.search(r"[\d,]+", str(s))
                rank_num = int(m.group(0).replace(",", "")) if m else None

        out.append({
            **r,
            "title": title,
            "type": typ,
            "brand": brand,
            "price": price,
            "image_url": image_url,
            "rank": rank_num,
        })
    return out

def _to_dataframe(recs: List[Dict[str, Any]]) -> pd.DataFrame:
    # Table now: item_id | score | title | type | brand | price | rank
    cols = ["item_id", "score", "title", "type", "brand", "price", "rank"]
    rows = []
    for r in recs:
        rows.append([
            r.get("item_id", ""),
            _safe_float(r.get("score"), 0.0),
            r.get("title"),
            r.get("type"),
            r.get("brand"),
            _safe_float(r.get("price")),
            r.get("rank"),
        ])
    return pd.DataFrame(rows, columns=cols)

def _render_cards_html(recs: List[Dict[str, Any]]) -> str:
    """
    Render a neat, non‑clickable 5‑card grid. Each card shows:
    Item ID, Title, Brand, Price, Score. Styles are forced to be
    high‑contrast so they’re readable on laptop dark themes.
    """
    css = (
        "<style>"
        ".cards-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:16px}"
        "@media(max-width:1200px){.cards-grid{grid-template-columns:repeat(3,1fr)}}"
        "@media(max-width:800px){.cards-grid{grid-template-columns:repeat(2,1fr)}}"
        ".card{background:#fff;border-radius:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);"
        " padding:12px;border:1px solid rgba(0,0,0,.06)}"
        ".card img{width:100%;height:auto;border-radius:12px;display:block;object-fit:contain;"
        " max-height:220px;margin:auto}"
        ".caption{font-family:ui-sans-serif,system-ui,-apple-system;line-height:1.35;margin-top:10px;"
        " color:#111}"
        ".caption .id{font-weight:800;font-size:1.05rem;margin-bottom:4px;color:#111}"
        ".caption .title{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
        " margin-bottom:6px;color:#222}"
        ".label{color:#333;font-weight:600}"
        ".muted{color:#555}"
        ".noimg{display:flex;align-items:center;justify-content:center;height:220px;background:#f6f6f8;"
        " color:#777;border-radius:12px;font-style:italic}"
        "</style>"
    )

    if not recs:
        return css + (
            "<div class='cards-grid'>"
            "<div class='card'><div class='noimg'>No items</div>"
            "<div class='caption muted' style='margin-top:8px'>—</div></div>"
            "</div>"
        )

    cards: List[str] = []
    for r in recs[:5]:
        iid = str(r.get("item_id") or "")
        title = (r.get("title") or "").strip()
        title_short = (title[:80] + "…") if len(title) > 80 else title

        brand = r.get("brand") or "-"
        price_txt = _fmt_price(r.get("price"))
        try:
            score_txt = f"{float(r.get('score') or 0):.4f}"
        except Exception:
            score_txt = "0.0000"

        img = (r.get("image_url") or "").strip()
        img_html = f'<img src="{img}" alt="item image" />' if img else "<div class='noimg'>No image</div>"

        cap_html = (
            "<div class='caption'>"
            f"<div class='id'>{iid}</div>"
            f"<div class='title'>{title_short}</div>"
            f"<div><span class='label'>Brand:</span> {brand}</div>"
            f"<div><span class='label'>Price:</span> {price_txt}</div>"
            f"<div class='muted'>Score: {score_txt}</div>"
            "</div>"
        )
        cards.append(f"<div class='card'>{img_html}{cap_html}</div>")

    grid = "".join(cards) or "<div class='muted'>—</div>"
    return css + f"<div class='cards-grid'>{grid}</div>"

# --------------------------------------------------------------------------------------
def _call_recommend(
    dataset: str,
    user_id: str,
    k: int,
    fusion: str,
    w_text: float,
    w_image: float,
    w_meta: float,
    use_faiss: bool,
    faiss_name: Optional[str],
    exclude_seen: bool,
    alpha: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], str, float]:
    k = max(1, min(int(k), 5))  # hard cap 5
    if use_faiss and not (faiss_name or "").strip():
        use_faiss = False

    payload: Dict[str, Any] = {
        "dataset": dataset,
        "user_id": user_id,
        "k": int(k),
        "fusion": fusion,
        "w_text": float(w_text),
        "w_image": float(w_image),
        "w_meta": float(w_meta),
        "use_faiss": bool(use_faiss),
        "exclude_seen": bool(exclude_seen),
    }
    if alpha is not None and fusion == "weighted":
        payload["alpha"] = float(alpha)
    if use_faiss and faiss_name:
        payload["faiss_name"] = str(faiss_name)

    def _post(pld):
        return _fetch_json(RECOMMEND_ENDPOINT, method="POST", payload=pld)

    t0 = time.time()
    status, body = _post(payload)

    # defensive retry for "index out of bounds"
    if status >= 500 and "out of bounds" in str(body.get("detail", "")).lower():
        for k_try in [3, 1]:
            payload["k"] = k_try
            status, body = _post(payload)
            if status == 200:
                break

    latency_ms = (time.time() - t0) * 1000.0
    if status != 200:
        msg = body.get("detail", "Unknown error")
        if "faiss" in str(msg).lower():
            msg = f"{msg} — tip: select a FAISS index or uncheck 'Use FAISS'."
        return [], f"Error {status}: {msg}", latency_ms

    recs = _sort_recs_desc(body.get("recommendations") or [])[:5]
    recs = _augment_with_meta(recs, dataset)
    return recs, f"OK • {len(recs)} items • {latency_ms:.0f} ms", latency_ms

def _call_chat_recommend(messages: List[Dict[str, str]], dataset: Optional[str] = None, user_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    payload = {"messages": messages}
    if dataset:
        payload["dataset"] = dataset
    if user_id:
        payload["user_id"] = user_id
    status, body = _fetch_json(CHAT_RECOMMEND_ENDPOINT, method="POST", payload=payload)
    if status != 200:
        return f"Error {status}: {body.get('detail','Unknown error')}", []
    recs = _augment_with_meta(body.get("recommendations") or [], dataset or "beauty")
    return body.get("reply") or "", recs

def _load_users(dataset: str) -> Tuple[List[str], str, Dict[str,str]]:
    status, body = _fetch_json(USERS_ENDPOINT, method="GET", payload={"dataset": dataset})
    if status == 200 and isinstance(body, dict) and "users" in body:
        users = [str(u) for u in body.get("users") or []]
        names = {str(k): str(v) for k, v in (body.get("names") or {}).items() if v}
        return users, f"Loaded {len(users)} users (API).", names
    users = _discover_users_local(dataset)
    return users, ("Loaded %d users (local parquet)." % len(users) if users else "No users found via API or local parquet."), {}

def _ping_api() -> str:
    status, body = _fetch_json(HEALTH_ENDPOINT, method="GET")
    if status == 200 and body.get("ok"):
        ver = body.get("version", "?")
        return f"✅ API healthy • {ver} • {API_BASE_URL}"
    if status == 0:
        return f"❌ Cannot reach API ({body.get('detail')}) at {API_BASE_URL}"
    return f"⚠️ API responded {status}: {body.get('detail','unknown')} • {API_BASE_URL}"

# --------------------------------------------------------------------------------------
# Event handlers
# --------------------------------------------------------------------------------------
def _run_reco(dataset: str, user_id: str, k: int, fusion: str, wt: float, wi: float, wm: float, use_faiss: bool, faiss_name: str, exclude_seen: bool, alpha: Optional[float]):
    user_id = (user_id or "").strip()
    if not user_id:
        empty_df = pd.DataFrame(columns=["item_id", "score", "title", "type", "brand", "price", "rank"])
        return (_render_cards_html([]), empty_df, "Error: please pick a User ID (or type one).", "—")

    recs, status_text, latency_ms = _call_recommend(
        dataset=dataset, user_id=user_id, k=max(1, min(k, 5)),
        fusion=fusion, w_text=wt, w_image=wi, w_meta=wm,
        use_faiss=use_faiss, faiss_name=faiss_name, exclude_seen=exclude_seen, alpha=alpha,
    )
    cards_html = _render_cards_html(recs)
    df = _to_dataframe(recs)
    return cards_html, df, status_text, (f"⏱️ {latency_ms:.0f} ms" if latency_ms > 0 else "—")

def _on_dataset_change(dataset: str):
    _load_item_meta(dataset)   # warm caches
    _load_user_names(dataset)

    users, u_msg, _ = _load_users(dataset)
    faiss = _discover_faiss_names(dataset)
    return (
        gr.update(choices=users, value=(users[0] if users else None)),
        gr.update(value=f"👤 {u_msg}"),
        gr.update(choices=faiss, value=(faiss[0] if faiss else None)),
        gr.update(value=bool(faiss)),
    )

def _on_refresh_users(dataset: str):
    users, u_msg, _ = _load_users(dataset)
    return (gr.update(choices=users, value=(users[0] if users else None)), gr.update(value=f"👤 {u_msg}"))

def _on_user_change(dataset: str, user_id: Optional[str]):
    user_id = (user_id or "").strip()
    if not user_id:
        return gr.update(value="—")
    names = _load_user_names(dataset)
    name = names.get(user_id)
    return gr.update(value=f"👤 {user_id} — {name}" if name else f"👤 {user_id}")

def _toggle_alpha(fusion: str):
    return gr.update(visible=(fusion == "weighted"))

def _chat_send(history: List[Dict[str, str]], chat_input: str, dataset: str, user_id: str):
    if not (chat_input or "").strip():
        empty_df = pd.DataFrame(columns=["item_id", "score", "title", "type", "brand", "price", "rank"])
        return history, "", "<div class='muted'>—</div>", empty_df

    history = (history or []) + [{"role": "user", "content": chat_input}]
    reply, items = _call_chat_recommend(history, dataset=dataset, user_id=(user_id or None))
    history = history + [{"role": "assistant", "content": reply or "—"}]

    items = _sort_recs_desc(items)[:5]
    cards_html = _render_cards_html(items)
    df = _to_dataframe(items)
    return history, "", cards_html, df

def _chat_clear():
    empty_df = pd.DataFrame(columns=["item_id", "score", "title", "type", "brand", "price", "rank"])
    return [], "", "<div class='muted'>—</div>", empty_df

def _on_check_api():
    return gr.update(value=_ping_api())

def _on_refresh_faiss(dataset: str):
    faiss = _discover_faiss_names(dataset)
    return gr.update(choices=faiss, value=(faiss[0] if faiss else None)), gr.update(value=bool(faiss))

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
with gr.Blocks(title="MMR-Agentic-CoVE • Recommender UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## MMR‑Agentic‑CoVE — Multimodal Recommender (Gradio UI)")

    with gr.Row():
        with gr.Column(scale=3, min_width=300):
            gr.Markdown("### Controls")

            health_md = gr.Markdown("—")
            check_api_btn = gr.Button("Check API", variant="secondary")

            dataset_dd = gr.Dropdown(label="Dataset", choices=["beauty"], value="beauty")

            user_dd = gr.Dropdown(label="User ID", choices=[], value=None, allow_custom_value=True)
            with gr.Row():
                refresh_users_btn = gr.Button("↻ Refresh users", variant="secondary")
                users_info = gr.Markdown("—")

            with gr.Accordion("Advanced", open=False):
                fusion_dd = gr.Radio(label="Fusion", choices=["concat", "weighted"], value="concat")
                alpha_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Alpha (weighted text↔image tilt)", visible=False)

                w_text = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Weight: text")
                w_image = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Weight: image")
                w_meta = gr.Slider(0.0, 2.0, value=0.4, step=0.1, label="Weight: meta")

                use_faiss_ck = gr.Checkbox(value=False, label="Use FAISS")
                faiss_dd = gr.Dropdown(label="FAISS index name", choices=[], value=None, allow_custom_value=True)
                faiss_refresh_btn = gr.Button("↻ Refresh FAISS", variant="secondary")
                exclude_seen_ck = gr.Checkbox(value=True, label="Exclude seen items")
                k_slider = gr.Slider(1, 5, value=5, step=1, label="Top‑K")  # hard‑limit 5

            recommend_btn = gr.Button("Get Recommendations", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### New user (cold start) / Chat")
            chat = gr.Chatbot(label="Tell me what you’re looking for", type="messages")
            chat_msg = gr.Textbox(placeholder="e.g., gentle shampoo under $20, unscented", label="Message")
            with gr.Row():
                chat_send = gr.Button("Send", variant="primary")
                chat_clear = gr.Button("Clear")

        with gr.Column(scale=9):
            gr.Markdown("### Results")
            cards_html = gr.HTML("<div class='muted'>—</div>")

            table = gr.Dataframe(
                headers=["item_id", "score", "title", "type", "brand", "price", "rank"],
                datatype=["str", "number", "str", "str", "str", "number", "number"],
                row_count=(0, "dynamic"),
                col_count=7,
                interactive=False,
                label="Details (sorted by score desc)"
            )

            with gr.Row():
                status_md = gr.Markdown("—")
                latency_md = gr.Markdown("—")

    # Wiring
    check_api_btn.click(fn=_on_check_api, inputs=[], outputs=[health_md])

    dataset_dd.change(fn=_on_dataset_change, inputs=[dataset_dd], outputs=[user_dd, users_info, faiss_dd, use_faiss_ck])

    user_dd.change(fn=_on_user_change, inputs=[dataset_dd, user_dd], outputs=[users_info])

    refresh_users_btn.click(fn=_on_refresh_users, inputs=[dataset_dd], outputs=[user_dd, users_info])

    fusion_dd.change(fn=_toggle_alpha, inputs=[fusion_dd], outputs=[alpha_slider])

    recommend_btn.click(
        fn=_run_reco,
        inputs=[dataset_dd, user_dd, k_slider, fusion_dd, w_text, w_image, w_meta, use_faiss_ck, faiss_dd, exclude_seen_ck, alpha_slider],
        outputs=[cards_html, table, status_md, latency_md],
    )

    chat_send.click(fn=_chat_send, inputs=[chat, chat_msg, dataset_dd, user_dd], outputs=[chat, chat_msg, cards_html, table])

    chat_clear.click(fn=_chat_clear, inputs=[], outputs=[chat, chat_msg, cards_html, table])

    def _on_initial_load(dataset: str):
        _load_item_meta(dataset)
        _load_user_names(dataset)
        return _on_dataset_change(dataset)

    demo.load(fn=_on_initial_load, inputs=[dataset_dd], outputs=[user_dd, users_info, faiss_dd, use_faiss_ck])

    faiss_refresh_btn.click(fn=_on_refresh_faiss, inputs=[dataset_dd], outputs=[faiss_dd, use_faiss_ck])

# --------------------------------------------------------------------------------------
# Launch (local)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT", "7860")))