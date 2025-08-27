# ui/app_gradio.py
from __future__ import annotations

import os
import math
import time
import re
import json
import subprocess, sys
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
# Local paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR      = PROJECT_ROOT / "logs"
PLOTS_DIR     = LOGS_DIR / "plots"

def _processed_path(dataset: str) -> Path:
    return PROCESSED_DIR / (dataset or "").lower().strip()

# --------------------------------------------------------------------------------------
# Discovery helpers
# --------------------------------------------------------------------------------------
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
    if dataset in _ITEM_META:
        return _ITEM_META[dataset]

    proc = _processed_path(dataset)
    candidates = [
        proc / "items_catalog.parquet",      # enriched catalog if present
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

    for c in ["item_id", "title", "brand", "price", "categories", "image_url", "rank", "rank_num", "rank_cat"]:
        if c not in df.columns:
            df[c] = None

    def _parse_rank_str(s):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return None, None
        s = str(s)
        m = re.search(r"([\d,]+)\s+in\s+(.+?)(?:\(|$)", s)
        if m:
            return int(m.group(1).replace(",", "")), m.group(2).strip()
        m2 = re.search(r"[\d,]+", s)
        return (int(m2.group(0).replace(",", "")), None) if m2 else (None, None)

    if "rank_num" in df.columns:
        need = df["rank_num"].isna()
        if "rank" in df.columns and need.any():
            parsed = df.loc[need, "rank"].apply(_parse_rank_str)
            df.loc[need, "rank_num"] = [x[0] for x in parsed]
            df.loc[need, "rank_cat"] = [x[1] for x in parsed]
    else:
        parsed = df["rank"].apply(_parse_rank_str)
        df["rank_num"] = [x[0] for x in parsed]
        df["rank_cat"] = [x[1] for x in parsed]

    df["item_id"] = df["item_id"].astype(str)
    _ITEM_META[dataset] = df.set_index("item_id", drop=False)
    return _ITEM_META[dataset]

def _load_user_names(dataset: str) -> Dict[str, str]:
    if dataset in _USER_NAMES:
        return _USER_NAMES[dataset]
    proc = _processed_path(dataset)
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
    meta = _load_item_meta(dataset)
    out = []
    for r in recs:
        iid = str(r.get("item_id", ""))
        row = meta.loc[iid] if iid in meta.index else None
        title = (None if row is None else row.get("title"))
        brand = r.get("brand") or (None if row is None else row.get("brand"))
        price = r.get("price") if r.get("price") is not None else (None if row is None else row.get("price"))
        image_url = r.get("image_url") or (None if row is None else row.get("image_url"))
        typ = _infer_type_from_title(title)
        rank_num = None if row is None else row.get("rank_num")
        if rank_num is None and row is not None:
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
    css = (
        "<style>"
        ".cards-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:16px}"
        "@media(max-width:1200px){.cards-grid{grid-template-columns:repeat(3,1fr)}}"
        "@media(max-width:800px){.cards-grid{grid-template-columns:repeat(2,1fr)}}"
        ".card{background:#fff;border-radius:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:12px;border:1px solid rgba(0,0,0,.06)}"
        ".card img{width:100%;height:auto;border-radius:12px;display:block;object-fit:contain;max-height:220px;margin:auto}"
        ".caption{font-family:ui-sans-serif,system-ui,-apple-system;line-height:1.35;margin-top:10px;color:#111}"
        ".caption .id{font-weight:800;font-size:1.05rem;margin-bottom:4px;color:#111}"
        ".caption .title{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:6px;color:#222}"
        ".label{color:#333;font-weight:600}"
        ".muted{color:#555}"
        ".noimg{display:flex;align-items:center;justify-content:center;height:220px;background:#f6f6f8;color:#777;border-radius:12px;font-style:italic}"
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
# Payload preview (JSON + cURL)
# --------------------------------------------------------------------------------------
def _build_payload(dataset, user_id, k, fusion, wt, wi, wm, use_faiss, faiss_name, exclude_seen, alpha):
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "user_id": (user_id or "").strip(),
        "k": int(max(1, min(int(k), 5))),  # UI hard-cap
        "fusion": fusion,
        "w_text": float(wt),
        "w_image": float(wi),
        "w_meta": float(wm),
        "use_faiss": bool(use_faiss),
        "exclude_seen": bool(exclude_seen),
    }
    if fusion == "weighted" and alpha is not None:
        payload["alpha"] = float(alpha)
    if use_faiss and (faiss_name or "").strip():
        payload["faiss_name"] = str(faiss_name).strip()
    return payload

def _curl_for_payload(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return (
        f"curl -s -X POST {RECOMMEND_ENDPOINT} \\\n"
        f"  -H 'content-type: application/json' \\\n"
        f"  -d '{data}'"
    )

# --------------------------------------------------------------------------------------
# API callers
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
) -> Tuple[List[Dict[str, Any]], str, float, Dict[str, Any], str]:
    # Prepare payload (also returned for the debug panel)
    payload = _build_payload(dataset, user_id, k, fusion, w_text, w_image, w_meta, use_faiss, faiss_name, exclude_seen, alpha)
    curl_cmd = _curl_for_payload(payload)

    # If FAISS requested but no index name, disable to avoid 400s.
    if payload["use_faiss"] and "faiss_name" not in payload:
        payload["use_faiss"] = False

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
        return [], f"Error {status}: {msg}", latency_ms, payload, curl_cmd

    recs = _sort_recs_desc(body.get("recommendations") or [])[:5]
    recs = _augment_with_meta(recs, dataset)
    return recs, f"OK • {len(recs)} items • {latency_ms:.0f} ms", latency_ms, payload, curl_cmd

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
# Metrics / plots helpers
# --------------------------------------------------------------------------------------
def _load_metrics_table(dataset: str, limit: int = 30) -> pd.DataFrame:
    csv_fp = LOGS_DIR / "metrics.csv"
    if not csv_fp.exists():
        return pd.DataFrame(columns=["timestamp","dataset","run_name","model","k","hit","ndcg","latency_ms","memory_mb","notes"])
    df = pd.read_csv(csv_fp)
    if dataset:
        df = df[df["dataset"] == dataset]
    # prefer newest first
    if "timestamp" in df.columns:
        try:
            df = df.sort_values("timestamp", ascending=False)
        except Exception:
            pass
    if limit and len(df) > limit:
        df = df.head(limit)
    return df

def _find_plot_images(dataset: str, k: int) -> List[str]:
    if not PLOTS_DIR.exists():
        return []
    pat = f"{dataset}_k{k}"
    images = []
    for p in sorted(PLOTS_DIR.glob("*.png")):
        if pat in p.name:
            images.append(str(p))
    # fallbacks to older names
    for alt in ["comparison", "quality", "trend"]:
        for p in sorted(PLOTS_DIR.glob(f"{dataset}_*{alt}*.png")):
            if str(p) not in images:
                images.append(str(p))
    return images[:6]

def _refresh_reports(dataset: str, k: int):
    df = _load_metrics_table(dataset)
    imgs = _find_plot_images(dataset, k)
    msg = f"Showing {len(df)} rows from logs/metrics.csv; {len(imgs)} plot(s) found."
    return df, imgs, msg

def _regenerate_plots(dataset: str, k: int):
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/plot_metrics.py"),
           "--dataset", dataset, "--k", str(k),
           "--metrics_csv", str(LOGS_DIR / "metrics.csv"),
           "--out_dir", str(PLOTS_DIR)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (cp.stdout or "") + "\n" + (cp.stderr or "")
    except Exception as e:
        out = f"Failed to run plot script: {e}"
    df, imgs, _ = _refresh_reports(dataset, k)
    return df, imgs, out.strip() or "Done."

# --------------------------------------------------------------------------------------
# Event handlers
# --------------------------------------------------------------------------------------
def _run_reco(dataset: str, user_id: str, k: int, fusion: str, wt: float, wi: float, wm: float, use_faiss: bool, faiss_name: str, exclude_seen: bool, alpha: Optional[float]):
    user_id = (user_id or "").strip()
    if not user_id:
        empty_df = pd.DataFrame(columns=["item_id", "score", "title", "type", "brand", "price", "rank"])
        payload = _build_payload(dataset, user_id, k, fusion, wt, wi, wm, use_faiss, faiss_name, exclude_seen, alpha)
        return (_render_cards_html([]), empty_df, "Error: please pick a User ID (or type one).", "—", payload, _curl_for_payload(payload))

    recs, status_text, latency_ms, payload, curl_cmd = _call_recommend(
        dataset=dataset, user_id=user_id, k=max(1, min(k, 5)),
        fusion=fusion, w_text=wt, w_image=wi, w_meta=wm,
        use_faiss=use_faiss, faiss_name=faiss_name, exclude_seen=exclude_seen, alpha=alpha,
    )
    cards_html = _render_cards_html(recs)
    df = _to_dataframe(recs)
    return cards_html, df, status_text, (f"⏱️ {latency_ms:.0f} ms" if latency_ms > 0 else "—"), payload, curl_cmd

def _on_dataset_change(dataset: str):
    _load_item_meta(dataset)
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
    gr.Markdown("## MMR-Agentic-CoVE — Multimodal Recommender (Gradio UI)")

    with gr.Row():
        # ========= LEFT: Controls =========
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
                k_slider = gr.Slider(1, 5, value=5, step=1, label="Top-K")  # hard-limit 5

            recommend_btn = gr.Button("Get Recommendations", variant="primary")

            with gr.Accordion("Debug / API payload", open=False):
                payload_json = gr.JSON(label="Payload sent to /recommend")
                curl_tb = gr.Textbox(label="Copy cURL", lines=3)

            gr.Markdown("---")
            gr.Markdown("### New user (cold start) / Chat")
            chat = gr.Chatbot(label="Tell me what you’re looking for", type="messages")
            chat_msg = gr.Textbox(placeholder="e.g., gentle shampoo under $20, unscented", label="Message")
            with gr.Row():
                chat_send = gr.Button("Send", variant="primary")
                chat_clear = gr.Button("Clear")

            gr.Markdown("---")
            with gr.Accordion("Metrics & reports", open=False):
                with gr.Row():
                    plots_k = gr.Slider(1, 50, value=10, step=1, label="K for plots")
                    refresh_plots_btn = gr.Button("↻ Refresh metrics/plots", variant="secondary")
                    regen_plots_btn = gr.Button("🛠️ Regenerate plots", variant="secondary")
                metrics_table = gr.Dataframe(
                    headers=["timestamp","dataset","run_name","model","k","hit","ndcg","latency_ms","memory_mb","notes"],
                    row_count=(0,"dynamic"), col_count=(10,"dynamic"),
                    interactive=False, label="metrics.csv (latest first)"
                )
                plots_gallery = gr.Gallery(label="Plots", show_label=True, columns=2, height=320)
                plots_console = gr.Textbox(label="Reports console", lines=4)

        # ========= RIGHT: Results =========
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

    # ----- Wiring -----
    check_api_btn.click(fn=_on_check_api, inputs=[], outputs=[health_md])

    dataset_dd.change(fn=_on_dataset_change, inputs=[dataset_dd], outputs=[user_dd, users_info, faiss_dd, use_faiss_ck])

    user_dd.change(fn=_on_user_change, inputs=[dataset_dd, user_dd], outputs=[users_info])

    refresh_users_btn.click(fn=_on_refresh_users, inputs=[dataset_dd], outputs=[user_dd, users_info])

    fusion_dd.change(fn=_toggle_alpha, inputs=[fusion_dd], outputs=[alpha_slider])

    recommend_btn.click(
        fn=_run_reco,
        inputs=[dataset_dd, user_dd, k_slider, fusion_dd, w_text, w_image, w_meta, use_faiss_ck, faiss_dd, exclude_seen_ck, alpha_slider],
        outputs=[cards_html, table, status_md, latency_md, payload_json, curl_tb],
    )

    chat_send.click(fn=_chat_send, inputs=[chat, chat_msg, dataset_dd, user_dd], outputs=[chat, chat_msg, cards_html, table])
    chat_clear.click(fn=_chat_clear, inputs=[], outputs=[chat, chat_msg, cards_html, table])

    # Reports panel
    refresh_plots_btn.click(fn=_refresh_reports, inputs=[dataset_dd, plots_k], outputs=[metrics_table, plots_gallery, plots_console])
    regen_plots_btn.click(fn=_regenerate_plots, inputs=[dataset_dd, plots_k], outputs=[metrics_table, plots_gallery, plots_console])

    # Populate on initial load
    def _on_initial_load(dataset: str):
        _load_item_meta(dataset)
        _load_user_names(dataset)
        u_dd, u_msg, f_dd, use_faiss = _on_dataset_change(dataset)
        # also prime reports
        df, imgs, msg = _refresh_reports(dataset, 10)
        return u_dd, u_msg, f_dd, use_faiss, df, imgs, msg

    demo.load(
        fn=_on_initial_load,
        inputs=[dataset_dd],
        outputs=[user_dd, users_info, faiss_dd, use_faiss_ck, metrics_table, plots_gallery, plots_console],
    )

    faiss_refresh_btn.click(fn=_on_refresh_faiss, inputs=[dataset_dd], outputs=[faiss_dd, use_faiss_ck])

# --------------------------------------------------------------------------------------
# Launch (local)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT", "7860")))