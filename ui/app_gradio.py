# ui/app_gradio.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import requests
from urllib.parse import quote

# -------------------------
# Backend endpoints
# -------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/") or "http://127.0.0.1:8000"
HEALTH_ENDPOINT         = f"{API_BASE_URL}/healthz"
USERS_ENDPOINT          = f"{API_BASE_URL}/users"
RECOMMEND_ENDPOINT      = f"{API_BASE_URL}/recommend"
CHAT_RECOMMEND_ENDPOINT = f"{API_BASE_URL}/chat_recommend"
FAISS_LIST_ENDPOINT     = f"{API_BASE_URL}/faiss"       # optional (UI degrades gracefully)
DEFAULTS_ENDPOINT       = f"{API_BASE_URL}/defaults"    # optional (UI degrades gracefully)
# cache of {dataset: {user_id: user_name}}
USER_NAMES: Dict[str, Dict[str, str]] = {}
# -------------------------
# Small helpers
# -------------------------
# --- Built-in SVG fallback (always available) ---
_PLACEHOLDER_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' width='480' height='360'>
  <defs>
    <linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>
      <stop offset='0%' stop-color='#f3f4f6'/>
      <stop offset='100%' stop-color='#e5e7eb'/>
    </linearGradient>
  </defs>
  <rect width='100%' height='100%' fill='url(#g)'/>
  <text x='50%' y='46%' dominant-baseline='middle' text-anchor='middle'
        font-family='Inter,ui-sans-serif,system-ui' font-size='26' fill='#374151'>
    Image not available
  </text>
  <text x='50%' y='62%' dominant-baseline='middle' text-anchor='middle'
        font-family='Inter,ui-sans-serif,system-ui' font-size='22' fill='#111827'>
    Beauty Product
  </text>
</svg>
""".strip()
PLACEHOLDER_DATA_URI = "data:image/svg+xml;utf8," + quote(_PLACEHOLDER_SVG)

def _load_png_placeholder(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return PLACEHOLDER_DATA_URI  # fall back to SVG

# Example: adjust path to wherever you put the PNG in your repo
# PLACEHOLDER_DATA_URI = _load_png_placeholder("ui/assets/beauty_placeholder.png")

def _fmt_price(p: Any) -> str:
    try:
        v = float(p)
        if v != v or v == float("inf") or v == float("-inf"):
            return "-"
        return f"${v:.2f}"
    except Exception:
        return "-"

def _to_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["item_id", "score", "title", "brand", "price", "rank"]
    rows = []
    for r in items or []:
        rows.append([
            r.get("item_id", ""),
            r.get("score", None),
            r.get("title"),
            r.get("brand"),
            r.get("price"),
            r.get("rank"),
        ])
    return pd.DataFrame(rows, columns=cols)

def _render_cards(items: List[Dict[str, Any]]) -> str:
    css = (
        "<style>"
        ".cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}"
        "@media(max-width:1200px){.cards{grid-template-columns:repeat(3,1fr)}}"
        "@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}}"

        ".card{background:var(--card-bg,#ffffff);border-radius:16px;"
        " box-shadow:0 1px 6px rgba(0,0,0,.08);padding:12px;border:1px solid rgba(0,0,0,.06);}"

        ".cap,* .cap{opacity:1 !important;}"
        ".cap{margin-top:10px;font-family:ui-sans-serif,system-ui,-apple-system;"
        " color:var(--card-fg,#111827);} "

        "@media(prefers-color-scheme:dark){"
        " .card{--card-bg:#0f172a; border-color:rgba(255,255,255,.08);}"
        " .cap{--card-fg:#e5e7eb;}"
        " .muted{color:#a1a1aa !important;}"
        "}"

        ".img{height:220px;display:flex;align-items:center;justify-content:center;"
        " background:linear-gradient(180deg,#f4f6f8,#eceff3);border-radius:12px;overflow:hidden}"
        "@media(prefers-color-scheme:dark){.img{background:linear-gradient(180deg,#111827,#0b1220);}}"
        ".img img{max-height:220px;max-width:100%;object-fit:contain;display:block}"

        ".id{font-weight:800;font-size:0.95rem;margin-bottom:2px;letter-spacing:.2px}"
        ".title{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
        "  margin:2px 0 6px 0;font-size:0.95rem}"
        ".muted{color:#4b5563;}"
        "</style>"
    )

    if not items:
        return css + "<div class='muted'>—</div>"

    cards = []
    for r in items[:8]:
        img = (r.get("image_url") or "").strip()
        if img:
            img_html = f"<img src='{img}' alt='product image'/>"
        else:
            img_html = (
                "<div class='img'>"
                "<div style='text-align:center;color:#555;'>"
                "Image not available<br/><span style='font-weight:600;'>Beauty Product</span>"
                "</div></div>"
            )

        title = (r.get("title") or "").strip()
        if len(title) > 90:
            title = title[:90] + "…"
        price = _fmt_price(r.get("price"))
        try:
            score = f"{float(r.get('score') or 0):.4f}"
        except Exception:
            score = "—"

        cards.append(
            "<div class='card'>"
            f"  <div class='img'>{img_html}</div>"
            "  <div class='cap'>"
            f"    <div class='id'>{r.get('item_id','')}</div>"
            f"    <div class='title'>{title}</div>"
            f"    <div>Brand: {r.get('brand') or '-'}</div>"
            f"    <div>Price: {price}</div>"
            f"    <div class='muted'>Score: {score}</div>"
            "  </div>"
            "</div>"
        )

    return css + f"<div class='cards'>{''.join(cards)}</div>"

# --- FAISS resolution helpers -----------------------------------------------
def _fmt_w(v: float) -> str:
    # match your index filenames: 1.0, 0.2, 0.1 -> trim trailing zeros smartly
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return s or "0"

def resolve_faiss_name(dataset: str, fusion: str, wt: float, wi: float, wm: float) -> tuple[Optional[str], bool]:
    """
    Decide which FAISS index to select using the new convention:
      items_<dataset>_concat.faiss        -> faiss_name = "<dataset>_concat"
      items_<dataset>_weighted.faiss      -> faiss_name = "<dataset>_weighted"
    Returns (index_name_or_None, enable_checkbox_bool)
    """
    names = get_faiss_list(dataset)  # list[str] like ["beauty_concat", "beauty_weighted", ...]
    if not names:
        return None, False

    if fusion == "concat":
        target = f"{dataset}_concat"
        if target in names:
            return target, True
        # fallback: any concat for this dataset
        for n in names:
            if n.startswith(f"{dataset}_") and n.endswith("_concat"):
                return n, True
        # final fallback: any concat
        for n in names:
            if n.endswith("_concat"):
                return n, True
        return names[0], True

    # weighted
    target = f"{dataset}_weighted"
    if target in names:
        return target, True
    # fallback: any weighted for this dataset
    for n in names:
        if n.startswith(f"{dataset}_") and n.endswith("_weighted"):
            return n, True
    # final fallback: any weighted
    for n in names:
        if n.endswith("_weighted"):
            return n, True
    return names[0], True

# -------------------------
# HTTP helpers
# -------------------------
def _fetch_json(url: str, method: str = "GET", payload: Optional[Dict] = None, timeout: int = 60) -> Tuple[int, Dict[str, Any]]:
    try:
        if method.upper() == "GET":
            r = requests.get(url, params=(payload or {}), timeout=timeout)
        else:
            r = requests.post(url, json=(payload or {}), timeout=timeout)
        # best-effort JSON
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"detail": r.text}
    except Exception as e:
        return 0, {"detail": f"Request failed: {e}"}
    
# ---- thin API wrappers ----
def ping_api() -> str:
    st, body = _fetch_json(HEALTH_ENDPOINT, "GET")
    if st == 200 and body.get("ok"):
        return f"✅ API ok • {body.get('version','?')} • {API_BASE_URL}"
    if st == 0:
        return f"❌ {body.get('detail')}"
    return f"⚠️ {st}: {body.get('detail','unknown')}"

def get_users(dataset: str) -> Tuple[List[str], str, Dict[str, str]]:
    st, body = _fetch_json(USERS_ENDPOINT, "GET", {"dataset": dataset})
    if st == 200 and isinstance(body, dict):
        users = [str(u) for u in (body.get("users") or [])]
        names = body.get("names") or {}
        # ensure keys/vals are strings
        names = {str(k): ("" if v is None else str(v)) for k, v in names.items()}
        return users, f"Loaded {len(users)} users.", names
    return [], (body.get("detail") or "Failed to load users."), {}

def get_faiss_list(dataset: str) -> List[str]:
    # Optional endpoint; degrade gracefully if not present
    st, body = _fetch_json(FAISS_LIST_ENDPOINT, "GET", {"dataset": dataset})
    if st == 200 and isinstance(body, dict):
        names = body.get("indexes") or body.get("indices") or []
        return [str(x) for x in names]
    return []  # no endpoint or empty

def get_defaults(dataset: str, fusion: str) -> Dict[str, Any]:
    """
    Tries the API first. Supports two shapes:
      1) { "defaults": { ... } }   (recommended)
      2) { ... }                    (legacy)
    Falls back to sensible UI defaults.
    """
    st, body = _fetch_json(DEFAULTS_ENDPOINT, "GET", {"dataset": dataset})
    if st == 200 and isinstance(body, dict):
        d = body.get("defaults")
        if isinstance(d, dict):
            return d
        if body:  # legacy: body is already the defaults dict
            return body

    # sensible fallbacks
    return {
        "w_text": 1.0,
        "w_image": 1.0 if fusion == "concat" else 0.2,
        "w_meta": 0.4 if fusion == "concat" else 0.2,
        "k": 10,
        "faiss_name": None
    }

def build_reco_payload(
    dataset: str,
    user_id: str,
    k: int,
    fusion: str,
    w_text: float, w_image: float, w_meta: float,
    use_faiss: bool,
    faiss_name: Optional[str],
    exclude_seen: bool,
    alpha: Optional[float],
    cove: Optional[str],   # UI hint only; API may ignore until implemented
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "user_id": (user_id or "").strip(),
        "k": int(max(1, min(int(k), 50))),
        "fusion": fusion,
        "w_text": float(w_text),
        "w_image": float(w_image),
        "w_meta": float(w_meta),
        "use_faiss": bool(use_faiss),
        "exclude_seen": bool(exclude_seen),
    }
    if alpha is not None and fusion == "weighted":
        payload["alpha"] = float(alpha)
    if use_faiss and (faiss_name or "").strip():
        payload["faiss_name"] = str(faiss_name).strip()
    if cove:
        payload["cove"] = cove  # harmless if API ignores
    return payload

def call_recommend(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, str, Dict[str, Any]]:
    """
    Returns: (items, status_text, metrics_text, raw_body)
    """
    t0 = time.time()
    st, body = _fetch_json(RECOMMEND_ENDPOINT, "POST", payload)
    latency = f"{(time.time() - t0)*1000.0:.0f} ms"
    if st != 200:
        return [], f"Error {st}: {body.get('detail','unknown')}", "—", body

    # API returns results (and/or recommendations). Prefer 'results'.
    items = body.get("results") or body.get("recommendations") or []
    # metrics is optional; API change recommended to return it
    m = body.get("metrics") or {}
    parts = []
    if "hit@k" in m: parts.append(f"hit@k {m['hit@k']:.3f}" if isinstance(m["hit@k"], (int,float)) else f"hit@k {m['hit@k']}")
    if "ndcg@k" in m: parts.append(f"ndcg@k {m['ndcg@k']:.3f}" if isinstance(m["ndcg@k"], (int,float)) else f"ndcg@k {m['ndcg@k']}")
    if "memory_mb" in m: parts.append(f"mem {int(m['memory_mb'])} MB" if isinstance(m["memory_mb"], (int,float)) else f"mem {m['memory_mb']}")
    metrics_text = " • ".join(parts) if parts else "—"

    return items, f"OK • {len(items)} items • ⏱️ {latency}", metrics_text, body

def call_chat(messages: List[Dict[str, str]], dataset: Optional[str], user_id: Optional[str]) -> Tuple[str, List[Dict[str, Any]]]:
    payload: Dict[str, Any] = {"messages": messages}
    if dataset: payload["dataset"] = dataset
    if user_id: payload["user_id"] = user_id
    st, body = _fetch_json(CHAT_RECOMMEND_ENDPOINT, "POST", payload)
    if st != 200:
        return f"Error {st}: {body.get('detail','unknown')}", []
    items = body.get("results") or body.get("recommendations") or []
    return body.get("reply") or "", items

# -------------------------
# Gradio event handlers
# -------------------------

def on_weights_change(dataset: str, fusion: str, wt: float, wi: float, wm: float):
    # only resolve when weighted; for concat we keep whatever on_dataset_change chose
    if fusion != "weighted":
        # disable FAISS if no concat index was found earlier
        names = get_faiss_list(dataset)
        name = next((n for n in names if "concat" in n), (names[0] if names else None))
        return gr.update(value=name), gr.update(value=bool(name))
    name, enable = resolve_faiss_name(dataset, fusion, wt, wi, wm)
    return gr.update(value=name), gr.update(value=enable)

def _ping_api_initial():
    """
    Called once when the UI loads.
    Tries a few quick pings so the banner doesn't flash 'Error' before the API is ready.
    """
    msg = ping_api()
    if msg.startswith("✅"):
        return gr.update(value=msg)

    # brief retry window (3 sec total) in case API container is still booting
    for _ in range(6):
        time.sleep(0.5)
        msg = ping_api()
        if msg.startswith("✅"):
            break
    return gr.update(value=msg)

def on_check_api():
    return gr.update(value=ping_api())

def on_dataset_change(dataset: str, fusion: str):
    users, msg, names = get_users(dataset)
    USER_NAMES[dataset] = names  # keep names cache in sync for the lower label
    # defaults pulled from API (or sensible fallback)
    d = get_defaults(dataset, fusion)
    wt = float(d.get("w_text", 1.0 if fusion == "concat" else 1.0))
    wi = float(d.get("w_image", 1.0 if fusion == "concat" else 0.2))
    wm = float(d.get("w_meta",  0.4 if fusion == "concat" else 0.2))

    # auto pick faiss using resolver (works even if /defaults returns no name)
    faiss_name, enable_faiss = resolve_faiss_name(dataset, fusion, wt, wi, wm)
    faiss_choices = get_faiss_list(dataset)

    return (
        gr.update(choices=users, value=(users[0] if users else None)),
        gr.update(value=f"👤 {msg}"),
        gr.update(value=wt),
        gr.update(value=wi),
        gr.update(value=wm),
        gr.update(choices=faiss_choices, value=faiss_name),
        gr.update(value=enable_faiss),
    )

def on_user_mode_change(mode: str, dataset: str):
    users, msg, names = get_users(dataset)
    USER_NAMES[dataset] = names  # refresh cache

    if mode == "new":
        return (
            gr.update(choices=users, value=None, interactive=True, label="User ID (optional)"),
            gr.update(value="🆕 New user: start with chat below (or type a custom id)."),
        )
    return (
        gr.update(choices=users, value=(users[0] if users else None), interactive=True, label="User ID"),
        gr.update(value=f"👤 {msg}"),
    )

def on_user_change(dataset: str, user_id: Optional[str]):
    uid = (user_id or "").strip()
    if not uid:
        return gr.update(value="🆕 New user")
    name = (USER_NAMES.get(dataset) or {}).get(uid, "")
    return gr.update(value=(f"👤 {uid} — {name}" if name else f"👤 {uid}"))

def on_toggle_alpha(fusion: str):
    return gr.update(visible=(fusion == "weighted"))

def on_refresh_faiss(dataset: str):
    names = get_faiss_list(dataset)
    return gr.update(choices=names, value=(names[0] if names else None)), gr.update(value=bool(names))

def do_recommend(
    dataset: str, user_mode: str, user_id: str, k: int, fusion: str,
    w_text: float, w_image: float, w_meta: float,
    use_faiss: bool, faiss_name: Optional[str], cove: str,
    exclude_seen: bool, alpha: Optional[float]
):
    uid = (user_id or "").strip()
    if user_mode == "existing" and not uid:
        empty = pd.DataFrame(columns=["item_id","score","title","brand","price","rank"])
        payload = build_reco_payload(dataset, uid, k, fusion, w_text, w_image, w_meta, use_faiss, faiss_name, exclude_seen, alpha, cove)
        return _render_cards([]), empty, "Pick an existing user or switch to 'New user'.", "—", "—", payload

    # For 'new' with blank id, we still allow (your backend may treat as cold-start via chat; here we call anyway)
    payload = build_reco_payload(dataset, uid, k, fusion, w_text, w_image, w_meta, use_faiss, faiss_name, exclude_seen, alpha, cove)
    items, status_text, metrics_text, body = call_recommend(payload)
    cards = _render_cards(items)
    df = _to_df(items)
    return cards, df, status_text, metrics_text, body, payload

def chat_send(history: List[Dict[str, str]], chat_input: str, dataset: str, user_id: str):
    if not (chat_input or "").strip():
        empty = pd.DataFrame(columns=["item_id","score","title","brand","price","rank"])
        return history, "", _render_cards([]), empty
    history = (history or []) + [{"role":"user","content":chat_input}]
    reply, items = call_chat(history, dataset, (user_id or None))
    history = history + [{"role":"assistant","content": (reply or "—")}]
    cards = _render_cards(items)
    df = _to_df(items)
    return history, "", cards, df

def chat_clear():
    empty = pd.DataFrame(columns=["item_id","score","title","brand","price","rank"])
    return [], "", _render_cards([]), empty

# -------------------------
# UI
# -------------------------
with gr.Blocks(title="MMR-Agentic-CoVE • Thin UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## MMR-Agentic-CoVE — Thin Gradio UI (API-driven)")

    with gr.Row():
        with gr.Column(scale=3, min_width=320):
            gr.Markdown("### Controls")

            health_md = gr.Markdown("—")
            check_api_btn = gr.Button("Check API", variant="secondary")

            dataset_dd = gr.Dropdown(label="Dataset", choices=["beauty"], value="beauty")

            user_mode = gr.Radio(
                label="User mode", choices=["existing","new"], value="existing",
                info="Pick 'existing' to select a known user or 'new' to jump into chat."
            )

            user_dd = gr.Dropdown(label="User ID", choices=[], value=None, allow_custom_value=True, interactive=True)
            with gr.Row():
                users_info = gr.Markdown("—")

            with gr.Accordion("Advanced", open=False):
                # FAISS + CoVE at the top
                with gr.Row():
                    use_faiss_ck = gr.Checkbox(value=False, label="Use FAISS")
                    faiss_dd = gr.Dropdown(label="FAISS index name", choices=[], value=None, allow_custom_value=True, scale=3)
                    faiss_refresh_btn = gr.Button("↻ Refresh FAISS", variant="secondary")
                cove_dd = gr.Dropdown(label="CoVE (pre-wired UI)", choices=["off","cove-small","cove-base","cove-large"], value="off")

                fusion_dd = gr.Radio(label="Fusion", choices=["concat","weighted"], value="concat")
                alpha_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Alpha (weighted tilt)", visible=False)

                w_text = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Weight: text")
                w_image = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Weight: image")
                w_meta  = gr.Slider(0.0, 2.0, value=0.4, step=0.1, label="Weight: meta")

                exclude_seen_ck = gr.Checkbox(value=True, label="Exclude seen items")
                k_slider = gr.Slider(1, 50, value=10, step=1, label="Top-K")
                # attach after w_text, w_image, w_meta, faiss_dd, use_faiss_ck are defined
                w_text.change(
                    on_weights_change,
                    [dataset_dd, fusion_dd, w_text, w_image, w_meta],
                    [faiss_dd, use_faiss_ck]
                )
                w_image.change(
                    on_weights_change,
                    [dataset_dd, fusion_dd, w_text, w_image, w_meta],
                    [faiss_dd, use_faiss_ck]
                )
                w_meta.change(
                    on_weights_change,
                    [dataset_dd, fusion_dd, w_text, w_image, w_meta],
                    [faiss_dd, use_faiss_ck]
                )
            recommend_btn = gr.Button("Get Recommendations", variant="primary")

            with gr.Accordion("Debug", open=False):
                payload_json = gr.JSON(label="Payload")
                raw_response = gr.JSON(label="API response (raw)")

            gr.Markdown("---")
            gr.Markdown("### New user (cold start) / Chat")
            chat = gr.Chatbot(label="Ask for products", type="messages")
            chat_msg = gr.Textbox(placeholder="e.g., gentle shampoo under $20", label="Message")
            with gr.Row():
                chat_send_btn = gr.Button("Send", variant="primary")
                chat_clear_btn = gr.Button("Clear")

        with gr.Column(scale=9):
            gr.Markdown("### Results")
            cards_html = gr.HTML(_render_cards([]))

            table = gr.Dataframe(
                headers=["item_id","score","title","brand","price","rank"],
                datatype=["str","number","str","str","number","number"],
                row_count=(0,"dynamic"), col_count=6, interactive=False,
                label="Details"
            )
            with gr.Row():
                status_md = gr.Markdown("—")
                metrics_md = gr.Markdown("—")

    # Wiring
    
    # Fill the health banner on initial page load
    demo.load(fn=_ping_api_initial, inputs=[], outputs=[health_md])
    
    check_api_btn.click(fn=on_check_api, inputs=[], outputs=[health_md])

    dataset_dd.change(
        fn=on_dataset_change,
        inputs=[dataset_dd, fusion_dd],
        outputs=[user_dd, users_info, w_text, w_image, w_meta, faiss_dd, use_faiss_ck]
    )

    fusion_dd.change(fn=on_toggle_alpha, inputs=[fusion_dd], outputs=[alpha_slider])
    # also refresh defaults for weights/faiss when fusion toggles
    fusion_dd.change(
        fn=on_dataset_change,
        inputs=[dataset_dd, fusion_dd],
        outputs=[user_dd, users_info, w_text, w_image, w_meta, faiss_dd, use_faiss_ck]
    )

    user_mode.change(fn=on_user_mode_change, inputs=[user_mode, dataset_dd], outputs=[user_dd, users_info])
    user_dd.change(fn=on_user_change, inputs=[dataset_dd, user_dd], outputs=[users_info])

    faiss_refresh_btn.click(fn=on_refresh_faiss, inputs=[dataset_dd], outputs=[faiss_dd, use_faiss_ck])

    recommend_btn.click(
        fn=do_recommend,
        inputs=[dataset_dd, user_mode, user_dd, k_slider, fusion_dd,
                w_text, w_image, w_meta, use_faiss_ck, faiss_dd, cove_dd,
                exclude_seen_ck, alpha_slider],
        outputs=[cards_html, table, status_md, metrics_md, raw_response, payload_json]
    )

    chat_send_btn.click(fn=chat_send, inputs=[chat, chat_msg, dataset_dd, user_dd], outputs=[chat, chat_msg, cards_html, table])
    chat_clear_btn.click(fn=chat_clear, inputs=[], outputs=[chat, chat_msg, cards_html, table])

    # initial load priming
    def _prime(dataset: str, fusion: str):
        u, msg, wt, wi, wm, faiss, enable = on_dataset_change(dataset, fusion)
        return u, msg, wt, wi, wm, faiss, enable, ping_api()

    demo.load(fn=_prime, inputs=[dataset_dd, fusion_dd], outputs=[user_dd, users_info, w_text, w_image, w_meta, faiss_dd, use_faiss_ck, health_md])

# local launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT","7860")))
    
