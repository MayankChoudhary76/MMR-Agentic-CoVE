import os
import time
import json
from typing import List, Dict, Any, Tuple, Optional

import requests
import gradio as gr

# ------------ Config ------------
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "8001"))
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "false").lower() == "true"

DEFAULT_DATASET = "beauty"
DEFAULT_USER_ID = "A3CIUOJXQ5VDQ2"
DEFAULT_K = 5

# Keep a small in-memory list of datasets (expand later programmatically if you add more)
DATASET_OPTIONS = ["beauty"]  # add more when you’ve processed them

# Hard fallback user ids by dataset (optional; we’ll also allow manual entry)
FALLBACK_USERS = {
    "beauty": [DEFAULT_USER_ID]
}

# ------------ Helpers ------------

def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/healthz", timeout=5)
        return r.ok
    except Exception:
        return False

def call_recommend(
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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    """
    Calls the /recommend API and returns (recs, raw, latency_seconds).
    """
    payload = {
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
    if use_faiss and faiss_name:
        payload["faiss_name"] = faiss_name

    t0 = time.time()
    r = requests.post(f"{API_BASE}/recommend", json=payload, timeout=60)
    latency = time.time() - t0

    r.raise_for_status()
    data = r.json()
    recs = data.get("recommendations", []) or []
    return recs, data, latency

def to_gallery_items(recs: List[Dict[str, Any]]) -> List[List[Any]]:
    """
    Gradio Gallery expects a list of [image (or None), label].
    We’ll put brand + item_id + price in the label. If image is None, Gallery still renders a card.
    """
    items = []
    for r in recs:
        img = r.get("image_url") or None
        brand = r.get("brand") or "-"
        iid = r.get("item_id") or "-"
        score = r.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        price = r.get("price")
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "-"
        cats = r.get("categories") or "-"
        label = f"{brand} · {iid}\nScore: {score_str} · Price: {price_str}\n{cats}"
        items.append([img, label])
    return items

def to_table(recs: List[Dict[str, Any]]) -> List[List[Any]]:
    """
    Build a tabular structure suitable for gr.Dataframe (list of rows).
    """
    rows = []
    for r in recs:
        rows.append([
            r.get("item_id") or "",
            f"{r.get('score'):.4f}" if isinstance(r.get("score"), (int, float)) else "",
            r.get("brand") or "",
            r.get("price") if isinstance(r.get("price"), (int, float)) else "",
            r.get("categories") or "",
            r.get("image_url") or "",
        ])
    return rows

def build_csv(recs: List[Dict[str, Any]]) -> str:
    """
    Return CSV text from recommendations.
    """
    import io, csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["item_id", "score", "brand", "price", "categories", "image_url"])
    for r in recs:
        writer.writerow([
            r.get("item_id") or "",
            f"{r.get('score'):.6f}" if isinstance(r.get("score"), (int, float)) else "",
            r.get("brand") or "",
            r.get("price") if isinstance(r.get("price"), (int, float)) else "",
            r.get("categories") or "",
            r.get("image_url") or "",
        ])
    return buf.getvalue()

# ------------ Cold Start (chat) placeholder agent ------------

def cold_start_agent(history: List[Tuple[str, str]], message: str) -> Tuple[List[Tuple[str, str]], List[List[Any]], List[List[Any]]]:
    """
    Chat-like cold start: for now this is a placeholder that responds conversationally
    and shows how to wire a real cold-start backend later.

    Returns:
      - updated chat history
      - cold-start gallery items (empty for now)
      - cold-start table rows (empty for now)
    """
    # Append user message
    history = history + [(message, None)]

    # Here’s where you would:
    #  - embed `message` with a text model
    #  - vector-search products (FAISS/ScaNN) by text embedding
    #  - or call a new API endpoint /coldstart {query, k, filters...}
    #
    # For now, we return a friendly placeholder assistant message,
    # and no recs so the UI stays consistent.
    assistant_reply = (
        "Thanks! I’ve captured your preferences. "
        "Cold‑start recommendations aren’t wired to the backend *yet* in this demo. "
        "To enable them, expose a `/coldstart` route that accepts a free‑text query, "
        "embeds it, and searches your item vectors. I’ll render the results here in the gallery/table."
    )

    # Replace last tuple’s assistant side
    history[-1] = (history[-1][0], assistant_reply)

    return history, [], []

# ------------ Gradio UI ------------

with gr.Blocks(title="MMR‑Agentic‑CoVE Recommender") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            dataset_dd = gr.Dropdown(choices=DATASET_OPTIONS, value=DEFAULT_DATASET, label="Dataset")
            user_dd = gr.Dropdown(choices=FALLBACK_USERS.get(DEFAULT_DATASET, []),
                                  value=DEFAULT_USER_ID if DEFAULT_USER_ID in FALLBACK_USERS.get(DEFAULT_DATASET, []) else None,
                                  allow_custom_value=True,
                                  label="User ID")
            username_md = gr.Markdown(value="**User name:** _(n/a)_", label=None)

            k_slider = gr.Slider(1, 50, value=DEFAULT_K, step=1, label="Top‑K")
            fusion_dd = gr.Radio(choices=["concat", "weighted"], value="concat", label="Fusion")
            w_text = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Weight: text")
            w_image = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Weight: image")
            w_meta = gr.Slider(0.0, 2.0, value=0.4, step=0.1, label="Weight: meta")
            use_faiss = gr.Checkbox(value=True, label="Use FAISS")
            faiss_name = gr.Textbox(value="beauty_concat_best", label="FAISS index name (when FAISS = true)")
            exclude_seen = gr.Checkbox(value=True, label="Exclude seen items")
            run_btn = gr.Button("Get Recommendations 🚀", variant="primary")

            # Metrics & health
            health_md = gr.Markdown(value="")
            latency_md = gr.Markdown(value="")
            count_md = gr.Markdown(value="")
            # a simple status line
            status_md = gr.Markdown(value="")

        with gr.Column(scale=2):
            gr.Markdown("### Results (Top images first, table below)")
            gallery = gr.Gallery(
                label="Images",
                columns=5,  # display up to 5 across
                preview=True,
                show_download_button=False,
            )

            tbl = gr.Dataframe(
                headers=["item_id", "score", "brand", "price", "categories", "image_url"],
                row_count=(DEFAULT_K if DEFAULT_K > 0 else 5),
                col_count=(6),
                wrap=True,
                interactive=False,
                label="Tabular results"
            )

            # Raw JSON (collapsible)
            raw_json = gr.JSON(label="Raw API JSON (debug)")

    # Cold start chat on its own row (wide)
    gr.Markdown("## Cold Start (New User) — Chat")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                "Describe what you’re looking for (e.g., *“vegan skincare under $20, fragrance‑free”*). "
                "I’ll collect your preferences and (once wired) return relevant products."
            )
            cold_chat = gr.Chatbot(label="Cold‑Start Assistant", type="messages")
            cold_msg = gr.Textbox(label="Your message")
            cold_submit = gr.Button("Send", variant="secondary")
        with gr.Column(scale=1):
            cold_gallery = gr.Gallery(
                label="Cold‑Start Images",
                columns=5,
                preview=True,
                show_download_button=False,
            )
            cold_tbl = gr.Dataframe(
                headers=["item_id", "score", "brand", "price", "categories", "image_url"],
                row_count=5,
                col_count=6,
                wrap=True,
                interactive=False,
                label="Cold‑Start Tabular results"
            )

    # ------------ Events ------------

    def on_dataset_change(ds: str) -> Tuple[List[str], Optional[str]]:
        # Populate user dropdown options when dataset changes (for now we use static fallback)
        users = FALLBACK_USERS.get(ds, [])
        default_val = users[0] if users else None
        return gr.update(choices=users, value=default_val), f"**User name:** _(n/a)_"

    dataset_dd.change(
        on_dataset_change,
        inputs=[dataset_dd],
        outputs=[user_dd, username_md]
    )

    def do_recommend(
        dataset: str, user_id: str, k: int, fusion: str,
        wt: float, wi: float, wm: float,
        faiss_on: bool, faiss_nm: str, ex_seen: bool
    ):
        # Ping API health
        healthy = api_health()
        health_line = "✅ API healthy" if healthy else "⚠️ API not reachable"

        try:
            recs, raw, latency = call_recommend(
                dataset=dataset,
                user_id=user_id,
                k=k,
                fusion=fusion,
                w_text=wt, w_image=wi, w_meta=wm,
                use_faiss=faiss_on,
                faiss_name=(faiss_nm if faiss_on else None),
                exclude_seen=ex_seen,
            )
        except requests.HTTPError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = str(e)
            return (
                [],  # gallery
                [],  # table
                {"error": detail},  # json
                health_line,
                "Latency: –",
                f"Count: –",
                f"❌ Error: {detail}",
            )
        except Exception as e:
            return (
                [],
                [],
                {"error": str(e)},
                health_line,
                "Latency: –",
                f"Count: –",
                f"❌ Error: {e}",
            )

        # Build displays
        gal = to_gallery_items(recs)
        table_rows = to_table(recs)
        latency_line = f"Latency: {latency*1000:.0f} ms"
        count_line = f"Count: {len(recs)}"
        status = "✅ OK"
        return gal, table_rows, raw, health_line, latency_line, count_line, status

    run_btn.click(
        do_recommend,
        inputs=[dataset_dd, user_dd, k_slider, fusion_dd, w_text, w_image, w_meta, use_faiss, faiss_name, exclude_seen],
        outputs=[gallery, tbl, raw_json, health_md, latency_md, count_md, status_md]
    )

    def on_cold_submit(history, message):
        return cold_start_agent(history or [], message or "")

    cold_submit.click(
        on_cold_submit,
        inputs=[cold_chat, cold_msg],
        outputs=[cold_chat, cold_gallery, cold_tbl]
    )
    cold_msg.submit(
        on_cold_submit,
        inputs=[cold_chat, cold_msg],
        outputs=[cold_chat, cold_gallery, cold_tbl]
    )

# ------------ Launch ------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=GRADIO_SHARE)