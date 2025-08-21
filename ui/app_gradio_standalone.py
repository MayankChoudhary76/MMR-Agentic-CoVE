# ui/app_gradio_standalone.py
from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Tuple

import gradio as gr

# Make sure we can import your src package when launched from project root
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the recommender service directly (no HTTP)
from src.service.recommender import recommend_for_user, RecommendConfig, FusionWeights


def get_recommendations(
    dataset: str,
    user_id: str,
    k: int,
    fusion: str,
    w_text: float,
    w_image: float,
    w_meta: float,
    use_faiss: bool,
    faiss_name: str,
    exclude_seen: bool,
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Returns:
      - gallery: list of (image_url, caption) for gr.Gallery
      - raw_json: nicely formatted JSON string (for the code viewer)
    """
    cfg = RecommendConfig(
        dataset=dataset.strip(),
        user_id=user_id.strip(),
        k=int(k),
        fusion=fusion,
        weights=FusionWeights(text=float(w_text), image=float(w_image), meta=float(w_meta)),
        use_faiss=bool(use_faiss),
        faiss_name=(faiss_name.strip() or None) if use_faiss else None,
        exclude_seen=bool(exclude_seen),
    )

    try:
        result: Dict[str, Any] = recommend_for_user(cfg)
    except Exception as e:
        # Show the error message in the JSON pane and an empty gallery
        err = {"error": str(e)}
        return [], json.dumps(err, indent=2)

    # Build a gallery of any items with images
    gallery: List[Tuple[str, str]] = []
    for rec in result.get("recommendations", []):
        img = rec.get("image_url")
        caption = f"{rec.get('item_id','?')} | score={rec.get('score',0):.4f}"
        if img:
            gallery.append((img, caption))

    raw_json = json.dumps(result, indent=2)
    return gallery, raw_json


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Multimodal Recommender (Beauty)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Multimodal Recommender (Beauty) — Standalone UI")

        with gr.Row():
            with gr.Column(scale=1):
                dataset = gr.Textbox(label="Dataset", value="beauty")
                user_id = gr.Textbox(label="User ID", value="A3CIUOJXQ5VDQ2")
                k = gr.Slider(label="Top‑K", minimum=1, maximum=50, step=1, value=10)
                fusion = gr.Radio(label="Fusion", choices=["concat", "weighted"], value="concat")

                w_text = gr.Slider(label="Weight: text", minimum=0.0, maximum=2.0, step=0.1, value=1.0)
                w_image = gr.Slider(label="Weight: image", minimum=0.0, maximum=2.0, step=0.1, value=1.0)
                w_meta = gr.Slider(label="Weight: meta", minimum=0.0, maximum=2.0, step=0.1, value=0.4)

                use_faiss = gr.Checkbox(label="Use FAISS", value=True)
                faiss_name = gr.Textbox(
                    label="FAISS name (used only if Use FAISS = true)",
                    value="beauty_concat_best",
                    placeholder="e.g. beauty_concat_best",
                )
                exclude_seen = gr.Checkbox(label="Exclude seen", value=True)

                btn = gr.Button("Get Recommendations", variant="primary")

            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Recommendations (images if available)",
                    show_label=True,
                    columns=[3],
                    height=400,
                    object_fit="contain",
                )
                raw = gr.Code(label="Raw response (JSON)", language="json")

        btn.click(
            fn=get_recommendations,
            inputs=[dataset, user_id, k, fusion, w_text, w_image, w_meta, use_faiss, faiss_name, exclude_seen],
            outputs=[gallery, raw],
        )

        gr.Markdown(
            "Tip: If you check **Use FAISS**, make sure the FAISS index files exist under "
            "`data/processed/<dataset>/index/` (e.g. `items_beauty_concat_best.faiss`, `*.npy`)."
        )

    return demo


if __name__ == "__main__":
    # You can override these with environment variables when launching
    GRADIO_PORT = int(os.getenv("GRADIO_PORT", "8010"))
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "true").lower() in ("1", "true", "yes")

    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=GRADIO_SHARE)