# ui/app_ui.py
import os
import math
import requests
import streamlit as st

st.set_page_config(page_title="MMR-Agentic-CoVE UI", layout="wide")
st.title("Multimodal Recommender (Beauty)")

# -----------------------------
# API base
# -----------------------------
# If you're on Paperspace, launch Streamlit with:
#   API_BASE="/proxy/8000" streamlit run ui/app_ui.py --server.baseUrlPath proxy/8001 ...
# Locally you can omit API_BASE and it will default to http://127.0.0.1:8000
API_BASE = os.environ.get("API_BASE", "").strip() or (
    "http://127.0.0.1:8000" if os.environ.get("LOCAL_API", "").strip() else "/proxy/8000"
)

# Make sure we don't end up with double slashes
def _url(path: str) -> str:
    if API_BASE.startswith("http"):
        return f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    # Relative (proxied) path
    return f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Recommendation Settings")
    dataset = st.text_input("Dataset", value="beauty")
    user_id = st.text_input("User ID", value="A3CIUOJXQ5VDQ2")
    k = st.number_input("Top-K", min_value=1, max_value=50, value=10)
    fusion = st.selectbox("Fusion", ["concat", "weighted"], index=0)
    w_text = st.number_input("Weight: text", value=1.0, step=0.1)
    w_image = st.number_input("Weight: image", value=1.0, step=0.1)
    w_meta = st.number_input("Weight: meta", value=0.4, step=0.1)
    use_faiss = st.checkbox("Use FAISS", value=True)
    faiss_name = st.text_input("FAISS name", value="beauty_concat_best") if use_faiss else ""
    exclude_seen = st.checkbox("Exclude seen", value=True)

    st.caption(f"API base: `{API_BASE}`")
    go = st.button("Get Recommendations")

# -----------------------------
# Health check
# -----------------------------
try:
    health = requests.get(_url("/healthz"), timeout=5)
    if health.ok:
        st.success("API: Healthy")
    else:
        st.warning(f"API health check failed (status {health.status_code})")
except Exception as e:
    st.error(f"API not reachable: {e}")

# -----------------------------
# Helpers
# -----------------------------
def _clean_float(x):
    try:
        f = float(x)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f

def call_api():
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
    if use_faiss:
        payload["faiss_name"] = faiss_name.strip()

    r = requests.post(_url("/recommend"), json=payload, timeout=60)
    r.raise_for_status()
    return r.json(), payload

# -----------------------------
# Run
# -----------------------------
if go:
    try:
        data, payload = call_api()
        with st.expander("Debug (request/response)", expanded=False):
            st.write("Payload:", payload)
            st.write("Response keys:", list(data.keys()))

        st.subheader("Top Recommendations")
        recs = data.get("recommendations", [])
        if not recs:
            st.info("No recommendations returned.")
        else:
            cols = st.columns(3)
            for i, rec in enumerate(recs):
                c = cols[i % 3]
                with c:
                    st.markdown(f"**{rec.get('item_id','?')}**")
                    score = _clean_float(rec.get("score"))
                    if score is not None:
                        st.caption(f"Score: {score:.4f}")
                    brand = rec.get("brand") or "-"
                    price = rec.get("price")
                    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "-"
                    cats = rec.get("categories") or "-"
                    img = rec.get("image_url")
                    if img:
                        st.image(img, use_container_width=True)
                    st.text(f"Brand: {brand}")
                    st.text(f"Price: {price_str}")
                    st.text(f"Categories: {cats}")

    except requests.HTTPError as e:
        try:
            st.error(f"API error: {e.response.status_code} - {e.response.json()}")
        except Exception:
            st.error(f"API error: {e}")
    except Exception as e:
        st.error(f"Request failed: {e}")