# api/app_api.py
from __future__ import annotations

import os
import sys
import math
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Make sure we can import from src/*
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -----------------------------------------------------------------------------
# Service import
# -----------------------------------------------------------------------------
try:
    from src.service.recommender import recommend_for_user, RecommendConfig, FusionWeights
except Exception as e:
    raise RuntimeError(
        "Could not import src.service.recommender.recommend_for_user. "
        "Make sure src/service/recommender.py exists and exports recommend_for_user(...)."
    ) from e


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class RecRequest(BaseModel):
    dataset: str = Field(default="beauty", description="Dataset key (e.g., 'beauty').")
    user_id: str = Field(description="User ID as seen in processed reviews.")
    k: int = Field(default=10, ge=1, le=1000)
    fusion: Literal["concat", "weighted"] = "concat"

    # fusion weights
    w_text: float = 1.0
    w_image: float = 1.0
    w_meta: float = 0.0  # set >0 if you built meta embeddings

    # retrieval
    use_faiss: bool = True
    faiss_name: Optional[str] = Field(
        default=None,
        description="Name used when saving the FAISS index (e.g., 'beauty_concat_best')."
    )
    exclude_seen: bool = True


class RecItem(BaseModel):
    item_id: str
    score: float
    brand: Optional[str] = None
    price: Optional[float] = None
    categories: Optional[str] = None
    image_url: Optional[str] = None


class RecResponse(BaseModel):
    dataset: str
    user_id: str
    fusion: Literal["concat", "weighted"]
    weights: Dict[str, float]
    k: int
    exclude_seen: bool
    use_faiss: bool
    faiss_name: Optional[str] = None
    recommendations: List[RecItem]


# -----------------------------------------------------------------------------
# Helpers to make JSON safe (avoid NaN/Inf leaking to JSON)
# -----------------------------------------------------------------------------
def _clean_float(x) -> Optional[float]:
    try:
        f = float(x)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f

def _clean_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x)
    return s if s.lower() != "nan" else None

def _clean_weights(w: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (w or {}).items():
        cf = _clean_float(v)
        if cf is None:
            continue
        out[k] = cf
    return out


# -----------------------------------------------------------------------------
# FastAPI app (works locally and behind Cloudflare Quick Tunnels)
# -----------------------------------------------------------------------------
app = FastAPI(
    title="MMR-Agentic-CoVE Recommender API",
    description="Dataset-aware recommendation endpoint (text/image/meta + FAISS).",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: open so the UI served from a different origin (e.g., trycloudflare.com)
# can call this API without issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for demo; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Global exception handlers → clean 4xx/5xx JSON
# -----------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Optional extra alias some monitors/tools use
@app.get("/health")
def health_alias():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecResponse)
def recommend(req: RecRequest):
    cfg = RecommendConfig(
        dataset=req.dataset,
        user_id=req.user_id,
        k=req.k,
        fusion=req.fusion,
        weights=FusionWeights(text=req.w_text, image=req.w_image, meta=req.w_meta),
        use_faiss=req.use_faiss,
        faiss_name=req.faiss_name,
        exclude_seen=req.exclude_seen,
    )

    try:
        result: Dict[str, Any] = recommend_for_user(cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    items: List[RecItem] = []
    for it in result.get("recommendations", []):
        items.append(
            RecItem(
                item_id=_clean_str(it.get("item_id")) or "",
                score=_clean_float(it.get("score")) or 0.0,
                brand=_clean_str(it.get("brand")),
                price=_clean_float(it.get("price")),
                categories=_clean_str(it.get("categories")),
                image_url=_clean_str(it.get("image_url")),
            )
        )

    return RecResponse(
        dataset=_clean_str(result.get("dataset")) or req.dataset,
        user_id=_clean_str(result.get("user_id")) or req.user_id,
        fusion=result.get("fusion", req.fusion),
        weights=_clean_weights(result.get("weights")),
        k=int(result.get("k", req.k)),
        exclude_seen=bool(result.get("exclude_seen", req.exclude_seen)),
        use_faiss=bool(result.get("use_faiss", req.use_faiss)),
        faiss_name=_clean_str(result.get("faiss_name")),
        recommendations=items,
    )


# -----------------------------------------------------------------------------
# Minimal HTML home for quick manual testing (works behind Cloudflare)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>MMR-Agentic-CoVE | API Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{font-family:system-ui,Arial,sans-serif;margin:20px;max-width:980px}
    label{display:block;margin:8px 0 4px}
    input,select{width:100%;padding:8px;font-size:14px}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    button{margin-top:14px;padding:10px 14px;font-size:14px;cursor:pointer}
    .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;margin-top:16px}
    .card{border:1px solid #ddd;border-radius:10px;padding:10px}
    .score{opacity:.7;font-size:12px}
    img{max-width:100%;border-radius:8px}
    .hint{margin-top:4px;color:#555;font-size:12px}
    pre{white-space:pre-wrap}
  </style>
</head>
<body>
  <h2>Multimodal Recommender (Beauty)</h2>
  <div class="hint">
    Behind a reverse proxy (e.g., Cloudflare Quick Tunnel) this page and
    <code>/recommend</code> share the same origin, so no extra config is needed.
  </div>

  <div class="row">
    <div>
      <label>Dataset</label>
      <input id="dataset" value="beauty"/>
    </div>
    <div>
      <label>User ID</label>
      <input id="user_id" value="A3CIUOJXQ5VDQ2"/>
    </div>
    <div>
      <label>Top-K</label>
      <input id="k" type="number" value="10" min="1" max="50"/>
    </div>
    <div>
      <label>Fusion</label>
      <select id="fusion">
        <option value="concat" selected>concat</option>
        <option value="weighted">weighted</option>
      </select>
    </div>
    <div>
      <label>Weight: text</label>
      <input id="w_text" type="number" step="0.1" value="1.0"/>
    </div>
    <div>
      <label>Weight: image</label>
      <input id="w_image" type="number" step="0.1" value="1.0"/>
    </div>
    <div>
      <label>Weight: meta</label>
      <input id="w_meta" type="number" step="0.1" value="0.4"/>
    </div>
    <div>
      <label>Use FAISS</label>
      <select id="use_faiss">
        <option value="true">true</option>
        <option value="false" selected>false</option>
      </select>
    </div>
    <div>
      <label>FAISS name</label>
      <input id="faiss_name" value="beauty_concat_best"/>
      <div class="hint">Used only when Use FAISS = true</div>
    </div>
    <div>
      <label>Exclude seen</label>
      <select id="exclude_seen">
        <option value="true" selected>true</option>
        <option value="false">false</option>
      </select>
    </div>
  </div>
  <button onclick="run()">Get Recommendations</button>
  <pre id="status"></pre>
  <div id="cards" class="cards"></div>

<script>
async function run(){
  const payload = {
    dataset: document.getElementById('dataset').value,
    user_id: document.getElementById('user_id').value,
    k: Number(document.getElementById('k').value),
    fusion: document.getElementById('fusion').value,
    w_text: Number(document.getElementById('w_text').value),
    w_image: Number(document.getElementById('w_image').value),
    w_meta: Number(document.getElementById('w_meta').value),
    use_faiss: document.getElementById('use_faiss').value === 'true',
    exclude_seen: document.getElementById('exclude_seen').value === 'true'
  };
  if (payload.use_faiss) payload.faiss_name = document.getElementById('faiss_name').value;

  const status = document.getElementById('status');
  const cards = document.getElementById('cards');
  status.textContent = 'Calling /recommend...';
  cards.innerHTML = '';
  try{
    const res = await fetch('/recommend', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const body = await res.json();
    if(!res.ok){
      status.textContent = 'Error: ' + (body.detail || res.status);
      return;
    }
    status.textContent = 'OK';
    (body.recommendations || []).forEach(rec => {
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `
        <div><b>${rec.item_id || ''}</b></div>
        <div class="score">score: ${(rec.score ?? 0).toFixed(4)}</div>
        ${rec.image_url ? `<img src="${rec.image_url}" />` : ''}
        <div>Brand: ${rec.brand ?? '-'}</div>
        <div>Price: ${typeof rec.price === 'number' ? '$' + rec.price.toFixed(2) : '-'}</div>
        <div>Categories: ${rec.categories ?? '-'}</div>
      `;
      cards.appendChild(div);
    });
  }catch(e){
    status.textContent = 'Request failed: ' + e;
  }
}
</script>
</body>
</html>
    """


# -----------------------------------------------------------------------------
# Run directly (use 0.0.0.0 so Cloudflare can reach it)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app_api:app", host="0.0.0.0", port=8000, reload=True)