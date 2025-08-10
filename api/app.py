# api/app.py
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import os
import sys

# Ensure we can import our src package when running `uvicorn api.app:app`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Service import
try:
    from src.service.recommender import recommend_for_user, RecommendConfig, FusionWeights
except Exception as e:
    raise RuntimeError(
        "Could not import src.service.recommender.recommend_for_user. "
        "Make sure src/service/recommender.py exists and exports recommend_for_user(...)."
    ) from e

# ---------------------------
# Request/Response schemas
# ---------------------------
class RecRequest(BaseModel):
    dataset: str = Field(default="beauty", description="Dataset key (e.g., 'beauty').")
    user_id: str = Field(description="User ID as seen in processed reviews.")
    k: int = Field(default=10, ge=1, le=1000)
    fusion: Literal["concat", "weighted"] = "concat"

    # fusion weights
    w_text: float = 1.0
    w_image: float = 1.0
    w_meta: float = 0.0  # set >0 if you built meta embeddings
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Only for 'weighted' fusion.")

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
    alpha: Optional[float] = None
    k: int
    exclude_seen: bool
    use_faiss: bool
    faiss_name: Optional[str] = None
    recommendations: list[RecItem]

# ---------------------------
# FastAPI app (local Swagger)
# ---------------------------
import swagger_ui_bundle  # provides local swagger assets

app = FastAPI(
    title="MMR-Agentic-CoVE Recommender API",
    description="Dataset-aware recommendation endpoint (text/image/meta + FAISS).",
    version="0.1.0",
    docs_url=None,   # disable default CDN docs
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Swagger assets locally so /docs works behind proxies
# app.mount("/static", StaticFiles(directory=swagger_ui_bundle.dist_path), name="static")

@app.get("/docs", include_in_schema=False)
def custom_docs():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="MMR-Agentic-CoVE Recommender API - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecResponse)
def recommend(req: RecRequest):
    # Build service config
    cfg = RecommendConfig(
        dataset=req.dataset,
        user_id=req.user_id,
        k=req.k,
        fusion=req.fusion,
        weights=FusionWeights(text=req.w_text, image=req.w_image, meta=req.w_meta),
        alpha=req.alpha,
        use_faiss=req.use_faiss,
        faiss_name=req.faiss_name,
        exclude_seen=req.exclude_seen,
    )

    result: Dict[str, Any] = recommend_for_user(cfg)

    # Normalize to response model
    items = [
        RecItem(
            item_id=it.get("item_id"),
            score=float(it.get("score", 0.0)),
            brand=it.get("brand"),
            price=it.get("price"),
            categories=it.get("categories"),
            image_url=it.get("image_url"),
        )
        for it in result.get("recommendations", [])
    ]
    return RecResponse(
        dataset=result["dataset"],
        user_id=result["user_id"],
        fusion=result["fusion"],
        weights=result["weights"],
        alpha=result.get("alpha"),
        k=result["k"],
        exclude_seen=result["exclude_seen"],
        use_faiss=result["use_faiss"],
        faiss_name=result.get("faiss_name"),
        recommendations=items,
    )

# Optional: run directly (use 0.0.0.0 so Paperspace proxy can reach it)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)