# src/service/recommender.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import json
import numpy as np
import pandas as pd
from pathlib import Path
from pathlib import Path
import pandas as pd
import ast
import math

from src.utils.paths import get_processed_path
from src.models.fusion import l2norm  # keep using the shared l2norm

# Optional FAISS import (we’ll guard usage)
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False


# -----------------------------
# Dataclasses & configuration
# -----------------------------

@dataclass
class FusionWeights:
    """Per‑modality weights. Used by both concat (pre-feature scale) and weighted (score-level)."""
    text: float = 1.0
    image: float = 1.0
    meta: float = 0.0


@dataclass
class RecommendConfig:
    dataset: str
    user_id: str
    k: int = 10
    fusion: str = "concat"          # "concat" or "weighted"
    weights: FusionWeights = field(default_factory=FusionWeights)
    # alpha (optional): for "weighted", tilt text vs image.
    alpha: Optional[float] = None
    use_faiss: bool = False
    faiss_name: Optional[str] = None
    exclude_seen: bool = True


__all__ = ["FusionWeights", "RecommendConfig", "recommend_for_user"]


# -----------------------------
# Small utilities
# -----------------------------

def _load_vectors(fp: Path, id_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load a parquet with columns [id_col, vector].
    Returns (df, matrix [N x D], ids list in same order).
    """
    df = pd.read_parquet(fp)
    if "vector" not in df.columns:
        raise ValueError(f"{fp} does not contain a 'vector' column.")
    mat = np.vstack(df["vector"].to_numpy()).astype(np.float32)
    ids = df[id_col].astype(str).tolist()
    return df, mat, ids


def _load_user_vectors(proc: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    fp = proc / "user_text_emb.parquet"
    df = pd.read_parquet(fp)
    if "vector" not in df.columns or "user_id" not in df.columns:
        raise ValueError(f"{fp} must have columns ['user_id','vector'].")
    U = np.vstack(df["vector"].to_numpy()).astype(np.float32)
    users = df["user_id"].astype(str).tolist()
    return df, U, users


def _cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity matrix between two L2‑normalized matrices.
    a: [N x D], b: [M x D]  => returns [N x M]
    """
    return a @ b.T


def _stack_and_weight(mats: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Horizontally stack matrices after optional per‑matrix scaling.
    All matrices must share the same number of rows.
    """
    if not mats:
        return np.zeros((0, 0), dtype=np.float32)
    if weights is not None:
        assert len(weights) == len(mats), "weights length must match number of matrices"
        mats = [w * m for w, m in zip(weights, mats)]
    return np.hstack(mats).astype(np.float32)

def _normalize_categories(cats):
    """Return categories as a clean Python list."""
    if isinstance(cats, list):
        return cats
    if isinstance(cats, str):
        try:
            v = ast.literal_eval(cats)  # handles "['Beauty']" → ['Beauty']
            if isinstance(v, list):
                return v
        except Exception:
            pass
        return [cats]
    return [] if cats is None else [str(cats)]

def _read_items_table(proc: Path) -> pd.DataFrame:
    """
    Returns a table with item_id + presentation fields to display alongside scores.
    Prefer items_catalog.parquet (has title/rank already parsed), then fall back.
    """
    # 1) Preferred enriched catalog
    fp0 = proc / "items_catalog.parquet"
    if fp0.exists():
        df = pd.read_parquet(fp0)
    else:
        # 2) Fallback to items_with_meta.parquet
        fp1 = proc / "items_with_meta.parquet"
        if fp1.exists():
            df = pd.read_parquet(fp1)
        else:
            # 3) Last resort: joined.parquet (dedup by item_id)
            fp2 = proc / "joined.parquet"
            if not fp2.exists():
                return pd.DataFrame(columns=["item_id","brand","price","categories","image_url","title","rank"])
            j = pd.read_parquet(fp2)
            cols = [c for c in ["item_id","brand","price","categories","image_url","title","rank"] if c in j.columns]
            df = j[cols].dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"])

    # Ensure expected columns exist
    for c in ["brand","price","categories","image_url","title","rank","rank_num","rank_cat"]:
        if c not in df.columns:
            df[c] = None

    # Normalize categories (fix stringified lists)
    
    def _fix_cat(v):
        # null-ish → []
        try:
            import pandas as pd, numpy as np
            if v is None or (isinstance(v, float) and math.isnan(v)):  # if math is not imported, use pd.isna
                return []
            if 'pd' in locals() and pd.isna(v):
                return []
        except Exception:
            pass

        # already list-like
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if x is not None and str(x).strip()]

        # numpy array
        try:
            import numpy as np
            if isinstance(v, np.ndarray):
                return [str(x).strip() for x in v.tolist() if x is not None and str(x).strip()]
        except Exception:
            pass

        # string cases
        if isinstance(v, str):
            s = v.strip()
            if not s or s == "[]":
                return []
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    import ast
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
                except Exception:
                    # fall through to single-label
                    pass
            return [s]

        # fallback
        return []

    
    df["categories"] = df["categories"].map(_fix_cat)

    df["item_id"] = df["item_id"].astype(str)
    return df[["item_id","brand","price","categories","image_url","title","rank","rank_num","rank_cat"]]

def _seen_items(proc: Path, user_id: str) -> set[str]:
    """Return a set of item_ids the user has interacted with (to optionally exclude)."""
    fp = proc / "reviews.parquet"
    if not fp.exists():
        return set()
    df = pd.read_parquet(fp, columns=["user_id", "item_id"])
    return set(df.loc[df["user_id"].astype(str) == str(user_id), "item_id"].astype(str).tolist())


def _pool_user_from_seen(
    mods: Dict[str, np.ndarray],
    id2pos: Dict[str, int],
    proc: Path,
    user_id: str,
    key: str
) -> Optional[np.ndarray]:
    """
    Average‑pool the user's seen items in a given modality (e.g., 'image' or 'meta').
    Returns an L2‑normalized [1 x D] vector, zeros if no valid items, or None if modality missing.
    """
    if key not in mods:
        return None
    seen = _seen_items(proc, user_id)
    if not seen:
        return np.zeros((1, mods[key].shape[1]), dtype=np.float32)
    take = [id2pos[s] for s in seen if s in id2pos]
    if not take:
        return np.zeros((1, mods[key].shape[1]), dtype=np.float32)
    up = np.mean(mods[key][take, :], axis=0, keepdims=True).astype(np.float32)
    return l2norm(up)


# -----------------------------
# Fusion building
# -----------------------------

def _load_item_modalities(proc: Path) -> Dict[str, np.ndarray]:
    """
    Load available item modality matrices keyed by name: 'text', 'image', 'meta'.
    Missing files simply skip the modality.
    """
    out: Dict[str, np.ndarray] = {}

    # Text
    fpt = proc / "item_text_emb.parquet"
    if fpt.exists():
        _, Mt, _ = _load_vectors(fpt, "item_id")
        out["text"] = l2norm(Mt)
    # Image
    fpi = proc / "item_image_emb.parquet"
    if fpi.exists():
        _, Mi, _ = _load_vectors(fpi, "item_id")
        out["image"] = l2norm(Mi)
    # Meta
    fpm = proc / "item_meta_emb.parquet"
    if fpm.exists():
        _, Mm, _ = _load_vectors(fpm, "item_id")
        out["meta"] = l2norm(Mm)

    if not out:
        raise RuntimeError("No item modality vectors found under processed directory.")
    return out


def _load_item_ids(proc: Path) -> List[str]:
    """
    Authoritative item id order from any available modality file.
    We pick text first, then image, then meta.
    """
    for name in ["item_text_emb.parquet", "item_image_emb.parquet", "item_meta_emb.parquet"]:
        fp = proc / name
        if fp.exists():
            _, _, ids = _load_vectors(fp, "item_id")
            return ids
    raise RuntimeError("No item embeddings found to derive item_id order.")


def _fuse_items(
    modalities: Dict[str, np.ndarray],
    scheme: str,
    weights: FusionWeights
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build a fused item matrix according to scheme.
      - "concat": feature‑level concat (pre‑normalize, weight per modality as feature scaling).
      - "weighted": keeps each item modality separate; use score‑level weighted sum later.
    Returns:
      - fused matrix (for "weighted", we return a stacked matrix but also return column slices)
      - dict with dims for reference
    """
    dims = {}
    if scheme == "concat":
        mats: List[np.ndarray] = []
        wts: List[float] = []
        if "text" in modalities:
            mats.append(modalities["text"])
            wts.append(weights.text)
            dims["Dt_text"] = modalities["text"].shape[1]
        if "image" in modalities:
            mats.append(modalities["image"])
            wts.append(weights.image)
            dims["Di_image"] = modalities["image"].shape[1]
        if "meta" in modalities:
            mats.append(modalities["meta"])
            wts.append(weights.meta)
            dims["Dm_meta"] = modalities["meta"].shape[1]

        # scale then concat → L2 normalize
        fused = l2norm(_stack_and_weight(mats, wts))
        dims["Df_fused"] = fused.shape[1]
        return fused, dims

    elif scheme == "weighted":
        parts: List[np.ndarray] = []
        if "text" in modalities:
            parts.append(modalities["text"])
            dims["Dt_text"] = modalities["text"].shape[1]
        if "image" in modalities:
            parts.append(modalities["image"])
            dims["Di_image"] = modalities["image"].shape[1]
        if "meta" in modalities:
            parts.append(modalities["meta"])
            dims["Dm_meta"] = modalities["meta"].shape[1]

        fused = _stack_and_weight(parts, None)  # no feature scaling here; weights applied at score time
        dims["Df_fused"] = fused.shape[1]
        return fused, dims

    else:
        raise ValueError(f"Unknown fusion scheme: {scheme}")


def _faiss_topk(index_fp: Path, ids_fp: Path, queries: np.ndarray, k: int) -> Tuple[np.ndarray, List[str]]:
    if not _FAISS_OK:
        raise RuntimeError("FAISS not available. Install faiss-cpu or disable use_faiss.")
    index = faiss.read_index(str(index_fp))
    D, I = index.search(queries.astype(np.float32), k)
    item_ids = np.load(ids_fp, allow_pickle=True).tolist()
    id_list = [str(x) for x in item_ids]
    flat_ids: List[str] = [id_list[i] for i in I[0]]
    return D[0], flat_ids


# -----------------------------
# Public API
# -----------------------------

def recommend_for_user(cfg: RecommendConfig) -> Dict:
    """
    Produce top-K recommendations for a user with either concat (feature-level) fusion
    or weighted (score-level) fusion. Returns a dict with 'recommendations' and metadata.
    """
    proc = get_processed_path(cfg.dataset)

    # --- Load user vector
    _, U, users = _load_user_vectors(proc)
    if cfg.user_id not in users:
        raise ValueError(f"user_id {cfg.user_id} not found in user_text_emb.parquet")
    u_idx = users.index(cfg.user_id)
    u = U[u_idx : u_idx + 1]  # [1 x D], expected L2-normalized already

    # --- Load items & modalities
    items_df = _read_items_table(proc)         # prefer items_with_meta / joined, normalized
    items_order = _load_item_ids(proc)         # authoritative id order
    id2pos = {iid: i for i, iid in enumerate(items_order)}

    mods = _load_item_modalities(proc)         # dict: text/image/meta -> [I x D], L2-normalized
    Vf, dims = _fuse_items(mods, cfg.fusion, cfg.weights)

    # --- Build user feature for concat, or compute score-level for weighted
    score_map: Dict[str, float] = {}
    candidate_ids: List[str] = []

    if cfg.fusion == "concat":
        parts_u: List[np.ndarray] = []
        wts: List[float] = []
        if "text" in mods:
            parts_u.append(u)
            wts.append(cfg.weights.text)
        if "image" in mods:
            u_img = _pool_user_from_seen(mods, id2pos, proc, cfg.user_id, "image")
            if u_img is None:
                u_img = np.zeros((1, mods["image"].shape[1]), dtype=np.float32)
            parts_u.append(u_img)
            wts.append(cfg.weights.image)
        if "meta" in mods:
            u_meta = _pool_user_from_seen(mods, id2pos, proc, cfg.user_id, "meta")
            if u_meta is None:
                u_meta = np.zeros((1, mods["meta"].shape[1]), dtype=np.float32)
            parts_u.append(u_meta)
            wts.append(cfg.weights.meta)

        uf = l2norm(_stack_and_weight(parts_u, wts))  # [1 x Df]

        # --- Search
        extra = 200 if cfg.exclude_seen else 0
        take_k = cfg.k + extra

        if cfg.use_faiss:
            if not cfg.faiss_name:
                raise ValueError("--use_faiss requires --faiss_name")
            idx_dir = proc / "index"
            D, faiss_ids = _faiss_topk(
                idx_dir / f"items_{cfg.faiss_name}.faiss",
                idx_dir / f"items_{cfg.faiss_name}.npy",
                uf, take_k
            )
            # FAISS scores are aligned with faiss_ids
            candidate_ids = faiss_ids
            score_map = {iid: float(s) for iid, s in zip(faiss_ids, D.tolist())}
        else:
            sims = _cosine_scores(uf, Vf)[0]  # [I]
            order = np.argsort(-sims)[:take_k]
            candidate_ids = [items_order[i] for i in order]
            # map from all items (or at least the candidates) to score
            score_map = {items_order[i]: float(sims[i]) for i in order}

    elif cfg.fusion == "weighted":
        # ----- effective weights with alpha (if provided)
        wt = cfg.weights.text
        wi = cfg.weights.image
        wm = cfg.weights.meta
        if cfg.alpha is not None:
            a = float(cfg.alpha)
            a = max(0.0, min(1.0, a))  # clamp
            wt = wt * a
            wi = wi * (1.0 - a)

        # ----- score-level similarities
        sims_total = None

        if "text" in mods and wt != 0:
            s = _cosine_scores(u, mods["text"])[0] * wt
            sims_total = s if sims_total is None else (sims_total + s)

        if "image" in mods and wi != 0:
            u_img = _pool_user_from_seen(mods, id2pos, proc, cfg.user_id, "image")
            if u_img is not None:
                s = _cosine_scores(u_img, mods["image"])[0] * wi
                sims_total = s if sims_total is None else (sims_total + s)

        if "meta" in mods and wm != 0:
            u_meta = _pool_user_from_seen(mods, id2pos, proc, cfg.user_id, "meta")
            if u_meta is not None:
                s = _cosine_scores(u_meta, mods["meta"])[0] * wm
                sims_total = s if sims_total is None else (sims_total + s)

        if sims_total is None:
            raise RuntimeError("No similarities could be computed in 'weighted' fusion.")

        extra = 200 if cfg.exclude_seen else 0
        order = np.argsort(-sims_total)[: cfg.k + extra]
        candidate_ids = [items_order[i] for i in order]
        score_map = {items_order[i]: float(sims_total[i]) for i in order}

        # NOTE: FAISS index is for concatenated vectors. For 'weighted' we use dense scoring only.

    else:
        raise ValueError(f"Unknown fusion scheme: {cfg.fusion}")

    # --- Exclude seen if requested, build final lists
    if cfg.exclude_seen:
        seen = _seen_items(proc, cfg.user_id)
        top_pairs: List[Tuple[str, float]] = []
        for iid in candidate_ids:
            if iid not in seen:
                sc = score_map.get(iid)
                if sc is not None:
                    top_pairs.append((iid, sc))
                if len(top_pairs) >= cfg.k:
                    break
        top_ids, top_scores = zip(*top_pairs) if top_pairs else ([], [])
    else:
        top_ids = candidate_ids[: cfg.k]
        top_scores = [score_map[iid] for iid in top_ids]

    # --- Attach presentation fields
    items_df = items_df.copy()
    items_df["item_id"] = items_df["item_id"].astype(str)
    meta_cols = ["brand", "price", "categories", "image_url", "title", "rank", "rank_num"]
    for c in meta_cols:
        if c not in items_df.columns:
            items_df[c] = None
    meta_map = items_df.set_index("item_id")[meta_cols].to_dict(orient="index")

    recs = []
    for iid, sc in zip(top_ids, top_scores):
        row = meta_map.get(str(iid), {}) or {}

        # normalize categories to a list
        cats = row.get("categories")
        if isinstance(cats, str):
            try:
                parsed = ast.literal_eval(cats)
                cats = list(parsed) if isinstance(parsed, (list, tuple)) else []
            except Exception:
                cats = []
        elif isinstance(cats, (list, tuple)):
            cats = list(cats)
        else:
            cats = []

        # parse rank to int if only string available
        rnum = row.get("rank_num")
        if rnum is None:
            r = row.get("rank")
            if isinstance(r, str):
                m = re.search(r"[\d,]+", r)
                rnum = int(m.group(0).replace(",", "")) if m else None
            elif isinstance(r, (int, float)):
                rnum = int(r)

        recs.append({
            "item_id": str(iid),
            "score": float(sc),
            "brand": row.get("brand"),
            "price": row.get("price"),
            "categories": cats,
            "image_url": row.get("image_url"),
            "title": row.get("title"),
            "rank": rnum,
        })

    return {
        "dataset": cfg.dataset,
        "user_id": cfg.user_id,
        "fusion": cfg.fusion,
        "weights": {"text": cfg.weights.text, "image": cfg.weights.image, "meta": cfg.weights.meta},
        "alpha": cfg.alpha,
        "k": cfg.k,
        "exclude_seen": cfg.exclude_seen,
        "use_faiss": cfg.use_faiss,
        "faiss_name": cfg.faiss_name,
        "recommendations": recs,
    }