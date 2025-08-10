# src/service/recommender.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import json
import numpy as np
import pandas as pd
from pathlib import Path

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
    # We assume inputs are already l2‑normalized. If not, normalize here.
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


def _read_items_table(proc: Path) -> pd.DataFrame:
    """
    Returns a table with item_id + presentation fields to display alongside scores.
    Expected columns in items_with_meta.parquet: ['item_id','brand','price','categories','image_url']
    """
    fp = proc / "items_with_meta.parquet"
    if not fp.exists():
        # Backoff: try joined.parquet and reduce to unique items
        j = pd.read_parquet(proc / "joined.parquet")
        cols = ["item_id"]
        for c in ["brand", "price", "categories", "image_url"]:
            if c in j.columns:
                cols.append(c)
        it = j[cols].drop_duplicates("item_id").reset_index(drop=True)
        return it
    return pd.read_parquet(fp)


def _seen_items(proc: Path, user_id: str) -> set[str]:
    """Return a set of item_ids the user has interacted with (to optionally exclude)."""
    fp = proc / "reviews.parquet"
    if not fp.exists():
        return set()
    df = pd.read_parquet(fp, columns=["user_id", "item_id"])
    return set(df.loc[df["user_id"].astype(str) == str(user_id), "item_id"].astype(str).tolist())


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
        # For weighted, we simply return a concatenation but keep dims to slice later.
        # Similarity is computed as weighted sum of cosines across slices.
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
    Produce top‑K recommendations for a user with either concat (feature‑level) fusion
    or weighted (score‑level) fusion. Returns a dict with 'recommendations' and metadata.
    """
    proc = get_processed_path(cfg.dataset)

    # --- Load user vector
    _, U, users = _load_user_vectors(proc)
    if cfg.user_id not in users:
        raise ValueError(f"user_id {cfg.user_id} not found in user_text_emb.parquet")
    u_idx = users.index(cfg.user_id)
    u = U[u_idx : u_idx + 1]  # [1 x 384], L2‑normalized already

    # --- Load items & modalities
    items_df = _read_items_table(proc)         # item details
    items_order = _load_item_ids(proc)         # authoritative id order
    id2pos = {iid: i for i, iid in enumerate(items_order)}

    mods = _load_item_modalities(proc)         # dict: text/image/meta -> [I x D], L2‑normalized
    Vf, dims = _fuse_items(mods, cfg.fusion, cfg.weights)

    # --- Build user feature for concat, or keep u for weighted
    if cfg.fusion == "concat":
        parts_u: List[np.ndarray] = []
        wts: List[float] = []
        if "text" in mods:
            parts_u.append(u)
            wts.append(cfg.weights.text)
        if "image" in mods:
            # No user image embedding; we pool user’s seen items’ image vectors (or zeros if none).
            # Simple approach: average image vectors of seen items.
            seen = _seen_items(proc, cfg.user_id)
            if seen and "image" in mods:
                take = [id2pos[s] for s in seen if s in id2pos]
                if take:
                    u_img = np.mean(mods["image"][take, :], axis=0, keepdims=True).astype(np.float32)
                    # Normalize that pooled vector
                    u_img = l2norm(u_img)
                else:
                    u_img = np.zeros((1, mods["image"].shape[1]), dtype=np.float32)
            else:
                u_img = np.zeros((1, mods["image"].shape[1]), dtype=np.float32)
            parts_u.append(u_img)
            wts.append(cfg.weights.image)
        if "meta" in mods:
            # Same idea: user meta preference from seen items' meta vectors
            seen = _seen_items(proc, cfg.user_id)
            if seen and "meta" in mods:
                take = [id2pos[s] for s in seen if s in id2pos]
                if take:
                    u_meta = np.mean(mods["meta"][take, :], axis=0, keepdims=True).astype(np.float32)
                    u_meta = l2norm(u_meta)
                else:
                    u_meta = np.zeros((1, mods["meta"].shape[1]), dtype=np.float32)
            else:
                u_meta = np.zeros((1, mods["meta"].shape[1]), dtype=np.float32)
            parts_u.append(u_meta)
            wts.append(cfg.weights.meta)

        uf = l2norm(_stack_and_weight(parts_u, wts))  # [1 x Df]

        # --- Search
        if cfg.use_faiss:
            if not cfg.faiss_name:
                raise ValueError("--use_faiss requires --faiss_name")
            idx_dir = proc / "index"
            D, top_ids = _faiss_topk(
                idx_dir / f"items_{cfg.faiss_name}.faiss",
                idx_dir / f"items_{cfg.faiss_name}.npy",
                uf, cfg.k + (200 if cfg.exclude_seen else 0)
            )
            scores = D  # already inner products / sims saved by FAISS builder
        else:
            sims = _cosine_scores(uf, Vf)  # [1 x I]
            scores = sims[0]
            # Rank by cosine
            top_idx = np.argsort(-scores)

            # map indices to item ids
            top_ids = [items_order[i] for i in top_idx[: cfg.k + (200 if cfg.exclude_seen else 0)]]

    elif cfg.fusion == "weighted":
        # Score‑level: compute per‑modality cosines and sum with weights
        sims_total = None
        count = 0

        # Always have text modality for user
        if "text" in mods:
            s = _cosine_scores(u, mods["text"])[0] * cfg.weights.text
            sims_total = s if sims_total is None else (sims_total + s)
            count += 1

        # Image score via pooled user image vector
        if "image" in mods:
            seen = _seen_items(proc, cfg.user_id)
            if seen:
                take = [id2pos[s] for s in seen if s in id2pos]
                if take:
                    u_img = np.mean(mods["image"][take, :], axis=0, keepdims=True).astype(np.float32)
                    u_img = l2norm(u_img)
                    s = _cosine_scores(u_img, mods["image"])[0] * cfg.weights.image
                    sims_total = s if sims_total is None else (sims_total + s)
                    count += 1

        # Meta score via pooled user meta vector
        if "meta" in mods:
            seen = _seen_items(proc, cfg.user_id)
            if seen:
                take = [id2pos[s] for s in seen if s in id2pos]
                if take:
                    u_meta = np.mean(mods["meta"][take, :], axis=0, keepdims=True).astype(np.float32)
                    u_meta = l2norm(u_meta)
                    s = _cosine_scores(u_meta, mods["meta"])[0] * cfg.weights.meta
                    sims_total = s if sims_total is None else (sims_total + s)
                    count += 1

        if sims_total is None:
            raise RuntimeError("No similarities could be computed in 'weighted' fusion.")

        scores = sims_total
        top_idx = np.argsort(-scores)
        top_ids = [items_order[i] for i in top_idx[: cfg.k + (200 if cfg.exclude_seen else 0)]]

    else:
        raise ValueError(f"Unknown fusion scheme: {cfg.fusion}")

    # --- Exclude seen if requested
    if cfg.exclude_seen:
        seen = _seen_items(proc, cfg.user_id)
        filtered: List[Tuple[str, float]] = []
        for iid in top_ids:
            if iid not in seen:
                filtered.append((iid, float(scores[id2pos[iid]])))
            if len(filtered) >= cfg.k:
                break
        top_ids, top_scores = zip(*filtered) if filtered else ([], [])
    else:
        top_scores = [float(scores[id2pos[iid]]) for iid in top_ids[: cfg.k]]

    # --- Attach presentation fields
    items_df = items_df.copy()
    items_df["item_id"] = items_df["item_id"].astype(str)
    meta_cols = ["brand", "price", "categories", "image_url"]
    for c in meta_cols:
        if c not in items_df.columns:
            items_df[c] = None

    meta_map = items_df.set_index("item_id")[meta_cols].to_dict(orient="index")

    recs = []
    for iid, sc in zip(top_ids, top_scores):
        row = meta_map.get(str(iid), {})
        recs.append({
            "item_id": str(iid),
            "score": float(sc),
            "brand": row.get("brand"),
            "price": row.get("price"),
            "categories": row.get("categories"),
            "image_url": row.get("image_url"),
        })

    return {
        "dataset": cfg.dataset,
        "user_id": cfg.user_id,
        "fusion": cfg.fusion,
        "weights": {"text": cfg.weights.text, "image": cfg.weights.image, "meta": cfg.weights.meta},
        "k": cfg.k,
        "exclude_seen": cfg.exclude_seen,
        "use_faiss": cfg.use_faiss,
        "faiss_name": cfg.faiss_name,
        "recommendations": recs,
    }