# src/service/recommender.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import json
import numpy as np
import pandas as pd

from src.utils.paths import get_processed_path
from src.models.fusion import concat_fusion, weighted_sum_fusion

ITEM_KEY = "item_id"


# ---------------------------- dataclasses ---------------------------- #

@dataclass
class FusionWeights:
    text: float = 1.0
    image: float = 0.0
    meta: float = 0.0


@dataclass
class RecommendConfig:
    dataset: str
    user_id: str
    k: int = 10
    fusion: str = "weighted"
    weights: FusionWeights = field(default_factory=FusionWeights)
    use_faiss: bool = False
    faiss_name: Optional[str] = None
    exclude_seen: bool = True
    alpha: Optional[float] = None  # legacy param accepted, ignored


# ---------------------------- IO helpers ---------------------------- #

def _proc_dir(dataset: str) -> Path:
    p = Path(get_processed_path(dataset))
    if not p.exists():
        raise FileNotFoundError(f"Processed dir not found: {p}")
    return p


def _read_parquet(fp: Path, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")
    df = pd.read_parquet(fp)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{fp} missing columns: {missing} | has {list(df.columns)}")
    return df


def _load_defaults(dataset: str) -> Dict[str, Dict[str, Any]]:
    fp = Path(f"data/processed/{dataset}/index/defaults.json")
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text())
    except Exception:
        return {}


def _load_user_vec(proc: Path, user_id: str) -> np.ndarray:
    dfu = _read_parquet(proc / "user_text_emb.parquet", ["user_id", "vector"])
    row = dfu[dfu["user_id"] == user_id]
    if row.empty:
        raise ValueError(
            f"user_id not found in user_text_emb.parquet: {user_id}. "
            f"Run scripts/build_text_emb.py."
        )
    v = np.asarray(row.iloc[0]["vector"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _load_item_dfs(proc: Path):
    Mt = _read_parquet(proc / "item_text_emb.parquet", [ITEM_KEY, "vector"])
    Mi = _read_parquet(proc / "item_image_emb.parquet", [ITEM_KEY, "vector"])
    meta_fp = proc / "item_meta_emb.parquet"
    Mm = _read_parquet(meta_fp, [ITEM_KEY, "vector"]) if meta_fp.exists() else None
    return Mt, Mi, Mm


def _load_items_table(proc: Path) -> pd.DataFrame:
    items = _read_parquet(proc / "items_with_meta.parquet")
    if ITEM_KEY not in items.columns:
        if items.index.name == ITEM_KEY:
            items = items.reset_index()
        else:
            raise KeyError(f"{ITEM_KEY} not found in items_with_meta.parquet")
    return items


def _user_seen_items(proc: Path, user_id: str) -> set:
    df = _read_parquet(proc / "reviews.parquet", ["user_id", ITEM_KEY])
    return set(df[df["user_id"] == user_id][ITEM_KEY].tolist())


# --------------------------- math helpers --------------------------- #

def _l2norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query.reshape(1, -1)
    q = q / (np.linalg.norm(q) + 1e-12)
    M = _l2norm_rows(matrix)
    return (q @ M.T).ravel()


def _align_modalities(Mt: pd.DataFrame, Mi: pd.DataFrame, Mm: Optional[pd.DataFrame]):
    base = Mt[[ITEM_KEY]].merge(Mi[[ITEM_KEY]], on=ITEM_KEY)
    if Mm is not None:
        base = base.merge(Mm[[ITEM_KEY]], on=ITEM_KEY)
    item_ids = base[ITEM_KEY].tolist()

    def reindex(mat_df: pd.DataFrame, ids: List[str]) -> np.ndarray:
        v = mat_df.set_index(ITEM_KEY).loc[ids, "vector"].to_numpy()
        return np.vstack(v).astype(np.float32)

    Vt = reindex(Mt, item_ids)
    Vi = reindex(Mi, item_ids)
    Vm = reindex(Mm, item_ids) if Mm is not None else None
    return item_ids, Vt, Vi, Vm

def _concat_user_vector(user_text_vec: np.ndarray,
                        dim_t: int, dim_i: int, dim_m: int,
                        w_text: float, w_image: float, w_meta: float) -> np.ndarray:
    ut = user_text_vec / (np.linalg.norm(user_text_vec) + 1e-12)
    parts = [w_text * ut]
    if dim_i > 0:
        parts.append(np.zeros((dim_i,), dtype=np.float32))
    if dim_m > 0:
        parts.append(np.zeros((dim_m,), dtype=np.float32))
    uf = np.concatenate(parts, axis=0).astype(np.float32)
    return uf / (np.linalg.norm(uf) + 1e-12)


def _weighted_user_vector(user_text_vec: np.ndarray, target_dim: int, w_text: float) -> np.ndarray:
    ut = (w_text * user_text_vec).astype(np.float32)
    ut = ut / (np.linalg.norm(ut) + 1e-12)
    d = ut.shape[0]
    if d == target_dim:
        uf = ut
    elif d < target_dim:
        pad = np.zeros((target_dim - d,), dtype=np.float32)
        uf = np.concatenate([ut, pad], axis=0)
    else:
        uf = ut[:target_dim]
    return uf / (np.linalg.norm(uf) + 1e-12)


# -------------------------- FAISS integration ----------------------- #

def _faiss_search(proc: Path, name: str, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS not available. Install faiss-cpu or disable use_faiss.") from e

    idx_dir = proc / "index"
    index_fp = idx_dir / f"items_{name}.faiss"
    ids_fp = idx_dir / f"items_{name}.npy"
    if not index_fp.exists() or not ids_fp.exists():
        raise FileNotFoundError(f"FAISS index or ids not found: {index_fp}, {ids_fp}")

    index = faiss.read_index(str(index_fp))
    q = query_vec.astype(np.float32).reshape(1, -1)
    D, I = index.search(q, k)
    ids = np.load(ids_fp, allow_pickle=True)
    return D[0], ids[I[0]]


def _resolve_faiss_name(dataset: str, fusion: str, faiss_name: Optional[str], defaults: Dict[str, Dict[str, Any]]) -> str:
    """
    Order of resolution:
      1) explicit faiss_name if provided
      2) defaults.json → defaults[fusion].faiss_name if present
      3) conventional fallback:
         - concat:   f"{dataset}_concat"
         - weighted: f"{dataset}_weighted_wt{wt}_wi{wi}_wm{wm}" (rounded)
    """
    if faiss_name:
        return faiss_name

    d = (defaults or {}).get(fusion, {})
    if isinstance(d, dict):
        n = d.get("faiss_name") or d.get("index_name")
        if isinstance(n, str) and n:
            return n

    if fusion == "concat":
        return f"{dataset}_concat"

    # weighted fallback uses weights baked into index filename
    wt = d.get("w_text", 1.0)
    wi = d.get("w_image", 0.0)
    wm = d.get("w_meta", 0.0)

    def _fmt(x: float) -> str:
        return f"{x:.1f}".rstrip("0").rstrip(".") if "." in f"{x:.1f}" else f"{x:.1f}"

    return f"{dataset}_weighted_wt{_fmt(wt)}_wi{_fmt(wi)}_wm{_fmt(wm)}"

# ----------------------------- core logic --------------------------- #

def _recommend_concat(proc: Path,
                      dataset: str,
                      user_id: str,
                      k: int,
                      exclude_seen: bool,
                      use_faiss: bool,
                      faiss_name: Optional[str],
                      w_text: float,
                      w_image: float,
                      w_meta: float) -> Tuple[pd.DataFrame, List[str]]:
    items_df = _load_items_table(proc)
    Mt, Mi, Mm = _load_item_dfs(proc)
    user_vec = _load_user_vec(proc, user_id)
    item_ids, Vt, Vi, Vm = _align_modalities(Mt, Mi, Mm)

    # Build fused item matrix and a compatible user vector
    Vf = concat_fusion(Vt, Vi, Vm, weights=(w_text, w_image, w_meta))
    uf = _concat_user_vector(
        user_text_vec=user_vec,
        dim_t=Vt.shape[1],
        dim_i=Vi.shape[1],
        dim_m=0 if Vm is None else Vm.shape[1],
        w_text=w_text, w_image=w_image, w_meta=w_meta
    )

    # Exclusions
    exclude = _user_seen_items(proc, user_id) if exclude_seen else set()

    # Search
    rec_ids: List[str]
    scores: np.ndarray
    if use_faiss:
        # Auto-resolve index name if missing
        defaults = _load_defaults(dataset)
        idx_name = _resolve_faiss_name(dataset, "concat", faiss_name, defaults)
        D, hits = _faiss_search(proc, idx_name, uf, k + 200)
        # Keep in catalog order map to fetch scores from Vf
        id2row = {iid: i for i, iid in enumerate(item_ids)}
        rec_ids = [iid for iid in hits.tolist() if iid not in exclude][:k]
        sel = np.array([id2row[i] for i in rec_ids], dtype=np.int64)
        scores = (uf.reshape(1, -1) @ _l2norm_rows(Vf[sel]).T).ravel()
    else:
        scores_all = (uf.reshape(1, -1) @ _l2norm_rows(Vf).T).ravel()
        mask = np.array([iid not in exclude for iid in item_ids], dtype=bool)
        scores_all = np.where(mask, scores_all, -np.inf)
        topk_idx = np.argpartition(scores_all, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores_all[topk_idx])[::-1]]
        rec_ids = [item_ids[i] for i in topk_idx]
        scores = scores_all[topk_idx]

    out = items_df.merge(
        pd.DataFrame({ITEM_KEY: rec_ids, "score": scores}),
        on=ITEM_KEY, how="right"
    )
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out, rec_ids

def _recommend_weighted(proc: Path,
                        dataset: str,
                        user_id: str,
                        k: int,
                        exclude_seen: bool,
                        use_faiss: bool,
                        faiss_name: Optional[str],
                        w_text: float,
                        w_image: float,
                        w_meta: float) -> Tuple[pd.DataFrame, List[str]]:
    items_df = _load_items_table(proc)
    Mt, Mi, Mm = _load_item_dfs(proc)
    user_vec = _load_user_vec(proc, user_id)
    item_ids, Vt, Vi, Vm = _align_modalities(Mt, Mi, Mm)

    # Fuse items with weighted-sum and create a compatible user vector
    Vf = weighted_sum_fusion(Vt, Vi, Vm, weights=(w_text, w_image, w_meta))
    uf = _weighted_user_vector(user_text_vec=user_vec, target_dim=Vf.shape[1], w_text=w_text)

    # Exclusions
    exclude = _user_seen_items(proc, user_id) if exclude_seen else set()

    # Search
    rec_ids: List[str]
    scores: np.ndarray
    if use_faiss:
        defaults = _load_defaults(dataset)
        idx_name = _resolve_faiss_name(dataset, "weighted", faiss_name, defaults)
        D, hits = _faiss_search(proc, idx_name, uf, k + 200)
        # filter seen, then clip
        filtered = [(float(d), iid) for d, iid in zip(D.tolist(), hits.tolist()) if iid not in exclude]
        filtered = filtered[:k]
        if filtered:
            scores = np.array([d for d, _ in filtered], dtype=np.float32)
            rec_ids = [iid for _, iid in filtered]
        else:
            scores = np.array([], dtype=np.float32)
            rec_ids = []
    else:
        scores_all = (uf.reshape(1, -1) @ _l2norm_rows(Vf).T).ravel()
        mask = np.array([iid not in exclude for iid in item_ids], dtype=bool)
        scores_all = np.where(mask, scores_all, -np.inf)
        topk_idx = np.argpartition(scores_all, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores_all[topk_idx])[::-1]]
        rec_ids = [item_ids[i] for i in topk_idx]
        scores = scores_all[topk_idx]

    out = items_df.merge(
        pd.DataFrame({ITEM_KEY: rec_ids, "score": scores}),
        on=ITEM_KEY, how="right"
    )
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out, rec_ids

# -------------------------- public API ------------------------------ #

def recommend_for_user(cfg: RecommendConfig) -> Dict[str, Any]:
    """
    Entry point used by api/app_api.py. Returns a dict ready for JSON response.
    It also auto-loads defaults.json to fill in weights/Faiss name when omitted.
    """
    proc = _proc_dir(cfg.dataset)
    defaults = _load_defaults(cfg.dataset)

    # Resolve weights: cfg.weights (if set) < defaults.json < fallback
    defw = defaults.get(cfg.fusion, {}) if defaults else {}
    wt = (cfg.weights.text
          if (cfg.weights and cfg.weights.text is not None)
          else defw.get("w_text", 1.0))
    wi = (cfg.weights.image
          if (cfg.weights and cfg.weights.image is not None)
          else defw.get("w_image", 0.0))
    wm = (cfg.weights.meta
          if (cfg.weights and cfg.weights.meta is not None)
          else defw.get("w_meta", 0.0))

    # Route to correct recommender
    if cfg.fusion == "concat":
        out, rec_ids = _recommend_concat(
            proc=proc,
            dataset=cfg.dataset,
            user_id=cfg.user_id,
            k=cfg.k,
            exclude_seen=cfg.exclude_seen,
            use_faiss=cfg.use_faiss,
            faiss_name=cfg.faiss_name,
            w_text=float(wt), w_image=float(wi), w_meta=float(wm),
        )
    elif cfg.fusion == "weighted":
        out, rec_ids = _recommend_weighted(
            proc=proc,
            dataset=cfg.dataset,
            user_id=cfg.user_id,
            k=cfg.k,
            exclude_seen=cfg.exclude_seen,
            use_faiss=cfg.use_faiss,
            faiss_name=cfg.faiss_name,
            w_text=float(wt), w_image=float(wi), w_meta=float(wm),
        )
    else:
        raise ValueError("fusion must be one of {'concat','weighted'}")

    # Ensure purely JSON-serializable payload
    cols = [c for c in [ITEM_KEY, "score", "brand", "price", "categories", "image_url"]
            if c in out.columns]
    if "score" in cols:
        out["score"] = out["score"].astype(float)

    records: List[Dict[str, Any]] = out[cols].head(int(cfg.k)).to_dict(orient="records")

    return {
        "dataset": cfg.dataset,
        "user_id": cfg.user_id,
        "fusion": cfg.fusion,
        "weights": {"text": float(wt), "image": float(wi), "meta": float(wm)},
        "k": int(cfg.k),
        "exclude_seen": bool(cfg.exclude_seen),
        "use_faiss": bool(cfg.use_faiss),
        "faiss_name": cfg.faiss_name,
        "results": records,
    }


__all__ = ["FusionWeights", "RecommendConfig", "recommend_for_user"]