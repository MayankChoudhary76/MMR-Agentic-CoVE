#!/usr/bin/env python3
"""
Recommend items for a given user from precomputed embeddings.

Supports:
- Vector-level CONCAT fusion  (optionally using a FAISS index)
- Vector-level WEIGHTED fusion (optionally using a FAISS index)
- Excluding items the user has already interacted with
- Pretty printing of top-K with metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from src.utils.paths import get_processed_path
from src.models.fusion import concat_fusion, weighted_sum_fusion

ITEM_KEY = "item_id"


# ---------------------------- IO helpers ---------------------------- #
def _load_proc_paths(dataset: str) -> Path:
    proc = Path(get_processed_path(dataset))
    if not proc.exists():
        raise FileNotFoundError(f"Processed dir not found: {proc}. Run data prep first.")
    return proc


def _read_parquet(fp: Path, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")
    df = pd.read_parquet(fp)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{fp} missing columns: {missing} | has {list(df.columns)}")
    return df


def _load_user_vector(proc: Path, user_id: str) -> np.ndarray:
    dfu = _read_parquet(proc / "user_text_emb.parquet", ["user_id", "vector"])
    row = dfu[dfu["user_id"] == user_id]
    if row.empty:
        raise ValueError(
            f"user_id not found: {user_id} in user_text_emb.parquet.\n"
            f"Build text embeddings first (scripts/build_text_emb.py)."
        )
    vec = np.asarray(row.iloc[0]["vector"], dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec


def _load_item_dfs(proc: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
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

# --------------------------- Math helpers --------------------------- #

def _stack_vectors(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    ids = df[ITEM_KEY].tolist()
    mat = np.vstack(df["vector"].to_numpy()).astype(np.float32, copy=False)
    return mat, ids


def _l2norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query.reshape(1, -1)
    q = q / (np.linalg.norm(q) + 1e-12)
    M = _l2norm_rows(matrix)
    return (q @ M.T).ravel()


# -------------------------- FAISS integration ----------------------- #

def _faiss_search(proc: Path, name: str, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search a FAISS index previously built with scripts/build_faiss.py.
    Index/dictionary are expected at:
      data/processed/<dataset>/index/items_<name>.faiss and .npy
    Returns: (scores_or_distances, item_ids)
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS not available. Install faiss-cpu or disable --use_faiss.") from e

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

# -------------------------- Alignment helpers ----------------------- #

def _align_modalities(Mt: pd.DataFrame, Mi: pd.DataFrame, Mm: Optional[pd.DataFrame]):
    """
    Inner-join item ids across modalities to ensure aligned rows for fusion.
    Returns: item_ids, Vt, Vi, Vm (Vm may be None)
    """
    base = Mt[[ITEM_KEY]].copy().merge(Mi[[ITEM_KEY]], on=ITEM_KEY)
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


def _concat_user_vector(
    user_text_vec: np.ndarray,
    dim_text: int,
    dim_image: int,
    dim_meta: int,
    w_text: float,
    w_image: float,
    w_meta: float,
) -> np.ndarray:
    """
    Build a user query vector in the same fused space as concat_fusion(item_mats).
    We only have a user TEXT vector; set image/meta parts to zeros but keep weights
    (weights matter because item vectors are weighted before concatenation).
    """
    ut = user_text_vec / (np.linalg.norm(user_text_vec) + 1e-12)

    parts = [w_text * ut]
    if dim_image > 0:
        parts.append(np.zeros((dim_image,), dtype=np.float32))  # no user image embedding
    if dim_meta > 0:
        parts.append(np.zeros((dim_meta,), dtype=np.float32))   # no user meta embedding

    uf = np.concatenate(parts, axis=0).astype(np.float32)
    return uf / (np.linalg.norm(uf) + 1e-12)


def _weighted_user_vector(user_text_vec: np.ndarray, target_dim: int, w_text: float) -> np.ndarray:
    """
    User vector for weighted-sum fused item space.
    We scale text part by w_text and pad/truncate to target_dim (image/meta parts are zeros).
    """
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

# ----------------------------- Recommenders ------------------------- #

def _recommend_concat(
    proc: Path,
    user_id: str,
    k: int,
    exclude_seen: bool,
    use_faiss: bool,
    faiss_name: Optional[str],
    w_text: float,
    w_image: float,
    w_meta: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Vector-level CONCAT fusion:
      Vf = concat_fusion(Vt, Vi, Vm; weights)
      uf = [w_t * u_text, zeros_img, zeros_meta]  (L2-normed)
      score = cosine(uf, Vf)
    If use_faiss, search FAISS index built on the *same* concat fusion and then
    re-score the returned candidates (to preserve ordering and show scores).
    """
    items_df = _load_items_table(proc)
    Mt, Mi, Mm = _load_item_dfs(proc)
    user_vec = _load_user_vector(proc, user_id)

    item_ids, Vt, Vi, Vm = _align_modalities(Mt, Mi, Mm)
    Vf = concat_fusion(Vt, Vi, Vm, weights=(w_text, w_image, w_meta))  # [I, Df]
    uf = _concat_user_vector(
        user_text_vec=user_vec,
        dim_text=Vt.shape[1],
        dim_image=Vi.shape[1],
        dim_meta=0 if Vm is None else Vm.shape[1],
        w_text=w_text, w_image=w_image, w_meta=w_meta,
    )

    exclude = _user_seen_items(proc, user_id) if exclude_seen else set()

    if use_faiss:
        if not faiss_name:
            raise ValueError("--use_faiss requires --faiss_name for concat mode")
        D, hits = _faiss_search(proc, faiss_name, uf, k + 200)
        # filter seen, keep order
        rec_ids = [iid for iid in hits.tolist() if iid not in exclude][:k]
        # re-score selected ids with cosine to display scores
        id2row = {iid: i for i, iid in enumerate(item_ids)}
        sel = np.array([id2row[i] for i in rec_ids], dtype=np.int64)
        scores = (uf.reshape(1, -1) @ _l2norm_rows(Vf[sel]).T).ravel()
    else:
        scores = (uf.reshape(1, -1) @ _l2norm_rows(Vf).T).ravel()
        if exclude:
            mask = np.array([iid not in exclude for iid in item_ids], dtype=bool)
            scores = np.where(mask, scores, -np.inf)
        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        rec_ids = [item_ids[i] for i in topk_idx]
        scores = scores[topk_idx]

    out = items_df.merge(
        pd.DataFrame({ITEM_KEY: rec_ids, "score": scores}),
        on=ITEM_KEY, how="right"
    ).sort_values("score", ascending=False).reset_index(drop=True)
    return out, rec_ids


def _recommend_weighted(
    proc: Path,
    user_id: str,
    k: int,
    exclude_seen: bool,
    use_faiss: bool,
    faiss_name: Optional[str],
    w_text: float,
    w_image: float,
    w_meta: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Vector-level WEIGHTED fusion:
      Vf = weighted_sum_fusion(Vt, Vi, Vm; weights)
      uf = padded/truncated user text vector scaled by w_text to match dim(Vf)
      score = cosine(uf, Vf)

    If use_faiss, search FAISS index built on the *same* weighted fusion and use
    returned inner-products (cosine on L2-normed) as scores (already sorted).
    """
    items_df = _load_items_table(proc)
    Mt, Mi, Mm = _load_item_dfs(proc)
    user_vec = _load_user_vector(proc, user_id)

    item_ids, Vt, Vi, Vm = _align_modalities(Mt, Mi, Mm)
    Vf = weighted_sum_fusion(Vt, Vi, Vm, weights=(w_text, w_image, w_meta))  # [I, D*]
    uf = _weighted_user_vector(user_vec, Vf.shape[1], w_text)

    exclude = _user_seen_items(proc, user_id) if exclude_seen else set()

    if use_faiss:
        if not faiss_name:
            raise ValueError("--use_faiss requires --faiss_name for weighted mode")
        D, hits = _faiss_search(proc, faiss_name, uf, k + 200)
        # filter seen while preserving order; trim to k
        filt = [(float(d), str(i)) for d, i in zip(D.tolist(), hits.tolist()) if i not in exclude]
        filt = filt[:k]
        if filt:
            scores, rec_ids = (np.array([d for d, _ in filt], dtype=np.float32),
                               [i for _, i in filt])
        else:
            scores, rec_ids = np.array([], dtype=np.float32), []
    else:
        scores = (uf.reshape(1, -1) @ _l2norm_rows(Vf).T).ravel()
        if exclude:
            mask = np.array([iid not in exclude for iid in item_ids], dtype=bool)
            scores = np.where(mask, scores, -np.inf)
        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        rec_ids = [item_ids[i] for i in topk_idx]
        scores = scores[topk_idx]

    out = items_df.merge(
        pd.DataFrame({ITEM_KEY: rec_ids, "score": scores}),
        on=ITEM_KEY, how="right"
    ).sort_values("score", ascending=False).reset_index(drop=True)
    return out, rec_ids

# ------------------------------ CLI -------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Recommend items for a user.")
    p.add_argument("--dataset", required=True, help="Dataset key (e.g., beauty)")
    p.add_argument("--user_id", required=True, help="User ID to recommend for")
    p.add_argument("--k", type=int, default=10, help="Top-K items")

    # Fusion options
    p.add_argument(
        "--fusion",
        choices=["concat", "weighted"],
        default="concat",
        help="Fusion strategy: 'concat' (vector concat) or 'weighted' (vector weighted sum)",
    )
    p.add_argument("--w_text", type=float, default=1.0, help="Weight for text modality")
    p.add_argument("--w_image", type=float, default=0.0, help="Weight for image modality")
    p.add_argument("--w_meta", type=float, default=0.0, help="Weight for metadata modality")

    # Defaults helper
    p.add_argument(
        "--use_defaults",
        action="store_true",
        help="Use defaults from data/processed/<dataset>/index/defaults.json for weights and FAISS names",
    )

    # FAISS options (supported for BOTH fusions—ensure the index was built accordingly)
    p.add_argument("--use_faiss", action="store_true", help="Use FAISS index for ANN search")
    p.add_argument(
        "--faiss_name",
        type=str,
        default=None,
        help="Name used when building FAISS index (e.g. 'beauty_concat' or 'beauty_weighted_wt1.0_wi0.2_wm0.2')",
    )

    # Filters / display
    p.add_argument("--exclude_seen", action="store_true", help="Exclude items the user already interacted with")
    p.add_argument("--show", type=int, default=None, help="Display first N rows (default: k)")
    return p.parse_args()


def _infer_faiss_name(dataset: str, fusion: str, w_text: float, w_image: float, w_meta: float) -> str:
    """
    Derive the FAISS base-name used by scripts/build_faiss.py.
    Files are: data/processed/<dataset>/index/items_<faiss_name>.faiss
    """
    if fusion == "concat":
        # build_faiss.py wrote items_<dataset>_concat.faiss
        return f"{dataset}_concat"
    # weighted
    # build_faiss.py wrote items_<dataset>_weighted_wt{w_text}_wi{w_image}_wm{w_meta}.faiss
    return f"{dataset}_weighted_wt{w_text}_wi{w_image}_wm{w_meta}"


def main():
    args = parse_args()
    proc = _load_proc_paths(args.dataset)

    # Optionally pull defaults (weights + faiss names) from defaults.json
    # File structure (example):
    # {
    #   "concat":   {"w_text": 1.0, "w_image": 1.0, "w_meta": 0.0, "faiss_name": "beauty_concat"},
    #   "weighted": {"w_text": 1.0, "w_image": 0.2, "w_meta": 0.2, "faiss_name": "beauty_weighted_wt1.0_wi0.2_wm0.2"}
    # }
    defaults = _load_defaults(args.dataset)

    if args.use_defaults and defaults and args.fusion in defaults:
        d = defaults[args.fusion]
        # Use stored defaults if present; fall back to existing CLI values otherwise.
        wt = float(d.get("w_text", args.w_text))
        wi = float(d.get("w_image", args.w_image))
        wm = float(d.get("w_meta", args.w_meta))
        faiss_default = d.get("faiss_name")
        # update args
        args.w_text, args.w_image, args.w_meta = wt, wi, wm
        if args.use_faiss and not args.faiss_name:
            # use explicit default if provided, else infer from weights/fusion
            args.faiss_name = faiss_default or _infer_faiss_name(args.dataset, args.fusion, wt, wi, wm)

    # If FAISS requested but name not given, try to infer from current settings
    if args.use_faiss and not args.faiss_name:
        args.faiss_name = _infer_faiss_name(args.dataset, args.fusion, args.w_text, args.w_image, args.w_meta)

    # Run pipeline
    if args.fusion == "concat":
        out, rec_ids = _recommend_concat(
            proc=proc,
            user_id=args.user_id,
            k=args.k,
            exclude_seen=args.exclude_seen,
            use_faiss=args.use_faiss,
            faiss_name=args.faiss_name,
            w_text=args.w_text,
            w_image=args.w_image,
            w_meta=args.w_meta,
        )
    else:
        out, rec_ids = _recommend_weighted(
            proc=proc,
            user_id=args.user_id,
            k=args.k,
            exclude_seen=args.exclude_seen,
            use_faiss=args.use_faiss,
            faiss_name=args.faiss_name,
            w_text=args.w_text,
            w_image=args.w_image,
            w_meta=args.w_meta,
        )

    n_show = args.show or args.k
    cols = [c for c in [ITEM_KEY, "score", "brand", "price", "categories", "image_url"] if c in out.columns]
    print(out[cols].head(n_show).to_string(index=False))

    # JSON summary (handy for UI)
    summary = {
        "dataset": args.dataset,
        "user_id": args.user_id,
        "fusion": args.fusion,
        "weights": {"text": args.w_text, "image": args.w_image, "meta": args.w_meta},
        "k": args.k,
        "exclude_seen": bool(args.exclude_seen),
        "use_faiss": bool(args.use_faiss),
        "faiss_name": args.faiss_name,
        "recommendations": out[cols].head(n_show).to_dict(orient="records"),
    }
    print("\nJSON:", json.dumps(summary, default=str)[:1200], "...")


if __name__ == "__main__":
    main()
