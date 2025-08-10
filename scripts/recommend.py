#!/usr/bin/env python3
"""
Recommend items for a given user from precomputed embeddings.

Supports:
- Vector-level concat fusion (optionally using a FAISS index)
- Score-level weighted fusion (no FAISS; combines per-modality cosine scores)
- Excluding items the user has already interacted with
- Pretty printing of top-K with metadata

Expected processed files under data/processed/<dataset>/:
- reviews.parquet                  (user_id, item_id, rating, ts, text)
- user_text_emb.parquet            (user_id, vector)
- item_text_emb.parquet            (item_id, vector)
- item_image_emb.parquet           (item_id, vector)
- item_meta_emb.parquet (optional) (item_id, vector)
- items_with_meta.parquet          (item_id, brand, price, categories, image_url)
- (optional) index/items_<name>.faiss + items_<name>.npy for FAISS searches

Usage example:
PYTHONPATH=$(pwd) python scripts/recommend.py \
  --dataset beauty \
  --user_id A3CIUOJXQ5VDQ2 \
  --exclude_seen \
  --use_faiss \
  --faiss_name beauty_concat_best \
  --fusion concat \
  --w_text 1.0 --w_image 1.0 --w_meta 0.4 \
  --k 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import get_processed_path
from src.models.fusion import concat_fusion, l2norm

ITEM_KEY = "item_id"


# ---------------------------- IO helpers ---------------------------- #

def _load_proc_paths(dataset: str) -> Path:
    proc = Path(get_processed_path(dataset))
    if not proc.exists():
        raise FileNotFoundError(f"Processed dir not found: {proc}. Run normalization first.")
    return proc


def _read_parquet(fp: Path, required_cols=None) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")
    df = pd.read_parquet(fp)
    if required_cols is not None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{fp} missing columns: {missing} | has {list(df.columns)}")
    return df


def _load_user_vector(proc: Path, user_id: str) -> np.ndarray:
    dfu = _read_parquet(proc / "user_text_emb.parquet", required_cols=["user_id", "vector"])
    row = dfu[dfu["user_id"] == user_id]
    if row.empty:
        raise ValueError(
            f"user_id not found in user_text_emb.parquet: {user_id}.\n"
            f"Run text embedding builder (scripts/build_text_emb.py)."
        )
    vec = np.asarray(row.iloc[0]["vector"], dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec


def _load_item_mats(proc: Path):
    Mt = _read_parquet(proc / "item_text_emb.parquet", required_cols=[ITEM_KEY, "vector"])
    Mi = _read_parquet(proc / "item_image_emb.parquet", required_cols=[ITEM_KEY, "vector"])
    meta_fp = proc / "item_meta_emb.parquet"
    Mm = _read_parquet(meta_fp, required_cols=[ITEM_KEY, "vector"]) if meta_fp.exists() else None
    return Mt, Mi, Mm


def _load_items_table(proc: Path) -> pd.DataFrame:
    items = _read_parquet(proc / "items_with_meta.parquet")
    # ensure ITEM_KEY column exists (handle old versions with index)
    if ITEM_KEY not in items.columns:
        if items.index.name == ITEM_KEY:
            items = items.reset_index()
        else:
            raise KeyError(f"{ITEM_KEY} not found in items_with_meta.parquet")
    return items


def _user_seen_items(proc: Path, user_id: str) -> set:
    # reviews.parquet has (user_id, item_id, rating, ts, text)
    df = _read_parquet(proc / "reviews.parquet", required_cols=["user_id", ITEM_KEY])
    return set(df[df["user_id"] == user_id][ITEM_KEY].tolist())


# --------------------------- Math helpers --------------------------- #

def _stack_vectors(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert a df with columns [item_id, vector] to (matrix, ids)."""
    ids = df[ITEM_KEY].tolist()
    mat = np.vstack(df["vector"].to_numpy()).astype(np.float32, copy=False)
    return mat, ids


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # query: [D], matrix: [N, D]
    q = query.reshape(1, -1)
    q = q / (np.linalg.norm(q) + 1e-12)
    M = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return (q @ M.T).ravel()


# -------------------------- FAISS integration ----------------------- #

def _faiss_search(proc: Path, name: str, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss
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


# ----------------------------- Recommender -------------------------- #

def _concat_user_vector(user_text_vec: np.ndarray,
                        item_text_mat: np.ndarray,
                        item_image_mat: np.ndarray | None,
                        item_meta_mat: np.ndarray | None,
                        w_text: float,
                        w_image: float,
                        w_meta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a user query vector in the same fused space as concat_fusion(item_mats).
    Since concat_fusion concatenates modality vectors (after optional weights & norm),
    we do the same for the user vector:
      user_fused = [ w_t * u_text_norm ; w_i * u_image? ; w_m * u_meta? ]
    For now, we only have user TEXT vector precomputed. We set user image/meta parts to zeros.
    """
    # Normalize user text
    ut = user_text_vec / (np.linalg.norm(user_text_vec) + 1e-12)

    parts = [w_text * ut]
    dim_text = ut.shape[0]

    dim_image = item_image_mat.shape[1] if item_image_mat is not None else 0
    dim_meta = item_meta_mat.shape[1] if item_meta_mat is not None else 0

    if dim_image > 0:
        parts.append(np.zeros((dim_image,), dtype=np.float32))  # we don't have a user image embedding
    if dim_meta > 0:
        parts.append(np.zeros((dim_meta,), dtype=np.float32))   # we don't have a user meta embedding

    uf = np.concatenate(parts, axis=0).astype(np.float32)
    uf = uf / (np.linalg.norm(uf) + 1e-12)
    return uf, np.array([dim_text, dim_image, dim_meta], dtype=np.int32)


def _recommend_concat(proc: Path,
                      user_id: str,
                      k: int,
                      exclude_seen: bool,
                      use_faiss: bool,
                      faiss_name: str | None,
                      w_text: float,
                      w_image: float,
                      w_meta: float) -> tuple[pd.DataFrame, list[str]]:
    """Concat fusion pipeline: build fused item matrix, user fused vector, and score/search."""
    # Load data
    items_df = _load_items_table(proc)
    Mt, Mi, Mm = _load_item_mats(proc)
    user_vec = _load_user_vector(proc, user_id)

    # Stack item modality matrices
    Vt, item_ids_t = _stack_vectors(Mt)
    Vi, item_ids_i = _stack_vectors(Mi)
    Vm, item_ids_m = (None, None)
    if Mm is not None:
        Vm, item_ids_m = _stack_vectors(Mm)

    # sanity: all item_id orders should match across modalities for clean concat
    # We'll inner-join on item_id to build aligned matrices.
    def df2(df): return df[[ITEM_KEY, "vector"]].copy()

    base = df2(Mt)
    base = base.merge(df2(Mi), on=ITEM_KEY, suffixes=("_t", "_i"))
    if Mm is not None:
        base = base.merge(df2(Mm), on=ITEM_KEY)
        base.rename(columns={"vector": "vector_m"}, inplace=True)  # text: _t, image: _i, meta: _m

    # Build aligned matrices
    item_ids = base[ITEM_KEY].tolist()
    Vt = np.vstack(base["vector_t"].to_numpy()).astype(np.float32)
    Vi = np.vstack(base["vector_i"].to_numpy()).astype(np.float32)
    Vm = np.vstack(base["vector_m"].to_numpy()).astype(np.float32) if "vector_m" in base.columns else None

    # Fused item matrix
    Vf = concat_fusion(Vt, Vi, Vm, weights=(w_text, w_image, w_meta))  # [I, Df]

    # User fused vector (text part + zeros for other modalities)
    uf, dims = _concat_user_vector(user_vec, Vt, Vi, Vm, w_text, w_image, w_meta)

    # Exclude seen items if requested
    exclude = set()
    if exclude_seen:
        exclude = _user_seen_items(proc, user_id)

    # FAISS or direct cosine
    if use_faiss:
        if not faiss_name:
            raise ValueError("--use_faiss requires --faiss_name")
        # Assumes FAISS index built over 'concat' fused item vectors with same weights/dims
        _, hits = _faiss_search(proc, faiss_name, uf, k + 200)  # search more to allow excluding seen
        rec_ids = [iid for iid in hits.tolist() if iid not in exclude][:k]
        # approximate scores: cosine with Vf subset
        id2row = {iid: i for i, iid in enumerate(item_ids)}
        sel = np.array([id2row[i] for i in rec_ids], dtype=np.int64)
        scores = (uf.reshape(1, -1) @ Vf[sel].T).ravel()
    else:
        # cosine scores against all items
        scores = (uf.reshape(1, -1) @ Vf.T).ravel()
        # mask seen
        mask = np.array([iid not in exclude for iid in item_ids], dtype=bool)
        scores_masked = np.where(mask, scores, -np.inf)
        topk_idx = np.argpartition(scores_masked, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores_masked[topk_idx])[::-1]]
        rec_ids = [item_ids[i] for i in topk_idx]
        scores = scores[topk_idx]

    # Compose output table
    out = items_df.merge(pd.DataFrame({ITEM_KEY: rec_ids, "score": scores}), on=ITEM_KEY, how="right")
    # Preserve sort by score
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out, rec_ids


def _recommend_weighted(proc: Path,
                        user_id: str,
                        k: int,
                        exclude_seen: bool,
                        w_text: float,
                        w_image: float,
                        w_meta: float) -> tuple[pd.DataFrame, list[str]]:
    """
    Weighted score fusion:
      score = w_t * cos(u_text, v_text) + w_i * cos(u_text, v_image) + w_m * cos(u_text, v_meta)
    (We only have a text vector for the user; this is a reasonable baseline.)
    """
    items_df = _load_items_table(proc)
    Mt, Mi, Mm = _load_item_mats(proc)
    user_vec = _load_user_vector(proc, user_id)

    Vt, ids_t = _stack_vectors(Mt)
    Vi, ids_i = _stack_vectors(Mi)
    Vm, ids_m = (None, None)
    if Mm is not None:
        Vm, ids_m = _stack_vectors(Mm)

    # Align on item_id
    base = Mt[[ITEM_KEY]].copy()
    base = base.merge(Mi[[ITEM_KEY]], on=ITEM_KEY)
    if Mm is not None:
        base = base.merge(Mm[[ITEM_KEY]], on=ITEM_KEY)
    item_ids = base[ITEM_KEY].tolist()

    # Reindex matrices
    def reindex(mat_df: pd.DataFrame, key_ids: list[str]) -> np.ndarray:
        m = mat_df.set_index(ITEM_KEY).loc[key_ids, "vector"].to_numpy()
        return np.vstack(m).astype(np.float32)

    Vt = reindex(Mt, item_ids)
    Vi = reindex(Mi, item_ids)
    Vm = reindex(Mm, item_ids) if Mm is not None else None

    # cosine scores
    st = _cosine_scores(user_vec, Vt) if w_text != 0 else 0.0
    si = _cosine_scores(user_vec, Vi) if w_image != 0 else 0.0
    sm = _cosine_scores(user_vec, Vm) if (Vm is not None and w_meta != 0) else 0.0

    scores = (w_text * st) + (w_image * si) + (w_meta * sm)

    # exclude seen
    if exclude_seen:
        exclude = _user_seen_items(proc, user_id)
        mask = np.array([iid not in exclude for iid in item_ids], dtype=bool)
        scores = np.where(mask, scores, -np.inf)

    topk_idx = np.argpartition(scores, -k)[-k:]
    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
    rec_ids = [item_ids[i] for i in topk_idx]
    top_scores = scores[topk_idx]

    out = items_df.merge(pd.DataFrame({ITEM_KEY: rec_ids, "score": top_scores}), on=ITEM_KEY, how="right")
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out, rec_ids


# ------------------------------ CLI -------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Recommend items for a user.")
    p.add_argument("--dataset", required=True, help="Dataset key (e.g., beauty)")
    p.add_argument("--user_id", required=True, help="User ID to recommend for")
    p.add_argument("--k", type=int, default=10, help="Top-K items")

    # Fusion options
    p.add_argument("--fusion", choices=["concat", "weighted"], default="concat",
                   help="Fusion strategy: 'concat' (vector concat) or 'weighted' (score fusion)")
    p.add_argument("--w_text", type=float, default=1.0, help="Weight for text modality")
    p.add_argument("--w_image", type=float, default=1.0, help="Weight for image modality")
    p.add_argument("--w_meta", type=float, default=0.0, help="Weight for metadata modality")

    # FAISS options (only for concat)
    p.add_argument("--use_faiss", action="store_true", help="Use FAISS index for ANN search (concat only)")
    p.add_argument("--faiss_name", type=str, default=None, help="Name used when building FAISS index")

    # Filters / display
    p.add_argument("--exclude_seen", action="store_true", help="Exclude items the user already interacted with")
    p.add_argument("--show", type=int, default=None, help="Display first N rows (default: k)")
    return p.parse_args()


def main():
    args = parse_args()
    proc = _load_proc_paths(args.dataset)

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
        if args.use_faiss:
            print("[warn] --use_faiss is ignored with fusion='weighted' (score-level fusion).", file=sys.stderr)
        out, rec_ids = _recommend_weighted(
            proc=proc,
            user_id=args.user_id,
            k=args.k,
            exclude_seen=args.exclude_seen,
            w_text=args.w_text,
            w_image=args.w_image,
            w_meta=args.w_meta,
        )

    n_show = args.show or args.k
    cols = [c for c in [ITEM_KEY, "score", "brand", "price", "categories", "image_url"] if c in out.columns]
    print(out[cols].head(n_show).to_string(index=False))

    # Also print a compact JSON summary (useful for UI later)
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
    print("\nJSON:", json.dumps(summary, default=str)[:1000], "...")
    

if __name__ == "__main__":
    main()