#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# If FAISS isn't installed:  pip install faiss-cpu
import faiss

from src.data.registry import get_paths
from src.models.fusion import concat_fusion  # FAISS is used with concat in online path


def _resolve_paths(dataset: str):
    paths = get_paths(dataset)
    if isinstance(paths, dict):
        raw = paths.get("raw") or paths.get("raw_dir") or paths.get("raw_path")
        proc = paths.get("processed") or paths.get("processed_dir") or paths.get("processed_path")
        return Path(raw), Path(proc)
    if isinstance(paths, (tuple, list)) and len(paths) >= 2:
        return Path(paths[0]), Path(paths[1])
    raw = getattr(paths, "raw", None) or getattr(paths, "raw_dir", None) or getattr(paths, "raw_path", None)
    proc = getattr(paths, "processed", None) or getattr(paths, "processed_dir", None) or getattr(paths, "processed_path", None)
    return Path(raw), Path(proc)


def _load_vectors(proc_dir: Path):
    # Text (required)
    text_df = pd.read_parquet(proc_dir / "item_text_emb.parquet")
    item_ids = text_df["item_id"].astype(str).to_numpy()
    Vt = np.stack(text_df["vector"].to_numpy()).astype(np.float32)

    # Image (optional)
    Vi = None
    img_fp = proc_dir / "item_image_emb.parquet"
    if img_fp.exists():
        img_df = pd.read_parquet(img_fp)
        img_map = {str(iid): vec for iid, vec in zip(img_df["item_id"].astype(str), img_df["vector"])}
        Di = len(next(iter(img_map.values())))
        Vi = np.stack([img_map.get(i, np.zeros(Di, dtype=np.float32)) for i in item_ids]).astype(np.float32)

    # Meta (optional)
    Vm = None
    meta_fp = proc_dir / "item_meta_emb.parquet"
    if meta_fp.exists():
        meta_df = pd.read_parquet(meta_fp)
        meta_map = {str(iid): vec for iid, vec in zip(meta_df["item_id"].astype(str), meta_df["vector"])}
        Dm = len(next(iter(meta_map.values())))
        Vm = np.stack([meta_map.get(i, np.zeros(Dm, dtype=np.float32)) for i in item_ids]).astype(np.float32)

    return item_ids, Vt, Vi, Vm


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def build_index(V: np.ndarray):
    # Cosine via inner product on L2-normalized vectors
    Vn = _l2norm(V.astype(np.float32, copy=False))
    index = faiss.IndexFlatIP(Vn.shape[1])
    index.add(Vn)
    return index


def main():
    ap = argparse.ArgumentParser(description="Build FAISS index for a dataset (concat fusion).")
    ap.add_argument("--dataset", required=True, help="Dataset key, e.g. beauty")
    # keep fusion arg for CLI parity, but we only honor concat for FAISS (matches online use)
    ap.add_argument("--fusion", choices=["concat", "weighted"], default="concat")
    ap.add_argument("--w_text", type=float, default=1.0)
    ap.add_argument("--w_image", type=float, default=1.0)
    ap.add_argument("--w_meta", type=float, default=0.0)
    ap.add_argument("--variant", type=str, default="concat_best",
                    help="Name part after <dataset>_ (e.g. concat_best → items_<dataset>_concat_best.faiss)")
    args = ap.parse_args()

    _, proc_dir = _resolve_paths(args.dataset)
    out_dir = proc_dir / "index"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_ids, Vt, Vi, Vm = _load_vectors(proc_dir)

    if args.fusion != "concat":
        print("[warn] FAISS index is used by the online recommender only for 'concat'. "
              "Proceeding with concat fusion to keep things consistent.")

    # Build concat fused matrix with weights (same as online concat path)
    V = concat_fusion(Vt, Vi, Vm, weights=(args.w_text, args.w_image, args.w_meta))

    index = build_index(V)

    # Naming: items_<dataset>_<variant>.{faiss,npy}
    base = f"items_{args.dataset}_{args.variant}"
    faiss_fp = out_dir / f"{base}.faiss"
    ids_fp   = out_dir / f"{base}.npy"

    faiss.write_index(index, str(faiss_fp))
    # Save ids with dtype=object to be robust
    np.save(ids_fp, np.array(item_ids, dtype=object))

    print(f"✅ Built FAISS index: {faiss_fp}")
    print(f"✅ Saved item id mapping: {ids_fp}")
    print(f"Items: {len(item_ids)} | Dim: {V.shape[1]}")
    print("Tip: In the UI/API, use faiss_name =", f"{args.dataset}_{args.variant!s}")

if __name__ == "__main__":
    main()