#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# You can use CPU FAISS everywhere; it’s fine for this dataset size
# If faiss isn't installed, run:  pip install faiss-cpu
import faiss

from src.data.registry import get_paths
from src.models.fusion import concat_fusion, weighted_sum_fusion


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
    item_ids = text_df["item_id"].to_numpy()
    Vt = np.stack(text_df["vector"].to_numpy()).astype(np.float32)

    # Image (optional)
    Vi = None
    img_fp = proc_dir / "item_image_emb.parquet"
    if img_fp.exists():
        img_df = pd.read_parquet(img_fp)
        img_map = {iid: vec for iid, vec in zip(img_df["item_id"], img_df["vector"])}
        Di = len(next(iter(img_map.values())))
        Vi = np.stack([img_map.get(i, np.zeros(Di, dtype=np.float32)) for i in item_ids]).astype(np.float32)

    # Meta (optional)
    Vm = None
    meta_fp = proc_dir / "item_meta_emb.parquet"
    if meta_fp.exists():
        meta_df = pd.read_parquet(meta_fp)
        meta_map = {iid: vec for iid, vec in zip(meta_df["item_id"], meta_df["vector"])}
        Dm = len(next(iter(meta_map.values())))
        Vm = np.stack([meta_map.get(i, np.zeros(Dm, dtype=np.float32)) for i in item_ids]).astype(np.float32)

    return item_ids, Vt, Vi, Vm


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def build_index(V: np.ndarray):
    # Cosine similarity via inner product on L2-normalized vectors
    Vn = _l2norm(V.astype(np.float32, copy=False))
    index = faiss.IndexFlatIP(Vn.shape[1])
    index.add(Vn)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fusion", choices=["concat", "weighted"], default="concat")
    ap.add_argument("--w_text", type=float, default=1.0)
    ap.add_argument("--w_image", type=float, default=1.0)
    ap.add_argument("--w_meta", type=float, default=0.0)
    ap.add_argument("--out_name", type=str, default="")
    args = ap.parse_args()

    _, proc_dir = _resolve_paths(args.dataset)
    out_dir = proc_dir / "index"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_ids, Vt, Vi, Vm = _load_vectors(proc_dir)

    if args.fusion == "concat":
        V = concat_fusion(Vt, Vi, Vm, weights=(args.w_text, args.w_image, args.w_meta))
    else:
        V = weighted_sum_fusion(Vt, Vi, Vm, weights=(args.w_text, args.w_image, args.w_meta))

    index = build_index(V)

    name = args.out_name or f"{args.fusion}_wt{args.w_text}_wi{args.w_image}_wm{args.w_meta}"
    faiss_fp = out_dir / f"items_{name}.faiss"
    ids_fp = out_dir / f"items_{name}.npy"

    faiss.write_index(index, str(faiss_fp))
    np.save(ids_fp, item_ids)

    print(f"✅ Built FAISS index: {faiss_fp}")
    print(f"✅ Saved item id mapping: {ids_fp}")
    print(f"Items: {len(item_ids)} | Dim: {V.shape[1]}")

if __name__ == "__main__":
    main()