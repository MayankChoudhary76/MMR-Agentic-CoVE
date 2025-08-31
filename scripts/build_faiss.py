#!/usr/bin/env python3
# scripts/build_faiss.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.models.fusion import concat_fusion, weighted_sum_fusion
from src.data.registry import get_paths


# ---------------------- Path helpers ---------------------- #

def _resolve_paths(dataset: str) -> Tuple[Path, Path]:
    paths = get_paths(dataset)
    if isinstance(paths, dict):
        raw = paths.get("raw") or paths.get("raw_dir") or paths.get("raw_path")
        proc = paths.get("processed") or paths.get("processed_dir") or paths.get("proc") or paths.get("processed_path")
        if raw and proc:
            return Path(raw), Path(proc)
    if isinstance(paths, (tuple, list)) and len(paths) >= 2:
        return Path(paths[0]), Path(paths[1])
    raw = getattr(paths, "raw", None) or getattr(paths, "raw_dir", None) or getattr(paths, "raw_path", None)
    proc = getattr(paths, "processed", None) or getattr(paths, "processed_dir", None) or getattr(paths, "processed_path", None)
    if raw and proc:
        return Path(raw), Path(proc)
    # fallback
    return Path("data/raw") / dataset, Path("data/processed") / dataset


# ---------------------- Data loading ---------------------- #

def _load_vectors(proc_dir: Path) -> Tuple[List[str], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    txt = pd.read_parquet(proc_dir / "item_text_emb.parquet")  # item_id, vector
    item_ids = txt["item_id"].tolist()
    Vt = np.stack(txt["vector"].to_numpy()).astype(np.float32)

    Vi = None
    img_fp = proc_dir / "item_image_emb.parquet"
    if img_fp.exists():
        img = pd.read_parquet(img_fp)
        img_map = {iid: vec for iid, vec in zip(img["item_id"], img["vector"])}
        first = next(iter(img_map.values()), None)
        if first is not None:
            Di = len(first)
            Vi = np.stack([img_map.get(i, np.zeros(Di, dtype=np.float32)) for i in item_ids]).astype(np.float32)

    Vm = None
    meta_fp = proc_dir / "item_meta_emb.parquet"
    if meta_fp.exists():
        met = pd.read_parquet(meta_fp)
        met_map = {iid: vec for iid, vec in zip(met["item_id"], met["vector"])}
        first = next(iter(met_map.values()), None)
        if first is not None:
            Dm = len(first)
            Vm = np.stack([met_map.get(i, np.zeros(Dm, dtype=np.float32)) for i in item_ids]).astype(np.float32)

    return item_ids, Vt, Vi, Vm


# ---------------------- Eval helpers (for sweep) ---------------------- #

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

def _leave_one_out(df: pd.DataFrame):
    df = df.sort_values(["user_id", "timestamp"])
    last = df.groupby("user_id").tail(1)[["user_id", "item_id"]].rename(columns={"item_id": "target"})
    train = df.merge(last, on="user_id", how="left")
    train = train[train["item_id"] != train["target"]][["user_id", "item_id"]]
    test = last
    return train, test

def _pad_to(a: np.ndarray, dim: int) -> np.ndarray:
    if a.shape[1] == dim:
        return a
    if a.shape[1] < dim:
        pad = np.zeros((a.shape[0], dim - a.shape[1]), dtype=a.dtype)
        return np.concatenate([a, pad], axis=1)
    return a[:, :dim]

def _evaluate_topk(user_vecs: np.ndarray,
                   item_vecs: np.ndarray,
                   item_ids: np.ndarray,
                   test: pd.DataFrame,
                   k: int) -> dict:
    S = _cosine_sim(user_vecs, item_vecs)  # [U x I]
    topk_idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    row_indices = np.arange(S.shape[0])[:, None]
    sorted_order = np.argsort(-S[row_indices, topk_idx], axis=1)
    topk_idx = topk_idx[row_indices, sorted_order]

    rec_items = item_ids[topk_idx]  # [U x k]
    test_items = test["target"].to_numpy()

    hits = (rec_items == test_items[:, None]).any(axis=1).astype(np.float32)

    # NDCG@k (binary relevance, single positive per user)
    gains = (rec_items == test_items[:, None]).astype(np.float32)
    ranks = np.argmax(gains, axis=1)
    ndcg = []
    for u, r in enumerate(ranks):
        if gains[u].any():
            ndcg.append(1.0 / np.log2(r + 2))
        else:
            ndcg.append(0.0)

    return {f"hit@{k}": float(hits.mean()),
            f"ndcg@{k}": float(np.mean(ndcg))}

def _candidate_grid() -> Iterable[Tuple[float, float, float]]:
    # Small grid; text anchored at 1.0
    ws = [0.0, 0.1, 0.2, 0.4]
    for wi in ws:
        for wm in ws:
            yield (1.0, wi, wm)

def sweep_weights(dataset: str,
                  proc_dir: Path,
                  Vt: np.ndarray,
                  Vi: Optional[np.ndarray],
                  Vm: Optional[np.ndarray],
                  k: int,
                  candidates: Iterable[Tuple[float, float, float]],
                  metric: str = "hit",
                  require_all_modalities: bool = False,
                  min_w_image: float = 0.0,
                  min_w_meta: float = 0.0) -> Tuple[Tuple[float, float, float], dict]:
    """
    Select best weights by maximizing primary `metric` (hit or ndcg),
    using the other metric as a tie-breaker. Optionally constrain the grid.
    """
    # Load reviews + user text only (as baseline user representation)
    reviews = pd.read_parquet(proc_dir / "reviews.parquet")
    train, test = _leave_one_out(reviews)

    U_text_df = pd.read_parquet(proc_dir / "user_text_emb.parquet")  # user_id, vector
    I_text_df = pd.read_parquet(proc_dir / "item_text_emb.parquet")  # item_id, vector

    # align users to evaluation subset
    eval_users = train["user_id"].drop_duplicates().to_numpy()
    u_map = {u: i for i, u in enumerate(U_text_df["user_id"])}
    keep_u = [u for u in eval_users if u in u_map]
    u_idx = np.array([u_map[u] for u in keep_u], dtype=np.int64)
    U_text = np.stack(U_text_df.loc[u_idx, "vector"].to_numpy()).astype(np.float32)  # [U x Dt]

    # align items
    item_ids = I_text_df["item_id"].to_numpy()

    # align test users to U_text rows
    test_users = test["user_id"].to_numpy()
    keep_mask = np.isin(test_users, keep_u)
    test_eval = test.loc[keep_mask].reset_index(drop=True)
    U_eval = U_text[: len(test_eval)]

    print(f"[sweep] Objective: {metric}@{k} (secondary tie-breaker is the other metric)")
    if require_all_modalities:
        print("[sweep] Constraint: w_image>0 and w_meta>0")
    if min_w_image > 0.0 or min_w_meta > 0.0:
        print(f"[sweep] Min weights: w_image≥{min_w_image}, w_meta≥{min_w_meta}")

    best_w = None
    best_primary = -1.0
    best_secondary = -1.0

    for wt, wi, wm in candidates:
        if wt != 1.0:
            wt = 1.0  # keep user text scale fixed

        # constraints
        if require_all_modalities and (wi <= 0.0 or wm <= 0.0):
            continue
        if wi < min_w_image or wm < min_w_meta:
            continue

        # fuse items
        Vf = weighted_sum_fusion(Vt, Vi, Vm, weights=(wt, wi, wm))
        # pad user vectors to fused dim
        U_fused = _pad_to(wt * U_eval, Vf.shape[1])

        m = _evaluate_topk(U_fused, Vf, item_ids, test_eval, k=k)
        hit = float(m[f"hit@{k}"])
        ndcg = float(m[f"ndcg@{k}"])
        print(f"→ tried wt={wt}, wi={wi}, wm={wm}  →  {m}")

        primary = hit if metric == "hit" else ndcg
        secondary = ndcg if metric == "hit" else hit

        if best_w is None or (primary, secondary) > (best_primary, best_secondary):
            best_w = (wt, wi, wm)
            best_primary = primary
            best_secondary = secondary

    if best_w is None:
        # In case constraints filtered everything, fall back to text-only
        best_w = (1.0, 0.0, 0.0)
        best_hit = float("nan")
        best_ndcg = float("nan")
    else:
        best_hit = best_primary if metric == "hit" else best_secondary
        best_ndcg = best_secondary if metric == "hit" else best_primary

    print(f"\n★ Best weights: wt={best_w[0]}, wi={best_w[1]}, wm={best_w[2]}  "
          f"with {{'hit@{k}': {best_hit}, 'ndcg@{k}': {best_ndcg}}}")
    return best_w, {f"hit@{k}": best_hit, f"ndcg@{k}": best_ndcg}


# ---------------------- FAISS build ---------------------- #

def _l2norm_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

def build_index(V: np.ndarray):
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS not available. Install faiss-cpu.") from e
    Vn = _l2norm_rows(V.astype(np.float32))
    index = faiss.IndexFlatIP(Vn.shape[1])  # cosine via IP on L2-normed
    index.add(Vn)
    return index, faiss


try:
    import csv, time, os
    sizes = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset,
        "fusion": args.fusion,
        "faiss_name": tip_name,
        "index_mb": round(os.path.getsize(faiss_fp) / (1024*1024), 3),
        "ids_mb": round(os.path.getsize(ids_fp) / (1024*1024), 3),
    }
    sizes_csv = Path("logs") / "index_sizes.csv"
    sizes_csv.parent.mkdir(parents=True, exist_ok=True)
    with sizes_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sizes.keys()))
        if sizes_csv.stat().st_size == 0:
            w.writeheader()
        w.writerow(sizes)
    print(f"🧱 Logged index sizes → {sizes_csv}")
except Exception as e:
    print(f"[warn] could not log index sizes: {e}")
    
# ---------------------- Defaults I/O ---------------------- #

def _read_defaults(proc_dir: Path) -> dict:
    fp = proc_dir / "index" / "defaults.json"
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text())
    except Exception:
        return {}

def _write_defaults(proc_dir: Path, fusion: str, w: Tuple[float, float, float], k: int, faiss_name: str):
    fp = proc_dir / "index" / "defaults.json"
    data = _read_defaults(proc_dir)
    data[fusion] = {
        "w_text": float(w[0]),
        "w_image": float(w[1]),
        "w_meta": float(w[2]),
        "k": int(k),
        "faiss_name": faiss_name,
    }
    fp.write_text(json.dumps(data, indent=2))
    print(f"✓ Saved defaults → {fp}")


# ---------------------- CLI main ---------------------- #

def main():
    ap = argparse.ArgumentParser(description="Build FAISS index for a dataset with concat/weighted fusion (plus optional weight sweep).")
    ap.add_argument("--dataset", required=True, help="Dataset key, e.g. beauty")
    ap.add_argument("--fusion", choices=["concat", "weighted"], default="concat",
                    help="Fusion strategy for item vectors in the index.")
    ap.add_argument("--w_text", type=float, default=1.0, help="Weight for text vectors")
    ap.add_argument("--w_image", type=float, default=0.0, help="Weight for image vectors")
    ap.add_argument("--w_meta", type=float, default=0.0, help="Weight for metadata vectors")
    ap.add_argument("--variant", type=str, default="", help="Custom variant suffix for file naming (optional)")
    ap.add_argument("--sweep", action="store_true", help="Auto-tune weights (weighted fusion only)")
    ap.add_argument("--k", type=int, default=10, help="k for hit@k / ndcg@k during sweep")
    ap.add_argument("--set_default", action="store_true", help="Write best weights to index/defaults.json")
    ap.add_argument("--use_defaults", action="store_true", help="Load weights from index/defaults.json for this fusion")

    # NEW: objective & constraints for sweep
    ap.add_argument("--metric", choices=["hit", "ndcg"], default="hit",
                    help="Primary objective for sweep (tie-breaker is the other metric).")
    ap.add_argument("--require_all_modalities", action="store_true",
                    help="Skip candidates where w_image==0 or w_meta==0 during sweep.")
    ap.add_argument("--min_w_image", type=float, default=0.0,
                    help="Minimum w_image allowed during sweep.")
    ap.add_argument("--min_w_meta", type=float, default=0.0,
                    help="Minimum w_meta allowed during sweep.")

    args = ap.parse_args()

    _, proc_dir = _resolve_paths(args.dataset)
    out_dir = proc_dir / "index"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load current item vectors
    item_ids, Vt, Vi, Vm = _load_vectors(proc_dir)

    # If requested, load weights from defaults.json (overrides CLI)
    if args.use_defaults:
        defaults = _read_defaults(proc_dir)
        d = defaults.get(args.fusion, {})
        args.w_text = float(d.get("w_text", args.w_text))
        args.w_image = float(d.get("w_image", args.w_image))
        args.w_meta = float(d.get("w_meta", args.w_meta))
        print(f"[defaults] Using weights → wt={args.w_text}, wi={args.w_image}, wm={args.w_meta}")

    has_img = Vi is not None
    has_meta = Vm is not None
    if args.w_image > 0.0 and not has_img:
        print("[warn] w_image > 0 but no item_image_emb.parquet found; image contributes zeros.")
    if args.w_meta > 0.0 and not has_meta:
        print("[warn] w_meta > 0 but no item_meta_emb.parquet found; meta contributes zeros.")

    # If sweeping (only meaningful for weighted fusion)
    chosen = (args.w_text, args.w_image, args.w_meta)
    chosen_metrics = None
    if args.sweep:
        if args.fusion != "weighted":
            print("[info] --sweep is intended for fusion='weighted'. Ignoring for concat.")
        else:
            candidates = list(_candidate_grid())
            chosen, chosen_metrics = sweep_weights(
                args.dataset, proc_dir, Vt, Vi, Vm, args.k, candidates,
                metric=args.metric,
                require_all_modalities=args.require_all_modalities,
                min_w_image=args.min_w_image,
                min_w_meta=args.min_w_meta,
            )
            if args.set_default:
                tip_name = f"{args.dataset}_weighted"
                _write_defaults(proc_dir, "weighted", chosen, args.k, tip_name)

    # Decide weights for the actual index build
    wt, wi, wm = chosen

    # Fuse item vectors
    if args.fusion == "concat":
        V = concat_fusion(Vt, Vi, Vm, weights=(wt, wi, wm))
    else:
        V = weighted_sum_fusion(Vt, Vi, Vm, weights=(wt, wi, wm))

    # Build FAISS cosine index
    index, faiss = build_index(V)

    # Output naming
    if args.variant:
        # Optional custom variant still gets dataset prefix and items_*
        base = f"items_{args.dataset}_{args.variant}"
        tip_name = f"{args.dataset}_{args.variant}"
    else:
        if args.fusion == "concat":
            base = f"items_{args.dataset}_concat"
            tip_name = f"{args.dataset}_concat"
            if args.set_default:
                _write_defaults(proc_dir, "concat", (wt, wi, wm), args.k, tip_name)
        else:
            # Always write a clean dataset-weighted name (no weight triple in filename)
            base = f"items_{args.dataset}_weighted"
            tip_name = f"{args.dataset}_weighted"
            if args.set_default:
                _write_defaults(proc_dir, "weighted", (wt, wi, wm), args.k, tip_name)
    faiss_fp = out_dir / f"{base}.faiss"
    ids_fp   = out_dir / f"{base}.npy"

    faiss.write_index(index, str(faiss_fp))
    np.save(ids_fp, np.array(item_ids, dtype=object))  # robust to str lengths

    print(f"✅ Built FAISS index: {faiss_fp}")
    print(f"✅ Saved item id mapping: {ids_fp}")
    print(f"Items: {len(item_ids)} | Dim: {V.shape[1]}")
    print(f"Modalities in index → text:✓  image:{'✓' if has_img else '×'}  meta:{'✓' if has_meta else '×'}")

    if chosen_metrics is not None:
        print(f"🏆 Chosen (k={args.k}) → {chosen_metrics}")

    print(f"Tip: Pass this base name to API/UI as --faiss_name: {tip_name}")


if __name__ == "__main__":
    main()