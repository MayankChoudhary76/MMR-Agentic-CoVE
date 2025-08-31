#!/usr/bin/env python3
# scripts/tune_weights.py
from __future__ import annotations
import argparse, itertools, json, subprocess, sys, shlex
from pathlib import Path

from src.data.registry import get_paths

def _resolve_paths(dataset: str):
    paths = get_paths(dataset)
    raw = None; proc = None
    if isinstance(paths, dict):
        raw = paths.get("raw") or paths.get("raw_dir") or paths.get("raw_path")
        proc = paths.get("processed") or paths.get("processed_dir") or paths.get("processed_path")
    elif isinstance(paths, (tuple, list)) and len(paths) >= 2:
        raw, proc = paths[0], paths[1]
    else:
        raw = getattr(paths, "raw", None) or getattr(paths, "raw_dir", None) or getattr(paths, "raw_path", None)
        proc = getattr(paths, "processed", None) or getattr(paths, "processed_dir", None) or getattr(paths, "processed_path", None)

    if not raw or not proc:
        raw = Path("data/raw") / dataset
        proc = Path("data/processed") / dataset
    return Path(raw), Path(proc)

def _fmt(x: float) -> str:
    return str(x).replace(".", "p")

def _auto_faiss_name(dataset: str, fusion: str, wt: float, wi: float, wm: float) -> str:
    return f"{dataset}_{fusion}_wt{_fmt(wt)}_wi{_fmt(wi)}_wm{_fmt(wm)}"

def _run_eval(dataset: str, fusion: str, wt: float, wi: float, wm: float, k: int, run_name: str) -> dict:
    cmd = [
        sys.executable, "scripts/eval_fusion.py",
        "--dataset", dataset,
        "--fusion", fusion,
        "--w_text", str(wt),
        "--w_image", str(wi),
        "--w_meta",  str(wm),
        "--k", str(k),
        "--run_name", run_name,
    ]
    print("▶", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1])
    # read JSON row that eval_fusion wrote
    with open(Path("logs") / f"{run_name}.json", "r") as f:
        return json.load(f)

def _build_faiss(dataset: str, fusion: str, wt: float, wi: float, wm: float, variant: str):
    cmd = [
        sys.executable, "scripts/build_faiss.py",
        "--dataset", dataset,
        "--fusion", fusion,
        "--w_text", str(wt),
        "--w_image", str(wi),
        "--w_meta",  str(wm),
        "--variant", variant,
    ]
    print("▶", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1])

def main():
    ap = argparse.ArgumentParser(description="Grid-search weights and set tuned FAISS default.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fusion", choices=["concat","weighted"], default="weighted")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", default="ndcg@10", help="metric column in eval JSON (e.g., ndcg@10, hit@10)")
    ap.add_argument("--w_text",  default="1",   help="comma list, e.g., 0.5,1")
    ap.add_argument("--w_image", default="0,1", help="comma list")
    ap.add_argument("--w_meta",  default="0,0.1,0.2", help="comma list")
    args = ap.parse_args()

    _, proc = _resolve_paths(args.dataset)
    (proc / "index").mkdir(parents=True, exist_ok=True)

    wts  = [float(x) for x in str(args.w_text).split(",") if x.strip()]
    wis  = [float(x) for x in str(args.w_image).split(",") if x.strip()]
    wms  = [float(x) for x in str(args.w_meta).split(",") if x.strip()]

    best = None
    rows = []
    for wt, wi, wm in itertools.product(wts, wis, wms):
        run_name = f"tune_{args.fusion}_wt{_fmt(wt)}_wi{_fmt(wi)}_wm{_fmt(wm)}_k{args.k}"
        res = _run_eval(args.dataset, args.fusion, wt, wi, wm, args.k, run_name)
        score = res.get(args.metric)
        rows.append((score, wt, wi, wm, run_name))
        if best is None or (score is not None and score > best[0]):
            best = (score, wt, wi, wm, run_name)

    if best is None:
        raise RuntimeError("No results produced by tuning sweep.")

    score, wt, wi, wm, run_name = best
    variant = f"{args.fusion}_wt{_fmt(wt)}_wi{_fmt(wi)}_wm{_fmt(wm)}"
    faiss_name = _auto_faiss_name(args.dataset, args.fusion, wt, wi, wm)

    print("\n🏆 Best config:")
    print(json.dumps({
        "dataset": args.dataset, "fusion": args.fusion, "k": args.k,
        "w_text": wt, "w_image": wi, "w_meta": wm,
        "metric": args.metric, "score": score,
        "faiss_name": faiss_name
    }, indent=2))

    # Build FAISS for the best config
    _build_faiss(args.dataset, args.fusion, wt, wi, wm, variant)

    # Persist default
    default_fp = proc / "index" / "default_faiss.json"
    with open(default_fp, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "fusion": args.fusion,
            "k": args.k,
            "w_text": wt,
            "w_image": wi,
            "w_meta": wm,
            "metric": args.metric,
            "score": score,
            "faiss_name": faiss_name
        }, f, indent=2)
    print(f"✅ Wrote tuned default → {default_fp}")

if __name__ == "__main__":
    main()