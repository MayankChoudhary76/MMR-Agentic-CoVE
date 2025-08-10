#!/usr/bin/env python3
# Evaluate multimodal fusion baseline (text + image) + optional report & plots.
#
# Inputs (must exist):
#   data/processed/<dataset>/reviews.parquet
#   data/processed/<dataset>/item_text_emb.parquet   (item_id, vector[Dt])
#   data/processed/<dataset>/user_text_emb.parquet   (user_id, vector[Dt])
#   data/processed/<dataset>/item_image_emb.parquet  (item_id, vector[Di])
#
# Output:
#   logs/<run_name>_eval.json
#   logs/<run_name>_report.json                    (if --report)
#   logs/metrics.csv                                (append)
#   logs/plots/<dataset>_k<k>_comparison.png       (if --plot)
#   logs/plots/<dataset>_k<k>_trend.png            (if --plot)
#
# Examples:
#   PYTHONPATH=$(pwd) python scripts/eval_fusion.py --dataset beauty --fusion concat --k 10 --run_name beauty_fusion_concat
#   PYTHONPATH=$(pwd) python scripts/eval_fusion.py --dataset beauty --fusion weighted --alpha 0.7 --k 10 --run_name beauty_fusion_wsum_a0p7 --report --plot

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from src.data.registry import get_paths
from src.models.fusion import concat_fusion, weighted_sum_fusion
from src.utils.paths import LOGS_DIR


# ------------------------ utilities ------------------------ #

def _resolve_processed_dir(dataset: str) -> Path:
    paths = get_paths(dataset)
    if isinstance(paths, dict) and "processed" in paths:
        return Path(paths["processed"])
    if isinstance(paths, (tuple, list)) and len(paths) >= 2:
        return Path(paths[1])
    return Path("data/processed") / dataset


def make_loo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-out split per user: last interaction as test, rest as train."""
    df = df.sort_values(["user_id", "timestamp"])
    test = df.groupby("user_id", as_index=False).tail(1)
    train = pd.concat([df, test]).drop_duplicates(keep=False)
    return train, test


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between row-wise vectors in a (U x D) and b (I x D)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def evaluate_topk(user_vecs_df: pd.DataFrame,
                  item_vec: np.ndarray,
                  item_ids: List[str],
                  test: pd.DataFrame,
                  k: int) -> Dict[str, float]:
    """
    Compute Hit@K and NDCG@K over users that exist in user_vecs_df and test.
    user_vecs_df: DataFrame with ['user_id','vector'] where vector is np.ndarray
    """
    true_item = test.set_index("user_id")["item_id"].to_dict()
    eval_users = [u for u in user_vecs_df["user_id"].tolist() if u in true_item]
    if not eval_users:
        return {f"hit@{k}": 0.0, f"ndcg@{k}": 0.0}

    U = np.vstack(user_vecs_df.set_index("user_id").loc[eval_users, "vector"].to_numpy()).astype(np.float32)
    S = _cosine_sim(U, item_vec)  # (|eval_users| x |items|)
    item_ids_arr = np.array(item_ids)

    hits = 0
    ndcgs = 0.0
    for i, u in enumerate(eval_users):
        target = true_item[u]
        order = np.argsort(-S[i])             # descending
        topk = item_ids_arr[order[:k]]
        if target in topk:
            hits += 1
            # position in the full ranking (1-based)
            rank = int(np.where(item_ids_arr[order] == target)[0][0]) + 1
            ndcgs += 1.0 / math.log2(rank + 1)

    n = max(1, len(eval_users))
    return {f"hit@{k}": hits / n, f"ndcg@{k}": ndcgs / n}


def append_metrics_csv(dataset: str,
                       run_name: str,
                       model: str,
                       k: int,
                       metrics: Dict[str, float],
                       notes: str = "fusion") -> None:
    """Append a row of metrics to logs/metrics.csv, creating the file if missing."""
    from datetime import datetime
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "dataset": dataset,
        "run_name": run_name,
        "model": model,
        "k": k,
        "hit": float(metrics.get(f"hit@{k}", 0.0)),
        "ndcg": float(metrics.get(f"ndcg@{k}", 0.0)),
        "notes": notes,
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOGS_DIR / "metrics.csv"
    write_header = not csv_path.exists()
    import csv as _csv
    with open(csv_path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"🧾 Appended → {csv_path}")


# ------------------------ plotting helpers ------------------------ #

def _plot_metrics(dataset: str, k: int):
    import matplotlib.pyplot as plt

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOGS_DIR / "metrics.csv"
    if not csv_path.exists():
        print("No logs/metrics.csv found; skipping plots.")
        return

    df = pd.read_csv(csv_path)
    df = df[(df["dataset"] == dataset) & (df["k"] == k)]
    if df.empty:
        print(f"No matching rows in metrics.csv for dataset={dataset}, k={k}.")
        return

    plots_dir = LOGS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Comparison bar chart: latest row per run_name
    latest = df.sort_values("timestamp").groupby("run_name").tail(1)
    latest = latest.sort_values("ndcg", ascending=False)

    # Bar: NDCG@K by run_name
    plt.figure(figsize=(10, 5))
    plt.bar(latest["run_name"], latest["ndcg"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"NDCG@{k}")
    plt.title(f"{dataset}: NDCG@{k} by run")
    plt.tight_layout()
    comp_path = plots_dir / f"{dataset}_k{k}_comparison.png"
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"📈 Saved comparison plot → {comp_path}")

    # Trend line: timestamp vs NDCG for each run
    plt.figure(figsize=(10, 5))
    for name, g in df.sort_values("timestamp").groupby("run_name"):
        plt.plot(pd.to_datetime(g["timestamp"]), g["ndcg"], marker="o", label=name)
    plt.legend()
    plt.ylabel(f"NDCG@{k}")
    plt.title(f"{dataset}: NDCG@{k} over time")
    plt.tight_layout()
    trend_path = plots_dir / f"{dataset}_k{k}_trend.png"
    plt.savefig(trend_path, dpi=150)
    plt.close()
    print(f"📈 Saved trend plot → {trend_path}")


# ------------------------ main ------------------------ #

def main(args):
    proc = _resolve_processed_dir(args.dataset)

    # 1) interactions
    reviews_fp = proc / "reviews.parquet"
    if not reviews_fp.exists():
        raise FileNotFoundError(f"Missing {reviews_fp}. Run normalization first.")
    df = pd.read_parquet(reviews_fp)
    for col in ("user_id", "item_id", "timestamp"):
        if col not in df.columns:
            raise ValueError("reviews.parquet missing required columns")
    train, test = make_loo(df)

    # 2) embeddings
    items_text = pd.read_parquet(proc / "item_text_emb.parquet")   # item_id, vector (384)
    users_text = pd.read_parquet(proc / "user_text_emb.parquet")   # user_id, vector (384)
    items_img  = pd.read_parquet(proc / "item_image_emb.parquet")  # item_id, vector (512)

    # 3) align items + zero-fill missing images
    item_ids: List[str] = items_text["item_id"].tolist()
    Vt = np.vstack(items_text["vector"].to_numpy()).astype(np.float32)

    img_map: Dict[str, np.ndarray] = {}
    img_dim = None
    for _, row in items_img.iterrows():
        v = row["vector"]
        if isinstance(v, np.ndarray):
            if img_dim is None:
                img_dim = int(v.shape[0])
            img_map[row["item_id"]] = v.astype(np.float32)
    if img_dim is None:
        img_dim = 512

    Vi_list = []
    nonzero = 0
    for iid in item_ids:
        vec = img_map.get(iid)
        if isinstance(vec, np.ndarray) and vec.size == img_dim:
            Vi_list.append(vec)
            if np.linalg.norm(vec) > 0:
                nonzero += 1
        else:
            Vi_list.append(np.zeros((img_dim,), dtype=np.float32))
    Vi = np.vstack(Vi_list).astype(np.float32)

    # 4) fuse ITEM vectors
    if args.fusion == "concat":
        Vf_items = concat_fusion(Vt, Vi)
        model_tag = "concat(text+image)"
        fused_dim = Vf_items.shape[1]
    else:
        Vf_items = weighted_sum_fusion(Vt, Vi, alpha=args.alpha)
        model_tag = f"wsum(a={args.alpha})"
        fused_dim = Vf_items.shape[1]

    # 5) build USER image vectors from TRAIN interactions (mean of item image vecs per user)
    # Map item_id -> row index in items_text
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}

    user_img_vecs: Dict[str, List[np.ndarray]] = {}
    for _, row in train.iterrows():
        uid = row["user_id"]
        iid = row["item_id"]
        idx = item_index.get(iid)
        if idx is None:
            continue
        v_img = Vi[idx]
        if np.linalg.norm(v_img) == 0:
            continue
        user_img_vecs.setdefault(uid, []).append(v_img)

    def _mean_or_zero(vecs: List[np.ndarray], d: int) -> np.ndarray:
        if not vecs:
            return np.zeros((d,), dtype=np.float32)
        m = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        n = np.linalg.norm(m) + 1e-12
        return (m / n).astype(np.float32)

    user_ids = users_text["user_id"].tolist()
    U_text = np.vstack(users_text["vector"].to_numpy()).astype(np.float32)
    U_img = np.vstack([_mean_or_zero(user_img_vecs.get(u, []), img_dim) for u in user_ids]).astype(np.float32)

    # fuse USER vectors using same method
    if args.fusion == "concat":
        Uf_users = concat_fusion(U_text, U_img)
    else:
        Uf_users = weighted_sum_fusion(U_text, U_img, alpha=args.alpha)

    # 6) evaluate users (fused) vs items (fused)
    metrics = evaluate_topk(
        pd.DataFrame({"user_id": user_ids, "vector": list(Uf_users)}),
        Vf_items,
        item_ids,
        test,
        k=args.k,
    )
    print("📊 Metrics:", json.dumps(metrics, indent=2))

    # 7) persist results
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.dataset}_fusion_{args.fusion}"
    with open(LOGS_DIR / f"{run_name}_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    append_metrics_csv(args.dataset, run_name, model_tag, args.k, metrics, notes="fusion")

    # 8) optional report
    if args.report:
        report = {
            "dataset": args.dataset,
            "fusion": args.fusion,
            "alpha": args.alpha if args.fusion == "weighted" else None,
            "k": args.k,
            "items_total": int(len(item_ids)),
            "image_nonzero": int(nonzero),
            "image_coverage": float(nonzero / max(1, len(item_ids))),
            "Dt_text": int(Vt.shape[1]),
            "Di_image": int(Vi.shape[1]),
            "Df_fused": int(fused_dim),
            "metrics": metrics,
        }
        with open(LOGS_DIR / f"{run_name}_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("📝 Report:", json.dumps(report, indent=2))

    # 9) optional plots
    if args.plot:
        _plot_metrics(args.dataset, args.k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset key (e.g., beauty)")
    ap.add_argument("--fusion", choices=["concat", "weighted"], default="concat",
                    help="concat = [text||image]; weighted = alpha*text + (1-alpha)*image (if same dims)")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight for text when fusion=weighted")
    ap.add_argument("--k", type=int, default=10, help="K for Hit@K / NDCG@K")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--report", action="store_true", help="print & save coverage/dim report JSON")
    ap.add_argument("--plot", action="store_true", help="generate comparison & trend plots from logs/metrics.csv")
    args = ap.parse_args()
    main(args)