# scripts/build_text_emb.py
"""
Builds baseline TEXT embeddings for items & users from data/processed/<dataset>/reviews.parquet
and runs a quick leave-one-out evaluation (Hit@K, nDCG@K).

Item embeddings  = mean of all review texts for that item
User embeddings  = mean of all review texts (train split) for that user

Outputs:
  data/processed/<dataset>/item_text_emb.parquet
  data/processed/<dataset>/user_text_emb.parquet
  logs/<dataset>_text_emb_eval.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DIR, LOGS_DIR
from src.data.registry import get_paths
from src.models.text_encoder import TextEncoder


def _mean_pool(group_texts, encoder: TextEncoder, batch_size: int = 256) -> np.ndarray:
    """Encode a list of texts and return the mean vector."""
    if len(group_texts) == 0:
        return np.zeros(encoder.dim, dtype=np.float32)
    emb = encoder.encode(group_texts, batch_size=batch_size)
    return emb.mean(axis=0)


def _build_item_embeddings(df: pd.DataFrame, encoder: TextEncoder, batch_size: int) -> pd.DataFrame:
    # item text = concat of all review texts for that item
    agg = df.groupby("item_id")["text"].apply(lambda s: " ".join(map(str, s))).reset_index()
    # encode in chunks to avoid huge memory
    vectors = []
    chunk = 2048
    for i in range(0, len(agg), chunk):
        texts = agg.iloc[i:i+chunk]["text"].tolist()
        vectors.append(encoder.encode(texts, batch_size=batch_size))
    vectors = np.vstack(vectors)
    out = pd.DataFrame({"item_id": agg["item_id"].values, "vector": list(vectors)})
    return out


def _leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each user, keep the last interaction as test, rest as train."""
    df = df.sort_values("timestamp")
    last = df.groupby("user_id").tail(1)
    rest = pd.concat([df, last]).drop_duplicates(keep=False)
    return rest, last


def _build_user_embeddings(train_df: pd.DataFrame, encoder: TextEncoder, batch_size: int) -> pd.DataFrame:
    # user text = concat of all their review texts (train-only)
    agg = train_df.groupby("user_id")["text"].apply(lambda s: " ".join(map(str, s))).reset_index()
    vectors = []
    chunk = 2048
    for i in range(0, len(agg), chunk):
        texts = agg.iloc[i:i+chunk]["text"].tolist()
        vectors.append(encoder.encode(texts, batch_size=batch_size))
    vectors = np.vstack(vectors)
    out = pd.DataFrame({"user_id": agg["user_id"].values, "vector": list(vectors)})
    return out


def _cosine_topk(user_vecs: np.ndarray, item_vecs: np.ndarray, item_ids: np.ndarray, k: int = 10):
    """Returns top-k item_ids per user by cosine (vectors assumed L2-normalized)."""
    # dot product == cosine since vectors are normalized
    scores = user_vecs @ item_vecs.T
    topk_idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
    # reorder each row by actual score
    row_scores = np.take_along_axis(scores, topk_idx, axis=1)
    order = np.argsort(-row_scores, axis=1)
    sorted_idx = np.take_along_axis(topk_idx, order, axis=1)
    topk_item_ids = item_ids[sorted_idx]
    return topk_item_ids


def _metrics_hit_ndcg(topk_item_ids: np.ndarray, true_item_ids: np.ndarray) -> dict:
    """Hit@K and nDCG@K (single-positive per user)."""
    K = topk_item_ids.shape[1]
    hits = []
    ndcgs = []
    for row, true in zip(topk_item_ids, true_item_ids):
        if true in row:
            hits.append(1.0)
            rank = np.where(row == true)[0][0] + 1  # 1-based
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
    return {"hit@{}".format(K): float(np.mean(hits)),
            "ndcg@{}".format(K): float(np.mean(ndcgs))}


def main(dataset: str, model_name: str, batch_size: int, k: int, limit_users: int | None):
    paths = get_paths(dataset)
    processed_dir = PROCESSED_DIR / dataset
    processed_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    reviews_path = processed_dir / "reviews.parquet"
    if not reviews_path.exists():
        raise FileNotFoundError(f"{reviews_path} not found. Run scripts/normalize.py first.")

    df = pd.read_parquet(reviews_path)
    if limit_users:
        keep_users = df["user_id"].drop_duplicates().sample(n=min(limit_users, df["user_id"].nunique()), random_state=42)
        df = df[df["user_id"].isin(keep_users)]

    # Build encoder
    encoder = TextEncoder(model_name=model_name)

    # Build item embeddings
    print("🔹 Building item text embeddings …")
    item_df = _build_item_embeddings(df, encoder, batch_size=batch_size)
    item_df.to_parquet(processed_dir / "item_text_emb.parquet", index=False)
    print(f"✅ Saved item vectors → {processed_dir/'item_text_emb.parquet'}  (dim={encoder.dim})")

    # Split and build user embeddings on train side only
    print("🔹 Leave-one-out split for users …")
    train_df, test_df = _leave_one_out_split(df)

    print("🔹 Building user text embeddings (train interactions only) …")
    user_df = _build_user_embeddings(train_df, encoder, batch_size=batch_size)
    user_df.to_parquet(processed_dir / "user_text_emb.parquet", index=False)
    print(f"✅ Saved user vectors → {processed_dir/'user_text_emb.parquet'}")

    # Eval: recommend topK for each test user
    print("🔹 Running quick evaluation …")
    # align arrays
    item_ids = item_df["item_id"].values
    item_mat = np.vstack(item_df["vector"].values)   # (I, d)

    # users that exist in user_df and appear in test
    test = test_df.merge(user_df[["user_id"]], on="user_id", how="inner")
    test = test.merge(item_df[["item_id"]], on="item_id", how="inner")
    test = test.merge(user_df, on="user_id", how="left")

    user_mat = np.vstack(test["vector"].values)      # (U, d)
    topk = _cosine_topk(user_mat, item_mat, item_ids=item_ids, k=k)
    metrics = _metrics_hit_ndcg(topk, test["item_id"].values)

    log_path = LOGS_DIR / f"{dataset}_text_emb_eval.json"
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("📊 Metrics:", json.dumps(metrics, indent=2))
    print(f"📝 Saved → {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beauty")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--k", type=int, default=10, help="Top-K for eval")
    ap.add_argument("--limit_users", type=int, default=None, help="Debug: limit #users")
    args = ap.parse_args()
    main(**vars(args))