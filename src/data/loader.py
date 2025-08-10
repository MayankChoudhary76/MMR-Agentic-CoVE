from pathlib import Path
import json
import pandas as pd

STANDARD_COLS = ["user_id", "item_id", "text", "rating", "timestamp"]

def read_reviews_json(path: Path) -> pd.DataFrame:
    """Read Amazon reviews (JSON lines) and map to a unified schema."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "user_id":   obj.get("reviewerID"),
                "item_id":   obj.get("asin"),
                "text":      " ".join([t for t in [obj.get("summary"), obj.get("reviewText")] if t]),
                "rating":    obj.get("overall"),
                "timestamp": obj.get("unixReviewTime"),
            })
    df = pd.DataFrame(rows, columns=STANDARD_COLS)
    # basic cleaning
    df = df.dropna(subset=["user_id", "item_id"]).reset_index(drop=True)
    return df

def normalize_dataset(raw_dir: Path, out_dir: Path) -> dict:
    """Normalize raw Beauty reviews.json -> processed/reviews.parquet"""
    out_dir.mkdir(parents=True, exist_ok=True)
    reviews_df = read_reviews_json(Path(raw_dir) / "reviews.json")
    out_path = Path(out_dir) / "reviews.parquet"
    if out_path.exists():
        out_path.unlink()
    reviews_df.to_parquet(out_path, index=False)
    return {
        "rows": int(len(reviews_df)),
        "users": int(reviews_df["user_id"].nunique()),
        "items": int(reviews_df["item_id"].nunique()),
        "ratings_mean": float(reviews_df["rating"].mean()),
    }