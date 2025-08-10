from pathlib import Path
import argparse
import json

def count_jsonl(path: Path, limit=3):
    n = 0
    print(f"\nSample rows from {path.name}:")
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if n <= limit:
                try:
                    obj = json.loads(line)
                    print({k: obj.get(k) for k in ("reviewerID", "asin", "overall", "summary")})
                except Exception:
                    print(line.strip()[:200])
    return n

def count_meta(path: Path, limit=3):
    return count_jsonl(path, limit)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beauty")
    args = ap.parse_args()

    raw_dir = Path("data/raw") / args.dataset
    reviews = raw_dir / "reviews.json"
    meta = raw_dir / "meta.json"

    print("Raw dir:", raw_dir.resolve())
    n_reviews = count_jsonl(reviews)
    n_meta = count_meta(meta)
    print(f"\n✅ Stats — reviews: {n_reviews:,} | metadata: {n_meta:,}")

if __name__ == "__main__":
    main()