from pathlib import Path
import argparse, json
from src.data.loader import normalize_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beauty")
    args = ap.parse_args()

    raw = Path("data/raw") / args.dataset
    out = Path("data/processed") / args.dataset
    stats = normalize_dataset(raw, out)

    print("✅ Saved:", (out / "reviews.parquet").resolve())
    print("📊 Stats:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()