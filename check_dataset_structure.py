#!/usr/bin/env python3
from pathlib import Path

BASE = Path("/notebooks/MMR-Agentic-CoVE/data/processed")
DATASETS = ["beauty"]  # extend with other datasets when you add them

expected_files = [
    "user_text_emb.parquet",
    "item_text_emb.parquet",   # required
    "items_with_meta.parquet", # preferred
    "reviews.parquet",         # optional but useful
]

def check_dataset(ds: str):
    ds_path = BASE / ds
    print(f"\n🔍 Checking dataset: {ds}")
    if not ds_path.exists():
        print(f"  ❌ Missing folder: {ds_path}")
        return

    # Check for unexpected nested dataset folder (like beauty/beauty)
    nested = ds_path / ds
    if nested.exists():
        print(f"  ⚠️ Found nested folder: {nested} — should delete if empty!")

    # Check expected files
    for fname in expected_files:
        fp = ds_path / fname
        if fp.exists():
            print(f"  ✅ Found {fname}")
        else:
            print(f"  ⚠️ Missing {fname}")

    # Check FAISS index
    index_dir = ds_path / "index"
    if not index_dir.exists():
        print(f"  ⚠️ Missing index folder: {index_dir}")
    else:
        faiss_files = list(index_dir.glob("items_*.faiss"))
        npy_files = list(index_dir.glob("items_*.npy"))
        if faiss_files and npy_files:
            print(f"  ✅ Found {len(faiss_files)} FAISS file(s) and {len(npy_files)} NPY file(s)")
        else:
            print("  ⚠️ Index folder present but missing .faiss or .npy files")

if __name__ == "__main__":
    for ds in DATASETS:
        check_dataset(ds)