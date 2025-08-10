#!/usr/bin/env python3
# Build item IMAGE embeddings (CLIP) with caching & retries.
# Input:  data/processed/<dataset>/items_with_meta.parquet  (must have: item_id, image_url)
# Output: data/processed/<dataset>/item_image_emb.parquet  (item_id, vector)

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.image_encoder import ImageEncoder
from src.utils.paths import get_dataset_paths, ensure_dir


def _download_to_cache(urls: List[str], cache_dir: Path, retries: int = 2, sleep: float = 0.5) -> List[Path]:
    """
    Download URLs to cache dir. Returns list of file paths (may be missing if failed).
    We keep filenames as sequential indices for simplicity.
    """
    from PIL import Image
    import requests
    import io

    cache_dir = ensure_dir(cache_dir)
    out_paths: List[Path] = []
    for i, url in enumerate(tqdm(urls, desc="Downloading")):
        fp = cache_dir / f"{i:06d}.jpg"
        if fp.exists() and fp.stat().st_size > 0:
            out_paths.append(fp)
            continue

        ok = False
        for _ in range(retries + 1):
            try:
                if not url or not isinstance(url, str):
                    break
                r = requests.get(url, timeout=10, stream=True)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img.save(fp, format="JPEG", quality=90, optimize=True)
                ok = True
                break
            except Exception:
                time.sleep(sleep)
                continue
        if not ok:
            # create empty placeholder so indexing stays aligned
            if not fp.exists():
                fp.touch()
        out_paths.append(fp)
    return out_paths


def _load_pils(filepaths: List[Path]):
    """Load PIL images from cached files (None for failures)."""
    from PIL import Image, UnidentifiedImageError
    pils = []
    for fp in filepaths:
        try:
            img = Image.open(fp).convert("RGB")
            pils.append(img)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
            pils.append(None)
    return pils


def main(dataset: str, batch_size: int, model_name: str, pretrained: str):
    paths = get_dataset_paths(dataset)
    proc_dir = ensure_dir(paths["processed"])

    items_fp = proc_dir / "items_with_meta.parquet"
    if not items_fp.exists():
        raise FileNotFoundError(f"Missing {items_fp}. Run scripts/join_meta.py first.")

    items = pd.read_parquet(items_fp)
    if "item_id" not in items.columns or "image_url" not in items.columns:
        raise ValueError("items_with_meta.parquet must have columns: item_id, image_url")

    item_ids = items["item_id"].tolist()
    urls = items["image_url"].fillna("").astype(str).tolist()

    cache_dir = Path("data/cache/images") / dataset
    filepaths = _download_to_cache(urls, cache_dir=cache_dir)

    pils = _load_pils(filepaths)
    ok_mask = [im is not None for im in pils]
    cov = float(np.mean(ok_mask))
    print(f"🖼️ Cached images: {sum(ok_mask)}/{len(ok_mask)} ({cov:.1%})")

    # Encode
    enc = ImageEncoder(model_name=model_name, pretrained=pretrained)
    # replace None with white blank to keep shape; encode_pils will handle
    from PIL import Image
    blank = Image.new("RGB", (224, 224), color=(255, 255, 255))
    safe_pils = [im if im is not None else blank for im in pils]

    vecs = enc.encode_pils(safe_pils, batch_size=batch_size, desc="CLIP encode")
    # zero-out failed ones (keep row count stable)
    vecs[np.logical_not(ok_mask)] = 0.0

    out = pd.DataFrame({"item_id": item_ids, "vector": [v.astype(np.float32) for v in vecs]})
    out_fp = proc_dir / "item_image_emb.parquet"
    out.to_parquet(out_fp, index=False)
    print(f"✅ Saved image vectors → {out_fp}  (dim={vecs.shape[1] if len(vecs) else 0})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset key (e.g., beauty)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = ap.parse_args()

    main(args.dataset, args.batch_size, args.model, args.pretrained)