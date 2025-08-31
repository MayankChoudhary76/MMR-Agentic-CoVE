#!/usr/bin/env python3
# scripts/normalize.py
from __future__ import annotations

import argparse
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.data.loader import normalize_dataset


def _utc_ts() -> str:
    # ISO-8601 with milliseconds, always UTC (Z)
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _get(stats: Dict[str, Any], *candidates: str) -> Any:
    """Pick the first present non-None key in stats (helps with different schemas)."""
    for k in candidates:
        if k in stats and stats[k] is not None:
            return stats[k]
    return None


def main():
    ap = argparse.ArgumentParser(description="Normalize a dataset and log normalization stats.")
    ap.add_argument("--dataset", default="beauty", help="Dataset key (e.g., beauty)")
    args = ap.parse_args()

    dataset = args.dataset
    raw_dir = Path("data/raw") / dataset
    out_dir = Path("data/processed") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- run normalization (your loader does the heavy lifting) ----
    stats: Dict[str, Any] = normalize_dataset(raw_dir, out_dir)

    reviews_fp = (out_dir / "reviews.parquet").resolve()

    print("✅ Saved:", reviews_fp)
    print("📊 Stats:", json.dumps(stats, indent=2))

    # ---- logging (CSV + JSON snapshot) ----
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = _utc_ts()

    # try to populate common counters if present; otherwise leave None
    n_users = _get(stats, "users", "n_users", "user_count")
    n_items = _get(stats, "items", "n_items", "item_count")
    n_interactions = _get(stats, "interactions", "n_interactions", "events", "rows", "ratings")

    row = {
        "timestamp": ts,
        "dataset": dataset,
        "reviews_parquet": str(reviews_fp),
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        # keep full stats as compact JSON for auditability
        "stats_json": json.dumps(stats, separators=(",", ":"), ensure_ascii=False),
    }

    csv_fp = logs_dir / "normalize.csv"
    header = ["timestamp", "dataset", "reviews_parquet", "n_users", "n_items", "n_interactions", "stats_json"]

    # append robustly; write header if file does not yet exist
    with csv_fp.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not csv_fp.exists() or csv_fp.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)

    # also write a per-run JSON snapshot for easy diffing
    snap_dir = logs_dir / "normalize"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_fp = snap_dir / f"{dataset}_{ts.replace(':','').replace('.','_')}.json"
    snap_fp.write_text(json.dumps(row, indent=2), encoding="utf-8")

    print(f"🧾 Appended → {csv_fp}")
    print(f"📝 Snapshot → {snap_fp}")


if __name__ == "__main__":
    main()