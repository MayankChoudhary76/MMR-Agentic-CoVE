# src/utils/metrics.py
from __future__ import annotations
import csv, os, time, subprocess
from pathlib import Path
from typing import Dict, Any

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"

def append_metrics(row: Dict[str, Any], csv_path: str | Path = "logs/metrics.csv", no_log: bool=False):
    """
    Append a single row of metrics/metadata to logs/metrics.csv.
    - Automatically adds timestamp + git_sha if missing.
    - Creates header on first write.
    """
    if no_log:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(row)  # shallow copy
    row.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    row.setdefault("git_sha", _git_sha())

    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)