# src/data/registry.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

# Canonical path helpers live in utils.paths
from src.utils.paths import (
    RAW_DIR,
    PROCESSED_DIR,
    get_dataset_paths as _get_dataset_paths,  # returns dict[str, str]
    get_raw_path,
    get_processed_path,
)


def get_paths(dataset: str) -> Dict[str, Path]:
    """
    Return raw and processed directories for a dataset name (as Path objects).
    Creates them if they do not exist.

    Example:
        d = get_paths("beauty")
        d["raw_dir"] -> Path(.../data/raw/beauty)
        d["processed_dir"] -> Path(.../data/processed/beauty)
    """
    name = (dataset or "").lower()
    raw_dir = RAW_DIR / name
    processed_dir = PROCESSED_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"raw_dir": raw_dir, "processed_dir": processed_dir}


def raw_file(dataset: str, filename: str) -> Path:
    """Convenience: Path to a file inside data/raw/<dataset>/"""
    return get_paths(dataset)["raw_dir"] / filename


def processed_file(dataset: str, filename: str) -> Path:
    """Convenience: Path to a file inside data/processed/<dataset>/"""
    return get_paths(dataset)["processed_dir"] / filename


# ---------------------------------------------------------------------
# Backwards‑compat shim used by older code/tests:
# test_registry.py imports get_dataset_paths from src.data.registry.
# Delegate to src.utils.paths.get_dataset_paths and return strings.
# The dict contains: raw, processed, cache, logs
# ---------------------------------------------------------------------
def get_dataset_paths(dataset: str) -> Dict[str, str]:
    """
    Returns absolute paths (as strings) for the given dataset:
    {
      "raw": ".../data/raw/<dataset>",
      "processed": ".../data/processed/<dataset>",
      "cache": ".../data/cache/<dataset>",
      "logs": ".../logs"
    }
    """
    return _get_dataset_paths(dataset)


__all__ = [
    "get_paths",
    "raw_file",
    "processed_file",
    "get_dataset_paths",  # keep public for tests
    "get_raw_path",
    "get_processed_path",
]