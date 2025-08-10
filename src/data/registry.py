# src/data/registry.py
from pathlib import Path
from typing import Dict
from src.utils.paths import RAW_DIR, PROCESSED_DIR

def get_paths(dataset: str) -> Dict[str, Path]:
    """
    Return raw and processed directories for a dataset name.
    Creates them if they do not exist.
    """
    name = dataset.lower()
    raw_dir = RAW_DIR / name
    processed_dir = PROCESSED_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"raw_dir": raw_dir, "processed_dir": processed_dir}

# (optional helpers if you want explicit file paths later)
def raw_file(dataset: str, filename: str) -> Path:
    return get_paths(dataset)["raw_dir"] / filename

def processed_file(dataset: str, filename: str) -> Path:
    return get_paths(dataset)["processed_dir"] / filename