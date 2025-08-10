# src/data/registry.py
from pathlib import Path
from src.utils.paths import DATA_DIR

# Central catalog of datasets -> raw/processed roots
DATA_REGISTRY = {
    "beauty": {
        "raw": DATA_DIR / "raw" / "beauty",
        "processed": DATA_DIR / "processed" / "beauty",
    },
    # Add more datasets here later (e.g., "toys", "electronics", ...)
}

def get_dataset_paths(dataset_name: str) -> dict:
    """
    Return dict with 'raw' and 'processed' paths.
    Ensures directories exist so later code can write safely.
    """
    key = dataset_name.lower()
    if key not in DATA_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry. "
                         f"Known: {list(DATA_REGISTRY.keys())}")

    paths = DATA_REGISTRY[key]

    # Auto-create directories (idempotent)
    for k in ("raw", "processed"):
        Path(paths[k]).mkdir(parents=True, exist_ok=True)

    return paths