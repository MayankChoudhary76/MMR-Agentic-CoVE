from pathlib import Path
from typing import Union, Dict

# --- project roots ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR     = DATA_DIR / "cache"
LOGS_DIR      = PROJECT_ROOT / "logs"
MODELS_DIR    = PROJECT_ROOT / "src" / "models"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists. Accepts either a str or a pathlib.Path.
    Returns a pathlib.Path.
    """
    p = Path(path) if not isinstance(path, Path) else path
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_raw_path(dataset: str) -> Path:
    """.../data/raw/<dataset>"""
    return ensure_dir(RAW_DIR / dataset)


def get_processed_path(dataset: str) -> Path:
    """.../data/processed/<dataset>"""
    return ensure_dir(PROCESSED_DIR / dataset)


def get_logs_path() -> Path:
    """.../logs"""
    return ensure_dir(LOGS_DIR)


def get_dataset_paths(dataset: str) -> Dict[str, Path]:
    """
    Convenience bundle of dataset-related paths.
    NOTE: returns Path objects (not strings) for consistency.
    """
    return {
        "raw": get_raw_path(dataset),
        "processed": get_processed_path(dataset),
        "cache": ensure_dir(CACHE_DIR / dataset),
        "logs": get_logs_path(),
    }


if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Data folder:", DATA_DIR)
    print("Raw:", get_raw_path("beauty"))
    print("Processed:", get_processed_path("beauty"))
    print("Logs:", get_logs_path())


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "CACHE_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "get_raw_path",
    "get_processed_path",
    "get_logs_path",
    "get_dataset_paths",
    "ensure_dir",
]