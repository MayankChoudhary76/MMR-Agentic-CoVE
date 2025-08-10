from pathlib import Path

# Base project directory (two levels up from utils/paths.py)
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Standard data and logs directories
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_DIR / "logs"
MODELS_DIR = PROJECT_DIR / "models"

def ensure_dir(p):
    """
    Create directory if it does not exist.
    Returns the Path object for the directory.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

# Ensure base dirs exist
for d in [RAW_DIR, PROCESSED_DIR, LOGS_DIR, MODELS_DIR]:
    ensure_dir(d)

def get_dataset_paths(dataset_name):
    """
    Returns a dictionary of key paths for a given dataset.
    """
    base = PROCESSED_DIR / dataset_name
    ensure_dir(base)
    return {
        "raw": RAW_DIR / dataset_name,
        "processed": base,
        "logs": LOGS_DIR / dataset_name,
        "models": MODELS_DIR / dataset_name
    }