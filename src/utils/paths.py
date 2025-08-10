from pathlib import Path

# --- project roots ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR     = DATA_DIR / "cache"
LOGS_DIR      = PROJECT_ROOT / "logs"
MODELS_DIR    = PROJECT_ROOT / "src" / "models"

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_raw_path(dataset: str) -> Path:
    """.../data/raw/<dataset>"""
    return ensure_dir(RAW_DIR / dataset)

def get_processed_path(dataset: str) -> Path:
    """.../data/processed/<dataset>"""
    return ensure_dir(PROCESSED_DIR / dataset)

def get_logs_path() -> Path:
    """.../logs"""
    return ensure_dir(LOGS_DIR)

# Optional: convenience for both
def get_dataset_paths(dataset: str) -> dict:
    return {
        "raw": str(get_raw_path(dataset)),
        "processed": str(get_processed_path(dataset)),
        "cache": str(ensure_dir(CACHE_DIR / dataset)),
        "logs": str(get_logs_path()),
    }

if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Data folder:", DATA_DIR)
    print("Raw:", get_raw_path("beauty"))
    print("Processed:", get_processed_path("beauty"))
    print("Logs:", get_logs_path())