from pathlib import Path

# Root of the repository (this file is in src/utils/, so go up two levels)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Core directories
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
CONFIGS_DIR = ROOT_DIR / "configs"
SCRIPTS_DIR = ROOT_DIR / "scripts"
DOCS_DIR = ROOT_DIR / "docs"

# Data subfolders
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don’t exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Project root:", ROOT_DIR)
    print("Data folder:", DATA_DIR)
    print("Configs folder:", CONFIGS_DIR)