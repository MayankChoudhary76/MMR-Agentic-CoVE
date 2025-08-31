from pathlib import Path
import yaml

def load_config(cfg_path: str | Path) -> dict:
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)
    
__all__ = ["load_config"]