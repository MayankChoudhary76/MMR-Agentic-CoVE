from pathlib import Path
import argparse, urllib.request, urllib.error, gzip, shutil, time
from src.utils.config import load_config
from src.data.registry import get_dataset_paths

def fetch_one(url: str, dest: Path, timeout=60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        print(f"↓ Trying: {url}")
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out)
        if url.endswith(".gz"):
            print(f"↪ Decompressing → {dest}")
            with gzip.open(tmp, "rb") as fin, open(dest, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            tmp.unlink()
        else:
            tmp.replace(dest)
        print(f"✅ Saved: {dest}")
        return True
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} on {url}")
    except Exception as e:
        print(f"Error on {url}: {e}")
    try:
        if tmp.exists(): tmp.unlink()
    except Exception:
        pass
    return False

def fetch_any(urls: list[str], dest: Path) -> None:
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Download attempt")
        if fetch_one(url, dest):
            return
        time.sleep(1)
    raise RuntimeError(f"All mirrors failed for {dest.name}")

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    reg_key = cfg.get("registry_key", cfg.get("dataset"))
    paths = get_dataset_paths(reg_key)
    raw_dir = Path(paths["raw"])

    for item in cfg.get("download_urls", []):
        name = item["name"]
        target = raw_dir / item["target"]
        mirrors = item.get("targets") or [item.get("url")]
        print(f"\n=== {name} ===")
        fetch_any(mirrors, target)

    print(f"\n✅ Done. Raw files in: {raw_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)