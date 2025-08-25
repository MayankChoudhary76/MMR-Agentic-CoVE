#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

IGNORE_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".idea", ".vscode"}

def should_skip(dirpath: Path) -> bool:
    parts = dirpath.parts
    return any(p in IGNORE_DIRS or p.startswith(".") for p in parts)

def find_missing_inits(root: Path):
    missing = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        # prune ignored dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if not should_skip(p / d)]
        if should_skip(p):
            continue
        if p == root:
            # don't force __init__.py at src root
            continue
        if "__init__.py" not in filenames:
            missing.append(p)
    return missing

def ensure_init(path: Path):
    fp = path / "__init__.py"
    if not fp.exists():
        fp.write_text("# auto-created to mark package\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Check (and optionally create) __init__.py in src/* packages.")
    ap.add_argument("--root", default="src", help="Root package directory (default: src)")
    ap.add_argument("--fix", action="store_true", help="Create missing __init__.py files")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"❌ Root not found: {root}")
        raise SystemExit(1)

    missing = find_missing_inits(root)

    if not missing:
        print("✅ All packages have __init__.py")
        return

    print("⚠️ Missing __init__.py in:")
    for d in missing:
        print("  -", d.relative_to(root.parent))

    if args.fix:
        if args.dry_run:
            print("\n--dry-run: would create __init__.py in the directories above.")
            return
        for d in missing:
            ensure_init(d)
        print("\n✅ Created __init__.py in all missing package directories.")
    else:
        print("\nTip: run with --fix to create them automatically.")

if __name__ == "__main__":
    main()
