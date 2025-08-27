#!/usr/bin/env python3
# scripts/plot_sweeps.py
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python")
    if "run_name" not in df.columns:
        df["run_name"] = ""
    df["run_name"] = df["run_name"].fillna("").astype(str)
    if "dataset" not in df.columns:
        df["dataset"] = ""
    return df

def extract_value(run_name: str, key: str, custom_regex: str | None = None) -> float | None:
    """
    Extract a numeric sweep value from run_name.
    Works for patterns like:
      wm0.3, alpha0.5, coveC8, comp16, hash_b12, lr1e-4
    You can override with --regex (must contain one capturing group for the number).
    """
    if custom_regex:
        m = re.search(custom_regex, run_name)
        return float(m.group(1)) if m else None

    # generic patterns to try (first match wins)
    pats = [
        rf"{re.escape(key)}([0-9]*\.?[0-9]+(?:e-?\d+)?)",   # key + float/scientific
        rf"{re.escape(key)}[A-Za-z]*([0-9]*\.?[0-9]+)",     # key + letters + number (e.g., coveC8)
    ]
    for pat in pats:
        m = re.search(pat, run_name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None

def main(args):
    csv_fp = Path(args.logs_dir) / "metrics.csv"
    if not csv_fp.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_fp}")

    df = load_metrics(csv_fp)

    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if args.prefix:
        df = df[df["run_name"].str.startswith(args.prefix)]

    if df.empty:
        print("No rows after filtering. Check --dataset and --prefix.")
        return

    # Pull sweep values
    df[args.sweep_key] = df["run_name"].apply(
        lambda s: extract_value(s, args.sweep_key, args.regex)
    )
    df = df.dropna(subset=[args.sweep_key]).copy()
    df.sort_values(by=args.sweep_key, inplace=True)

    if df.empty:
        print(f"No rows contained sweep key '{args.sweep_key}'.")
        return

    # Metric column fallback names
    metric_alias = args.metric
    if metric_alias not in df.columns:
        # common aliases
        aliases = {
            "hit@10": "hit", "ndcg@10": "ndcg", "mrr@10": "mrr",
            "latency_p50": "p50_ms", "latency_p95": "p95_ms",
        }
        if metric_alias in aliases and aliases[metric_alias] in df.columns:
            metric_alias = aliases[metric_alias]
        else:
            raise ValueError(f"Metric '{args.metric}' not found. Available: {list(df.columns)}")

    cols_show = ["run_name", args.sweep_key, metric_alias]
    print(df[cols_show].to_string(index=False))

    plots_dir = Path(args.logs_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,4))
    plt.plot(df[args.sweep_key], df[metric_alias], marker="o")
    for x, y, name in zip(df[args.sweep_key], df[metric_alias], df["run_name"]):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.xlabel(args.sweep_key)
    plt.ylabel(metric_alias)
    title_bits = []
    if args.dataset: title_bits.append(f"dataset={args.dataset}")
    if args.prefix:  title_bits.append(f"prefix={args.prefix}")
    plt.title(f"{metric_alias} vs {args.sweep_key}" + (f" ({', '.join(title_bits)})" if title_bits else ""))
    out = plots_dir / f"{(args.dataset or 'all')}_{args.sweep_key}_{metric_alias}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs", help="Directory containing metrics.csv")
    ap.add_argument("--dataset", default="", help="Filter by dataset value (optional)")
    ap.add_argument("--prefix",  default="", help="Filter runs whose run_name starts with this prefix")
    ap.add_argument("--sweep_key", default="wm", help="Key encoded in run_name (e.g., wm, alpha, coveC, comp)")
    ap.add_argument("--metric", default="ndcg", help="Metric column to plot (e.g., hit, ndcg, mrr, p50_ms)")
    ap.add_argument("--regex", default="", help="Optional custom regex with ONE capture group for the value")
    args = ap.parse_args()
    main(args)