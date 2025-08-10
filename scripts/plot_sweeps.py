#!/usr/bin/env python3
# scripts/plot_sweeps.py
from __future__ import annotations
import argparse, re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python")
    # normalize columns
    if "run_name" not in df.columns:
        df["run_name"] = ""
    df["run_name"] = df["run_name"].fillna("").astype(str)
    if "dataset" not in df.columns:
        df["dataset"] = ""
    return df


def extract_numeric_suffix(run_name: str, key_prefix: str) -> float | None:
    """
    Extract the last floating number after a given key (e.g., 'wm0.3' -> 0.3)
    Example pattern in run_name: ..._wm0.3
    """
    # find e.g. '_wm0.3' where key_prefix='wm'
    m = re.search(rf"{re.escape(key_prefix)}([0-9.]+)(?:$|[^0-9.])", run_name)
    return float(m.group(1)) if m else None


def main(args):
    logs_dir = Path(args.logs_dir)
    csv_fp = logs_dir / "metrics.csv"
    if not csv_fp.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_fp}")

    df = load_metrics(csv_fp)

    # filter by dataset and optional run_name prefix
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if args.prefix:
        df = df[df["run_name"].str.startswith(args.prefix)]

    if df.empty:
        print("No rows after filtering. Check --dataset and --prefix.")
        return

    # extract sweep variable from run_name (e.g., 'wm')
    df[args.sweep_key] = df["run_name"].apply(lambda s: extract_numeric_suffix(s, f"{args.sweep_key}"))
    df = df.dropna(subset=[args.sweep_key]).copy()
    df.sort_values(by=args.sweep_key, inplace=True)

    if df.empty:
        print(f"No rows contained sweep key '{args.sweep_key}'.")
        return

    # choose metric column
    metric_col = args.metric
    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric_col}' not found in CSV columns: {list(df.columns)}")

    # print compact table
    cols_show = ["run_name", args.sweep_key, metric_col]
    print(df[cols_show].to_string(index=False))

    # plot
    plots_dir = logs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df[args.sweep_key], df[metric_col], marker="o")
    plt.xlabel(args.sweep_key)
    plt.ylabel(metric_col)
    title_parts = []
    if args.dataset: title_parts.append(f"dataset={args.dataset}")
    if args.prefix: title_parts.append(f"prefix={args.prefix}")
    plt.title(f"{metric_col} vs {args.sweep_key}" + (f" ({', '.join(title_parts)})" if title_parts else ""))
    out = plots_dir / f"{(args.dataset or 'all')}_{args.sweep_key}_{metric_col}.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs", help="Directory containing metrics.csv")
    ap.add_argument("--dataset", default="", help="Filter by dataset value (optional)")
    ap.add_argument("--prefix", default="", help="Filter runs whose run_name starts with this prefix")
    ap.add_argument("--sweep_key", default="wm", help="Key used in run_name to encode sweep value (e.g., wm0.3)")
    ap.add_argument("--metric", default="ndcg@10", help="Metric column to plot (e.g., 'hit@10' or 'ndcg@10')")
    args = ap.parse_args()
    main(args)