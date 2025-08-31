#!/usr/bin/env python3
# scripts/plot_sweeps.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------
# Helpers
# -----------------------
ALIASES = {
    # sweep key aliases
    "wm": "w_meta",
    "wi": "w_image",
    "wt": "w_text",
    # metric aliases (fallbacks)
    "hit": "hit@10",
    "ndcg": "ndcg@10",
    "mrr": "mrr@10",
    "p50_ms": "latency_p50",
    "p95_ms": "latency_p95",
}


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python")
    if "run_name" not in df.columns:
        df["run_name"] = ""
    df["run_name"] = df["run_name"].fillna("").astype(str)
    if "dataset" not in df.columns:
        df["dataset"] = ""
    return df


def extract_value_from_run_name(run_name: str, key: str, custom_regex: str | None = None) -> float | None:
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
        rf"{re.escape(key)}[A-Za-z_]*([0-9]*\.?[0-9]+)",    # key + letters + number (e.g., coveC8)
    ]
    for pat in pats:
        m = re.search(pat, run_name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def resolve_sweep_key(df: pd.DataFrame, sweep_key: str) -> str:
    """
    Map alias → real key, and return the key we will use.
    """
    key = ALIASES.get(sweep_key, sweep_key)
    return key


def pick_metric_column(df: pd.DataFrame, requested: str | None) -> str:
    """
    Choose a metric column to plot. Priority:
      1) If requested is present, use it.
      2) If requested is an alias (e.g., 'ndcg' -> 'ndcg@10'), use the first existing alias target.
      3) Otherwise, prefer the last ndcg@K column, else last hit@K column, else raise.
    """
    if requested and requested in df.columns:
        return requested

    if requested and requested in ALIASES:
        alias = ALIASES[requested]
        # alias can itself be a metric or a pattern like 'ndcg@10'
        if alias in df.columns:
            return alias

    # prefer ndcg@K, then hit@K
    ndcgs = [c for c in df.columns if c.startswith("ndcg@")]
    hits = [c for c in df.columns if c.startswith("hit@")]

    if ndcgs:
        return sorted(ndcgs, key=lambda c: int(c.split("@")[1]))[-1]
    if hits:
        return sorted(hits, key=lambda c: int(c.split("@")[1]))[-1]

    # final fallback to generic names if present
    for fallback in ("ndcg", "hit", "mrr", "latency_p50", "latency_p95"):
        if fallback in df.columns:
            return fallback

    raise ValueError(
        f"Could not find a metric column. Available columns: {list(df.columns)}"
    )


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Plot metric sweeps from logs/metrics.csv")
    ap.add_argument("--logs_dir", default="logs", help="Directory containing metrics.csv")
    ap.add_argument("--dataset", default="", help="Filter by dataset value (optional)")
    ap.add_argument("--prefix", default="", help="Filter runs whose run_name starts with this prefix")
    ap.add_argument(
        "--sweep_key",
        default="w_meta",
        help="Column to sweep (e.g., w_meta, w_image, w_text, k) OR a token to parse from run_name. Aliases: wm, wi, wt",
    )
    ap.add_argument(
        "--metric",
        default="ndcg",
        help="Metric column to plot (e.g., ndcg@10, hit@10). Aliases: ndcg→ndcg@10, hit→hit@10, mrr→mrr@10",
    )
    ap.add_argument(
        "--regex",
        default="",
        help="Optional custom regex with ONE capture group for extracting sweep values from run_name",
    )
    ap.add_argument(
        "--out_dir",
        default="",
        help="Directory to save plots (default: <logs_dir>/plots)",
    )
    args = ap.parse_args()

    csv_fp = Path(args.logs_dir) / "metrics.csv"
    if not csv_fp.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_fp}")

    df = load_metrics(csv_fp)

    # filter
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if args.prefix:
        df = df[df["run_name"].str.startswith(args.prefix)]

    if df.empty:
        print("No rows after filtering. Check --dataset and --prefix.")
        return

    # resolve sweep key (alias-friendly)
    sweep_key = resolve_sweep_key(df, args.sweep_key)

    # Decide output dir
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.logs_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # If the sweep key exists as a proper column, use it directly; otherwise parse from run_name
    if sweep_key in df.columns:
        df_sweep = df.copy()
    else:
        df_sweep = df.copy()
        df_sweep[sweep_key] = df_sweep["run_name"].apply(
            lambda s: extract_value_from_run_name(s, sweep_key, args.regex or None)
        )

    df_sweep = df_sweep.dropna(subset=[sweep_key]).copy()
    if df_sweep.empty:
        print(f"No rows contained sweep key '{sweep_key}'.")
        return

    # pick a metric column
    metric_col = pick_metric_column(df_sweep, args.metric if args.metric else None)

    # simple view in console
    cols_show = ["run_name", sweep_key, metric_col]
    print(df_sweep[cols_show].sort_values(sweep_key).to_string(index=False))

    # plot
    df_plot = df_sweep.sort_values(sweep_key)
    plt.figure(figsize=(8, 4))
    plt.plot(df_plot[sweep_key], df_plot[metric_col], marker="o")
    for x, y, name in zip(df_plot[sweep_key], df_plot[metric_col], df_plot["run_name"]):
        try:
            yval = float(y)
        except Exception:
            yval = y
        if isinstance(yval, float):
            label = f"{yval:.3f}"
        else:
            label = str(yval)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    plt.grid(True, alpha=0.25)
    plt.xlabel(sweep_key)
    plt.ylabel(metric_col)
    title_bits = []
    if args.dataset:
        title_bits.append(f"dataset={args.dataset}")
    if args.prefix:
        title_bits.append(f"prefix={args.prefix}")
    plt.title(f"{metric_col} vs {sweep_key}" + (f" ({', '.join(title_bits)})" if title_bits else ""))

    out = out_dir / f"{(args.dataset or 'all')}_{sweep_key}_{metric_col}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    main()