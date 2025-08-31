#!/usr/bin/env python3
"""
Plot experiment metrics from logs/metrics.csv

Expected (but optional) columns:
- run_name, dataset, k, timestamp
- hit, ndcg, mrr
- p50_ms, p95_ms
- index_mb, emb_text_mb, emb_image_mb, cove_mb

The script is robust to missing columns: charts that can't be built are skipped.
"""

import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

def annotate_bars(ax, fmt="{:.3f}", ypad=0.0, fontsize=9):
    for p in ax.patches:
        value = p.get_height()
        ax.annotate(fmt.format(value),
                    (p.get_x() + p.get_width()/2.0, value + ypad),
                    ha="center", va="bottom", fontsize=fontsize)

def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        try:
            df = df.copy()
            df["__ts"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("__ts")
            df = df.drop(columns=["__ts"])
            return df
        except Exception:
            pass
    return df

def _metric_cols(df: pd.DataFrame, k: int) -> tuple[str|None, str|None]:
    hitc  = f"hit@{k}"  if f"hit@{k}"  in df.columns else ("hit"  if "hit"  in df.columns else None)
    ndcgc = f"ndcg@{k}" if f"ndcg@{k}" in df.columns else ("ndcg" if "ndcg" in df.columns else None)
    return hitc, ndcgc

def _maybe_plot_quality(df: pd.DataFrame, k: int, out_dir: pathlib.Path, dataset: str):
    hit_col, ndcg_col = _metric_cols(df, k)
    if not hit_col and not ndcg_col:
        print("• Skipping quality comparison (no hit/ndcg columns).")
        return

    labels = df["run_name"].astype(str).tolist()
    x = range(len(labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    plotted = False
    if hit_col:
        hit = df[hit_col].astype(float).tolist()
        ax.bar([i - (width/2 if ndcg_col else 0) for i in x], hit, width, label=f"{hit_col}")
        plotted = True
    if ndcg_col:
        ndcg = df[ndcg_col].astype(float).tolist()
        ax.bar([i + (width/2 if hit_col else 0) for i in x], ndcg, width, label=f"{ndcg_col}")
        plotted = True

    if not plotted:
        print("• Skipping quality comparison (no plottable series).")
        return

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Quality — {dataset} (K={k})")
    ax.legend()
    annotate_bars(ax, "{:.3f}")
    fig.tight_layout()
    out = out_dir / f"{dataset}_k{k}_quality.png"
    fig.savefig(out, dpi=150)
    print(f"📈 Saved {out}")

    # Trend lines if timestamp is present
    if "timestamp" in df.columns and (hit_col or ndcg_col):
        try:
            dft = df.copy()
            dft["ts"] = pd.to_datetime(dft["timestamp"], format="ISO8601", utc=True, errors="coerce")
            dft = dft.dropna(subset=["ts"])
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            if hit_col:
                ax2.plot(dft["ts"], dft[hit_col],  marker="o", label=hit_col)
            if ndcg_col:
                ax2.plot(dft["ts"], dft[ndcg_col], marker="o", label=ndcg_col)
            ax2.set_ylim(0, 1.0)
            ax2.set_ylabel("Score")
            ax2.set_title(f"Quality Trend — {dataset} (K={k})")
            ax2.legend()
            fig2.autofmt_xdate()
            out2 = out_dir / f"{dataset}_k{k}_quality_trend.png"
            fig2.savefig(out2, dpi=150)
            print(f"📉 Saved {out2}")
        except Exception as e:
            print(f"• Skipping trend plot (time parse failed): {e}")

def _maybe_plot_latency(df: pd.DataFrame, out_dir: pathlib.Path, dataset: str, k: int):
    if not {"p50_ms", "p95_ms"}.issubset(set(df.columns)):
        print("• Skipping latency plots (missing p50_ms/p95_ms).")
        return

    labels = df["run_name"].astype(str).tolist()
    p50 = df["p50_ms"].astype(float).tolist()
    p95 = df["p95_ms"].astype(float).tolist()
    x = range(len(labels)); width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], p50, width, label="p50 (ms)")
    ax.bar([i + width/2 for i in x], p95, width, label="p95 (ms)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency — {dataset} (K={k})")
    ax.legend()
    annotate_bars(ax, "{:.0f}")
    fig.tight_layout()
    out = out_dir / f"{dataset}_k{k}_latency.png"
    fig.savefig(out, dpi=150)
    print(f"⚡ Saved {out}")

def _maybe_plot_sizes(df: pd.DataFrame, out_dir: pathlib.Path, dataset: str, k: int):
    # Any of these present → make a chart
    present = [c for c in ["index_mb", "emb_text_mb", "emb_image_mb", "cove_mb"] if c in df.columns]
    if not present:
        print("• Skipping size plots (no size columns found).")
        return

    labels = df["run_name"].astype(str).tolist()
    x = range(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    offset = - (len(present)-1) * width / 2.0
    for i, col in enumerate(present):
        y = df[col].astype(float).tolist()
        ax.bar([j + offset + i*width for j in x], y, width, label=col.replace("_mb", "").replace("_", " "))
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Size (MB)")
    ax.set_title(f"Artifact Sizes — {dataset} (K={k})")
    ax.legend()
    annotate_bars(ax, "{:.1f}", ypad=0.5, fontsize=8)
    fig.tight_layout()
    out = out_dir / f"{dataset}_k{k}_sizes.png"
    fig.savefig(out, dpi=150)
    print(f"🧱 Saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", default="logs/metrics.csv")
    ap.add_argument("--dataset", default="beauty")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out_dir", default="logs/plots")
    args = ap.parse_args()

    csv_path = pathlib.Path(args.metrics_csv)
    out_dir  = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run an eval and append a row first.")

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    # Basic filtering
    if "dataset" in df.columns:
        df = df[df["dataset"] == args.dataset]
    if "k" in df.columns:
        df = df[df["k"] == args.k]
    if df.empty:
        raise SystemExit("No rows match dataset/K; check logs/metrics.csv.")

    # Keep only required columns + run_name
    if "run_name" not in df.columns:
        raise SystemExit("metrics.csv must contain a 'run_name' column.")

    df = _ensure_sorted(df)

    # Plots
    _maybe_plot_quality(df, args.k, out_dir, args.dataset)
    _maybe_plot_latency(df, out_dir, args.dataset, args.k)
    _maybe_plot_sizes(df, out_dir, args.dataset, args.k)

if __name__ == "__main__":
    main()