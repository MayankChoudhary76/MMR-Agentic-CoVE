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

def _maybe_plot_quality(df: pd.DataFrame, k: int, out_dir: pathlib.Path, dataset: str):
    needed = {"hit", "ndcg"}
    if not needed.issubset(set(df.columns)):
        print("• Skipping quality comparison (missing hit/ndcg).")
        return

    labels = df["run_name"].astype(str).tolist()
    hit  = df["hit"].astype(float).tolist()
    ndcg = df["ndcg"].astype(float).tolist()
    x = range(len(labels)); width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], hit,  width, label=f"Hit@{k}")
    ax.bar([i + width/2 for i in x], ndcg, width, label=f"NDCG@{k}")
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

    # Optional MRR
    if "mrr" in df.columns:
        mrr = df["mrr"].astype(float).tolist()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(x, mrr, width=0.6, label="MRR")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labels, rotation=15)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Score")
        ax2.set_title(f"MRR — {dataset} (K={k})")
        annotate_bars(ax2, "{:.3f}")
        fig2.tight_layout()
        out2 = out_dir / f"{dataset}_k{k}_mrr.png"
        fig2.savefig(out2, dpi=150)
        print(f"📈 Saved {out2}")

    # Trend lines if timestamp available
    if "timestamp" in df.columns:
        try:
            dft = df.copy()
            dft["ts"] = pd.to_datetime(dft["timestamp"])
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(dft["ts"], dft["hit"],  marker="o", label=f"Hit@{k}")
            ax3.plot(dft["ts"], dft["ndcg"], marker="o", label=f"NDCG@{k}")
            if "mrr" in dft.columns:
                ax3.plot(dft["ts"], dft["mrr"], marker="o", label="MRR")
            ax3.set_ylim(0, 1.0)
            ax3.set_ylabel("Score")
            ax3.set_title(f"Quality Trend — {dataset} (K={k})")
            ax3.legend()
            fig3.autofmt_xdate()
            out3 = out_dir / f"{dataset}_k{k}_quality_trend.png"
            fig3.savefig(out3, dpi=150)
            print(f"📉 Saved {out3}")
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

    df = pd.read_csv(csv_path)
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