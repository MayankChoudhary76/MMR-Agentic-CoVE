#!/usr/bin/env python3
import argparse, pathlib
import pandas as pd
import matplotlib.pyplot as plt

def annotate_bars(ax):
    for p in ax.patches:
        value = p.get_height()
        ax.annotate(f"{value:.3f}", 
                    (p.get_x() + p.get_width()/2, value),
                    ha="center", va="bottom", fontsize=9, rotation=0)

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
        raise SystemExit(f"Missing {csv_path}. Run a build script first.")

    df = pd.read_csv(csv_path)
    df = df[(df["dataset"] == args.dataset) & (df["k"] == args.k)].copy()

    # Sort by timestamp so runs appear chronologically
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # One grouped bar chart: Hit@K and NDCG@K for each run_name
    labels = df["run_name"].tolist()
    hit = df["hit"].tolist()
    ndcg = df["ndcg"].tolist()

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar([i - width/2 for i in x], hit, width, label=f"Hit@{args.k}")
    ax.bar([i + width/2 for i in x], ndcg, width, label=f"NDCG@{args.k}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Comparison of Runs on '{args.dataset}' (K={args.k})")
    ax.legend()
    annotate_bars(ax)
    fig.tight_layout()
    out = out_dir / f"{args.dataset}_k{args.k}_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"📈 Saved {out}")

    # Optional: time trend lines
    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"])
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(df["ts"], df["hit"], marker="o", label=f"Hit@{args.k}")
        ax2.plot(df["ts"], df["ndcg"], marker="o", label=f"NDCG@{args.k}")
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Score")
        ax2.set_title(f"Metric Trend over Time — {args.dataset} (K={args.k})")
        ax2.legend()
        fig2.autofmt_xdate()
        out2 = out_dir / f"{args.dataset}_k{args.k}_trend.png"
        fig2.savefig(out2, dpi=150)
        print(f"📉 Saved {out2}")

if __name__ == "__main__":
    main()