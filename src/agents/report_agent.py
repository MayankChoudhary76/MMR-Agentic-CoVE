#!/usr/bin/env python3
# src/agents/report_agent.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repo root (/notebooks/MMR-Agentic-CoVE)
LOGS = ROOT / "logs"
PLOTS = LOGS / "plots"
REPORTS_ROOT = ROOT / "reports"

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_metrics(csv_fp: Path) -> pd.DataFrame:
    if not csv_fp.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_fp}")
    df = pd.read_csv(csv_fp, engine="python")
    # normalize
    for col in ["run_name", "dataset", "fusion"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    for wcol in ["w_text", "w_image", "w_meta"]:
        if wcol not in df.columns:
            df[wcol] = float("nan")
    # convenience flags
    if "faiss" not in df.columns:
        df["faiss"] = df["run_name"].str.contains("faiss", case=False, na=False).astype(bool)
    return df

def _metric_cols(df: pd.DataFrame, k: int) -> Dict[str, Optional[str]]:
    # prefer explicit @k
    hit_col = f"hit@{k}" if f"hit@{k}" in df.columns else ("hit" if "hit" in df.columns else None)
    ndcg_col = f"ndcg@{k}" if f"ndcg@{k}" in df.columns else ("ndcg" if "ndcg" in df.columns else None)
    return {"hit": hit_col, "ndcg": ndcg_col}

def _top_n_table(
    df: pd.DataFrame, dataset: str, k: int, top_n: int = 5,
    prefer_faiss: bool = True
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    df = df[df["dataset"] == dataset] if "dataset" in df.columns else df

    cols = _metric_cols(df, k)
    hitc, ndcgc = cols["hit"], cols["ndcg"]
    if not ndcgc and not hitc:
        raise ValueError(f"No hit/ndcg columns found for k={k}. Available: {list(df.columns)}")

    # sort keys: ndcg desc, then hit desc; optional FAISS preference when tied
    sort_cols = []
    if ndcgc: sort_cols.append(ndcgc)
    if hitc:  sort_cols.append(hitc)
    if not sort_cols:
        raise ValueError("No sortable metric columns.")
    df["_faiss"] = df.get("faiss", df["run_name"].str.contains("faiss", case=False, na=False)).astype(int)
    by = [c for c in sort_cols] + (["_faiss"] if prefer_faiss else [])
    df_sorted = df.sort_values(by=by, ascending=[False]*len(by))

    # build a compact table for the report
    keep_cols = ["run_name", "dataset", "fusion", "w_text", "w_image", "w_meta"]
    if hitc:  keep_cols.append(hitc)
    if ndcgc: keep_cols.append(ndcgc)

    top = df_sorted[keep_cols].head(top_n).reset_index(drop=True)

    # choose recommendation = first row
    rec_row = top.iloc[0].to_dict()
    rec = {
        "dataset": dataset,
        "k": k,
        "recommended_run": rec_row["run_name"],
        "fusion": rec_row.get("fusion"),
        "weights": {
            "w_text": float(rec_row.get("w_text")) if pd.notna(rec_row.get("w_text")) else None,
            "w_image": float(rec_row.get("w_image")) if pd.notna(rec_row.get("w_image")) else None,
            "w_meta": float(rec_row.get("w_meta")) if pd.notna(rec_row.get("w_meta")) else None,
        },
        "metrics": {
            (hitc or "hit"): float(rec_row.get(hitc)) if hitc else None,
            (ndcgc or "ndcg"): float(rec_row.get(ndcgc)) if ndcgc else None,
        },
    }
    return top, rec

def _md_table(df: pd.DataFrame) -> str:
    """
    Return a markdown-ish table. Falls back to a preformatted text block if
    pandas' to_markdown requires 'tabulate' and it's not installed.
    """
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback: plain text inside code fences so the report still renders.
        return "```\n" + df.to_string(index=False) + "\n```"
def _copy_plots_into(out_dir: Path, dataset: str) -> list[str]:
    """
    Return the list of plot filenames that were copied into out_dir.
    Only copies files that exist under logs/plots.
    """
    wanted = [
        f"{dataset}_k10_quality.png",
        f"{dataset}_k10_quality_trend.png",
        f"{dataset}_k10_latency.png",
        f"{dataset}_w_meta_ndcg@10.png",
        f"{dataset}_w_meta_hit@10.png",
        f"{dataset}_k_ndcg@10.png",
    ]
    copied: list[str] = []
    for name in wanted:
        src = PLOTS / name
        if src.exists():
            try:
                import shutil
                dst = out_dir / name
                shutil.copy2(src, dst)
                copied.append(name)
            except Exception:
                pass
    return copied

def _baseline_quadrant(df: pd.DataFrame, dataset: str, k: int) -> Optional[pd.DataFrame]:
    """
    Build a compact 2x2 comparison if rows exist:
      No-FAISS / FAISS   ×   concat / weighted
    """
    cols = _metric_cols(df, k)
    hitc, ndcgc = cols["hit"], cols["ndcg"]
    if not ndcgc and not hitc:
        return None

    d = df.copy()
    if "dataset" in d.columns:
        d = d[d["dataset"] == dataset]
    if "fusion" not in d.columns:
        return None
    if "faiss" not in d.columns:
        d["faiss"] = d["run_name"].str.contains("faiss", case=False, na=False).astype(bool)

    # For each quadrant, pick the best row (by ndcg then hit)
    rows = []
    for fa in [False, True]:
        for fu in ["concat", "weighted"]:
            sub = d[(d["fusion"].str.lower()==fu) & (d["faiss"]==fa)]
            if ndcgc: sub = sub.sort_values(ndcgc, ascending=False)
            if hitc:
                sub = sub.sort_values([ndcgc, hitc], ascending=[False, False]) if ndcgc else sub.sort_values(hitc, ascending=False)
            if sub.empty:
                rows.append({"faiss": "Yes" if fa else "No", "fusion": fu, "run_name": "—",
                             "hit@k": None if not hitc else None, "ndcg@k": None if not ndcgc else None})
            else:
                r = sub.iloc[0]
                rows.append({
                    "faiss": "Yes" if fa else "No",
                    "fusion": fu,
                    "run_name": r.get("run_name", ""),
                    "hit@k": (float(r[hitc]) if hitc else None),
                    "ndcg@k": (float(r[ndcgc]) if ndcgc else None),
                })
    out = pd.DataFrame(rows, columns=["faiss","fusion","run_name","hit@k","ndcg@k"])
    # Return None if literally no metrics found
    if out[["hit@k","ndcg@k"]].isna().all().all():
        return None
    return out

def _write_report(
    out_dir: Path,
    tag: str,
    dataset: str,
    k: Optional[int],
    include_compare: bool,
    top_n: int,
    prefer_faiss: bool,
    metrics_csv: Path,
) -> None:
    _ensure_dir(out_dir)

    # Self-contained: copy plots into the report directory
    copied_plots = _copy_plots_into(out_dir, dataset)

    # optional compare section + recommendation
    compare_md = ""
    summary_json: Dict[str, Any] = {}
    if include_compare and k is not None:
        df_all = _load_metrics(metrics_csv)
        try:
            top, rec = _top_n_table(df_all, dataset=dataset, k=k, top_n=top_n, prefer_faiss=prefer_faiss)
            compare_md = (
                "## Top runs (auto)\n\n"
                + _md_table(top.rename(columns={
                    f"hit@{k}": "hit@k", f"ndcg@{k}": "ndcg@k"
                })) + "\n\n"
                "### Recommendation (auto)\n\n"
                "```json\n" + json.dumps(rec, indent=2) + "\n```\n"
            )
            summary_json["recommendation"] = rec
            summary_json["top_runs"] = json.loads(top.to_json(orient="records"))

            # Add a 4-way baseline quadrant if possible
            quad = _baseline_quadrant(df_all, dataset=dataset, k=k)
            if quad is not None:
                compare_md += "\n### Baseline 4-way comparison (FAISS × Fusion)\n\n"
                compare_md += _md_table(quad) + "\n"
                summary_json["baseline_quadrant"] = json.loads(quad.to_json(orient="records"))
        except Exception as e:
            compare_md = f"> Could not compute comparison for k={k}: {e}\n"

    # build markdown
    md_parts = [f"# {dataset} — {tag}\n"]
    if include_compare and k is not None:
        md_parts.append(compare_md)

    if copied_plots:
        md_parts.append("## Plots\n")
        for name in copied_plots:
            md_parts.append(f"![{name}](./{name})\n")

    # metrics snapshot (also save a filtered CSV into the report for grading)
    try:
        dfm = _load_metrics(metrics_csv)
        snap = dfm[dfm["dataset"] == dataset] if "dataset" in dfm.columns else dfm
        md_parts.append("## Metrics snapshot\n\n")
        show_cols = [c for c in ["run_name","dataset","fusion","w_text","w_image","w_meta",
                                 "k","hit","ndcg","hit@5","ndcg@5","hit@10","ndcg@10","hit@20","ndcg@20","p50_ms","p95_ms"]
                     if c in snap.columns]
        if not show_cols:
            show_cols = list(snap.columns)[:10]
        md_parts.append(_md_table(snap[show_cols].tail(20)) + "\n")
        # Save a compact CSV snapshot into the report folder
        snap.to_csv(out_dir / "metrics.csv", index=False)
    except Exception as e:
        md_parts.append(f"> Could not include metrics snapshot: {e}\n")

    # write index.md
    md_path = out_dir / "index.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    # render HTML (pretty if markdown package available; otherwise fallback)
    html_path = out_dir / "index.html"
    try:
        import markdown  # type: ignore
        html = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables"])
        html_full = [
            "<html><head><meta charset='utf-8'><title>Report</title>",
            "<style>body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;max-width:900px;margin:40px auto;padding:0 16px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px 8px}</style>",
            "</head><body>",
            html,
            "</body></html>",
        ]
        html_path.write_text("\n".join(html_full), encoding="utf-8")
    except Exception:
        # simple fallback
        html = [
            "<html><head><meta charset='utf-8'><title>Report</title></head><body>",
            f"<pre style='font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'>{md_path.read_text(encoding='utf-8')}</pre>",
            "</body></html>",
        ]
        html_path.write_text("\n".join(html), encoding="utf-8")

    # write summary.json
    (out_dir / "summary.json").write_text(json.dumps({
        "dataset": dataset,
        "tag": tag,
        "k": k,
        "include_compare": include_compare,
        **summary_json
    }, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="report")
    ap.add_argument("--k", type=int, default=10, help="k to use for comparison tables")
    ap.add_argument("--include-compare", action="store_true", help="Include Top runs + Recommendation section")
    ap.add_argument("--top-n", type=int, default=3, help="How many runs to show in the Top table")
    ap.add_argument("--prefer-faiss", action="store_true", help="Prefer FAISS runs when metrics tie")
    ap.add_argument("--metrics_csv", default=str(LOGS / "metrics.csv"))
    ap.add_argument("--plots_dir", default=str(PLOTS))
    ap.add_argument("--out", default="", help="Optional explicit out path (file or directory)")
    ap.add_argument("--no-plots", action="store_true", help="(kept for back-compat; plots are referenced if present)")
    ap.add_argument("--zip", action="store_true", help="Zip the report folder")
    args = ap.parse_args()

    dataset = args.dataset
    tag = args.tag
    out_dir = Path(args.out) if args.out else (REPORTS_ROOT / dataset / f"{pd.Timestamp.now():%Y%m%d_%H%M%S} {tag}")
    _ensure_dir(out_dir)

    # Create report
    _write_report(
        out_dir=out_dir,
        tag=tag,
        dataset=dataset,
        k=args.k if args.include_compare else None,
        include_compare=args.include_compare,
        top_n=args.top_n,
        prefer_faiss=args.prefer_faiss,
        metrics_csv=Path(args.metrics_csv),
    )

    print(f"→ Assembling report at {out_dir}")
    print(f"✓ Report ready: {out_dir}")

    if args.zip:
        import shutil
        zpath = out_dir.with_suffix(".zip")
        base = out_dir.name
        shutil.make_archive(str(zpath.with_suffix("")), "zip", out_dir.parent, base)
        print(f"📦 Zipped → {zpath}")

if __name__ == "__main__":
    main()