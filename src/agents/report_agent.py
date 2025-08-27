# src/agents/report_agent.py
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]  # repo root
LOGS_DIR = ROOT / "logs"
PLOTS_DIR = LOGS_DIR / "plots"
REPORTS_DIR = ROOT / "reports"


@dataclass
class RunRow:
    run_name: str
    dataset: str
    k: Optional[int]
    hit: Optional[float]
    ndcg: Optional[float]
    latency_ms_p50: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    recall: Optional[float] = None
    mrr: Optional[float] = None
    timestamp: Optional[str] = None


def _sf(x: str | float | None) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _read_metrics_csv(csv_path: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(
                RunRow(
                    run_name=d.get("run_name", ""),
                    dataset=d.get("dataset", ""),
                    k=int(d["k"]) if d.get("k") else None,
                    hit=_sf(d.get("hit") or d.get("hit@10") or d.get("hit@k")),
                    ndcg=_sf(d.get("ndcg") or d.get("ndcg@10") or d.get("ndcg@k")),
                    latency_ms_p50=_sf(d.get("latency_ms_p50")),
                    latency_ms_p95=_sf(d.get("latency_ms_p95")),
                    recall=_sf(d.get("recall")),
                    mrr=_sf(d.get("mrr")),
                    timestamp=d.get("timestamp"),
                )
            )
    return rows


def _best(rows: List[RunRow], key: str) -> Optional[RunRow]:
    rows2 = [r for r in rows if getattr(r, key) is not None]
    if not rows2:
        return None
    return max(rows2, key=lambda r: getattr(r, key) or -1)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_matching(src_dir: Path, dst_dir: Path, dataset: str, patterns: Tuple[str, ...]) -> List[Path]:
    """Copy files that start with '<dataset>_' and end with any of patterns."""
    copied: List[Path] = []
    if not src_dir.exists():
        return copied
    for fp in sorted(src_dir.iterdir()):
        if not fp.is_file():
            continue
        name = fp.name
        if not name.startswith(f"{dataset}_"):
            continue
        if any(name.endswith(suf) for suf in patterns):
            out = dst_dir / name
            shutil.copy2(fp, out)
            copied.append(out)
    return copied


class ReportAgent:
    """
    Generates a self-contained report folder:
      reports/<dataset>/<YYYYmmdd_HHMMSS>/
        - metrics.csv
        - all PNGs + JSONs from logs/plots matching '<dataset>_*'
        - summary.json (top runs, best metrics)
        - index.md + index.html (easy to browse)
        - (optional) archive.zip
    """

    def build_plots(self, dataset: str = "beauty") -> None:
        """Run your existing plotting scripts."""
        print(f"→ Generating plots for dataset='{dataset}'")
        subprocess.check_call([sys.executable, "scripts/plot_metrics.py", "--dataset", dataset])
        # These defaults expect you to encode a sweep key in run_name (e.g. wm0.3)
        # You can call plot_sweeps.py multiple times with different sweep keys if desired.
        for sweep_key, metric in [("wm", "ndcg"), ("wm", "hit")]:
            try:
                subprocess.check_call([
                    sys.executable, "scripts/plot_sweeps.py",
                    "--dataset", dataset,
                    "--sweep_key", sweep_key,
                    "--metric", metric,
                ])
            except subprocess.CalledProcessError:
                # It's okay if no rows match; continue.
                pass

    def assemble_report(self, dataset: str = "beauty", tag: Optional[str] = None, zip_report: bool = False) -> Path:
        """
        Bundle artifacts into a versioned folder and write a summary.
        Returns the report directory path.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{ts}" + (f"_{tag}" if tag else "")
        out_dir = _ensure_dir(REPORTS_DIR / dataset / label)
        print(f"→ Assembling report at {out_dir}")

        # 1) Copy metrics.csv
        metrics_csv_src = LOGS_DIR / "metrics.csv"
        if metrics_csv_src.exists():
            shutil.copy2(metrics_csv_src, out_dir / "metrics.csv")

        # 2) Copy plots & JSONs for this dataset
        copied_png = _copy_matching(PLOTS_DIR, out_dir, dataset, (".png",))
        copied_json = _copy_matching(PLOTS_DIR, out_dir, dataset, (".json",))

        # 3) Build summary from metrics.csv filtered by dataset
        all_rows = _read_metrics_csv(metrics_csv_src)
        rows = [r for r in all_rows if r.dataset == dataset]
        best_hit = _best(rows, "hit")
        best_ndcg = _best(rows, "ndcg")

        summary = {
            "dataset": dataset,
            "created_at": ts,
            "num_runs": len(rows),
            "artifacts": {
                "png": [p.name for p in copied_png],
                "json": [p.name for p in copied_json],
                "metrics_csv": "metrics.csv" if metrics_csv_src.exists() else None,
            },
            "best_hit": asdict(best_hit) if best_hit else None,
            "best_ndcg": asdict(best_ndcg) if best_ndcg else None,
            "runs": [asdict(r) for r in rows],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # 4) Write index.md and index.html
        self._write_index_md(out_dir, summary)
        self._write_index_html(out_dir, summary)

        # 5) Optional zip
        if zip_report:
            archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
            print(f"✓ Zipped to {archive}")

        print("✓ Report ready:", out_dir)
        return out_dir

    # ---------- helpers for index files ----------

    def _write_index_md(self, out_dir: Path, summary: Dict) -> None:
        lines = []
        lines.append(f"# Report — dataset: **{summary['dataset']}**")
        lines.append(f"_Created at: {summary['created_at']}_\n")
        lines.append("## Best runs")
        if summary.get("best_hit"):
            b = summary["best_hit"]
            lines.append(f"- **Best Hit@K**: `{b.get('hit')}` • run: `{b.get('run_name')}` • k={b.get('k')}")
        if summary.get("best_ndcg"):
            b = summary["best_ndcg"]
            lines.append(f"- **Best NDCG@K**: `{b.get('ndcg')}` • run: `{b.get('run_name')}` • k={b.get('k')}")
        lines.append("\n## Artifacts")
        arts = summary.get("artifacts", {})
        for k, arr in arts.items():
            if not arr:
                continue
            if isinstance(arr, list):
                lines.append(f"- **{k.upper()}**: " + ", ".join(f"`{a}`" for a in arr))
            else:
                lines.append(f"- **{k.upper()}**: `{arr}`")
        lines.append("\n## Plots")
        for name in arts.get("png", []):
            lines.append(f"![{name}]({name})")
        (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_index_html(self, out_dir: Path, summary: Dict) -> None:
        """Tiny static HTML index so you can open the folder in a browser and click around."""
        def esc(s: str) -> str:
            return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        arts = summary.get("artifacts", {})
        rows = summary.get("runs", [])
        html = []
        html.append("<!doctype html><meta charset='utf-8'>")
        html.append("<title>Report</title>")
        html.append("<style>body{font-family:ui-sans-serif,system-ui;margin:24px;} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px 8px} th{background:#f8f8f8}</style>")
        html.append(f"<h1>Report — dataset: <b>{esc(summary['dataset'])}</b></h1>")
        html.append(f"<p><i>Created at: {esc(summary['created_at'])}</i></p>")

        # Best section
        html.append("<h2>Best runs</h2><ul>")
        if summary.get("best_hit"):
            b = summary["best_hit"]
            html.append(f"<li><b>Best Hit@K</b>: {esc(str(b.get('hit')))} — run: <code>{esc(b.get('run_name',''))}</code> — k={esc(str(b.get('k')))}</li>")
        if summary.get("best_ndcg"):
            b = summary["best_ndcg"]
            html.append(f"<li><b>Best NDCG@K</b>: {esc(str(b.get('ndcg')))} — run: <code>{esc(b.get('run_name',''))}</code> — k={esc(str(b.get('k')))}</li>")
        html.append("</ul>")

        # Table of runs
        if rows:
            html.append("<h2>All runs</h2>")
            html.append("<table><tr><th>run_name</th><th>k</th><th>hit</th><th>ndcg</th><th>p50(ms)</th><th>p95(ms)</th><th>timestamp</th></tr>")
            for r in rows:
                html.append(
                    "<tr>"
                    f"<td><code>{esc(r.get('run_name',''))}</code></td>"
                    f"<td>{esc(str(r.get('k')))}</td>"
                    f"<td>{esc(str(r.get('hit')))}</td>"
                    f"<td>{esc(str(r.get('ndcg')))}</td>"
                    f"<td>{esc(str(r.get('latency_ms_p50')))}</td>"
                    f"<td>{esc(str(r.get('latency_ms_p95')))}</td>"
                    f"<td>{esc(str(r.get('timestamp')))}</td>"
                    "</tr>"
                )
            html.append("</table>")

        # Plots
        if arts.get("png"):
            html.append("<h2>Plots</h2>")
            for name in arts["png"]:
                html.append(f"<div><img src='{esc(name)}' style='max-width:100%;height:auto;'/><div><code>{esc(name)}</code></div></div><hr/>")

        (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


# ------------------- CLI -------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beauty")
    ap.add_argument("--tag", default="", help="Optional label appended to report folder name")
    ap.add_argument("--no-plots", action="store_true", help="Skip running plotting scripts; just gather artifacts")
    ap.add_argument("--zip", dest="zip_report", action="store_true", help="Zip the final report folder")
    args = ap.parse_args()

    agent = ReportAgent()
    if not args.no_plots:
        agent.build_plots(dataset=args.dataset)
    agent.assemble_report(dataset=args.dataset, tag=(args.tag or None), zip_report=args.zip_report)


if __name__ == "__main__":
    _cli()