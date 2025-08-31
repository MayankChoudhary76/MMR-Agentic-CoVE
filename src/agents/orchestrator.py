#!/usr/bin/env python3
# src/agents/orchestrator.py
from __future__ import annotations
import argparse, json, os, re, shlex, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Paths
ROOT = Path(__file__).resolve().parents[2]   # repo root (/notebooks/MMR-Agentic-CoVE)
LOGS = ROOT / "logs"
PLOTS = LOGS / "plots"
REPORTS = ROOT / "reports"
PLOTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1","true","yes","y","on"}

def run(cmd: List[str], dry: bool = False) -> int:
    print("▶", " ".join(shlex.quote(c) for c in cmd))
    if dry:
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(ROOT)}:{env.get('PYTHONPATH','')}".rstrip(":")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT), env=env
    )
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()

def auto_run_name(cfg: Dict[str, Any], k: int) -> str:
    parts = [f"{cfg['fusion']}_k{k}"]
    if len(cfg['w_texts']) == 1:  parts.append(f"wt{cfg['w_texts'][0]:g}")
    if len(cfg['w_images']) == 1: parts.append(f"wi{cfg['w_images'][0]:g}")
    if len(cfg['w_metas']) == 1:  parts.append(f"wm{cfg['w_metas'][0]:g}")
    if cfg.get("use_faiss"):      parts.append("faiss")
    return "_".join(parts)

# ---------------------------
# NL → config parser
# ---------------------------
def _grab_float_list(text: str, key: str, default: List[float]) -> List[float]:
    m = re.search(fr"{key}\s*=\s*([0-9eE\.\,\s\-]+)", text)
    if not m:
        return default
    vals = [v for v in re.split(r"[,\s]+", m.group(1).strip()) if v]
    out: List[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out or default

def parse_command(text: str) -> Dict[str, Any]:
    """
    Examples:
      - run dataset=beauty fusion=concat k=5,10 use_faiss=1
      - run baselines on beauty k=5,10,20 fusion=weighted w_meta=0,0.1,0.2
      - plot metrics for beauty k=10
      - compare best for beauty k=10
      - report beauty title="FAISS baselines"
    """
    raw = text.strip()
    t = raw.lower()

    # dataset
    m_dataset = re.search(r"(?:on|for|dataset=)\s*([a-z0-9_]+)", t)
    dataset = m_dataset.group(1) if m_dataset else "beauty"

    # ks
    m_k = re.search(r"\bk\s*=\s*([0-9,\s]+)", t)
    ks = [10]
    if m_k:
        ks = [int(x) for x in re.split(r"[,\s]+", m_k.group(1).strip()) if x]

    # fusion
    m_fusion = re.search(r"(fusion|mode)\s*=\s*(concat|weighted|wsum)", t)
    fusion = m_fusion.group(2) if m_fusion else ("weighted" if ("weighted" in t or "wsum" in t) else "concat")

    # weights (lists accepted)
    w_texts  = _grab_float_list(t, "w_text",  [1.0])
    w_images = _grab_float_list(t, "w_image", [0.0])
    w_metas  = _grab_float_list(t, "w_meta",  [0.0])

    # faiss
    use_faiss = False
    m_use = re.search(r"use_faiss\s*=\s*(\w+)", t)
    if "faiss" in t:
        use_faiss = True
    if m_use:
        use_faiss = ensure_bool(m_use.group(1))
    m_fname = re.search(r"(faiss_name|index)\s*=\s*([\w\-\.]+)", raw)
    faiss_name = m_fname.group(2) if m_fname else f"{dataset}_{fusion}"

    # intent
    if re.search(r"\b(plot|charts|figure)\b", t):
        intent = "plot"
    elif re.search(r"\b(report|summarize)\b", t):
        intent = "report"
    elif re.search(r"\b(compare|best|recommend)\b", t):
        intent = "compare"
    else:
        intent = "run_grid" if re.search(r"\bbaseline|grid|sweep|many\b", t) or len(ks) > 1 or any(len(x) > 1 for x in (w_texts,w_images,w_metas)) else "run_one"

    # optional title
    m_title = re.search(r"title\s*=\s*'([^']+)'", raw) or re.search(r'title\s*=\s*"([^"]+)"', raw)
    title = m_title.group(1) if m_title else None

    return {
        "intent": intent,
        "dataset": dataset,
        "ks": ks,
        "fusion": fusion,
        "w_texts": w_texts,
        "w_images": w_images,
        "w_metas": w_metas,
        "use_faiss": use_faiss,
        "faiss_name": faiss_name,
        "sweep_key": "wm",  # default used by plot_sweeps
        "title": title,
    }

# ---------------------------
# Executors
# ---------------------------
def exec_run_one(cfg: Dict[str, Any], dry: bool = False) -> int:
    k = cfg["ks"][0] if cfg["ks"] else 10
    wt = cfg["w_texts"][0]
    wi = cfg["w_images"][0]
    wm = cfg["w_metas"][0]
    run_name = auto_run_name(cfg, k)

    cmd = [
        sys.executable, "scripts/eval_fusion.py",
        "--dataset", cfg["dataset"],
        "--fusion", cfg["fusion"],
        "--w_text", str(wt),
        "--w_image", str(wi),
        "--w_meta",  str(wm),
        "--k", str(k),
        "--run_name", run_name,
    ]
    if cfg["use_faiss"]:
        cmd += ["--use_faiss", "--faiss_name", cfg["faiss_name"]]
    return run(cmd, dry=dry)

def exec_run_grid(cfg: Dict[str, Any], dry: bool = False) -> int:
    rc = 0
    for k in cfg["ks"]:
        for wt in cfg["w_texts"]:
            for wi in cfg["w_images"]:
                for wm in cfg["w_metas"]:
                    one = dict(cfg)
                    one["ks"] = [k]
                    one["w_texts"] = [wt]
                    one["w_images"] = [wi]
                    one["w_metas"] = [wm]
                    rn = auto_run_name(one, k)
                    print(f"\n=== Run: {rn} (fusion={one['fusion']}, k={k}, wt={wt}, wi={wi}, wm={wm}, faiss={one['use_faiss']}) ===")
                    rc |= exec_run_one(one, dry=dry)
    return rc

def exec_plot(cfg: Dict[str, Any], dry: bool = False) -> int:
    rc = 0
    rc |= run([sys.executable, "scripts/plot_metrics.py",
               "--dataset", cfg["dataset"], "--k", str(cfg["ks"][0] if cfg['ks'] else 10),
               "--out_dir", str(PLOTS)], dry=dry)
    # sweep plot defaults to w_meta vs ndcg@10
    rc |= run([sys.executable, "scripts/plot_sweeps.py",
               "--logs_dir", str(LOGS),
               "--dataset", cfg["dataset"],
               "--sweep_key", "w_meta",
               "--metric", f"ndcg@{cfg['ks'][0] if cfg['ks'] else 10}"], dry=dry)
    return rc

def exec_report(cfg: Dict[str, Any], dry: bool = False) -> int:
    title = cfg["title"] or f"{cfg['dataset']} — results"
    out_dir = REPORTS / cfg["dataset"] / f"{ts()}_{title}"
    k = (cfg["ks"][0] if cfg["ks"] else 10)

    cmd = [
        sys.executable, "-m", "src.agents.report_agent",
        "--dataset", cfg["dataset"],
        "--tag", title,
        "--k", str(k),
        "--include-compare",
        "--top-n", "3",
        "--prefer-faiss",
        "--metrics_csv", str(LOGS / "metrics.csv"),
        "--plots_dir", str(PLOTS),
        "--out", str(out_dir),
    ]
    rc = run(cmd, dry=dry)
    if rc == 0:
        print(f"📄 Report → {out_dir}")
    return rc

def exec_compare(cfg: Dict[str, Any], dry: bool = False) -> int:
    """Quick best-model recommendation from logs/metrics.csv filtered by dataset and k."""
    import pandas as pd
    csv_fp = LOGS / "metrics.csv"
    if not csv_fp.exists():
        print(f"Missing metrics CSV: {csv_fp}")
        return 1
    k = cfg["ks"][0] if cfg["ks"] else 10
    hit_col = f"hit@{k}"
    ndcg_col = f"ndcg@{k}"

    df = pd.read_csv(csv_fp)
    if "dataset" in df.columns:
        df = df[df["dataset"] == cfg["dataset"]]
    keep_cols = [c for c in ["run_name","dataset","fusion","w_text","w_image","w_meta",hit_col,ndcg_col] if c in df.columns]
    df = df[keep_cols].dropna(subset=[ndcg_col], how="any")
    if df.empty:
        print("No comparable rows found.")
        return 0

    # rank by ndcg desc, tiebreak by hit desc
    df["_rank"] = df[[ndcg_col, hit_col]].apply(tuple, axis=1)
    best = df.sort_values(by=[ndcg_col, hit_col], ascending=[False, False]).head(3)
    print("\n🏆 Top runs:")
    print(best[keep_cols].to_string(index=False))

    top = best.iloc[0].to_dict()
    print("\n✅ Recommendation:")
    print(json.dumps({
        "dataset": cfg["dataset"],
        "k": k,
        "recommended_run": top.get("run_name"),
        "fusion": top.get("fusion"),
        "weights": {
            "w_text": top.get("w_text"),
            "w_image": top.get("w_image"),
            "w_meta": top.get("w_meta"),
        },
        "metrics": {
            hit_col: top.get(hit_col),
            ndcg_col: top.get(ndcg_col),
        }
    }, indent=2))
    return 0

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Natural-language orchestrator for evals/plots/reports/compare.")
    ap.add_argument("--cmd", required=True, help="Free-form text command.")
    ap.add_argument("--dry", action="store_true", help="Print commands without executing.")
    args = ap.parse_args()

    cfg = parse_command(args.cmd)
    print("Parsed:", json.dumps(cfg, indent=2))

    intent = cfg["intent"]
    if intent == "run_one":
        rc = exec_run_one(cfg, dry=args.dry)
    elif intent == "run_grid":
        rc = exec_run_grid(cfg, dry=args.dry)
    elif intent == "plot":
        rc = exec_plot(cfg, dry=args.dry)
    elif intent == "report":
        rc = exec_report(cfg, dry=args.dry)
    elif intent == "compare":
        rc = exec_compare(cfg, dry=args.dry)
    else:
        print(f"Unknown intent: {intent}")
        rc = 2

    sys.exit(rc)

if __name__ == "__main__":
    main()