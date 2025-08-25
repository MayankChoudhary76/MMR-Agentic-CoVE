from __future__ import annotations
import argparse

from src.agents.orchestrator import Orchestrator
from src.agents.types import Task


def parse_args():
    ap = argparse.ArgumentParser(description="Agentic orchestrator entrypoint")
    ap.add_argument("--intent", required=True, help="prepare | index")
    ap.add_argument("--dataset", required=True, help="dataset key, e.g. beauty")

    # knobs for index / shared
    ap.add_argument("--fusion", choices=["concat", "weighted"], default="concat")
    ap.add_argument("--w_text", type=float, default=1.0)
    ap.add_argument("--w_image", type=float, default=1.0)
    ap.add_argument("--w_meta", type=float, default=0.0)

    # use faiss_name everywhere (not out_name)
    ap.add_argument("--faiss_name", type=str, default=None,
                    help="FAISS index suffix, e.g. beauty_concat_best")

    # placeholders for future intents
    ap.add_argument("--user", type=str, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--use_faiss", action="store_true", default=True)
    ap.add_argument("--exclude_seen", action="store_true", default=True)

    return ap.parse_args()


def main():
    args = parse_args()
    orch = Orchestrator()

    task = Task(
        intent=args.intent,
        dataset=args.dataset,
        user=args.user,
        k=args.k,
        fusion=args.fusion,
        w_text=args.w_text,
        w_image=args.w_image,
        w_meta=args.w_meta,
        use_faiss=args.use_faiss,
        faiss_name=args.faiss_name,   # correct field
        exclude_seen=args.exclude_seen,
    )

    orch.run(task)


if __name__ == "__main__":
    main()
