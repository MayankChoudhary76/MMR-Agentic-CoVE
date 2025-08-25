# src/agents/orchestrator.py
from __future__ import annotations

from src.agents.types import Task
from src.agents.data_agent import DataAgent
from src.agents.index_agent import IndexAgent, IndexConfig


class Orchestrator:
    def __init__(self):
        self.data = DataAgent()
        self.index = IndexAgent()

    def run(self, task: Task):
        intent = (task.intent or "").lower()

        if intent == "prepare":
            # data pipeline: join_meta -> text_emb -> image_emb -> meta_emb
            self.data.prepare(task.dataset)
            print("✓ Data preparation complete.")
            return

        if intent == "index":
            # use Task.faiss_name as the output index name
            cfg = IndexConfig(
                dataset=task.dataset,
                fusion=task.fusion or "concat",
                w_text=task.w_text if task.w_text is not None else 1.0,
                w_image=task.w_image if task.w_image is not None else 1.0,
                w_meta=task.w_meta if task.w_meta is not None else 0.0,
                out_name=task.faiss_name or "",   # <-- map faiss_name → out_name
            )
            self.index.build(cfg)
            return

        raise ValueError(f"Unknown intent: {task.intent!r}")