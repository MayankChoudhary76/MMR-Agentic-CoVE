# src/agents/index_agent.py
from __future__ import annotations
import subprocess
import sys
from dataclasses import dataclass

@dataclass
class IndexConfig:
    dataset: str
    fusion: str = "concat"          # "concat" | "weighted"
    w_text: float = 1.0
    w_image: float = 1.0
    w_meta: float = 0.0
    out_name: str = ""              # e.g. "beauty_concat_best"

class IndexAgent:
    def _run(self, argv: list[str]) -> None:
        # Run the CLI step in the same interpreter/venv
        subprocess.check_call(argv)

    def build(self, cfg: IndexConfig) -> None:
        args = [
            sys.executable, "scripts/build_faiss.py",
            "--dataset", cfg.dataset,
            "--fusion", cfg.fusion,
            "--w_text", str(cfg.w_text),
            "--w_image", str(cfg.w_image),
            "--w_meta", str(cfg.w_meta),
        ]
        if cfg.out_name:
            args += ["--out_name", cfg.out_name]
        print("→", " ".join(args))
        self._run(args)
        print("✓ Index build complete.")