import subprocess, sys
from typing import Literal

class DataAgent:
    """
    Runs data prep scripts for a dataset:
    - join_meta.py
    - build_text_emb.py
    - build_image_emb.py
    - build_meta_emb.py (optional; comment out if not needed)
    """

    def _run(self, argv):
        print("→", " ".join(argv))
        subprocess.check_call(argv)

    def prepare(self, dataset: Literal["beauty"]="beauty"):
        # 1) join reviews + meta
        self._run([sys.executable, "scripts/join_meta.py", "--dataset", dataset])
        # 2) build embeddings
        self._run([sys.executable, "scripts/build_text_emb.py", "--dataset", dataset])
        self._run([sys.executable, "scripts/build_image_emb.py", "--dataset", dataset])
        # If you use meta embeddings, keep this; otherwise comment it:
        self._run([sys.executable, "scripts/build_meta_emb.py", "--dataset", dataset])
        print("✓ Data preparation complete.")
