import subprocess, sys

class ModelAgent:
    """Runs evaluation / sweeps for fusion strategies."""
    def eval(self, dataset: str="beauty"):
        print("→ eval fusion on", dataset)
        subprocess.check_call([sys.executable, "scripts/eval_fusion.py", "--dataset", dataset])
        print("✓ Evaluation complete.")
