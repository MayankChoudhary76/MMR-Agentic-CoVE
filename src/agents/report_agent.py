import subprocess, sys

class ReportAgent:
    """Generates plots/reports from logs/metrics."""
    def build_reports(self, dataset: str="beauty"):
        print("→ generate plots/reports", dataset)
        subprocess.check_call([sys.executable, "scripts/plot_metrics.py", "--dataset", dataset])
        subprocess.check_call([sys.executable, "scripts/plot_sweeps.py", "--dataset", dataset])
        print("✓ Reports complete.")
