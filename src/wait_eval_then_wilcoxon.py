"""Wait for repeated-eval CSVs, then run Wilcoxon + comparison_2.md."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results"
CSVS = (
    RESULTS_DIR / "lstm_results.csv",
    RESULTS_DIR / "bilstm_results.csv",
    RESULTS_DIR / "transformer_results.csv",
)
def all_fresh() -> bool:
    if not all(p.is_file() for p in CSVS):
        return False
    lstm_mtime = datetime.fromtimestamp(CSVS[0].stat().st_mtime)
    for path in CSVS[1:]:
        if datetime.fromtimestamp(path.stat().st_mtime) < lstm_mtime:
            return False
    return True


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    print("Waiting for BiLSTM + Transformer repeated-eval CSVs...", flush=True)
    while not all_fresh():
        time.sleep(120)
    print("All CSVs fresh — running Wilcoxon.", flush=True)
    from src.run_wilcoxon_comparison import main as run_wilcoxon

    run_wilcoxon(write_markdown=True)


if __name__ == "__main__":
    main()
