"""Build output/comparison_2.md from updated results CSVs (post engine-split protocol)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    compare_models_wilcoxon,
    load_experiment_results,
    summarize_experiment_results,
)

RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results"
OUTPUT = PROJECT_ROOT / "output" / "comparison_2.md"


def _fmt_mean_std(summary, metric: str) -> str:
    return f"{summary.loc[metric, 'Média']:.3f} ± {summary.loc[metric, 'Desvio padrão']:.3f}"


def _wilcoxon_cell(p: float, alpha: float = 0.05) -> str:
    tag = "significant" if p < alpha else "not sig."
    return f"p = {p:.3f} ({tag})"


def main() -> None:
    lstm = load_experiment_results(RESULTS_DIR / "lstm_results.csv")
    bilstm = load_experiment_results(RESULTS_DIR / "bilstm_results.csv")
    transformer = load_experiment_results(RESULTS_DIR / "transformer_results.csv")
    sum_lstm = summarize_experiment_results(lstm)
    sum_bi = summarize_experiment_results(bilstm)
    sum_tr = summarize_experiment_results(transformer)

    pairs = (
        ("LSTM vs BiLSTM", lstm, bilstm),
        ("LSTM vs Transformer", lstm, transformer),
        ("BiLSTM vs Transformer", bilstm, transformer),
    )
    wilcoxon_rows: list[tuple[str, dict[str, float]]] = []
    for label, left, right in pairs:
        w = compare_models_wilcoxon(left, right)
        wilcoxon_rows.append((label, {row["metrica"]: row["p_value"] for _, row in w.iterrows()}))

    def iter_table(df) -> str:
        lines = [
            "| Iteration | MSE | RMSE | MAE | R² | Time (s) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"| {int(row['Iteração'])} | {row['MSE']:.3f} | {row['RMSE']:.3f} | "
                f"{row['MAE']:.3f} | {row['R2']:.3f} | {row['tempo']:.3f} |"
            )
        return "\n".join(lines)

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body = f"""# Comparison 2 — LSTM vs BiLSTM vs Transformer (FD001)

Second three-model comparison after **smooth validation curves** changes: engine-held-out internal validation, stricter callbacks (`restore_best_weights`, `ReduceLROnPlateau`, best checkpoint reload after each iteration), and dedicated history-fit plots. **Not comparable to Comparison 1** (random window split and older callback defaults).

Generated: {now}

## Run

| Item | Value |
| --- | --- |
| Repeated eval | `src/rerun_repeated_eval.py` |
| Wilcoxon | `src/comparacao_wilcoxon.ipynb` logic |
| Dataset | C-MAPSS FD001 |
| RUL cap | 130 |
| Window | length 30, step 1 |
| Internal split | 80% / 20% **by engine ID**, `random_state=1` |

## Protocol

- Hyperband winners unchanged (existing `hyper_*_hyperband` projects; LR grid in code narrowed for future searches).
- Final evaluation: 10 continued fits per model, up to 30 epochs, batch 200, early stopping patience 5, with best-weight reload after each `fit`.
- Wilcoxon signed-rank, paired by iteration, α = 0.05 (`tempo` excluded).

## Test metrics — mean ± standard deviation (10 iterations)

| Metric | LSTM | BiLSTM | Transformer |
| --- | ---: | ---: | ---: |
| MSE | {_fmt_mean_std(sum_lstm, 'MSE')} | {_fmt_mean_std(sum_bi, 'MSE')} | {_fmt_mean_std(sum_tr, 'MSE')} |
| RMSE | {_fmt_mean_std(sum_lstm, 'RMSE')} | {_fmt_mean_std(sum_bi, 'RMSE')} | {_fmt_mean_std(sum_tr, 'RMSE')} |
| MAE | {_fmt_mean_std(sum_lstm, 'MAE')} | {_fmt_mean_std(sum_bi, 'MAE')} | {_fmt_mean_std(sum_tr, 'MAE')} |
| R² | {_fmt_mean_std(sum_lstm, 'R2')} | {_fmt_mean_std(sum_bi, 'R2')} | {_fmt_mean_std(sum_tr, 'R2')} |

## Running times (`tempo`, seconds per iteration)

| Model | Mean | Std | Sum of 10 iterations |
| --- | ---: | ---: | ---: |
| LSTM | {sum_lstm.loc['tempo', 'Média']:.3f} | {sum_lstm.loc['tempo', 'Desvio padrão']:.3f} | {lstm['tempo'].sum():.1f} s |
| BiLSTM | {sum_bi.loc['tempo', 'Média']:.3f} | {sum_bi.loc['tempo', 'Desvio padrão']:.3f} | {bilstm['tempo'].sum():.1f} s |
| Transformer | {sum_tr.loc['tempo', 'Média']:.3f} | {sum_tr.loc['tempo', 'Desvio padrão']:.3f} | {transformer['tempo'].sum():.1f} s |

## Per-iteration tables

### LSTM

{iter_table(lstm)}

### BiLSTM

{iter_table(bilstm)}

### Transformer

{iter_table(transformer)}

## Wilcoxon signed-rank (α = 0.05)

| Pair | MSE | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
"""
    for label, pmap in wilcoxon_rows:
        body += (
            f"| {label} | {_wilcoxon_cell(pmap['MSE'])} | {_wilcoxon_cell(pmap['RMSE'])} | "
            f"{_wilcoxon_cell(pmap['MAE'])} | {_wilcoxon_cell(pmap['R2'])} |\n"
        )

    body += """
## Source artifacts

- `data/processed/results/lstm_results.csv`
- `data/processed/results/bilstm_results.csv`
- `data/processed/results/transformer_results.csv`
- `src/rerun_repeated_eval.py`
- `src/comparacao_wilcoxon.ipynb`
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(body, encoding="utf-8")
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
