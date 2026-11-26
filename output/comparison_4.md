# Comparison 4 — LSTM vs BiLSTM vs Transformer (FD001)

Fourth comparison: **same Hyperband v2 winners as Comparison 3** (fixed `*_best_params.json`), **no new search** — repeated train/valid/test with updated figures (no MA val overlay; MAE ticks every 5; scatter ticks every 10). **Not comparable** to Comparisons 1–2. Comparison 3 metrics/figures preserved under `data/processed/comparison_3/`.

Generated: 2026-11-23 (`src/run_comparison_4.py`)

## Protocol

- Hyperparameters: frozen from Comparison 3 (`data/processed/comparison_3/results/*_best_params.json` snapshot at run start).
- Internal split: 80% / 20% **by engine ID**, `random_state=1`.
- `fit_model_for_history` then `run_repeated_train_eval`: 10×, up to 30 epochs, batch 200, patience 5.

## Test metrics — mean ± standard deviation (10 iterations)

| Metric | LSTM | BiLSTM | Transformer |
| --- | ---: | ---: | ---: |
| MSE | 232.605 ± 3.725 | 220.319 ± 2.335 | 253.009 ± 3.759 |
| RMSE | 15.251 ± 0.121 | 14.843 ± 0.079 | 15.906 ± 0.118 |
| MAE | 10.385 ± 0.302 | 10.544 ± 0.055 | 10.859 ± 0.120 |
| R² | 0.765 ± 0.004 | 0.778 ± 0.002 | 0.745 ± 0.004 |

## Running times (`tempo`, seconds per iteration)

| Model | Mean | Std | Sum of 10 iterations |
| --- | ---: | ---: | ---: |
| LSTM | 181.799 | 107.401 | 1818.0 s |
| BiLSTM | 806.053 | 510.299 | 8060.5 s |
| Transformer | 54.593 | 23.186 | 545.9 s |

## Wilcoxon signed-rank (α = 0.05)

| Pair | MSE | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
| LSTM vs BiLSTM | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.025 (significant) | p = 0.002 (significant) |
| LSTM vs Transformer | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.006 (significant) | p = 0.002 (significant) |
| BiLSTM vs Transformer | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) |

## Source artifacts

- `output/comparison_4.md` (this file)
- `data/processed/comparison_3/` — frozen Comparison 3 results and figures
- `data/processed/results/*_results.csv` — Comparison 4 metrics
- `data/processed/figures/*_training_history.png`, `*_rul_scatter.png`
- Log: `data/processed/comparison_4.log`
