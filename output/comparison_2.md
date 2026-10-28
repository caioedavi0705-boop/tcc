# Comparison 2 — LSTM vs BiLSTM vs Transformer (FD001)

Second three-model comparison after **smooth validation curves** changes: engine-held-out internal validation, stricter callbacks (`restore_best_weights`, `ReduceLROnPlateau`, best checkpoint reload after each iteration), and dedicated history-fit plots. **Not comparable to Comparison 1** (random window split and older callback defaults).

Generated: 2026-10-28 13:28 Hora oficial do Brasil

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
| MSE | 214.904 ± 4.937 | 228.372 ± 6.773 | 261.002 ± 2.922 |
| RMSE | 14.659 ± 0.168 | 15.111 ± 0.222 | 16.155 ± 0.090 |
| MAE | 10.481 ± 0.134 | 10.150 ± 0.166 | 11.848 ± 0.071 |
| R² | 0.783 ± 0.005 | 0.769 ± 0.007 | 0.736 ± 0.003 |

## Running times (`tempo`, seconds per iteration)

| Model | Mean | Std | Sum of 10 iterations |
| --- | ---: | ---: | ---: |
| LSTM | 200.849 | 75.253 | 2008.5 s |
| BiLSTM | 159.089 | 50.188 | 1590.9 s |
| Transformer | 60.146 | 18.061 | 601.5 s |

## Per-iteration tables

### LSTM

| Iteration | MSE | RMSE | MAE | R² | Time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 222.820 | 14.927 | 10.718 | 0.775 | 345.057 |
| 2 | 222.820 | 14.927 | 10.718 | 0.775 | 149.081 |
| 3 | 210.126 | 14.496 | 10.371 | 0.788 | 269.856 |
| 4 | 210.126 | 14.496 | 10.371 | 0.788 | 283.794 |
| 5 | 210.126 | 14.496 | 10.371 | 0.788 | 135.761 |
| 6 | 210.126 | 14.496 | 10.371 | 0.788 | 134.249 |
| 7 | 215.724 | 14.688 | 10.472 | 0.782 | 182.927 |
| 8 | 215.724 | 14.688 | 10.472 | 0.782 | 133.137 |
| 9 | 215.724 | 14.688 | 10.472 | 0.782 | 156.449 |
| 10 | 215.724 | 14.688 | 10.472 | 0.782 | 218.178 |

### BiLSTM

| Iteration | MSE | RMSE | MAE | R² | Time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 245.811 | 15.678 | 10.447 | 0.752 | 223.309 |
| 2 | 222.695 | 14.923 | 10.362 | 0.775 | 249.870 |
| 3 | 219.700 | 14.822 | 9.818 | 0.778 | 151.547 |
| 4 | 227.930 | 15.097 | 10.125 | 0.770 | 141.706 |
| 5 | 227.930 | 15.097 | 10.125 | 0.770 | 86.642 |
| 6 | 227.930 | 15.097 | 10.125 | 0.770 | 138.745 |
| 7 | 227.930 | 15.097 | 10.125 | 0.770 | 167.306 |
| 8 | 227.930 | 15.097 | 10.125 | 0.770 | 169.046 |
| 9 | 227.930 | 15.097 | 10.125 | 0.770 | 167.266 |
| 10 | 227.930 | 15.097 | 10.125 | 0.770 | 95.455 |

### Transformer

| Iteration | MSE | RMSE | MAE | R² | Time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 261.435 | 16.169 | 11.890 | 0.736 | 82.224 |
| 2 | 260.551 | 16.142 | 11.919 | 0.737 | 47.421 |
| 3 | 260.551 | 16.142 | 11.919 | 0.737 | 35.782 |
| 4 | 260.551 | 16.142 | 11.919 | 0.737 | 36.033 |
| 5 | 269.110 | 16.405 | 11.923 | 0.728 | 70.165 |
| 6 | 259.564 | 16.111 | 11.781 | 0.738 | 46.949 |
| 7 | 259.564 | 16.111 | 11.781 | 0.738 | 71.504 |
| 8 | 259.564 | 16.111 | 11.781 | 0.738 | 82.154 |
| 9 | 259.564 | 16.111 | 11.781 | 0.738 | 74.191 |
| 10 | 259.564 | 16.111 | 11.781 | 0.738 | 55.039 |

## Wilcoxon signed-rank (α = 0.05)

| Pair | MSE | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
| LSTM vs BiLSTM | p = 0.004 (significant) | p = 0.004 (significant) | p = 0.002 (significant) | p = 0.004 (significant) |
| LSTM vs Transformer | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) |
| BiLSTM vs Transformer | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) |

## Source artifacts

- `data/processed/results/lstm_results.csv`
- `data/processed/results/bilstm_results.csv`
- `data/processed/results/transformer_results.csv`
- `src/rerun_repeated_eval.py`
- `src/comparacao_wilcoxon.ipynb`
