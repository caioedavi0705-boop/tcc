# Comparison 3 — LSTM vs BiLSTM vs Transformer (FD001)

Third three-model comparison: **fresh Hyperband v2** (`hyper_*_hyperband_v2`, `overwrite=True`) on the smooth-validation protocol (engine-held-out internal validation, `ReduceLROnPlateau`, `restore_best_weights`, `fit_model_for_history`, best-weight reload after each repeated-eval iteration). **Not comparable** to Comparison 1 (random split / old callbacks) or Comparison 2 (recycled v1 Hyperband winners).

Generated: 2026-11-08 (after `nbconvert --execute` on all four notebooks)

## Run

| Item | Value |
| --- | --- |
| Workflow | `src/lstm.ipynb` → `bilstm.ipynb` → `transformer.ipynb` → `comparacao_wilcoxon.ipynb` |
| Wall time (nbconvert chain) | ~22.5 h on CPU (Nov 7–8, 2026) |
| Log | `data/processed/notebook_full_hyperband_v2.log` |
| Dataset | C-MAPSS FD001 |
| RUL cap | 130 |
| Window | length 30, step 1 |
| Internal split | 80% / 20% **by engine ID**, `random_state=1` |
| Tuner projects | `hyper_lstm_hyperband_v2`, `hyper_bi_hyperband_v2`, `hyper_transformer_hyperband_v2` |

## Protocol

- **New Hyperband** with `LEARNING_RATE_CHOICES = [0.001, 0.0001, 0.00001]` and `create_search_callbacks` (no v1 trial reuse).
- Final evaluation: 10 continued fits per model, up to 30 epochs, batch 200, early stopping patience 5, best checkpoint reload each iteration.
- Wilcoxon signed-rank, paired by iteration, α = 0.05 (`tempo` excluded).

## Selected hyperparameters (v2 winners)

| Model | Architecture (summary) | LR |
| --- | --- | ---: |
| LSTM | LSTM 64 → 128, dropout 0.5 / 0.2, dense 64 | 0.001 |
| BiLSTM | BiLSTM 128 → 64 → 32, dropout 0.4 / 0.4 / 0.3, dense 64 | 0.001 |
| Transformer | 2 encoder blocks, `d_model` 32, 2 heads, FF 64, dropout 0.2, dense 64 | 0.001 |

Full JSON: `data/processed/results/*_best_params.json`

## Test metrics — mean ± standard deviation (10 iterations)

| Metric | LSTM | BiLSTM | Transformer |
| --- | ---: | ---: | ---: |
| MSE | 227.822 ± 1.493 | **217.541 ± 1.090** | 255.367 ± 2.053 |
| RMSE | 15.094 ± 0.050 | **14.749 ± 0.037** | 15.980 ± 0.064 |
| MAE | 10.332 ± 0.215 | **10.182 ± 0.036** | 11.372 ± 0.034 |
| R² | 0.770 ± 0.002 | **0.780 ± 0.001** | 0.742 ± 0.002 |

## Running times (`tempo`, seconds per iteration)

| Model | Mean | Std | Sum of 10 iterations |
| --- | ---: | ---: | ---: |
| LSTM | 317.045 | 217.857 | 3170.5 s |
| BiLSTM | 785.162 | 446.698 | 7851.6 s |
| Transformer | **49.685** | 16.514 | 496.9 s |

## Wilcoxon signed-rank (α = 0.05)

| Pair | MSE | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
| LSTM vs BiLSTM | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.008 (significant) | p = 0.002 (significant) |
| LSTM vs Transformer | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) |
| BiLSTM vs Transformer | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) | p = 0.002 (significant) |

## Takeaways

1. Under **new v2 Hyperband**, BiLSTM has the best mean test MSE/RMSE/MAE/R²; LSTM is close; Transformer is weakest on error metrics but fastest in the repeated-eval loop.
2. All three winning trials selected **lr = 0.001** (new grid; no 0.01 trials).
3. Every pairwise Wilcoxon comparison is significant at α = 0.05 on MSE, RMSE, MAE, and R².

## Source artifacts

- `src/lstm.ipynb`, `src/bilstm.ipynb`, `src/transformer.ipynb`, `src/comparacao_wilcoxon.ipynb`
- `data/processed/results/lstm_results.csv`, `bilstm_results.csv`, `transformer_results.csv`
- `data/processed/keras_tuner/hyper_*_hyperband_v2/`
