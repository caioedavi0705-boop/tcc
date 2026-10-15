# Comparison 1 — LSTM vs BiLSTM vs Transformer (FD001)

First three-model comparison after Hyperband search and the encoder-only Transformer. Numbers come from `data/processed/results/{lstm,bilstm,transformer}_results.csv` and `src/comparacao_wilcoxon.ipynb`.

## Run

| Item | Value |
| --- | --- |
| Transformer notebook | 2026-10-14 18:27 → 19:43 (-03); wall ~1 h 15 min |
| LSTM then BiLSTM notebooks | 2026-10-15 13:48 → 15:02 (-03); wall ~1 h 14 min |
| Wilcoxon notebook | 2026-10-15 (~18:15 -03); exit 0 |
| Hardware | CPU only (TensorFlow 2.21; no GPU on native Windows) |
| Dataset | C-MAPSS FD001 |
| Windows | `(17731, 30, 17)` train, `(10196, 30, 17)` test |
| RUL cap | 130 |
| Window | length 30, step 1 |
| Internal split | 80% train / 20% validation, `random_state=1` |

## Protocol

- Search: Keras Tuner **Hyperband**, `max_epochs=15`, `factor=3`, `hyperband_iterations=1`, batch size 200, early stopping on `val_loss` with patience 3. Objective: validation MAE. Notebooks used `overwrite=False` (LSTM/BiLSTM resumed prior Hyperband projects where present).
- Final evaluation: 10 continued fits per model, up to 30 epochs, batch size 200, early stopping patience 5.
- The same model object is reused across the 10 iterations (training continues; each iteration is not a cold start).
- Statistical test: Wilcoxon signed-rank, paired by iteration, α = 0.05. Column `tempo` is excluded from the test.

## Selected hyperparameters

| Model | Architecture |
| --- | --- |
| LSTM | LSTM 128, dropout 0.4; LSTM 64, dropout 0.4; dense 32; linear output; RMSprop lr 0.01 |
| BiLSTM | BiLSTM 32, dropout 0.2; BiLSTM 64, dropout 0.3; linear output; RMSprop lr 0.01 |
| Transformer | 2 encoder blocks; `d_model` 32; 2 heads; FF 256; dropout 0.3; dense 32; linear output; RMSprop lr 0.001 |

## Test metrics — mean ± standard deviation (10 iterations)

| Metric | LSTM | BiLSTM | Transformer |
| --- | ---: | ---: | ---: |
| MSE | 326.887 ± 76.697 | 329.008 ± 41.034 | 346.868 ± 37.479 |
| RMSE | 17.974 ± 2.056 | 18.104 ± 1.180 | 18.599 ± 1.032 |
| MAE | 13.753 ± 2.472 | **12.558 ± 1.120** | 14.040 ± 0.932 |
| R² | 0.670 ± 0.077 | 0.668 ± 0.041 | 0.650 ± 0.038 |

## Running times (`tempo`, seconds per iteration)

| Model | Mean | Std | Sum of 10 iterations |
| --- | ---: | ---: | ---: |
| LSTM | 215.971 | 25.955 | 2159.7 s (~36.0 min) |
| BiLSTM | 203.214 | 124.210 | 2032.1 s (~33.9 min) |
| Transformer | **153.473** | 45.472 | 1534.7 s (~25.6 min) |

On average, the Transformer continued-training loop was the fastest. BiLSTM had the largest spread across iterations. These figures cover only the 10 `fit` + evaluate loops, not Hyperband search wall time.

## Per-iteration tables

### LSTM

| Iteration | MSE | RMSE | MAE | R² | Time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 433.827 | 20.829 | 17.461 | 0.562 | 236.403 |
| 2 | 274.329 | 16.563 | 13.557 | 0.723 | 175.770 |
| 3 | 458.527 | 21.413 | 18.131 | 0.537 | 248.062 |
| 4 | 238.603 | 15.447 | 11.429 | 0.759 | 174.959 |
| 5 | 272.673 | 16.513 | 11.772 | 0.725 | 209.141 |
| 6 | 302.625 | 17.396 | 12.562 | 0.694 | 228.717 |
| 7 | 288.504 | 16.985 | 11.639 | 0.709 | 233.262 |
| 8 | 287.750 | 16.963 | 12.349 | 0.709 | 208.261 |
| 9 | 407.933 | 20.197 | 15.769 | 0.588 | 204.360 |
| 10 | 304.104 | 17.439 | 12.863 | 0.693 | 240.770 |

### BiLSTM

| Iteration | MSE | RMSE | MAE | R² | Time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 230.382 | 15.178 | 10.051 | 0.767 | 337.685 |
| 2 | 326.326 | 18.064 | 12.462 | 0.670 | 378.319 |
| 3 | 319.499 | 17.875 | 11.700 | 0.677 | 363.915 |
| 4 | 331.338 | 18.203 | 12.707 | 0.665 | 286.977 |
| 5 | 334.220 | 18.282 | 12.517 | 0.663 | 119.919 |
| 6 | 347.322 | 18.637 | 13.035 | 0.649 | 120.661 |
| 7 | 329.120 | 18.142 | 12.880 | 0.668 | 92.814 |
| 8 | 333.973 | 18.275 | 12.964 | 0.663 | 171.736 |
| 9 | 398.496 | 19.962 | 14.481 | 0.598 | 79.716 |
| 10 | 339.407 | 18.423 | 12.787 | 0.657 | 80.399 |

### Transformer

| Iteration | MSE | RMSE | MAE | R² | Time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 271.551 | 16.479 | 12.039 | 0.726 | 249.224 |
| 2 | 371.906 | 19.285 | 14.156 | 0.624 | 123.706 |
| 3 | 384.117 | 19.599 | 14.441 | 0.612 | 150.708 |
| 4 | 312.312 | 17.672 | 13.398 | 0.685 | 185.480 |
| 5 | 309.793 | 17.601 | 13.241 | 0.687 | 183.875 |
| 6 | 360.018 | 18.974 | 14.179 | 0.636 | 110.423 |
| 7 | 344.379 | 18.557 | 14.167 | 0.652 | 94.983 |
| 8 | 361.745 | 19.020 | 14.677 | 0.635 | 142.365 |
| 9 | 365.518 | 19.119 | 14.817 | 0.631 | 123.357 |
| 10 | 387.338 | 19.681 | 15.285 | 0.609 | 170.612 |

## Wilcoxon signed-rank (α = 0.05)

| Pair | MSE | RMSE | MAE | R² |
| --- | --- | --- | --- | --- |
| LSTM vs BiLSTM | p = 0.492 (not sig.) | p = 0.492 (not sig.) | p = 0.625 (not sig.) | p = 0.492 (not sig.) |
| LSTM vs Transformer | p = 0.432 (not sig.) | p = 0.275 (not sig.) | p = 0.557 (not sig.) | p = 0.432 (not sig.) |
| BiLSTM vs Transformer | p = 0.160 (not sig.) | p = 0.160 (not sig.) | **p = 0.002 (significant)** | p = 0.160 (not sig.) |

## Takeaways

1. On mean test metrics, BiLSTM has the best MAE; LSTM is close on MSE/RMSE/R²; the Transformer is slightly worse on average error metrics but fastest in the repeated-training loop.
2. The only statistically significant pairwise difference at α = 0.05 is **MAE for BiLSTM vs Transformer** (favoring BiLSTM on the observed means). All other metric–pair combinations are not significant.
3. LSTM vs BiLSTM differences are not significant on any reported metric in this protocol.

## Source artifacts

- `src/lstm.ipynb`, `src/bilstm.ipynb`, `src/transformer.ipynb`
- `src/comparacao_wilcoxon.ipynb`
- `data/processed/results/lstm_results.csv`
- `data/processed/results/bilstm_results.csv`
- `data/processed/results/transformer_results.csv`
- `data/processed/results/*_best_params.json`
