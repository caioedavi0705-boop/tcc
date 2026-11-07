# Transformer encoder

A third model joins the FD001 comparison: a small encoder-only Transformer in Keras, trained and searched with the **same protocol as LSTM and BiLSTM** (including engine-based validation and smooth-validation-curve training helpers). Architectural detail (PE, attention, `d_model`) lives in [transformer-architecture.md](transformer-architecture.md).

## Architecture

`build_model_transformer` in `src/models.py` uses the Functional API.

- A dense layer projects the 17 sensors to `d_model`.
- `SinusoidalPositionalEncoding` adds a fixed sinusoidal encoding over the 30 cycles. It is a registered Keras layer so checkpoints can reload the full graph.
- One to three `TransformerEncoderBlock` layers. Each block is multi-head self-attention, dropout, a residual connection and layer normalization, then a ReLU feed-forward block with the same residual pattern.
- Global average pooling reduces the sequence. An optional dense layer may follow. The output is one linear unit (RUL).
- Compile settings match the recurrent models: **`_rmsprop(learning_rate)`** (RMSprop with **`clipnorm=1.0`**), MAE loss, MSE metric, `jit_compile=True`.

## Data and internal validation

Windowing is unchanged: `(30, 17)` tensors from `create_fd001_windowed_features`. Internal train/validation split uses **`split_windowed_data_by_engine`** on `windowed.train_window_engine_ids` (80% / 20% of **engines**, `random_state=1`), not a random split of overlapping windows. See [smooth-validation-curves.md](smooth-validation-curves.md).

## Search

Hyperband: `max_epochs=15`, `factor=3`, one iteration, batch size 200, objective `val_loss`.

Search callbacks (`create_search_callbacks`): early stopping with **`restore_best_weights=True`**, **`min_delta=1e-3`**, plus **`ReduceLROnPlateau`** (`factor=0.5`, `patience=2`, `min_lr=1e-5`). No checkpoint during search.

Transformer Hyperband choices:

- Encoder blocks: 1, 2, or 3
- `d_model`: 32, 64, 128
- Heads: 2 or 4 (`d_model` must be divisible by the head count)
- Feed-forward width: 64, 128, 256
- Dropout: same list as LSTM/BiLSTM
- Learning rate: **`LEARNING_RATE_CHOICES = [0.001, 0.0001, 0.00001]`** (0.01 removed for stability)

`src/transformer.ipynb` runs search under **`hyper_transformer_hyperband`** (`overwrite=False` resumes prior trials). Existing projects may still reflect trials from the old LR grid until you use a new project name or `overwrite=True` once.

## Training, history plots, and repeated evaluation

After Hyperband, the notebook:

1. **`fit_model_for_history`** — clones the best model, trains from **fresh weights** (up to 80 epochs, batch 200, patience 10) with full training callbacks. Does **not** mutate the Hyperband `best_model` used for repeated eval. Saves:
   - `data/processed/transformer_history_fit.weights.h5`
   - `data/processed/transformer_history_fit.weights_history.json`
2. **`plot_training_history`** → `data/processed/figures/transformer_training_history.png` (best-`val_loss` marker, 3-epoch validation MA overlay).
3. **`run_repeated_train_eval`** — 10 continued fits on the **same** `best_model` (up to 30 epochs, batch 200, patience 5), **`capture_first_history=False`**. After each iteration, reloads best weights from the checkpoint path.

Checkpoints: the notebook passes `CHECKPOINT = data/processed/transformer_model.h5`; `create_training_callbacks` normalizes this to **`transformer_model.weights.h5`** for Keras `save_weights_only=True`. Repeated-eval loops use the same normalized path for `load_weights`.

## Saved artifacts

| Path | Role |
| --- | --- |
| `data/processed/transformer_model.weights.h5` | Best weights during 10× repeated eval |
| `data/processed/transformer_history_fit.weights.h5` | Best weights from history-fit run (plots only) |
| `data/processed/results/transformer_results.csv` | 10 iterations for Wilcoxon |
| `data/processed/results/transformer_best_params.json` | Winning Hyperband hyperparameters |
| `data/processed/figures/transformer_training_history.png` | Smoothed validation curves (history fit) |
| `data/processed/figures/transformer_rul_*.png` | Test diagnostics after repeated eval |

Legacy `transformer_model_first_history.json` (from old `capture_first_history=True` on iteration 1) is no longer produced by the notebook; use `*_history_fit.weights_history.json` for curve JSON.

## Comparison

Run **`src/comparacao_wilcoxon.ipynb`** after the three model notebooks. It loads the CSVs and runs Wilcoxon signed-rank at α = 0.05 (pairs: LSTM–BiLSTM, LSTM–Transformer, BiLSTM–Transformer). Summaries live under **`output/`** (e.g. `comparison_1.md`, `comparison_2.md`).

**Comparison 1** used random window validation and older callbacks; **Comparison 2** reflects the engine split and callback changes — metrics are **not directly comparable** across those reports.

## Related docs

- [transformer-architecture.md](transformer-architecture.md) — `d_model`, positional encoding, self-attention (no training protocol).
- [smooth-validation-curves.md](smooth-validation-curves.md) — motivation and shared `src/models.py` / notebook checklist.
- [faster-hyperparameter-search.md](faster-hyperparameter-search.md) — Hyperband and batch 200 baseline.
