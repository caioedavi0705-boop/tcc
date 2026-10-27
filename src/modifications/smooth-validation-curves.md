# Smooth validation curves (training history)

Validation MAE/MSE in the saved `*_training_history.png` figures oscillated epoch-to-epoch instead of declining smoothly. This change targets **optimization noise** (learning rate, no schedule), **noisy validation** (random window split), and **misleading plots** (warm-started first `fit` of the 10-iteration loop).

## Symptoms

- `val_loss` and `val_mean_squared_error` zig-zag every epoch.
- MSE panel often looks rougher than MAE (loss is MAE; MSE is only a metric).
- Curves reflect one continued `fit` after Hyperband, not a clean train-from-scratch segment.

## Root causes (this repo)

1. **RMSprop with lr up to 0.01** (LSTM/BiLSTM winners) and **no `ReduceLROnPlateau`**.
2. **`split_windowed_data`**: random 80/20 on overlapping windows (same engine in train and val).
3. **`plot_training_history`** uses `history` from **iteration 1** of `run_repeated_train_eval` on **Hyperband weights**.
4. **Callbacks**: no `restore_best_weights`; checkpoint not restricted to best epoch (secondary for zig-zag, still fixes evaluation).

## Implementation checklist

### 1. `src/features.py`

- Extend `FD001WindowedFeatures` with `train_window_engine_ids: np.ndarray`.
- Change `time_window` to return `(x, y, engine_ids)` (one motor ID per window).
- In `create_fd001_windowed_features`, pass `train_window_engine_ids` into the named tuple.

### 2. `src/models.py`

**Optimizer**

- Add `_rmsprop(learning_rate: float) -> RMSprop` with `clipnorm=1.0`.
- Use it in `build_model_lstm`, `build_model_bilstm`, and transformer `compile`.
- Narrow tuner grid: `LEARNING_RATE_CHOICES = [0.001, 0.0001, 0.00001]` (drop `0.01`).

**Callbacks**

- Import `ReduceLROnPlateau`.
- `create_training_callbacks`:
  - `EarlyStopping(monitor="val_loss", patience=..., restore_best_weights=True, min_delta=1e-3)`
  - `ModelCheckpoint(..., save_best_only=True, save_weights_only=True, mode="min")`
  - `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)`
- `create_search_callbacks(patience=3)` — same EarlyStopping + ReduceLROnPlateau (no checkpoint) for `search_hyperparameters`.

**Split**

- Add `split_windowed_data_by_engine(x, y, engine_ids, test_size=0.2, random_state=1)`:
  - `train_test_split` on **unique engine IDs**, mask windows by held-out engines.

**History fit (plots)**

- Add `fit_model_for_history(model, x_train, y_train, x_val, y_val, checkpoint_path, *, epochs=80, batch_size=200, patience=10)`:
  - `fresh = keras.models.clone_model(model)`
  - `fresh.compile_from_config(model.get_compile_config())` (fresh weights)
  - Single `fit` with `create_training_callbacks`
  - `fresh.load_weights(checkpoint_path)` before return (optional: return `fresh` only for plotting; keep `best_model` for repeated eval)

**Repeated eval**

- After each `model.fit` in `run_repeated_train_eval`, `model.load_weights(checkpoint_path)`.
- Default `capture_first_history=False`; notebooks use `fit_model_for_history` for plots.

**Plotting**

- In `plot_training_history`: vertical line at epoch of minimum `val_loss`; optional 3-epoch moving average on val series (dashed, labeled diagnostic).

### 3. Model notebooks (`src/lstm.ipynb`, `src/bilstm.ipynb`, `src/transformer.ipynb`)

Imports:

```python
from src.models import (
    ...
    fit_model_for_history,
    split_windowed_data_by_engine,
)
```

After `windowed = create_fd001_windowed_features(...)`:

```python
x1, x2, y1, y2 = split_windowed_data_by_engine(
    x_train,
    y_train,
    windowed.train_window_engine_ids,
    test_size=0.2,
    random_state=1,
)
```

After Hyperband, **before** or **after** `run_repeated_train_eval` (recommended: before, so repeated eval still starts from search weights):

```python
history = fit_model_for_history(
    best_model,
    x1,
    y1,
    x2,
    y2,
    CHECKPOINT.with_name("lstm_history_fit.weights.h5"),  # per model
)
plot_training_history(history, "LSTM", save_path=FIGURES_DIR / "lstm_training_history.png")

resultados, _ = run_repeated_train_eval(
    best_model,
    ...
    capture_first_history=False,
)
```

Use distinct history checkpoint filenames per model (`bilstm_history_fit.weights.h5`, `transformer_history_fit.weights.h5`).

### 4. Hyperband project names (optional)

If LR grid changes, use new tuner project names (e.g. `hyper_lstm_hyperband_v2`) or `overwrite=True` once to avoid mixing trials with the old 0.01 lr space.

## What stays the same

- 10× `run_repeated_train_eval` on the **same** `best_model` object (continued training protocol for Wilcoxon CSVs).
- FD001 test hold-out; batch size 200 for final loops unless you experiment later.
- Figures still under `data/processed/figures/` (gitignored).

## Verification

1. Run `src/lstm.ipynb` only; inspect `lstm_training_history.png` (smoother val MAE; MSE may still be noisier than MAE).
2. Confirm `*_history_fit.weights.h5` matches best `val_loss` epoch in saved JSON if you log history.
3. Re-run three models + `comparacao_wilcoxon.ipynb`; write `output/comparison_2.md` (not comparable to Comparison 1 if engine split + LR grid changed).

## Relation to other modification docs

- Builds on [faster-hyperparameter-search.md](faster-hyperparameter-search.md) (Hyperband, batch 200).
- Independent of [transformer-model.md](transformer-model.md).
