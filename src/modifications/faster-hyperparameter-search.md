# Faster hyperparameter search

The hyperparameter search in `src/models.py` was the slow part of the FD001 comparison. This change keeps the same model family and the same 10 full-data training repetitions, and spends less time discarding weak architectures.

## What was slow

`search_hyperparameters` used Keras Tuner `BayesianOptimization` with 15 trials and 5 epochs each. The search call did not set `batch_size`, so Keras used its default of 32. Final training in `run_repeated_train_eval` already used 200. On the FD001 windows (17,731 sequences of length 30), that meant about six times more optimizer steps per search epoch.

Every trial, including 4-layer networks with 512 units, ran for the full 5 epochs. The 512-unit stacks, especially BiLSTM, dominated the runtime. The unit list in code was `[32, 64, 128, 256, 512]`, while `readme.md` already described the range `[32, 256]`.

## What changed

In `search_hyperparameters`:

- `BayesianOptimization` was replaced by `Hyperband`.
- `max_epochs=15`, `factor=3`, `hyperband_iterations=1`.
- Most trials stop after 1 or 5 epochs. Only promoted configurations continue, up to 15 epochs.
- Search uses `batch_size=200`, the same size as the final training.
- Each trial uses `EarlyStopping` on `val_loss` with patience 3.
- The search no longer passes `epochs=` into `tuner.search`. Hyperband sets the epoch budget of each trial.
- The old arguments `max_trials` and `search_epochs` were replaced by `max_epochs`, `factor`, `hyperband_iterations`, `batch_size`, and `patience`.

Other search settings:

- `LSTM_UNIT_CHOICES` is `[32, 64, 128, 256]`. The 512-unit option was removed.
- LSTM and BiLSTM models are compiled with `jit_compile=True`, so both the search and the later `model.fit` calls use the compiled model. If XLA fails on a machine, set `jit_compile=False` in the two `compile()` calls.
- The objective is still `val_loss`. That loss is MAE, not MSE.

In `src/comparacao_final.ipynb` the two search calls use project names `hyper_lstm_hyperband` and `hyper_bi_hyperband`, with `overwrite=False`. Old Bayesian logs under `hyper_lstm` and `hyper_bi` cannot be resumed by Hyperband. The new names avoid that clash and let a later rerun continue finished trials.

`readme.md` section 6 documents Hyperband, batch size 200, early stopping, the unit set `{32, 64, 128, 256}`, and validation MAE as the search objective.

## What stayed the same

- Final comparison: 10 repetitions, full FD001 windows, up to 30 epochs, batch size 200, early stopping patience 5.
- Architecture family: 1–4 recurrent layers, one optional dense layer, dropout in `[0.2, 0.5]`, and RMSprop learning rates `{0.01, 0.001, 0.0001}`.
- The same model object is still reused across the 10 repetitions. Each repetition continues training; it is not a new random initialization.

## First measured run

The first execution after this change, on CPU with TensorFlow 2.21 and no GPU, finished in 4 h 49 min 34 s. Hyperband reported 1 h 17 min for the LSTM search and 2 h 12 min for the BiLSTM search. The per-iteration training times are in `data/processed/summaries/run_001_hyperband_fd001.md`. There is no earlier full Bayesian run in this repository, so that file is a baseline for later runs rather than a before/after speedup.
