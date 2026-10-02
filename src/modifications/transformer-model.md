# Transformer encoder

A third model joins the FD001 comparison: a small encoder-only Transformer in Keras, trained and searched with the same protocol as LSTM and BiLSTM.

## Architecture

`build_model_transformer` in `src/models.py` uses the Functional API.

- A dense layer projects the 17 sensors to `d_model`.
- `SinusoidalPositionalEncoding` adds a fixed sinusoidal encoding over the 30 cycles. It is a registered Keras layer so the `.h5` checkpoint can reload.
- One to three `TransformerEncoderBlock` layers. Each block is multi-head self-attention, dropout, a residual connection and layer normalization, then a ReLU feed-forward block with the same residual pattern.
- Global average pooling reduces the sequence. An optional dense layer may follow. The output is one linear unit (RUL).
- Compile settings match the recurrent models: RMSprop, MAE loss, MSE metric, `jit_compile=True`.

## Search

Hyperband is unchanged: `max_epochs=15`, `factor=3`, one iteration, batch size 200, early stopping on `val_loss` with patience 3.

Transformer choices:

- Encoder blocks: 1, 2, or 3
- `d_model`: 32, 64, 128
- Heads: 2 or 4 (`d_model` is divisible by the head count)
- Feed-forward width: 64, 128, 256
- Dropout and learning rate: the same lists as LSTM and BiLSTM

`src/transformer.ipynb` runs this search under `hyper_transformer_hyperband` and saves:

- `data/processed/transformer_model.h5`
- `data/processed/results/transformer_results.csv`
- `data/processed/results/transformer_best_params.json`
- `data/processed/transformer_model_first_history.json`

## Comparison

`src/comparacao_wilcoxon.ipynb` loads the three iteration tables and runs Wilcoxon signed-rank tests at alpha 0.05 for LSTM vs BiLSTM, LSTM vs Transformer, and BiLSTM vs Transformer. Each pair is reported on its own. LSTM and BiLSTM do not need to be trained again.
