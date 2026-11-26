# Comparative Analysis of Deep Learning Models (LSTM vs BiLSTM vs Encoder-Only Transformer) for Turbofan Engine Remaining Useful Life (RUL) Estimation

## 1. Project Overview
This repository contains an end-to-end Machine Learning pipeline developed in Python to estimate the **Remaining Useful Life (RUL)** of commercial aircraft turbofan engines using sensor time-series data. The core objective is to support **Predictive Maintenance (PdM)** by comparing deep learning architectures for temporal regression:
- **Long Short-Term Memory (LSTM)**
- **Bidirectional Long Short-Term Memory (BiLSTM)**
- **Encoder-only Transformer**

The codebase is built using **TensorFlow/Keras**, **Scikit-Learn**, and **SciPy**.

---

## Project Folder Structure

The repository separates raw data, reusable Python modules, and exploratory notebooks:

```
tcc/
├── data/
│   ├── raw/
│   │   ├── training/     # train_FD001.txt (engine run-to-failure trajectories)
│   │   ├── test/         # test_FD001.txt (truncated engine trajectories)
│   │   └── result/       # RUL_FD001.txt (ground-truth RUL for test engines)
│   └── processed/        # Intermediate or derived artifacts from preprocessing
├── notebooks/            # Exploratory / experimental Jupyter notebooks
├── src/
│   ├── data.py           # Load, clean, and prepare C-MAPSS FD001 tables
│   ├── features.py       # Scaling, feature selection, and sliding-window tensors
│   ├── models.py         # LSTM/BiLSTM/Transformer build, tuning, training, and evaluation
│   ├── lstm.ipynb        # LSTM search, repeated training, and plots
│   ├── bilstm.ipynb      # BiLSTM search, repeated training, and plots
│   ├── transformer.ipynb # Transformer search, repeated training, and plots
│   └── comparacao_wilcoxon.ipynb  # Pairwise Wilcoxon comparison of the saved runs
├── output/               # Human-readable experiment comparison summaries (comparison_1.md, …)
├── readme.md
└── .gitignore
```

**How the folders are organized**

- **`data/`** — Local dataset storage (ignored by Git). `raw/` mirrors the NASA C-MAPSS FD001 split into training runs, test runs, and final RUL labels; `processed/` is reserved for cleaned or engineered outputs produced by the pipeline.
- **`src/`** — Core pipeline code, split by responsibility: ingestion (`data.py`), feature engineering and windowing (`features.py`), and model search/training/metrics (`models.py`). The experiment is split into `lstm.ipynb`, `bilstm.ipynb`, `transformer.ipynb`, and `comparacao_wilcoxon.ipynb`. The model notebooks save iteration tables under `data/processed/results/` for the Wilcoxon notebook.
- **`notebooks/`** — Space for additional exploratory analysis without mixing ad-hoc work into the reusable `src/` modules.
- **`output/`** — Written comparison reports (`comparison_*.md`) documenting protocol changes, mean metrics, and Wilcoxon tests across LSTM, BiLSTM, and Transformer runs. Each file is self-contained; read the header before comparing numbers across versions.

Typical flow: load from `data/raw/` via `src/data.py` → build windows with `src/features.py` → train LSTM, BiLSTM, and the Transformer from their notebooks with `src/models.py` → compare the saved tables in `comparacao_wilcoxon.ipynb` → summarize results under `output/comparison_*.md`.

---

## 2. Problem Formulation & Target Modeling

The physical degradation of a turbofan engine is framed as a **time-series regression task**.

For a given engine $i$ at operational cycle $t$:
$$N(t) = \text{current cycle count}$$
$$N_{\text{total}}^{(i)} = \text{total life cycles until failure}$$

### Piece-wise Linear Degradation Model
Initial operation cycles typically present negligible physical degradation. To reflect realistic failure dynamics, a **piece-wise linear target function** is used with a maximum target threshold set at $RUL_{\text{max}} = 130$ cycles:

$$RUL(t) = \begin{cases} 
130, & \text{if } N_{\text{total}}^{(i)} - N(t) \ge 130 \\
N_{\text{total}}^{(i)} - N(t), & \text{if } N_{\text{total}}^{(i)} - N(t) < 130 
\end{cases}$$

---

## 3. Dataset Specification (NASA C-MAPSS - FD001)

- **Source:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS).
- **Subdataset:** **FD001**
  - **Operating Conditions:** 1 (Constant operational setting)
  - **Fault Modes:** 1 (HPT - High Pressure Turbine degradation)
  - **Training Engines:** 100 units (20,631 total sample observations)
  - **Test Engines:** 100 units (13,096 total sample observations + final ground-truth RUL values)

---

## 4. Preprocessing Pipeline

1. **Data Cleaning:** Removal of missing values or invalid entries (if present).
2. **Feature Elimination:**
   - 24 original features (3 flight parameters + 21 sensor channels).
   - Features with zero variance across all time steps are dropped:
     - Flight parameter: `TRA` (Throttle Actuator Control)
     - Sensors: `Sensor 1`, `Sensor 5`, `Sensor 10`, `Sensor 16`, `Sensor 18`, `Sensor 19`
   - **Retained Features ($N_{\text{features}}$):** 17 dynamic variables.
3. **Feature Scaling (MinMax Normalization):**
   $$x_f = \frac{x_f - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$$
4. **Sliding Windowing Strategy:**
   - Sequential input tensors of shape `(batch_size, window_length, features)` = `(N, L, 17)`.
   - **Sequence Length ($L$):** $30$ cycles (selected based on the minimum engine run length of 31 cycles in test data).
   - **Stride ($step$):** $1$ cycle (maximizes window sample count and mitigates overfitting).

---

## 5. Hyperparameter Tuning & Network Architectures

Hyperparameter optimization was performed via **Hyperband** (`KerasTuner`) with `max_epochs=15`, `factor=3`, and one Hyperband iteration. Search training used **batch size 200** and **early stopping** on validation loss (patience 3). Objective minimized during search: validation **MAE** (`val_loss`).

### Search Space
- **Recurrent Layers:** Up to 4 LSTM or BiLSTM layers.
- **Dense Layers:** Up to 1 optional fully connected hidden layer (plus linear output).
- **Neurons per Layer:** $\{32, 64, 128, 256\}$.
- **Dropout Rate:** $[0.2, 0.5]$ with step $0.1$.
- **Learning Rate:** $\{0.01, 0.001, 0.0001\}$ using the `RMSprop` optimizer.
- **Activation Functions:**
  - Recurrent Layers: $\tanh$
  - Hidden Dense Layers: $\text{ReLU}$
  - Final Output Layer: Linear

### Transformer search space
The encoder-only Transformer uses the same Hyperband settings, loss, optimizer, dropout rates, and learning rates. Its own choices are:

- **Encoder blocks:** 1 to 3
- **Model dimension (`d_model`):** $\{32, 64, 128\}$
- **Attention heads:** $\{2, 4\}$
- **Feed-forward width:** $\{64, 128, 256\}$
- **Optional dense layer:** 0 or 1 hidden layer, with the same unit choices as the recurrent models

Each encoder block is multi-head self-attention plus a ReLU feed-forward block, both with residual connections and layer normalization. A fixed sinusoidal positional encoding is added after projecting the 17 sensors to `d_model`. The sequence is pooled by global average pooling before the linear RUL output.

---

## 6. Training & Validation Setup

- **Max Epochs:** 30
- **Batch Size:** 200 samples
- **Monitored Training Loss:** Mean Absolute Error (MAE)
- **Early Stopping Callback:** Patience = 5 epochs monitoring validation loss.
- **Statistical Replication Protocol:**
  - Each model is independently trained and validated across **10 random initialization seeds** to evaluate stability and performance distribution.

---

## 7. Evaluation Metrics & Statistical Testing

### 7.1 Performance Metrics
Given $n$ samples, ground-truth $y_i$, predicted $\hat{y}_i$, and ground-truth mean $\bar{y}$:

- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

### 7.2 Non-Parametric Statistical Inference
To evaluate statistical significance between paired model runs without assuming normality, the **Wilcoxon Signed-Rank Test** is applied on paired metrics ($n = 10$):

1. Compute paired differences $d_i = \text{Metric}_{\text{LSTM}}^{(i)} - \text{Metric}_{\text{BiLSTM}}^{(i)}$.
2. Rank absolute differences $|d_i|$ and compute sum of positive/negative ranks:
   $$R^+ = \sum_{d_i > 0} \text{rank}(|d_i|), \quad R^- = \sum_{d_i < 0} \text{rank}(|d_i|)$$
3. Test Statistic $T = \min(R^+, R^-)$.
4. Standardized $z$-score calculation:
   $$z = \frac{T - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}}$$
5. Significance Threshold: **$p\text{-value} < 0.05$** rejects the null hypothesis ($H_0$: median difference between models equals 0).

Pairwise Wilcoxon tests for all three models are reported in the latest `output/comparison_*.md` files (LSTM vs BiLSTM, LSTM vs Transformer, BiLSTM vs Transformer).

---

## 8. Experiment Comparison Reports (`output/`)

| Report | Summary |
| --- | --- |
| `comparison_1.md` | First three-model run after Hyperband search and the encoder-only Transformer. |
| `comparison_2.md` | Smooth validation-curve protocol (engine-held-out split, updated callbacks). **Not comparable** to Comparison 1. |
| `comparison_3.md` | Fresh Hyperband v2 winners on the smooth-validation protocol. **Not comparable** to Comparisons 1–2. |
| **`comparison_4.md`** | **Same frozen hyperparameters as Comparison 3** (`*_best_params.json`); no new search—10× repeated train/valid/test with updated training-history and scatter figures (no MA overlay on validation; rescaled MAE/scatter ticks). **Not comparable** to Comparisons 1–2; Comparison 3 artifacts remain under `data/processed/comparison_3/`. |

Use **`comparison_4.md`** for the current narrative on mean test MSE/RMSE/MAE/R², iteration runtimes, and Wilcoxon significance under the frozen-v3 protocol.

---
