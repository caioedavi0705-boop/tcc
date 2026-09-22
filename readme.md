# Comparative Analysis of Deep Learning Models (LSTM vs BiLSTM) for Turbofan Engine Remaining Useful Life (RUL) Estimation

## 1. Project Overview
This repository contains an end-to-end Machine Learning pipeline developed in Python to estimate the **Remaining Useful Life (RUL)** of commercial aircraft turbofan engines using sensor time-series data. The core objective is to support **Predictive Maintenance (PdM)** by comparing two recurrent deep learning architectures:
- **Long Short-Term Memory (LSTM)**
- **Bidirectional Long Short-Term Memory (BiLSTM)**

The codebase is built using **TensorFlow/Keras**, **Scikit-Learn**, and **SciPy**.

---

## 2. Theoretical Background & Mathematical Formulation

### 2.1 Long Short-Term Memory (LSTM)
LSTM networks address the vanishing/exploding gradient problems inherent in standard Recurrent Neural Networks (RNNs) through a gating mechanism that controls information flow over long sequences.

For time step $t$, input vector $x_t$, previous hidden state $h_{t-1}$, and previous cell state $C_{t-1}$:

1. **Input Gate ($i_t$):** Controls how much new information enters the cell state.
   $$i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)$$

2. **Forget Gate ($f_t$):** Determines what information to discard from the previous cell state.
   $$f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)$$

3. **Candidate Cell State ($\tilde{C}_t$):** Generates new candidate values.
   $$\tilde{C}_t = \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)$$

4. **Cell State Update ($C_t$):** Combines forgotten past information and weighted new input information.
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

5. **Output Gate ($o_t$):** Decides the next hidden state.
   $$o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)$$

6. **Hidden State Update ($h_t$):**
   $$h_t = o_t \odot \tanh(C_t)$$

*Note:* $\sigma$ represents the Sigmoid activation function, and $\odot$ denotes element-wise Hadamard multiplication.

### 2.2 Bidirectional LSTM (BiLSTM)
The BiLSTM processes temporal sequences using two separate LSTM layers:
- **Forward Layer ($\vec{h}_t$):** Reads inputs from past to future ($t = 1 \rightarrow T$).
- **Backward Layer ($\overleftarrow{h}_t$):** Reads inputs from future to past ($t = T \rightarrow 1$).

The combined output layer aggregates representations from both temporal directions:
$$h_t = [\vec{h}_t, \overleftarrow{h}_t]$$

---

## 3. Problem Formulation & Target Modeling

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

## 4. Dataset Specification (NASA C-MAPSS - FD001)

- **Source:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS).
- **Subdataset:** **FD001**
  - **Operating Conditions:** 1 (Constant operational setting)
  - **Fault Modes:** 1 (HPT - High Pressure Turbine degradation)
  - **Training Engines:** 100 units (20,631 total sample observations)
  - **Test Engines:** 100 units (13,096 total sample observations + final ground-truth RUL values)

---

## 5. Preprocessing Pipeline

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

## 6. Hyperparameter Tuning & Network Architectures

Hyperparameter optimization was performed via **Bayesian Optimization** using `KerasTuner` over 15 trials with 5 epochs each. Loss optimized during search: **MSE**.

### Search Space
- **Recurrent Layers:** Up to 4 LSTM or BiLSTM layers.
- **Dense Layers:** Up to 2 Fully Connected layers.
- **Neurons per Layer:** Potentials of 2 in range $[32, 256]$.
- **Dropout Rate:** $[0.2, 0.5]$ with step $0.1$.
- **Learning Rate:** $\{0.01, 0.001, 0.0001\}$ using the `RMSprop` optimizer.
- **Activation Functions:**
  - Recurrent Layers: $\tanh$
  - Hidden Dense Layers: $\text{ReLU}$
  - Final Output Layer: Linear

---

## 7. Training & Validation Setup

- **Max Epochs:** 30
- **Batch Size:** 200 samples
- **Monitored Training Loss:** Mean Absolute Error (MAE)
- **Early Stopping Callback:** Patience = 5 epochs monitoring validation loss.
- **Statistical Replication Protocol:**
  - Each model is independently trained and validated across **10 random initialization seeds** to evaluate stability and performance distribution.

---

## 8. Evaluation Metrics & Statistical Testing

### 8.1 Performance Metrics
Given $n$ samples, ground-truth $y_i$, predicted $\hat{y}_i$, and ground-truth mean $\bar{y}$:

- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

### 8.2 Non-Parametric Statistical Inference
To evaluate statistical significance between paired model runs without assuming normality, the **Wilcoxon Signed-Rank Test** is applied on paired metrics ($n = 10$):

1. Compute paired differences $d_i = \text{Metric}_{\text{LSTM}}^{(i)} - \text{Metric}_{\text{BiLSTM}}^{(i)}$.
2. Rank absolute differences $|d_i|$ and compute sum of positive/negative ranks:
   $$R^+ = \sum_{d_i > 0} \text{rank}(|d_i|), \quad R^- = \sum_{d_i < 0} \text{rank}(|d_i|)$$
3. Test Statistic $T = \min(R^+, R^-)$.
4. Standardized $z$-score calculation:
   $$z = \frac{T - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}}$$
5. Significance Threshold: **$p\text{-value} < 0.05$** rejects the null hypothesis ($H_0$: median difference between models equals 0).

---
