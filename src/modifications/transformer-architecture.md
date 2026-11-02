# Transformer architecture (FD001 encoder-only)

Deep-dive notes for the model in `build_model_transformer` (`src/models.py`). **§2** covers **`d_model`** and multi-head sizing; **§3** covers sinusoidal positional encoding and how cycle order is handled in the recurrent baselines. Later sections (encoder blocks, self-attention, pooling) may be added here or live in the TCC draft.

---

## 2. `d_model`: the model’s working dimension

### 2.1 Definition in your code

`d_model` is the **size of the vector that represents one cycle** inside the Transformer stack. Hyperband chooses it from **`{32, 64, 128}`**. It appears in:

1. **Input projection** — `Dense(d_model)` maps **17 → d_model** at each of the 30 timesteps.
2. **Encoder blocks** — self-attention and FFN operate in **d_model**-dimensional space; residuals require matching shapes.
3. **Pooling and head** — after `GlobalAveragePooling1D`, the vector has length **d_model** (before an optional extra `Dense`).

`d_model` is **capacity**: how many dimensions the network uses to encode sensor combinations and cross-cycle context at each timestep. It is not the window length (30) and not the raw sensor count (17).

### 2.2 The projection layer as “embedding”

```python
inputs = Input(shape=input_shape)
x = Dense(d_model, name="project")(inputs)
x = SinusoidalPositionalEncoding(name="positional_encoding")(x)
```

For each cycle \(t\) and sensor vector \(x_t \in \mathbb{R}^{17}\):

\[
e_t = W_{\text{proj}} x_t + b_{\text{proj}} \in \mathbb{R}^{d_{\text{model}}}
\]

The same \(W_{\text{proj}}\) is applied at every timestep; **which cycle is which** is added later via positional encoding and attention.

### 2.3 Relationship to multi-head attention

In each `TransformerEncoderBlock`:

```python
self.attention = MultiHeadAttention(
    num_heads=self.num_heads,
    key_dim=self.d_model // self.num_heads,
)
```

Keras `MultiHeadAttention` uses **`key_dim` per head**. The attention output width is **`num_heads × key_dim`**. Your code enforces:

\[
\text{num\_heads} \times \text{key\_dim} = \text{num\_heads} \times \frac{d_{\text{model}}}{\text{num\_heads}} = d_{\text{model}}
\]

Example from the tuned model: **`d_model = 32`**, **`num_heads = 2`** → **`key_dim = 16`**. After all heads run, their outputs are concatenated back to **32** dimensions so the residual **`inputs + attention`** remains shape-compatible.

**Search constraint:** `d_model` must be divisible by `num_heads`. Choices **32 / 64 / 128** with **2 or 4** heads always satisfy this.

#### Split vs projection (what “16 per head” means)

With **`d_model = 32`** and two heads, **`key_dim = 16`** does **not** mean the 32-D vector is **cut in half** so head 1 uses coordinates 0–15 and head 2 uses 16–31.

Each head receives the **full** **`d_model`** vector at every timestep and applies **its own learned linear maps** into a **16-D** space for attention. The 16 is the size of **Q, K, and V inside that head**, not a fixed slice of the original channels.

For each cycle, let \(x_t \in \mathbb{R}^{32}\) be the block input at that timestep. For head \(h \in \{1, 2\}\) (same idea as in “Attention Is All You Need”):

\[
Q_h = x_t \, W^Q_h \in \mathbb{R}^{16}, \quad
K_h = x_t \, W^K_h \in \mathbb{R}^{16}, \quad
V_h = x_t \, W^V_h \in \mathbb{R}^{16}
\]

Each \(W^Q_h, W^K_h, W^V_h\) is conceptually **`32 × 16`**: all **32** input components feed each head, but each head **projects** into its own **16-D** subspace for scaled dot-product attention. Head \(h\) outputs \(z_h \in \mathbb{R}^{16}\); heads are **concatenated** to \([z_1; z_2] \in \mathbb{R}^{32}\), then typically mixed with an output projection **`W^O`** ( **`32 × 32`** ) back to **`d_model`**.

| Idea | Meaning | This project |
|------|---------|--------------|
| **Literal split** | Head 1 uses dims 0–15, head 2 uses 16–31 | **No** |
| **Learned projection** | Each head maps \(\mathbb{R}^{32} \to \mathbb{R}^{16}\) with different weights | **Yes** |

**Intuition:** two experts each read the **same** 32-number cycle vector; each writes a **16-number** summary for matching and blending. They do **not** each read only half of the coordinates. Head 1 and head 2 can emphasize **different linear combinations** of the same sensors and context because \(W^Q_1 \neq W^Q_2\), etc.

**Why `num_heads × key_dim = d_model`:** this is a **design choice** so concatenated head outputs match **`d_model`** for residuals. It does **not** imply that the input vector was physically partitioned into equal chunks.

**Short answer:** each head builds a **16-D representation from the full 32-D vector** via learned Q/K/V projections; those spaces are **parallel views**, then merged back to **32-D**—not a hard split of the 32 original dimensions.

### 2.4 Scale and regularization intuition

Larger **`d_model`** increases parameters in projection, attention, and FFN, which raises capacity and overfitting risk on FD001. The tuned **`d_model = 32`** is small relative to typical NLP Transformers and fits short windows (**L = 30**) and comparison with LSTM hidden sizes in **`{32, 64, 128, 256}`**.

---

## 3. Sinusoidal positional encoding and cycle index

### 3.1 Role in the Transformer

Self-attention is **permutation-equivariant** over timesteps unless the model knows **where** each cycle sits in the window. After `Dense(d_model)`, every cycle is a **`d_model`** vector built only from **that cycle’s 17 sensors** (same projection weights at all timesteps). `SinusoidalPositionalEncoding` adds a **fixed**, **non-learned** vector that depends on index **`pos ∈ {0, …, 29}`** inside the window:

```python
return inputs + encoding  # element-wise; sin/cos by dimension (see models.py)
```

That marks **absolute position within the 30-cycle snippet**, not a separate “cycle ID” feature in the raw data. Attention can then use both **sensor patterns** and **when** they occurred in the window.

### 3.2 Comparison with LSTM and BiLSTM (positional tracking)

The LSTM and BiLSTM in `build_model_lstm` / `build_model_bilstm` use the **same inputs**: **`(30, 17)`** windows from `src/features.py`, with **no** extra channel for cycle index and **no** positional encoding layer. Cycle order is handled differently—not by an explicit PE-like signal, but by **recurrence**.

**Unidirectional LSTM**

Keras consumes timesteps in order **0 → 29**. Each step updates hidden and cell state:

\[
h_t = \mathrm{LSTM}(x_t,\, h_{t-1},\, c_{t-1})
\]

“Which index we are in the window” is **implicit**: it is not added as a vector, but encoded in **how many recurrent steps have run** and **what is stored in \(h_t\)** after seeing cycles \(0…t\). On the **last** LSTM layer, `return_sequences=False`, so the regression head uses **one** vector—the state after the full forward pass—rather than 30 explicit position-tagged vectors like the Transformer encoder output before pooling.

**Bidirectional LSTM**

`Bidirectional(LSTM(...))` runs a **forward** LSTM (0 → 29) and a **backward** LSTM (29 → 0), then **concatenates** the two outputs. Each direction still respects **processing order**; the backward pass also carries **distance-from-the-end-of-the-window** information. That is richer positional context than a single LSTM, but still **learned recurrent mixing**, not a fixed sin/cos map of index \(t\).

**Side-by-side (this project)**

| Aspect | Transformer | LSTM / BiLSTM |
|--------|-------------|---------------|
| Explicit cycle index in inputs | Yes — sinusoidal PE on projected vectors | No |
| How order enters the model | Full window self-attention; PE labels **index in window** | **Sequential** state updates (two directions in BiLSTM) |
| Where “position” mainly lives | **Input representation** (projection + PE) before attention | **Hidden state evolution** over time |
| Representation before the RUL head | **30** context vectors → `GlobalAveragePooling1D` | Typically **one** summary vector (final LSTM / BiLSTM states) |

**Same inputs, fair comparison:** all three models see identical **`(30, 17)`** tensors. You did **not** concatenate a normalized cycle number or sin/cos index to the 17 sensors for the RNNs—that would be the closest analogue to PE for recurrent models; omitting it keeps the comparison focused on **architecture**, not extra positional features for one family only.

**Conceptual takeaway:** LSTM and BiLSTM **do** track progression through the window, but through **step order and recurrent memory**, not a mechanism **equivalent** to `SinusoidalPositionalEncoding`. The Transformer makes **position explicit and fixed** in the representation; RNNs make it **implicit and learned** in the state dynamics.

---

## Other sections

Encoder blocks, self-attention, `GlobalAveragePooling1D`, and end-to-end data flow: project chat / TCC draft. Implementation summary: `src/modifications/transformer-model.md` and `readme.md` §6.
