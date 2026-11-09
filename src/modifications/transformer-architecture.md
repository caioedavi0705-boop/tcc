# Transformer architecture (FD001 encoder-only)

Deep-dive notes for the model in `build_model_transformer` (`src/models.py`). **§2** — **`d_model`** and multi-head sizing; **§3** — sinusoidal PE (implementation, math, Vaswani formula, relative position) vs LSTM/BiLSTM; **§5** — self-attention (weighted sum of values). Encoder blocks and pooling may be added in later sections.

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

Self-attention is **permutation-equivariant** over timesteps unless the model knows **where** each cycle sits in the window. After `Dense(d_model)`, every cycle is a **`d_model`** vector built only from **that cycle’s 17 sensors** (same projection weights at all timesteps). `SinusoidalPositionalEncoding` adds a **fixed**, **non-learned** vector that depends on index **`pos ∈ {0, …, 29}`** inside the window.

That marks **absolute position within the 30-cycle snippet**, not NASA’s global engine cycle counter as a separate raw feature (unless that information is already reflected in the sensors). Attention can then use both **sensor patterns** and **when** they occurred in the window.

**What PE is not (this project):**

- Not learned (no trainable weights in the PE layer).
- Not an extra 18th sensor column — it is added in **`d_model`** space after the 17→`d_model` projection.
- Not used in LSTM/BiLSTM (see **§3.5**).

### 3.2 Implementation: `SinusoidalPositionalEncoding.call`

In Keras, custom layers implement **`call(...)`** for the forward pass. Source: `src/models.py`.

```python
def call(self, inputs, training=None):
    seq_len = ops.shape(inputs)[1]
    d_model = ops.shape(inputs)[2]
    position = ops.cast(ops.arange(seq_len)[:, None], inputs.dtype)
    dimension = ops.cast(ops.arange(d_model)[None, :], inputs.dtype)
    depth = ops.cast(d_model, inputs.dtype)
    angles = position / ops.power(10000.0, (2 * (dimension // 2)) / depth)
    even = ops.equal(ops.cast(dimension, "int32") % 2, 0)
    encoding = ops.where(even, ops.sin(angles), ops.cos(angles))
    return inputs + encoding
```

**Input shape:** `inputs` is **`(batch, 30, d_model)`**. Positional encoding is **identical for every sample** in the batch; only **`pos`** along the window changes.

| Line / variable | Shape (typical) | Meaning |
|-----------------|-----------------|---------|
| `seq_len` | scalar **30** | Window length (time axis). |
| `d_model` | scalar **32** (or 64/128) | Channels per cycle after projection. |
| `position` | **(30, 1)** | Window index **0…29** (`[:, None]` = column for broadcasting). |
| `dimension` | **(1, d_model)** | Channel index **0…d_model−1**. |
| `depth` | scalar | Same numeric value as **`d_model`**; used in the wavelength exponent. |
| `angles` | **(30, d_model)** | Angle fed to sin/cos for each **(pos, channel)** pair. |
| `even` | **(1, d_model)** | True for even channel indices → use **sin**; odd → **cos**. |
| `encoding` | **(30, d_model)** | Fixed positional matrix. |
| `return inputs + encoding` | **(batch, 30, d_model)** | Element-wise add: projected sensors + position fingerprint per cycle. |

**`training=None`:** unused here (no dropout); Keras still passes it.

**Python `//` in this layer:** **`dimension // 2`** is **integer (floor) division**, not a comment. Examples: `0//2=0`, `1//2=0`, `2//2=1`, `3//2=1`. It assigns dimension pairs **(0,1), (2,3), …** to the same wavelength index **k**. Elsewhere in the project, **`d_model // num_heads`** is the same operator (e.g. `32 // 2 → 16`).

### 3.3 Mathematics: angles, sin/cos pairs, and addition

**Paper definition (Vaswani et al., *Attention Is All You Need*, 2017).** For dimension pair index **i** (i = 0, 1, …, d_model/2 − 1) and position **pos**:

\[
PE_{(pos,\, 2i)}   = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

Same angle for the sin/cos pair; your code uses **`k = dimension // 2`** instead of **i**, and **`depth`** instead of **`d_model`** in code.

**Wavelength and frequency form.** Define:

\[
\lambda_i = 10000^{\,2i/d_{\text{model}}}, \qquad
\text{angle}(pos, i) = \frac{pos}{\lambda_i} = pos \cdot \omega_i, \quad \omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}
\]

Your implementation computes `angles = position / 10000**((2 * (dimension // 2)) / depth)`, i.e. **`pos / λ_k`** with **k = dimension // 2**.

**Why several frequencies?** Small **i** (small **λ**): encoding changes quickly as **pos** increases (fine resolution). Large **i** (large **λ**): slow change along the window (coarse resolution). Each position vector mixes many scales.

**Tiny example (`d_model = 4`, three positions).** Pairs (dim 0,1) use **λ₀ = 1**; (dim 2,3) use **λ₁ = 10000^(2/4) = 100**.

| pos | dim 0 (sin) | dim 1 (cos) | dim 2 (sin pos/100) | dim 3 (cos) |
|-----|-------------|-------------|---------------------|-------------|
| 0 | 0 | 1 | 0 | 1 |
| 1 | sin(1) | cos(1) | sin(0.01)≈0.01 | cos(0.01)≈1 |
| 2 | sin(2) | cos(2) | sin(0.02) | cos(0.02) |

Row **pos** is the vector **added** to the projected representation of cycle **pos** in the window.

**Why add to `inputs`?** Similar sensor readings at different cycles would yield similar projected vectors; **adding** PE shifts them differently so self-attention can distinguish **“same sensors, different index in the window.”**

### 3.4 Origin of the formula, base 10000, and relative position

**Where ωₖ comes from.** The frequency/wavelength law is **not** derived from C-MAPSS or RUL physics. It is the **positional encoding scheme from Vaswani et al. (2017)** — an architectural choice so that:

1. **Absolute** index **pos** is encoded without extra learned parameters.
2. **Sin/cos pairs** at multiple wavelengths mix **fine** and **coarse** position scales in one vector.
3. For a fixed offset **k**, **PE(pos + k)** can be written as a **linear function** of **PE(pos)** (trigonometric addition formulas), which helps models represent **relative** offsets, not only absolute indices.

**Why base 10000?** In the original paper with **`d_model = 512`**, **λ₀ = 1** and the largest **λ** is on the order of **10⁴**, so the slowest dimensions vary little over typical sentence lengths but still differ over very long ranges. **10000** is a **hyperparameter of the encoding**, not a fitted constant. This project keeps the same formula for **`d_model ∈ {32, 64, 128}`** (fewer sin/cos pairs when **d_model** is small; exponents **2i/d_model** still geometrically space wavelengths across pairs).

**Relative position hint (dot products).** **Absolute** position: “cycle **7** in the window.” **Relative** position: “these two cycles are **k** steps apart.” RUL patterns often depend on **lags** (e.g. trend over the last few cycles).

Each **pos** has a vector **PE(pos)**. The dot product **PE(pos) · PE(pos + k)** measures how aligned two position vectors are. With sinusoids at multiple frequencies, that alignment can depend strongly on **k** (the gap), not only on **pos** — so the geometry encodes **distance in the window**, not only “where on the line.”

**Link to self-attention (qualitative).** Queries and keys are built from **projected sensors + PE** through learned layers. Attention scores involve **q · k**, so position can influence which cycles match. The model **learns** whether to use lag-like patterns; PE supplies a **standard inductive bias** toward relative offsets. It is **not** guaranteed that every head implements literal “look k cycles back” — that is learned behavior aided by this encoding.

**One-sentence summary:** Sinusoidal PE from Vaswani et al. uses **pos / 10000^(2i/d_model)** so each index gets a fixed fingerprint and pairwise comparisons can emphasize **separation k**; your **`call`** implements the same angles and adds them to projected cycle vectors.

**TCC citation:** Vaswani et al. (2017), positional encoding (Section 3.5 in the paper).

### 3.5 Comparison with LSTM and BiLSTM (positional tracking)

The LSTM and BiLSTM in `build_model_lstm` / `build_model_bilstm` use the **same inputs**: **`(30, 17)`** windows from `src/features.py`, with **no** extra channel for cycle index and **no** `SinusoidalPositionalEncoding` layer. Cycle order is handled by **recurrence**, not by a fixed sin/cos map added to features.

**Unidirectional LSTM**

Keras feeds timesteps in order **0 → 29**. Each step updates hidden and cell state:

\[
h_t = \mathrm{LSTM}(x_t,\, h_{t-1},\, c_{t-1})
\]

| Symbol | Meaning |
|--------|---------|
| **\(x_t\)** | Scaled **17** sensor values at cycle **t** in the window. |
| **\(h_{t-1}\)** | Hidden state after cycle **t − 1** (compressed past). |
| **\(c_{t-1}\)** | Cell state (longer internal memory). |
| **\(h_t\)** | New summary after observing cycle **t**. |

“Which index we are in the window” is **implicit**: the same **\(x_t\)** at a different step would produce a different **\(h_t\)** because **\(h_{t-1}\)** differs. There is no **`call`** that adds **PE(pos)** to **\(x_t\)**.

On the **last** LSTM layer, **`return_sequences=False`**, so the RUL head uses **one** vector (state after cycle **29**), not **30** position-tagged vectors like the Transformer encoder before pooling.

**Bidirectional LSTM**

`Bidirectional(LSTM(...))` runs:

1. **Forward** LSTM: **0 → 29**.
2. **Backward** LSTM: **29 → 0** on the same **`(30, 17)`** sequence.

Keras **concatenates** the two outputs (e.g. forward and backward hidden sizes sum on the feature axis). That adds **distance from window start** and **distance from window end**, but still via **learned gates and states**, not **`SinusoidalPositionalEncoding.call`**.

**Side-by-side (this project)**

| Aspect | Transformer | LSTM / BiLSTM |
|--------|-------------|---------------|
| Explicit cycle index in inputs | Yes — sinusoidal PE on projected vectors (**§3.2–3.4**) | No |
| How order enters the model | Self-attention + PE labels **index in window** | **Sequential** state updates (two directions in BiLSTM) |
| Where “position” mainly lives | **Input representation** (projection + PE) | **Hidden state evolution** over time |
| Representation before the RUL head | **30** vectors → `GlobalAveragePooling1D` | Typically **one** summary vector (final LSTM / BiLSTM states) |

**Same inputs, fair comparison:** all three models see identical **`(30, 17)`** tensors. You did **not** concatenate a normalized cycle index or sin/cos features to the **17** sensors for the RNNs (that would be the closest PE analogue for RNNs).

**Conceptual takeaway:** LSTM and BiLSTM **do** track progression through the window via **step order and memory**; the Transformer adds **explicit, fixed** position in **`d_model`** space before attention. Different mechanisms, same window tensor.

**Section map**

```text
§3.2  SinusoidalPositionalEncoding.call  → build encoding[pos, :] → inputs + encoding
§3.5  LSTM / BiLSTM                      → order via h_t chain (and two directions in BiLSTM)
```

---

## 5. Self-attention (inside the encoder block)

Self-attention runs inside each `TransformerEncoderBlock` in `src/models.py`:

```python
attention = self.attention(inputs, inputs, training=training)
```

Query, key, and value all come from the **same** sequence **`(batch, 30, d_model)`** — that is **self-attention** (no decoder, no cross-attention to a second sequence).

### 5.1 “For each timestep, a weighted sum of all value vectors”

**Timestep** means **one cycle** in the window. After projection and positional encoding you have **30 timesteps**, each represented by a **`d_model`** vector.

For every cycle, the layer builds three vectors from that cycle’s data (via learned linear maps):

| Symbol | Role |
|--------|------|
| **Query (Q)** | “What am I looking for?” |
| **Key (K)** | “What do I offer for matching?” |
| **Value (V)** | “What information do I pass on if another cycle selects me?” |

In self-attention, Q, K, and V are all derived from **the same 30 cycles**. The **value** at cycle \(s\) is \(v_s\): the content cycle \(s\) contributes when other cycles “listen” to it.

**Core idea:** pick one cycle \(t\) (e.g. cycle 25). The layer does **not** leave that cycle’s vector unchanged in isolation. It **replaces** it with a **new** vector built as follows:

1. Compare cycle \(t\)’s **query** to **every** cycle’s **key** (including \(t\)) → scores for “how relevant is cycle \(s\) when updating cycle \(t\)?”
2. Turn scores into **weights** \(\alpha_{t,0}, \ldots, \alpha_{t,29}\) that are **non-negative** and **sum to 1** (softmax).
3. **New vector at cycle \(t\)**:

\[
\text{new}_t = \alpha_{t,0}\, v_0 + \alpha_{t,1}\, v_1 + \cdots + \alpha_{t,29}\, v_{29}
\]

So at cycle \(t\) the output is a **blend of all 30 value vectors**. If \(\alpha_{t,28}\) is large, the representation at \(t\) depends strongly on cycle 28; if \(\alpha_{t,5}\) is small, cycle 5 contributes little.

**“For each timestep”** means this procedure runs for **every** \(t \in \{0,\ldots,29\}\). Each cycle can use **different** weights over the same 30 values, so cycle 10 and cycle 29 can “look at” the window differently.

**One-line summary:** for **each** cycle, self-attention asks how much to copy from **every** cycle in the window, and sets that cycle’s new vector to that **weighted average of all value vectors**. Weights are **learned** (through Q/K projections and training), not fixed by hand.

**RUL intuition (30 cycles, 17 sensors):** raw input at cycle 20 is mostly “sensors at cycle 20.” After one self-attention layer, the vector at cycle 20 can encode “sensors at 20 **plus** weighted evidence from other cycles” (e.g. a trend over 15–19, or the last cycles if the model learns that). That mixing **is** the weighted sum of values.

### 5.2 Scores, scaling by \(\sqrt{d_k}\), the attention matrix, and \(\alpha_{t,s}\)

**Step 1 — raw compatibility (score).** For one head, with query vector \(q_t\) and key vector \(k_s\) (both length **`d_k`**, e.g. **`d_k = 16`** when **`d_model = 32`** and **2 heads**):

\[
e_{t,s} = \frac{q_t \cdot k_s}{\sqrt{d_k}}
\]

**\(e_{t,s}\)** is the **score** (logit): “how well does key at cycle **s** match query at cycle **t**?” before normalization.

**Why divide by \(\sqrt{d_k}\)?** The dot product **adds** **`d_k`** terms: \(q_t \cdot k_s = \sum_{j=1}^{d_k} q_{t,j}\, k_{s,j}\). If those terms are roughly order **1** on average, their **sum** tends to grow with **`d_k`** (roughly like **\(\mathcal{O}(\sqrt{d_k})\)** to **\(\mathcal{O}(d_k)\)** in magnitude depending on correlations). **Large scores** make **softmax** extremely **peaky** (one weight ≈ **1**, all others ≈ **0**), so small changes in inputs barely change the output — **gradients** during training can **vanish**. Dividing by **\(\sqrt{d_k}\)** **rescales** scores so their typical size stays **stable** when you change head width (same trick as in Vaswani et al.; Keras `MultiHeadAttention` uses it internally). Intuition: you are **counteracting** summing **`d_k`** products so softmax sees **moderate** logits, not giants that grow automatically with **`key_dim`**.

**Step 2 — attention weights.** For a **fixed** query cycle **t**, softmax runs over all source cycles **s**:

\[
\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'=0}^{29} \exp(e_{t,s'})}
\]

**Properties:** every **\(\alpha_{t,s} \geq 0\)** and **\(\sum_s \alpha_{t,s} = 1\)** for each **t**. So for cycle **t**, the **\(\alpha_{t,\cdot}\)** row is a **probability distribution** over “which cycles to listen to.”

**What a high \(\alpha_{t,s}\) means.** Fix **t** (the cycle being **updated**). **Large \(\alpha_{t,s}\)** means: when building the new representation at cycle **t**, the layer takes a **large fraction** of its information from the **value vector** **\(v_s\)** at cycle **s**. In words: **“cycle t is strongly attending to cycle s.”** It does **not** by itself mean “s caused t” in a physical sense — it is what **learned** Q/K matching chose for this window. If **\(\alpha_{t,s}\)** is high, other weights **\(\alpha_{t,s'}\)** in the same row must be smaller because the row sums to **1**. Low **\(\alpha_{t,s}\)** means cycle **s** contributes little to **t**’s new vector.

**Output (same as §5.1):** \(\text{new}_t = \sum_s \alpha_{t,s}\, v_s\).

**Attention matrix — scores or weights?** For one head, both are **30 × 30** tables (one row per query cycle **t**, one column per key/source cycle **s**):

| Matrix | Entry | Name | Typical use |
|--------|--------|------|-------------|
| **Pre-softmax** | **\(e_{t,s}\)** | **Scores** (scaled dot products) | Debugging; can be any real number |
| **Post-softmax** | **\(\alpha_{t,s}\)** | **Attention weights** | Weighted sum of values; heatmaps / interpretability |

People say “attention matrix” for **either**; in papers and plots it usually means **\(\alpha\)** (weights). Implementation order: **scores → softmax (row-wise over s) → weights → multiply/add values**. Your code calls `MultiHeadAttention` once; that bundle happens inside the layer.

**Multi-head note:** each head has its own **\(e^{(h)}\)** and **\(\alpha^{(h)}\)**; head outputs are concatenated back to **`d_model`** (§2.3).

### 5.3 Tiny numeric example (3 cycles, scalar values)

Suppose there are only **3** cycles and value scalars \(v_0=1\), \(v_1=10\), \(v_2=100\). For cycle \(t=2\), imagine softmax weights \(\alpha_{2,0}=0.1\), \(\alpha_{2,1}=0.2\), \(\alpha_{2,2}=0.7\) (these are **\(\alpha\)**, not raw scores):

\[
\text{new}_2 = 0.1 \cdot 1 + 0.2 \cdot 10 + 0.7 \cdot 100 = 73
\]

The representation at timestep 2 is **73**, not **100** — a **mix** driven by the attention weights. High **\(\alpha_{2,2}=0.7\)** means cycle 2’s output is mostly built from its **own** value **\(v_2\)**, with smaller pulls from cycles 0 and 1. With **`d_model = 32`**, each \(v_s\) is a 32-D vector; the same weighted-sum logic runs inside `MultiHeadAttention` with **§5.2** scores and scaling.

### 5.4 Contrast with one LSTM step (same window)

- **LSTM at cycle 20:** updates state mainly from **\(x_{20}\)** and **\(h_{19}\)**; cycle 25 is not in the formula for \(h_{20}\) directly (only through earlier chained states).
- **Self-attention at cycle 20:** cycle **25** can contribute **in one layer** with weight \(\alpha_{20,25}\).

That is why attention is described as **global within the window**: any cycle can interact with any other cycle in a single block (your window uses **no causal mask**, so “future” cycles inside the 30-window are allowed — appropriate when the full 30-cycle history is observed at once).

### 5.5 Where this sits in the block

Attention output is dropped out, added to the block input (**residual**), and layer-normalized; then the feed-forward sublayer runs **per timestep**. See `TransformerEncoderBlock.call` in `src/models.py`. Per-head Q/K/V projection (not a channel split) is in **§2.3**.

### 5.6 Summary: multi-head attention and learned head roles

**“Heads are not manually assigned.”** The model never fixes roles such as “head 1 = local trends, head 2 = end of window.” Each head has its **own learned** maps \(W^Q, W^K, W^V\) into **`key_dim`** (§2.3). Training on RUL updates those weights; if one head favors neighbors and another favors distant cycles, that **emerges from data**, not from hand-written rules.

**Why several heads in one layer?**

| Idea | Role in this project |
|------|----------------------|
| **Parallel views of the window** | Several attentions run **at once** on the same 30 cycles, each in a different subspace, then concatenate back to **`d_model`**. Each cycle’s vector can reflect **multiple cycle–cycle patterns** in **one** encoder block. |
| **Depth vs heads** | **Depth** = stacking **encoder blocks** (e.g. 2 in the tuned model). **Heads** add **breadth** (diverse attention maps per layer), not extra layers. |
| **Speed** | Multi-head **does not** make attention faster. It usually **increases** compute (H attentions + merge). Parallelism on GPU is throughput, not fewer training steps or a smaller model. |

**Main reasons (Vaswani-style motivation):**

1. **Capacity as parallel specialists** — several smaller attentions can capture **different** match types (e.g. short lag vs early/late contrast) before merging.
2. **Inductive bias** — one large dot-product is a single mixing pattern; multiple heads encourage **diverse** \(\alpha\) maps in the **same** layer.

**One sentence:** Multi-head self-attention applies **parallel** scaled dot-product attentions over the window with **learned, not hand-defined** roles, then **concatenates** so each cycle mixes information through **several relationship patterns** at once; **depth** comes from **encoder blocks**, while **heads** widen **per-layer** expressiveness, **not** training speed.

---

## Other sections

Encoder block overview (residual + FFN), `GlobalAveragePooling1D`, and end-to-end data flow: TCC draft or future additions here. Implementation summary: `src/modifications/transformer-model.md` and `readme.md` §6.
