# Character-Level Language Modelling with RNNs
### Next-Character Prediction using Seq-to-One Architecture

---

## What We're Building

A **character-level language model** that learns to predict the next character in a sequence of text.

- Trained on Shakespeare's complete works (~1 million characters)
- Uses a **Gated Recurrent Unit (GRU)** neural network
- Seq-to-One architecture: given $n$ characters, predict the next **one** character
- Can generate new Shakespeare-style text character by character

---

## Why Character-Level?

Instead of treating **words** as tokens, we treat every individual **character** as a token.

### Advantages
- **Tiny vocabulary** — only ~68 unique characters vs ~25,000 unique words
- **No out-of-vocabulary (OOV) problem** — every character is always known
- **No complex tokenization rules** — just `list(text)`
- **Can generate any word**, including names and invented words

### Trade-offs
- Sequences are much longer (need ~500 steps to generate ~80 words)
- Model must learn spelling, grammar, and meaning all from scratch
- More steps required to span the same semantic distance

---

## Part 1 — Data Preprocessing Pipeline

### Step 1: Load the Dataset

We use the **Tiny Shakespeare** dataset (~1 million characters of Shakespeare's plays).

```
Total characters: 1,115,393
Unique characters: 65
```

### Step 2: Character Tokenization

The simplest tokenizer possible:

```python
def char_tokenize(text):
    return list(text)
```

Every space, newline, letter, and punctuation mark becomes its own token.

### Step 3: Build the Vocabulary

Map every unique character to a unique integer index.

Special tokens added:
- `<PAD>` → index 0 (padding for variable-length batches)
- `<UNK>` → index 1 (unknown characters)
- `<EOS>` → index 2 (end of sequence)

**Result: Vocabulary of ~68 tokens** (compare: ~25,000 for word-level)

---

## Part 1 — Sliding Window Dataset

### The Seq-to-One Approach

We slide a window of `seq_length = 100` characters across the text.

Each window creates one training sample:

```
Full text:  "To be or not to be, that is..."

Window 1:
  Input:  "To be or not to be, that is t"  (100 chars)
  Target: "h"                                (1 char)

Window 2:
  Input:  "o be or not to be, that is th"  (100 chars)
  Target: "e"                                (1 char)
```

- **Input shape per sample:** `(100,)` — a sequence of 100 character indices
- **Target shape per sample:** scalar — a single character index

**Total training samples: ~1.1 million**

---

## Part 1 — Batching with Padding

### The `collate_fn`

When sequences in a batch have different lengths, we **pad** shorter ones with `<PAD>` (index 0).

```
Batch of 3 sequences:
  Seq 1: [45, 12, 33, ...]  length = 100  → no padding needed
  Seq 2: [7,  22, 6,  ...]  length = 80   → 20 zeros appended
  Seq 3: [18, 5,  44, ...]  length = 60   → 40 zeros appended
```

An **attention mask** (1 = real character, 0 = padding) is created alongside each sequence so the model knows what to ignore.

**Targets need NO padding** — they are already a 1D tensor of single character indices.

---

## Part 2 — The Model Architecture

### Three Layers

```
Input (batch_size, seq_length)
        ↓
[Embedding Layer]  64-dim character vectors
        ↓
[GRU Layer × 2]   512-dim hidden state — processes sequence with memory
        ↓
[Last Hidden State]  Take ONLY the final time step (seq-to-one!)
        ↓
[Linear Layer]     Map to vocabulary logits (batch_size, vocab_size)
```

### Why Only the Last Hidden State?

In **seq-to-one**, we only need **one prediction** per input sequence.

- We feed 100 characters through the GRU
- The GRU maintains a hidden state that compresses everything it has seen
- After the last character, we use that hidden state to predict the next character
- **No need to compute outputs at every time step** → more efficient

---

## Part 2 — GRU: Gated Recurrent Unit

### How GRUs Work

At each character position $t$, the GRU computes:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Where:
- $z_t$ = **update gate** — how much of the old memory to keep
- $\tilde{h}_t$ = **candidate hidden state** — new information from current character
- $h_t$ = **new hidden state** — the updated memory

### Why GRU over simple RNN?

Simple RNNs suffer from the **vanishing gradient problem** — gradients shrink to zero when backpropagating through many time steps, so the model forgets long-term patterns.

GRU's gates allow it to **selectively remember or forget** information, handling longer sequences much better.

---

## Part 2 — Hyperparameters

| Parameter | Character-Level | Word-Level | Reason |
|---|---|---|---|
| `vocab_size` | ~68 | ~25,000 | Only characters |
| `embedding_dim` | 64 | 128 | Tiny vocab needs small embeddings |
| `hidden_dim` | 512 | 256 | Must learn spelling + grammar |
| `num_layers` | 2 | 2 | Same depth |
| `seq_length` | 100 chars | 20 words | More steps for same context |
| `batch_size` | 64 | 32 | Cheaper forward pass |

---

## Part 2 — Training Loop

### Steps per Batch

1. **Forward pass** — feed 100-character input through model → get `(batch_size, vocab_size)` logits
2. **Compute loss** — `CrossEntropyLoss(output, target)` — no reshaping needed! (seq-to-one advantage)
3. **Backward pass** — compute gradients
4. **Gradient clipping** — clip norm to 5.0 to prevent exploding gradients
5. **Optimizer step** — Adam updates the weights

### Truncated Backpropagation Through Time (TBPTT)

The hidden state is **detached** between batches:

```python
if hidden is not None:
    hidden = hidden.detach()
```

This breaks the computational graph so gradients don't flow beyond a single batch, saving memory while still passing information forward.

---

## Part 2 — What the Model Learns

The model starts knowing nothing about language. Through training it learns:

| Phase | What it learns |
|---|---|
| Early epochs | Common character pairs (`th`, `he`, `in`) |
| Mid epochs | Common words (`the`, `and`, `that`) |
| Late epochs | Word boundaries, punctuation, style |
| After many epochs | Shakespearean vocabulary (`thou`, `thy`, `hath`) |

Loss decreases from ~4.2 → ~1.8 over 10 epochs (crossentropy over ~68 classes).

---

## Part 3 — Text Generation

### Auto-Regressive Character Generation

Generation is a loop:

```
Seed: "To be or not to be"

Step 1: Feed "To be or not to be" → predict ","
Step 2: Feed "o be or not to be," → predict " "
Step 3: Feed " be or not to be, " → predict "t"
...
```

At each step we keep a sliding window of the last `seq_length` characters.

### Temperature Sampling

Instead of always picking the most likely character (which becomes repetitive), we sample from a scaled distribution:

$$P_{\text{temp}}(c_i) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

| Temperature | Effect |
|---|---|
| `T = 0.5` | Conservative — repetitive but correct spelling |
| `T = 1.0` | Balanced — creative and mostly coherent |
| `T = 1.5` | Creative — novel sequences, some gibberish |

---

## Part 3 — Example Generated Text

**Seed:** `"First Citizen:\n"` | **Temperature:** `1.0`

```
First Citizen:
What say you to this? What say the king?
I have not seen the father of the heart,
That we have seen the people of the state,
And with the world is not the king of all,
The duke of York is not to be the lord.
```

The model has learned:
- ✅ Correct spelling of common words
- ✅ Line breaks and verse structure
- ✅ Shakespearean vocabulary
- ✅ Basic subject-verb agreement
- ⚠️ Long-range coherence is limited

---

## Part 3 — The Fundamental Limitation: Information Bottleneck

### The Sequential Bottleneck

Information from early characters must travel through every hidden state to influence the final prediction:

$$h_1 \rightarrow h_2 \rightarrow h_3 \rightarrow \cdots \rightarrow h_{100}$$

At each step, the hidden state must:
1. Remember everything important from before
2. Incorporate the new character
3. Pass updated information forward

**Information from position 1 is diluted by the time we reach position 100** — like a game of telephone across 100 steps.

### Why This Matters More at Character Level

- 100 characters ≈ only 15–20 words of semantic context
- The model has a shorter effective memory for meaning
- Relating the subject of a sentence to its verb many characters later is hard

---

## The Solution: Attention Mechanisms

### What We Need

| Problem | Solution |
|---|---|
| Sequential processing is slow | **Parallel processing** in Transformers |
| Long-range information decay | **Direct connections** between any two positions |
| Hidden state bottleneck | **Attention** — selectively focus on relevant characters |

### The Attention Idea

Instead of passing information through a chain of hidden states, **every position can directly attend to every other position**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This is exactly what **Transformers** use — and why they outperform RNNs on long-range tasks.

**This is the motivation for Days 7–9!**

---

## Word-Level vs Character-Level: Full Comparison

| **Aspect** | **Word-Level RNN** | **Character-Level RNN** |
|---|---|---|
| Tokenizer | Split on whitespace + punctuation | `list(text)` |
| Vocabulary size | ~25,000 | ~68 |
| Sequence length | 20 tokens | 100 tokens |
| Embedding dim | 128 | 64 |
| Hidden dim | 256 | 512 |
| OOV problem | Yes | No |
| Can generate new words | No | Yes |
| Steps to generate text | Fewer (word per step) | More (char per step) |
| Training convergence | Faster (more info per token) | Slower |
| Saved model | `shakespeare_rnn_model.pth` | `shakespeare_char_rnn_model.pth` |

---

## When to Use Each Approach

### Use Character-Level When:
- Vocabulary is dynamic (names, URLs, code, mixed languages)
- Morphologically rich languages (many word endings/forms)
- Tasks involving spelling (correction, generation of novel terms)
- Memory is constrained (tiny embedding table)

### Use Word-Level When:
- Standard English NLP with fixed vocabulary
- Faster training convergence is needed
- Semantic meaning per token is important

### Use Transformers When:
- Long-range dependencies matter (they always do!)
- Parallel training is needed for speed
- State-of-the-art results are required

---

## Key Takeaways

1. **Character tokenization is trivial** — `list(text)` is the entire tokenizer
2. **Tiny vocabulary eliminates OOV** — every character is always in the vocab
3. **The same seq-to-one architecture works at both granularities** — only data changes
4. **Longer sequences are needed** to capture the same semantic context as word-level
5. **Temperature controls creativity** — lower = safer, higher = more imaginative
6. **The RNN bottleneck is universal** — both word and character RNNs suffer from it
7. **Attention mechanisms are the solution** — Transformers attend to all positions at once

---

## What's Next?

- **Day 7:** Bigram language models and the attention mechanism from scratch
- **Day 8:** Multi-head attention and the Transformer block
- **Day 9:** Scaling up — full Transformer language models

The character-level RNN is a powerful baseline, but the limitations we've seen here are exactly why the field moved to Transformers. Everything we've built today is the foundation!
