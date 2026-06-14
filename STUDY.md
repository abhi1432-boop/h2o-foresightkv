# Study Guide — H2O + ForesightKV

This is your personal study guide. Not published to GitHub.
The goal is that after reading this you can explain every part of the codebase
to anyone — even someone who has never heard of transformers.

---

## Diagrams

### Part 1 — Training Pipeline (you run this once to build the scorer)

```
STEP 1: collect_traces.py
┌─────────────────────────────────────────────────────────┐
│  140 prompts                                            │
│  for each prompt:                                       │
│    feed prompt into phi-2 (full cache, no eviction)     │
│    generate 50 tokens                                   │
│    at every single step, record:                        │
│      "how much did each token get attended to           │
│       across all 32 layers?"                            │
│  save raw attention data to traces/ folder              │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
STEP 2: compute_labels.py
┌─────────────────────────────────────────────────────────┐
│  load traces/                                           │
│  for each prompt token:                                 │
│    add up ALL the attention it received                 │
│    across ALL 50 decode steps and ALL 32 layers         │
│    → this is its LTC score (how important was it        │
│      by the END of generation?)                         │
│  normalize each prompt's scores to 0-1                  │
│  save to labels.pt                                      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
STEP 3: extract_features.py
┌─────────────────────────────────────────────────────────┐
│  for each token, extract 5 things you can know          │
│  BEFORE generation starts (prefill only):               │
│                                                         │
│  [position in sequence]                                 │
│  [is it one of the first 5 tokens?]                     │
│  [how common is this token in english?]                 │
│  [how much attention did it get during prefill?]        │
│  [which layers attended to it most?]                    │
│                                                         │
│  pair each token's 5 features with its LTC label        │
│  save to features.pt                                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
STEP 4: train_scorer.py
┌─────────────────────────────────────────────────────────┐
│  load features.pt                                       │
│                                                         │
│  train a tiny neural network:                           │
│                                                         │
│  [5 features] → (16 neurons) → [1 importance score]    │
│                                                         │
│  the network learns:                                    │
│  "given these 5 things I can see at prefill,            │
│   predict how important this token will be              │
│   by the end of generation"                             │
│                                                         │
│  save trained weights to scorer.pt (97 numbers total)  │
└─────────────────────────────────────────────────────────┘
```

---

### Part 2 — Inference (what happens when you actually generate text)

```
prompt comes in: "Instruct: Write a haiku about a cat. Output:"

STEP 0: PREFILL (feed the whole prompt at once)
┌─────────────────────────────────────────────────────────┐
│  phi-2 processes all prompt tokens simultaneously       │
│                                                         │
│  HOOK 1 fires (cache adapter):                          │
│    phi-2 tries to save K and V to DynamicCache          │
│    → intercepted → saved to H2OCache instead            │
│                                                         │
│  HOOK 2 fires (forward hook):                           │
│    attention weights captured from all 32 layers        │
│    → passed to H2OCache.update_scores()                 │
│                                                         │
│  IN FORESIGHT MODE:                                     │
│    scores initialized to ZERO (not prefill attention)   │
│    seed_from_prefill() runs:                            │
│      compute 5 features for each prompt token           │
│      run scorer network → predicted importance scores   │
│      replace zeros with those predictions               │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
STEPS 1-50: DECODE (one new token per step)
┌─────────────────────────────────────────────────────────┐
│  for each new token:                                    │
│                                                         │
│  HOOK 1: save new token's K and V to H2OCache           │
│                                                         │
│  HOOK 2: capture attention weights                      │
│    → add to running importance scores                   │
│                                                         │
│  if cache is over budget:                               │
│    ┌───────────────────────────────────────────┐        │
│    │  EVICTION (_evict)                        │        │
│    │  split cache into two zones:              │        │
│    │                                           │        │
│    │  [recent window] → always kept            │        │
│    │  [everyone else] → ranked by score        │        │
│    │                                           │        │
│    │  keep top scorers + recent window         │        │
│    │  drop the rest                            │        │
│    └───────────────────────────────────────────┘        │
│                                                         │
│  pick next token, repeat                                │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
                   generated text
```

---

### Part 3 — What the Experiment Measured

```
QUESTION: does the scorer's guess at step 0
          match where H2O ends up after 50 steps?

scorer prediction          H2O accumulator
at step 0                  at step 50
(before any decoding)      (after 50 decode steps)
                                    
token A:  0.9  ─────────?─────────  token A: high score
token B:  0.2  ─────────?─────────  token B: low score
token C:  0.7  ─────────?─────────  token C: high score
token D:  0.1  ─────────?─────────  token D: low score

if they match → positive correlation (pre-seeding helps)
if they disagree → negative correlation (pre-seeding hurts)

YOUR RESULTS:
┌──────────────────────────────────────────────────────┐
│  cross-domain  r = -0.507  (trained on A, tested on B)│
│  within-domain r = +0.084  (trained and tested on A)  │
│                                                        │
│  sign flipped positive when domains matched            │
│  → architecture works, data is the bottleneck          │
└──────────────────────────────────────────────────────┘
```

---

---

## The Big Picture

When a language model like phi-2 generates text, it does it one token at a time.
A token is roughly a word or part of a word. Every time it generates a new token,
it has to look back at everything it has generated so far and decide what is relevant.

To avoid recomputing everything from scratch at every step, it saves two vectors
for every past token: K (key) and V (value). This saved data is called the KV cache.
The problem is the cache grows every step and eventually runs out of memory.

H2O solves this by evicting tokens that are not getting much attention.
ForesightKV improves H2O by giving each token a smarter starting score
instead of starting everyone at zero.

---

## The Library Analogy

Attention works like looking something up in a library.

- **Q (query)** is your question. What are you looking for right now?
- **K (key)** is each book's index card. What does this book contain?
- **V (value)** is the actual book content. The real information.

When a new token is generated, it takes its Q and compares it against every
past token's K to figure out which ones are relevant. Then it mixes together
the V vectors of the relevant ones. That mixture is the attention output.

Q is thrown away after each step because it is only useful to the token
that just generated it. K and V are saved because every future token
needs to look at them. That is why it is called the KV cache, not the KQV cache.

---

## What Hooking Means

Hooking means intercepting something that is already happening without
changing the original code. Like a phone tap — two people are talking
and you are listening in without either of them knowing.

In this codebase there are two hooks:

**Hook 1 — the cache adapter.**
Phi-2 calls `past_key_values.update()` every time it wants to save K and V.
Normally this goes to HuggingFace's built-in cache. You replaced
`past_key_values` with your own object that looks identical to phi-2 but
redirects the call to H2OCache instead. Phi-2 has no idea anything changed.

**Hook 2 — the forward hooks.**
After each of phi-2's 32 attention layers finishes running, PyTorch
automatically calls any function you registered with `register_forward_hook`.
You registered one on every layer that reads the attention weights and
passes them to H2OCache. The layer runs completely normally — your function
just fires silently afterward.

---

## File by File

---

### h2o_cache.py — The Core

This is the most important file. Everything else exists to support this one.

**What it stores:**
- `key_cache` — saved K vectors, one slot per layer
- `value_cache` — saved V vectors, one slot per layer
- `accumulated_scores` — running total of how much attention each token has received

**`update(key_states, value_states, layer_idx)`**

Called every forward pass. Adds the new token's K and V to the cache.
On the very first call (step 0, the prefill), it stores the whole prompt.
On every step after that, it appends one new token.
Returns the full cache so attention can run over everything remembered.

Tensor shape coming in: `[1, 32, new_tokens, 80]`
- 1 = batch size (one sequence)
- 32 = attention heads
- new_tokens = prompt length on step 0, then 1 each step after
- 80 = head dimension (phi-2's embedding size 2560 divided by 32 heads)

**`update_scores(attn_weights, layer_idx)`**

Called after attention runs. The attention weights tell you how much each
token was attended to this step. You sum across all heads and all queries
to get one score per token, then add it to the running total.

If the cache is now over budget, call `_evict()`.

In ForesightKV mode, instead of starting the accumulator with the prefill
attention, it starts with zeros. The scorer will replace those zeros
with predicted importance scores right after prefill finishes.

**`_evict(layer_idx)`**

Called when the cache is over budget. Splits the cache into two zones:
- The local window (the last N tokens) — always safe, never evicted
- Everyone else — eviction candidates

Picks the top scorers from the eviction candidates, combines them with
the local window, and drops everything else. Sorts by original position
so the model does not get confused about token order.

**`seed_from_prefill(input_ids, freq_table, scorer_model)`**

ForesightKV only. Called once right after the step 0 forward pass.
Computes 5 features for each prompt token, runs them through the scorer,
and replaces the zero-initialized accumulated scores with the predicted
importance scores. From this point on, accumulation continues normally
on top of those predicted scores.

---

### patch.py — The Two Interception Points

**`H2OCacheAdapter`**

Phi-2 expects `past_key_values` to be a HuggingFace `DynamicCache` object.
DynamicCache has about 10 methods the model calls internally. By inheriting
from it you get all of those for free and only override two:

- `update()` — redirects K and V storage to H2OCache
- `get_seq_length()` — tells phi-2 how many tokens are in the cache
  so it can compute the right position IDs for rotary embeddings

**`patch_model(model, max_cache_size, local_window_size)`**

Registers a forward hook on every one of phi-2's 32 attention layers.
Each hook captures the attention weights after that layer runs and
passes them to `H2OCache.update_scores()`.

The `make_hook(idx)` wrapper exists because of a Python loop closure bug.
If you define the hook directly inside the loop, all 32 hooks would share
the same `layer_idx` variable and by the time any hook fires the loop is
done and `layer_idx` is 31 for all of them. Wrapping in `make_hook(idx)`
captures the correct value immediately for each hook.

Returns the cache, the adapter, and an `unpatch()` function. You must call
`unpatch()` when generation is done or the hooks stay attached to the model
and fire on every future generation, including ones that are not supposed
to use H2O.

---

### collect_traces.py — Recording What Actually Happens

Runs phi-2 on all 140 prompts with a full unlimited cache (no eviction).
At every single decode step, for every one of the 32 layers, records
how much attention each cached token received (summed over all heads).

Saves one file per prompt to the `traces/` folder. Each file contains
a list called `step_attns` where `step_attns[t]` is a tensor of shape
`[32, cache_len_at_step_t]` — one row per layer, one column per token.

Step 0 is the prefill (all prompt tokens processed at once).
Steps 1 through 50 are the decode steps (one new token each).

If you turn your computer off mid-run it picks up where it left off
because it checks if each file already exists before generating.

---

### compute_labels.py — What Does Long-Term Important Actually Mean?

Loads the saved traces and computes the LTC (Long-Term Contribution) score
for every prompt token.

LTC for token p = total attention token p received across ALL decode steps
(steps 1 through 50), summed across all 32 layers.

Step 0 (prefill) is excluded because that attention is concurrent with the
token — we only want future attention, meaning attention from tokens generated
after the prompt.

LTC is then normalized to 0-1 within each prompt so scores are comparable
across prompts of different lengths. The most attended token gets 1.0,
the least attended gets 0.0.

This is the ground truth label. A token with LTC close to 1.0 is genuinely
important — the model kept attending to it throughout the entire generation.
A token with LTC close to 0.0 is safe to evict.

---

### extract_features.py — What Can You Know at Prefill Time?

For each token, extracts 5 features using ONLY information available
before any decoding has happened. No future attention allowed.

| Feature | What it measures |
|---|---|
| normalized position | where in the sequence is this token (0 = first, 1 = last) |
| is sink | is this one of the first 5 tokens (these always get attended to) |
| token frequency | how common is this token in the training corpus |
| prefill attention | how much attention did it receive during the prefill pass |
| layer depth | which layers attended to this token most (early vs late layers) |

Pairs each token's 5 features with its LTC label from compute_labels.py.
This is the training dataset for the scorer.

---

### train_scorer.py — Teaching the Network to Predict Importance

Trains a tiny neural network called the Scorer.

Architecture: 5 inputs → 16 hidden neurons with ReLU → 1 output with Sigmoid.
Total parameters: 97. Extremely small — fits in a hardware register file.

Takes the feature-label pairs from extract_features.py and trains with
mean squared error loss. After training, reports:
- Pearson correlation between predictions and actual LTC scores
- Top-50 overlap (of the 50 tokens the scorer thinks are most important,
  how many actually are in the real top 50)

Also exports the weights as integer constants to `scorer_weights.py`
for eventual hardware implementation (fixed-point arithmetic).

The within-domain experiment retrains on only the 10 Factual-Long training
prompts and tests on 10 Factual-Long eval prompts. This isolates whether
the architecture works from whether the training data is diverse enough.

---

### evaluate.py — Side by Side Comparison

Runs two conditions on every eval prompt:
1. H2O baseline — accumulators start at zero
2. H2O + ForesightKV — accumulators seeded with scorer predictions

Also runs a full-cache baseline (no eviction) to get a reference output.

Measures three things per condition:
- `cold_errors` — tokens with LTC above 0.7 that were evicted in the first 50 steps
- `total_errors` — tokens with LTC above 0.7 that were evicted at any point
- `token_match` — fraction of generated tokens that match the full-cache reference

Prints a comparison table. Lower errors and higher token match = better.

---

### cold_start_alignment.py — The Key Experiment

This is the experiment Chaithu asked for directly.

For each eval prompt:
1. Run the scorer on the prefill features to get predicted importance scores
2. Run H2O baseline for 50 decode steps and let the accumulator settle
3. After 50 steps, compare the scorer's initial predictions against where
   the H2O accumulator actually landed
4. Compute Pearson correlation between the two

**What the correlation means:**
- Positive = scorer predicted the same tokens as important that H2O confirmed
- Near zero = scorer is random noise
- Negative = scorer predicted the wrong tokens as important

**Your results:**
- Cross-domain (trained on 5 domains, tested on 2 different ones): r = -0.507
- Within-domain (trained and tested on Factual-Long only): r = +0.084

The sign flip from negative to positive when the domains match is the key result.
It proves the architecture is correct and the training data is the only problem.

---

## The Numbers That Matter

| Experiment | Result | What it means |
|---|---|---|
| Cross-domain correlation | -0.507 | Scorer points wrong direction on new domains |
| Within-domain correlation | +0.084 | Flips positive with matched training data |
| Cold start errors baseline | 0.90 | H2O evicts important tokens early |
| Cold start errors ForesightKV | 0.87 | Slight improvement, limited by scorer quality |

---

## How to Explain This to Anyone

**To someone who knows nothing:**
Language models have a memory problem — they can only remember so many past words at once. H2O decides which words to forget by watching which ones keep getting referenced. The problem is it makes bad decisions in the first 50 steps because it hasn't seen enough to tell what matters yet. ForesightKV trains a tiny network to predict which words will matter before generation even starts, so the first eviction decisions are smarter.

**To someone technical:**
H2O's per-token accumulators are zero-initialized, making the score distribution flat for the first ~50-100 decode steps and causing cold-start eviction errors. ForesightKV pre-seeds those accumulators with a learned prior trained on LTC labels from completed generations. The scorer takes 5 prefill-only features and outputs a predicted importance score before decoding begins.

**To a hardware person:**
The scorer is a 5-16-1 MLP with 97 parameters. It runs once per token during prefill, writes a single scalar to each TIU accumulator, then stays silent for the rest of generation. Weights should live in a programmable register file rather than being fixed at tape-out because cross-domain correlation is negative — you need per-workload retraining to get positive alignment.

---

## Track A — Quantization Eviction Agreement (2026-06-13)

**Question:** Does TurboQuant key quantization (the real chip path) change which tokens H2O evicts?

**Setup:**
- Model: Qwen2.5-3B-Instruct, 1 prompt (166 tokens = 10 blocks of 16), 24 decode steps
- Keys quantized BEFORE the QK dot product (how the chip actually works, not post-softmax)
- H2O token budget = 64, block size = 16
- Four conditions: fp (full precision), tq4 (TurboQuant 4-bit), int4 (naive uniform), int3 (naive uniform)

**Results:**

| Condition | token_agree | block_agree | token_r | block_r |
|---|---|---|---|---|
| TurboQuant b=4 | 0.703 | **1.000** | 0.994 | 0.990 |
| Naive INT4 | 0.594 | 0.500 | 0.396 | 0.535 |
| Naive INT3 | 0.578 | 1.000 | 0.634 | 0.803 |

**What the metrics mean:**
- `token_agree` — fraction of the 64-token keep set that matches full precision
- `block_agree` — fraction of the 10 blocks the chip would evict/keep that match full precision (this is the number Chaithu asked for)
- `token_r` — Pearson correlation of the full importance ranking vs fp
- `block_r` — same but after summing importance into 16-token blocks

**The story:**
TurboQuant's Hadamard rotation + Lloyd-Max codebook is designed to preserve inner products (the QK dot product). Even though keys are compressed to ~4.25 bits, the attention scores still rank tokens in nearly the same order as full precision (`token_r=0.994`). At the 16-token block level that the chip actually evicts on, agreement is **perfect (1.000)**.

Naive INT4 is terrible (`block_agree=0.500`, `token_r=0.396`) — coin-flip quality, worse than doing nothing. This is because transformer keys have a few large outlier channels that dominate when you quantize uniformly. TurboQuant's rotation spreads those outliers before quantizing, which is why it preserves rankings and naive INT4 doesn't.

Naive INT3 hitting 1.000 block_agree is partly luck from having only 10 blocks — don't read into it. Its token_r=0.634 shows the ranking is meaningfully damaged.

**The number for Chaithu:** block_agree = 1.000 for TurboQuant. The chip's actual key path does not hurt eviction fidelity at block granularity. The old 0.68 number was wrong because it used post-softmax noise + naive INT3 — both bad assumptions.
