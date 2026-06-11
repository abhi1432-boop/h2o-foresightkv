# H2O + ForesightKV

H2O (Heavy Hitter Oracle) KV cache eviction extended with ForesightKV pre-seeding and beta decay, implemented on Microsoft phi-2. Also includes quantization impact analysis motivated by TurboQuant+ 3-bit KV compression.

Paper: [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71645) · [arXiv](https://arxiv.org/abs/2306.14048)

## What it does

During autoregressive generation, transformer models cache the K and V tensors from past tokens so they don't have to recompute them. This cache grows linearly with sequence length and becomes the main memory bottleneck for long contexts.

H2O exploits the observation that attention is concentrated on a small subset of tokens — the "heavy hitters." Instead of keeping the full cache, H2O keeps only:

1. **Heavy hitters** — tokens with the highest accumulated attention scores
2. **A local window** — the most recent N tokens (always kept)

Everything else gets evicted.

## ForesightKV pre-seeding

Standard H2O initializes eviction scores to zero and relies entirely on observed attention. This causes cold-start errors — tokens evicted before they've had time to accumulate attention are gone forever even if they would have become important.

ForesightKV replaces the zero initialization with a learned prior. A small MLP (Scorer) is trained to predict each token's Long-Term Contribution (LTC) score from 5 prefill-only features:

- Normalized position in the prompt
- Whether the token is a sink (first 5 positions)
- Token frequency in the training corpus
- Total prefill attention (summed across layers)
- Weighted mean layer index (where in the network the token receives attention)

After the prefill step, the Scorer runs once and seeds the accumulator with its predictions. From that point, real attention accumulates on top of the predictions. Beta decay fades out the prior over time so that by decode step 50 the prior has nearly vanished and real attention drives eviction decisions on its own.

## Quantization impact

Motivated by TurboQuant+'s 3-bit KV compression, we measure how much quantizing K and V degrades the eviction signal H2O depends on.

**quantization_impact.py** — approximation: applies quantization to saved attention weights post-softmax and measures how much the LTC ranking and top-k eviction decisions shift.

**quantization_direct.py** — direct method: re-runs phi-2 with a `QuantizedDynamicCache` that quantizes K and V at storage time (matching TurboQuant+'s actual behavior). Requires float32; float16 produces NaN on MPS during decode steps. For scale, run on a CUDA device.

Results from the approximation across 30 eval prompts:

| Precision | LTC correlation | top-k evict agree |
|-----------|----------------|-------------------|
| FP32 (baseline) | 1.000 | 1.000 |
| INT8 | 1.000 | 0.992 |
| INT4 | 0.999 | 0.689 |
| INT3 | 0.997 | 0.681 |

The importance ranking survives 3-bit well (r=0.997) but ~1 in 3 borderline tokens flip at the eviction cutoff. Most noise appears at INT4 — going from 4 to 3 bits barely makes it worse.

## Files

| File | What it does |
|------|-------------|
| `h2o_cache.py` | `H2OCache`: append, score accumulation, eviction, ForesightKV seeding, beta decay |
| `patch.py` | `H2OCacheAdapter` (inherits `DynamicCache`) + `patch_model` hooks |
| `model.py` | Loads phi-2, runs baseline vs H2O comparison |
| `collect_traces.py` | Runs phi-2 on all prompts, saves attention weights at every decode step |
| `compute_labels.py` | Computes LTC scores from saved traces, saves labels.pt |
| `extract_features.py` | Extracts 5 prefill-only features per token, saves features.pt |
| `train_scorer.py` | Trains the Scorer MLP, exports weights |
| `evaluate.py` | Runs H2O baseline vs H2O+ForesightKV, measures cold-start errors |
| `quantization_impact.py` | Post-softmax quantization approximation across 30 eval prompts |
| `quantization_direct.py` | Direct K/V quantization at storage time (CUDA recommended) |

## Run it

```bash
pip install transformers torch scipy
python3 model.py                # H2O baseline vs ForesightKV on one prompt
python3 collect_traces.py       # collect attention traces (runs phi-2, takes time)
python3 compute_labels.py       # compute LTC labels from traces
python3 extract_features.py     # extract scorer features
python3 train_scorer.py         # train the Scorer MLP
python3 evaluate.py             # measure cold-start improvement
python3 quantization_impact.py  # quantization impact (fast, uses saved traces)
```

## Notes

- Uses `attn_implementation="eager"` — sdpa hides attention weights from hooks
- Phi-2 is a base model, prompts use `Instruct: ... Output:` format
- Greedy decoding throughout
