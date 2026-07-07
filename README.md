# H2O + ForesightKV

H2O (Heavy Hitter Oracle) KV cache eviction extended with ForesightKV pre-seeding and beta decay, targeting the **LonghornSilicon** LLM-inference accelerator. Originally prototyped on Microsoft phi-2 (`model.py`, `evaluate.py`); the current scorer training and quantization experiments (Tracks A/B/C) run on **Qwen2.5-3B-Instruct**. Also measures how the real KV-cache compression path affects H2O's eviction decisions.

Paper: [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71645) · [arXiv](https://arxiv.org/abs/2306.14048)

## What it does

During autoregressive generation, transformer models cache the K and V tensors from past tokens so they don't have to recompute them. This cache grows linearly with sequence length and becomes the main memory bottleneck for long contexts.

H2O exploits the observation that attention is concentrated on a small subset of tokens — the "heavy hitters." Instead of keeping the full cache, H2O keeps only:

1. **Heavy hitters** — tokens with the highest accumulated attention scores
2. **A local window** — the most recent N tokens (always kept)

Everything else gets evicted.

## ForesightKV pre-seeding

Standard H2O initializes eviction scores to zero and relies entirely on observed attention. This causes cold-start errors — tokens evicted before they've had time to accumulate attention are gone forever even if they would have become important.

ForesightKV replaces the zero initialization with a learned prior. A small MLP (Scorer, ~97 parameters) is trained to predict each token's Long-Term Contribution (LTC) score from 5 prefill-only features:

- Normalized position in the prompt
- Whether the token is a sink (first 5 positions)
- Late-layer attention (attention from the top 1/3 of layers at prefill)
- Early-layer attention (attention from the bottom 1/3 of layers)
- Layer consistency (how uniformly the token is attended across layer depths)

After the prefill step, the Scorer runs once and seeds the accumulator with its predictions. From that point, real attention accumulates on top of the predictions. Beta decay fades out the prior over time so that by ~decode step 50 the prior has nearly vanished and real attention drives eviction decisions on its own. LTC labels use a 200-step, multi-horizon window (50/100/150/200) so the target reflects genuine long-term importance, not just cold-start attention.

## Quantization vs. eviction (Track A)

The headline question: does compressing the KV **keys** change which tokens H2O evicts? `quant_eviction_blocks.py` models the **real chip key path** — `TurboQuantProd(bits=4)` = rotation → 3-bit Lloyd-Max → 1-bit QJL on the residual ≈ 4.25 bpv — applied to keys *before* the QK^T product (so the noise compounds every decode step), on Qwen2.5-3B.

Result (10 prompts, 16-token blocks, budget 64):

| Condition | block agreement | rank correlation (token_r) |
|-----------|-----------------|----------------------------|
| **TurboQuant b=4 (real path)** | **0.725** | **0.996** |
| naive INT4 | 0.600 | 0.793 |
| naive INT3 | 0.575 | 0.797 |

TurboQuant preserves the importance *ranking* almost perfectly (r = 0.996); the only real loss is at the discrete top-k block boundary, where near-tied blocks flip. Naive uniform INT3/INT4 do far worse because they mangle the outlier channels TurboQuant's rotation spreads out first. **block agreement is the number the chip cares about** (the TIU evicts whole 16-token blocks). See `STUDY.md` for the full analysis and the recent-token-buffer sweep.

> An earlier approximation (`quantization_impact.py` / `quantization_direct.py`, phi-2) applied quantization *post-softmax* and estimated INT3 top-k agreement at ~0.68. That was the *optimistic* version (noise added after the math); Track A models the real *before*-QK^T path and is the number to trust.

## Upstream / hardware references (LonghornSilicon accelerator)

This work targets the LonghornSilicon LLM inference accelerator. Three upstream
repos are connected:

| Repo | Role | How it's wired in here |
|------|------|------------------------|
| [themoddedcube/turboquant-plus](https://github.com/themoddedcube/turboquant-plus) | TurboQuant+ reference (the real KV quantization) | `turboquant/quantizer.py` is byte-identical to theirs; `turboquant/kv_cache.py` (`TurboQuantKVCache`) vendored. Drives the Track A key path. |
| [LonghornSilicon/kv-cache-engine](https://github.com/LonghornSilicon/kv-cache-engine) | Block 2 — the silicon the quantization experiment probes (keys 4.25 bpv) | `kv_cache_engine_ref.py` vendored (bit-accurate Python ref, verified round-trip at dim=128). The block-agreement number is this engine's eviction fidelity. |
| [LonghornSilicon/adaptive-precision-attention](https://github.com/LonghornSilicon/adaptive-precision-attention) | The ACU (INT8/FP16 tile routing) — sibling block | Not vendored (no ACU code is run here). Connected conceptually: the scorer is designed as a VecU epilogue seeding 128 block registers, so scorer output is pooled per 16-token block. |

The real key path = `TurboQuantProd(bits=4)` = Hadamard/random rotation → 3-bit
Lloyd-Max → 1-bit QJL on the residual ≈ 4.25 bpv. `dequantize(quantize(K))`→QK^T
is algebraically identical to the reference's asymmetric `attention_score()`
estimator, so the simulation is faithful. See `STUDY.md` for the full results.

### Papers

- **H2O** (this project's basis) — [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71645) · [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- **TurboQuant+** — [paper (PDF)](https://github.com/themoddedcube/turboquant-plus/blob/turboquant-plus/paper/turboquant_plus_v2.pdf)
- **Adaptive Precision Attention (ACU)** — [paper (PDF)](https://github.com/LonghornSilicon/adaptive-precision-attention/blob/master/paper/adaptive_precision_attention.pdf)
- **KV Cache Engine** — [ISA spec (PDF)](https://github.com/LonghornSilicon/kv-cache-engine/blob/master/docs/isa/kv_cache_engine_isa.pdf)

## Files

| File | What it does |
|------|-------------|
| `h2o_cache.py` | `H2OCache`: append, score accumulation, eviction, ForesightKV seeding, beta decay |
| `quant_eviction_blocks.py` | Track A: real TurboQuant key path, per-token + 16-token block eviction agreement |
| `quant_eviction_real.py` | Track A: `TurboQuantKVCache` path (TurboQuantProd + recent-token buffer), buffer sweep |
| `train_domain_bank.py` | Track C: per-domain scorer bank (register-file experiment) |
| `eval_domain_bank.py` | Track C: z-score router + oracle/learned/general/wrong policy eval |
| `train_general_clean.py` | Track C: leakage-free general-vs-oracle comparison |
| `patch.py` | `H2OCacheAdapter` (inherits `DynamicCache`) + `patch_model` hooks |
| `kv_cache_engine_ref.py` | Vendored bit-accurate reference model of the kv-cache-engine (silicon ground truth) |
| `collect_traces.py` | Runs **Qwen2.5-3B** on all prompts (200 decode steps), saves attention at every step |
| `compute_labels.py` | Computes multi-horizon LTC scores from saved traces, saves labels.pt |
| `extract_features.py` | Extracts 5 prefill-only features per token, saves features.pt |
| `train_scorer.py` | Trains the Scorer MLP (seeded), exports fixed-point weights |
| `model.py` / `evaluate.py` | Original phi-2 H2O demo: baseline vs H2O+ForesightKV, cold-start errors |
| `quantization_impact.py` / `quantization_direct.py` | Early phi-2 post-softmax quantization approximation (superseded by Track A) |

## Run it

```bash
pip install "transformers==5.8.1" torch scipy

# Scorer pipeline (Qwen2.5-3B) — collect_traces is the slow part, best on a GPU
python3 collect_traces.py       # runs Qwen2.5-3B on all prompts (200 steps), saves traces/
python3 compute_labels.py       # multi-horizon LTC labels → labels.pt
python3 extract_features.py     # 5 prefill-only features → features.pt
python3 train_scorer.py         # train the Scorer MLP (Track B)
python3 train_domain_bank.py    # per-domain scorer bank (Track C)
python3 eval_domain_bank.py     # routing / oracle / general / wrong policies (Track C)

# Track A — quantization vs eviction (Qwen2.5-3B)
python3 quant_eviction_blocks.py  # real TurboQuant key path, block-level agreement

# Original phi-2 H2O demo (separate)
python3 model.py                # H2O baseline vs ForesightKV on one prompt
python3 evaluate.py             # measure cold-start improvement
```

## Notes

- Uses `attn_implementation="eager"` — sdpa hides attention weights (needed for `output_attentions`)
- Scorer + Track A run on **Qwen2.5-3B** (head_dim 128); the original H2O demo (`model.py`/`evaluate.py`) is on **phi-2**
- Prompts use the `Instruct: ... Output:` format; greedy decoding throughout
- Heavy runs (traces, quantization) are done on Colab T4 — TurboQuant's `searchsorted` is flaky on Apple MPS
