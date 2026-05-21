# H2O + ForesightKV

H2O (Heavy Hitter Oracle) KV cache eviction extended with ForesightKV pre-seeding, implemented from scratch on Microsoft's phi-2.

Paper: [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71645) · [arXiv](https://arxiv.org/abs/2306.14048)

## What it does

During autoregressive generation, transformer models cache the K and V tensors from past tokens so they don't have to recompute them. This cache grows linearly with sequence length and becomes the main memory bottleneck for long contexts.

H2O exploits the observation that attention is concentrated on a small subset of tokens — the "heavy hitters." Instead of keeping the full cache, H2O keeps only:

1. **Heavy hitters** — tokens with the highest accumulated attention scores
2. **A local window** — the most recent N tokens (always kept)

Everything else gets evicted.

## Files

- `h2o_cache.py` — the `H2OCache` class: append, score accumulation, eviction logic
- `patch.py` — `H2OCacheAdapter` (inherits from HuggingFace `DynamicCache`) and `patch_model` which registers forward hooks on each attention layer to capture attention weights
- `model.py` — loads phi-2, runs a baseline vs H2O comparison with greedy generation

## Run it

```bash
pip install transformers torch
python3 model.py
```

The script generates 60 tokens with the baseline cache and with H2O, then prints both. With a small budget you'll see the H2O output diverge from baseline once eviction kicks in.

## Notes

- Uses `attn_implementation="eager"` because sdpa hides attention weights
- Phi-2 is a base model (not instruction-tuned), so prompts use the `Instruct: ... Output:` format
- Greedy decoding for simplicity — production would use sampling
