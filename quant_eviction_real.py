"""
Track A, real key path — uses turboquant-plus's TurboQuantKVCache behavior.

Difference from quant_eviction_blocks.py
----------------------------------------
quant_eviction_blocks.py quantized EVERY key. The real TurboQuantKVCache
(turboquant/kv_cache.py, vendored from turboquant-plus) keeps the most recent
`buffer_size` tokens UNQUANTIZED as a quality buffer, and quantizes only the
older keys with TurboQuantProd (the same 4.25-bpv key path: 3-bit Lloyd-Max +
1-bit QJL on the residual).

NOTE: the "outlier channels kept FP" line in kv_cache.py's docstring is NOT
implemented in the reference class — only the recent-token buffer is real. So
the buffer is the only faithful difference from the quantize-everything run.

Why a sweep
-----------
On these short prompts (157-199 tokens) buffer_size=128 would leave most keys
full-precision and trivialise the metric. The chip's real regime is long
context (128 blocks x 16 = 2048 tokens, buffer ~6%). So we SWEEP buffer_size to
show the transition: buffer=0 is the pure-quantization stress test (reproduces
quant_eviction_blocks.py's ~0.725); larger buffers show how much the recent-
token buffer protects eviction fidelity.

Reports block_agree (the number the chip evicts on) per buffer_size, averaged
over all 10 LONG_PROMPTS, with full precision as the ground-truth ranking.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from prompts import LONG_PROMPTS
from turboquant.quantizer import TurboQuantProd

MODEL_NAME     = "Qwen/Qwen2.5-3B-Instruct"
BLOCK_SIZE     = 16
MAX_NEW_TOKENS = 24
NUM_PROMPTS    = 10
TOKEN_BUDGET   = 64
BUFFER_SWEEP   = [0, 16, 64, 128]   # 0 = quantize all (stress); 128 = kv_cache default


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    attn_implementation="eager",
).to(device)
model.eval()
head_dim = model.config.hidden_size // model.config.num_attention_heads
print(f"done loading  head_dim={head_dim}")

# The real key path: TurboQuantProd(bits=4) == 3-bit Lloyd-Max + 1-bit QJL.
# Quantization runs on CPU (searchsorted is flaky on MPS) then moves back.
_tq4 = TurboQuantProd(dim=head_dim, bits=4, device=torch.device("cpu"))


def quant_keys(K):
    """Reconstruct keys through the 4.25-bpv path. dequant(quant(K)) and QK^T is
    algebraically the reference's asymmetric attention_score estimator."""
    dev, dt = K.device, K.dtype
    return _tq4.forward(K.detach().cpu().float()).to(device=dev, dtype=dt)


class RealKeyCache(DynamicCache):
    """Replicates TurboQuantKVCache's key path inside an HF Cache: keep the most
    recent `buffer_size` keys full-precision, quantize the rest. buffer_size<=0
    quantizes everything (the old stress test). Re-quantizing the older keys each
    step is deterministic, so it equals the reference's quantize-once-on-flush."""

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer_size = buffer_size

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        b = self.buffer_size
        S = keys.shape[-2]
        if b <= 0:
            mod = quant_keys(keys)
        elif S <= b:
            mod = keys
        else:
            n_q = S - b
            mod = torch.cat([quant_keys(keys[..., :n_q, :]), keys[..., n_q:, :]], dim=-2)
        return mod, values


def run(prompt_ids, cache):
    """Per-token accumulated attention importance over prompt tokens (decode steps only)."""
    prompt_len = prompt_ids.shape[1]
    generated = prompt_ids
    importance = torch.zeros(prompt_len, device=device)
    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS + 1):
            model_inputs = generated if step == 0 else generated[:, -1:]
            out = model(model_inputs, past_key_values=cache, use_cache=True,
                        output_attentions=True, return_dict=True)
            if step > 0:
                for a in out.attentions:
                    importance += a[0, :, 0, :prompt_len].sum(dim=0)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
    return importance.cpu()


def to_blocks(vec):
    n = (len(vec) // BLOCK_SIZE) * BLOCK_SIZE
    if n == 0:
        return vec.clone()
    return vec[:n].reshape(-1, BLOCK_SIZE).sum(dim=1)


def topk_agreement(a, b, k):
    k = min(k, len(a))
    if k == 0:
        return float("nan")
    ta = set(torch.topk(a, k).indices.tolist())
    tb = set(torch.topk(b, k).indices.tolist())
    return len(ta & tb) / k


def main():
    prompts = LONG_PROMPTS[:NUM_PROMPTS]
    results = {b: [] for b in BUFFER_SWEEP}     # buffer_size -> list of block_agree

    for i, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = ids.shape[1]
        print(f"\n[{i+1}/{len(prompts)}] prompt_len={prompt_len}  blocks={prompt_len // BLOCK_SIZE}")

        fp_blk = to_blocks(run(ids, DynamicCache()))         # ground-truth ranking
        tok_k = min(TOKEN_BUDGET, prompt_len)
        blk_k = max(1, tok_k // BLOCK_SIZE)

        for b in BUFFER_SWEEP:
            q_blk = to_blocks(run(ids, RealKeyCache(b)))
            ba = topk_agreement(fp_blk, q_blk, blk_k)
            results[b].append(ba)
            print(f"   buffer={b:>3}  block_agree={ba:.3f}")

    print("\n" + "=" * 56)
    print(f"AVERAGE over {len(prompts)} prompts  "
          f"(budget={TOKEN_BUDGET}, block={BLOCK_SIZE}, real TurboQuant key path)")
    print(f"{'buffer_size':>12}{'block_agree':>14}{'note':>26}")
    print("-" * 56)
    notes = {0: "quantize all (stress)", 128: "kv_cache.py default"}
    for b in BUFFER_SWEEP:
        m = float(np.nanmean(results[b]))
        print(f"{b:>12}{m:>14.3f}{notes.get(b, ''):>26}")
    print("=" * 56)
    print("buffer=0 should match quant_eviction_blocks.py (~0.725). Higher buffer")
    print("= more recent keys kept FP = higher agreement (but only meaningful in")
    print("long context, where the buffer is a small fraction of the sequence).")


if __name__ == "__main__":
    main()
