"""
Track A — does real TurboQuant key quantization change H2O eviction decisions,
and does it matter at 16-token block granularity?

Why this experiment exists
--------------------------
quantization_impact.py quantized attention weights AFTER softmax. That is the
"optimistic" path: the noise is added after the math is already done. In the
real chip the keys are quantized BEFORE the query*key dot product, so the noise
enters the attention scores themselves and compounds at every decode step.

Also, the old experiment used plain uniform INT3. The real key path is smarter:
  Hadamard rotation -> 3-bit Lloyd-Max -> 1-bit QJL on the residual (~4.25 bpv).
That is TurboQuantProd(bits=4). It is built to preserve inner products, so it
should damage the attention ranking far less than naive uniform INT3.

What we measure
---------------
For each prompt we run generation 4 ways, quantizing the KEYS only (values do
not affect which tokens get attention, so they do not affect eviction):
  - fp     : full precision keys (the ground-truth ranking)
  - tq4    : TurboQuant bits=4  (the real key path)
  - int4   : naive uniform INT4 (dumb 4-bit, for contrast)
  - int3   : naive uniform INT3 (the old method that gave 0.68)

H2O importance for a token = total attention it received, summed over all heads,
all queries, all layers, across the decode steps (prefill excluded, same as LTC).

Agreement is reported two ways:
  - per-token : do the same individual tokens fall in the keep set?
  - per-block : group tokens into 16-token blocks (sum importance per block),
                do the same blocks fall in the keep set? This is what the chip
                actually evicts on, so it is the headline number.
"""

import torch
import numpy as np
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from prompts import LONG_PROMPTS
from turboquant.quantizer import TurboQuantProd

MODEL_NAME     = "Qwen/Qwen2.5-3B-Instruct"
BLOCK_SIZE     = 16     # the chip tracks 16-token blocks
MAX_NEW_TOKENS = 24     # decode steps — more = more meaningful importance signal
NUM_PROMPTS    = 5      # scale up for more meaningful averages
TOKEN_BUDGET   = 64     # H2O keep budget (how many tokens survive) for agreement


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, attn_implementation="eager"
).to(device)
model.eval()
head_dim = model.config.hidden_size // model.config.num_attention_heads
print(f"done loading  head_dim={head_dim}")


# ── key quantizers ──────────────────────────────────────────────────────
# Each takes a key tensor [batch, kv_heads, n_new, head_dim] and returns it
# with quantization error baked in. Quantization runs on CPU (TurboQuant uses
# searchsorted which is flaky on MPS) then moves back to the key's device.

_tq4 = TurboQuantProd(dim=head_dim, bits=4, device=torch.device("cpu"))


def quant_tq4(K):
    dev, dt = K.device, K.dtype
    return _tq4.forward(K.detach().cpu().float()).to(device=dev, dtype=dt)


def _uniform(K, bits):
    dev, dt = K.device, K.dtype
    t = K.detach().cpu().float()
    m = t.abs().max()
    if m == 0:
        return K
    lv = 2 ** (bits - 1) - 1
    q = (t / m * lv).round().clamp(-lv, lv) / lv * m
    return q.to(device=dev, dtype=dt)


def quant_int4(K):
    return _uniform(K, 4)


def quant_int3(K):
    return _uniform(K, 3)


QUANTIZERS = {"fp": None, "tq4": quant_tq4, "int4": quant_int4, "int3": quant_int3}


# ── cache that smudges keys on write ────────────────────────────────────
class QuantKeyCache(DynamicCache):
    """Quantize each new key vector the moment it is stored — once, like the
    chip. Old keys keep the error they were stored with; it feeds every future
    attention step. Values pass through untouched (they do not affect ranking)."""

    def __init__(self, key_quantizer):
        super().__init__()
        self.key_quantizer = key_quantizer

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.key_quantizer is not None:
            key_states = self.key_quantizer(key_states)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def run(prompt_ids, key_quantizer):
    """Generate and return per-token accumulated attention importance over the
    original prompt tokens (decode steps only, prefill excluded)."""
    prompt_len = prompt_ids.shape[1]
    cache = QuantKeyCache(key_quantizer)
    generated = prompt_ids
    importance = torch.zeros(prompt_len, device=device)

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS + 1):
            model_inputs = generated if step == 0 else generated[:, -1:]
            out = model(model_inputs, past_key_values=cache, use_cache=True,
                        output_attentions=True, return_dict=True)
            if step > 0:  # skip prefill, same convention as LTC
                # a: [1, heads, 1, k_len] at decode steps — accumulate on MPS
                # in-place per layer, one CPU transfer at the very end
                for a in out.attentions:
                    importance += a[0, :, 0, :prompt_len].sum(dim=0)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)

    return importance.cpu()


# ── agreement metrics ───────────────────────────────────────────────────
def topk_agreement(a, b, k):
    k = min(k, len(a))
    if k == 0:
        return float("nan")
    ta = set(torch.topk(a, k).indices.tolist())
    tb = set(torch.topk(b, k).indices.tolist())
    return len(ta & tb) / k


def to_blocks(vec):
    n = (len(vec) // BLOCK_SIZE) * BLOCK_SIZE
    if n == 0:
        return vec.clone()
    return vec[:n].reshape(-1, BLOCK_SIZE).sum(dim=1)


def corr(a, b):
    a, b = a.numpy(), b.numpy()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return pearsonr(a, b)[0]


def main():
    prompts = LONG_PROMPTS[:NUM_PROMPTS]
    # results[cond] = list of (tok_agree, blk_agree, tok_r, blk_r)
    results = {c: [] for c in QUANTIZERS if c != "fp"}

    for i, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = ids.shape[1]
        n_blocks = prompt_len // BLOCK_SIZE
        print(f"\n[{i+1}/{len(prompts)}] prompt_len={prompt_len}  blocks={n_blocks}")

        imp = {c: run(ids, q) for c, q in QUANTIZERS.items()}
        fp_tok = imp["fp"]
        fp_blk = to_blocks(fp_tok)

        tok_k = min(TOKEN_BUDGET, prompt_len)
        blk_k = max(1, tok_k // BLOCK_SIZE)

        for c in results:
            q_tok, q_blk = imp[c], to_blocks(imp[c])
            ta = topk_agreement(fp_tok, q_tok, tok_k)
            ba = topk_agreement(fp_blk, q_blk, blk_k)
            tr = corr(fp_tok, q_tok)
            br = corr(fp_blk, q_blk)
            results[c].append((ta, ba, tr, br))
            print(f"   {c:<5} token_agree={ta:.3f}  block_agree={ba:.3f}  "
                  f"token_r={tr:.3f}  block_r={br:.3f}")

    print("\n" + "=" * 70)
    print(f"AVERAGE over {len(prompts)} prompts   "
          f"(token budget={TOKEN_BUDGET}, block size={BLOCK_SIZE})")
    print(f"{'condition':<18}{'token_agree':>13}{'block_agree':>13}"
          f"{'token_r':>10}{'block_r':>10}")
    print("-" * 70)
    names = {"tq4": "TurboQuant b=4", "int4": "naive INT4", "int3": "naive INT3"}
    for c in ["tq4", "int4", "int3"]:
        rows = np.array(results[c])
        m = np.nanmean(rows, axis=0)
        print(f"{names[c]:<18}{m[0]:>13.3f}{m[1]:>13.3f}{m[2]:>10.3f}{m[3]:>10.3f}")
    print("=" * 70)
    print("\ntoken_agree / block_agree = fraction of the keep set that matches "
          "full precision\nblock_agree is the number the chip cares about "
          "(it evicts whole 16-token blocks)")


if __name__ == "__main__":
    main()
