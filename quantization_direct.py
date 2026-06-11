"""
Direct quantization experiment: re-run phi-2 with K and V quantized at storage time.

Unlike quantization_impact.py (which approximates by quantizing saved attention weights
after the fact), this runs phi-2 with a QuantizedDynamicCache — K and V are quantized
the moment they are stored, so every attention computation uses noisy K/V vectors.
This is what TurboQuant+ actually does in hardware.

NOTE: requires float32 — float16 produces NaN during decode steps on MPS due to
Q*K^T overflow. For scale, run on a CUDA device.

Compares against a float32 FP baseline run (not saved traces) so precision is consistent.
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from scipy.stats import pearsonr
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS

MODEL_NAME     = "microsoft/phi-2"
TRACE_DIR      = "traces"
BITS_TO_TEST   = [8, 4, 3]
EVAL_INDICES   = list(range(110, 115))   # 5 eval prompts
MAX_NEW_TOKENS = 15


def quantize(tensor, bits):
    if bits == 32:
        return tensor
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return tensor
    levels    = 2 ** (bits - 1) - 1
    scaled    = tensor / max_val * levels
    quantized = scaled.round().clamp(-levels, levels)
    return quantized / levels * max_val


class QuantizedDynamicCache(DynamicCache):
    """DynamicCache that quantizes K and V the moment they are stored.

    When phi-2 reads them back for the next attention step it sees noisy values,
    simulating what happens with compressed KV storage in hardware.
    """

    def __init__(self, bits):
        super().__init__()
        self.bits = bits

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return super().update(
            quantize(key_states, self.bits),
            quantize(value_states, self.bits),
            layer_idx,
            cache_kwargs,
        )


def run_generation(model, input_ids, num_layers, bits):
    """Run phi-2 generation with a (possibly quantized) cache.

    Returns step_attns: list of [num_layers, cache_len] tensors, one per step.
    Reads attention weights from outputs.attentions (requires output_attentions=True).
    """
    cache      = QuantizedDynamicCache(bits) if bits < 32 else DynamicCache()
    step_attns = []
    generated  = input_ids.clone()

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS + 1):
            model_inputs = generated if step == 0 else generated[:, -1:]
            outputs = model(
                model_inputs, past_key_values=cache,
                use_cache=True, output_attentions=True, return_dict=True,
            )

            stacked = torch.stack([
                a[0].sum(dim=(0, 1)).detach().cpu().float()
                for a in outputs.attentions
            ])  # [num_layers, cache_len]
            step_attns.append(stacked)

            if step < MAX_NEW_TOKENS:
                next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)

    return step_attns


def compute_ltc(step_attns, prompt_len):
    device  = step_attns[0].device
    ltc_raw = torch.zeros(prompt_len, device=device)
    for t, stack in enumerate(step_attns):
        if t == 0:
            continue
        per_token = stack.sum(dim=0)
        ltc_raw  += per_token[:prompt_len]
    lo, hi = ltc_raw.min(), ltc_raw.max()
    return (ltc_raw - lo) / (hi - lo) if hi > lo else torch.zeros_like(ltc_raw)


def topk_agreement(a, b, k):
    k = min(k, len(a))
    top_a = set(torch.topk(a, k).indices.tolist())
    top_b = set(torch.topk(b, k).indices.tolist())
    return len(top_a & top_b) / k


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading phi-2 on {device} (float32)...")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    model      = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    model.eval()
    num_layers = len(model.model.layers)
    print(f"done  num_layers={num_layers}\n")

    results = {b: [] for b in BITS_TO_TEST}

    for idx in EVAL_INDICES:
        trace      = torch.load(os.path.join(TRACE_DIR, f"trace_{idx:04d}.pt"), weights_only=False)
        prompt_len = trace["prompt_len"]
        input_ids  = trace["input_ids"].unsqueeze(0).to(device)

        print(f"  [{idx}] FP baseline: running...", end="\r", flush=True)
        baseline_attns = run_generation(model, input_ids, num_layers, bits=32)
        ltc_fp32       = compute_ltc(baseline_attns, prompt_len)

        for bits in BITS_TO_TEST:
            print(f"  [{idx}] INT{bits}: running...", end="\r", flush=True)
            step_attns = run_generation(model, input_ids, num_layers, bits)
            ltc_quant  = compute_ltc(step_attns, prompt_len)

            a, b = ltc_fp32.numpy(), ltc_quant.numpy()
            if len(a) >= 3 and a.std() > 1e-6 and b.std() > 1e-6:
                r     = float(pearsonr(a, b)[0])
                agree = topk_agreement(
                    torch.tensor(a), torch.tensor(b), k=max(1, prompt_len // 4)
                )
                results[bits].append((r, agree))

        row = {b: results[b][-1][0] if results[b] else float("nan") for b in BITS_TO_TEST}
        print(f"  [{idx}] prompt_len={prompt_len}  "
              f"INT8 r={row[8]:.3f}  INT4 r={row[4]:.3f}  INT3 r={row[3]:.3f}")

    print()
    print("=" * 58)
    print(f"{'Precision':<16} {'LTC corr (direct)':>18} {'top-k agree':>12}")
    print("-" * 58)
    print(f"{'FP32 (baseline)':<16} {'1.000':>18} {'1.000':>12}")
    for bits in BITS_TO_TEST:
        rows = results[bits]
        if not rows:
            continue
        avg_r     = float(np.mean([r for r, _ in rows]))
        avg_agree = float(np.mean([a for _, a in rows]))
        print(f"{'INT' + str(bits):<16} {avg_r:>18.3f} {avg_agree:>12.3f}")
    print("=" * 58)
    print()
    print("Direct method: K and V quantized at storage time.")
    print("Compare to quantization_impact.py for the post-softmax approximation.")
