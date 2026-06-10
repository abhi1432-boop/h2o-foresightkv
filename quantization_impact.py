"""
Measures how KV cache quantization affects H2O eviction decisions.

Uses already-saved traces instead of re-running phi-2, so this runs in seconds.

Quantizing K and V introduces noise into the attention dot products, which
shows up as noise in the attention weights. We approximate this by quantizing
the saved attention weights directly — a valid first-order approximation since
the effect of K/V quantization is to add noise to those weights.

Bit widths tested: FP32 (baseline), INT8, INT4, INT3
INT3 is the TurboQuant+ target — that row directly answers whether 3-bit
KV compression degrades the signal H2O and ForesightKV depend on.
"""

import os
import torch
import numpy as np
from scipy.stats import pearsonr
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS

LABELS_PATH = "labels.pt"
TRACE_DIR   = "traces"
NUM_TRAIN   = len(TRAIN_PROMPTS)
BIT_WIDTHS  = [32, 8, 4, 3]


def quantize(tensor, bits):
    """Simulate storing a tensor at `bits` bit precision and reading it back.

    Steps:
      1. Find the largest absolute value (the range of the tensor)
      2. Scale the tensor so it fits into ±(2^(bits-1) - 1) integer levels
      3. Round to the nearest integer (this is where precision is lost)
      4. Scale back to the original range

    The rounding step introduces small errors — more bits = smaller errors.
    32 bits means no quantization, tensor returned unchanged.
    """
    if bits == 32:
        return tensor
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return tensor
    levels    = 2 ** (bits - 1) - 1
    scaled    = tensor / max_val * levels
    quantized = scaled.round().clamp(-levels, levels)
    return quantized / levels * max_val


def compute_ltc_from_attns(step_attns, prompt_len):
    """Compute LTC scores from a list of step attention tensors.

    step_attns[0] = prefill (excluded from LTC — not future attention)
    step_attns[1:] = decode steps (this is the future attention that defines LTC)
    """
    ltc_raw = torch.zeros(prompt_len)
    for t, stack in enumerate(step_attns):
        if t == 0:
            continue
        per_token = stack.sum(dim=0)       # sum over layers
        ltc_raw  += per_token[:prompt_len]
    lo, hi = ltc_raw.min(), ltc_raw.max()
    return (ltc_raw - lo) / (hi - lo) if hi > lo else torch.zeros_like(ltc_raw)


def topk_agreement(a, b, k=5):
    """Fraction of top-k tokens that agree between score vectors a and b."""
    k = min(k, len(a))
    top_a = set(torch.topk(a, k).indices.tolist())
    top_b = set(torch.topk(b, k).indices.tolist())
    return len(top_a & top_b) / k


def main():
    labels    = torch.load(LABELS_PATH, weights_only=False)
    label_map = {r["prompt_idx"]: r for r in labels if r["prompt_idx"] >= NUM_TRAIN}
    eval_indices = sorted(label_map.keys())

    print(f"Running quantization impact on {len(eval_indices)} eval prompts")
    print(f"Using saved traces — no model rerun needed\n")

    # results[bits] = list of (correlation, topk_agreement) per prompt
    results = {b: [] for b in BIT_WIDTHS}

    for idx in eval_indices:
        trace_path = os.path.join(TRACE_DIR, f"trace_{idx:04d}.pt")
        if not os.path.exists(trace_path):
            print(f"  [{idx}] trace missing, skipping")
            continue

        trace      = torch.load(trace_path, weights_only=False)
        prompt_len = trace["prompt_len"]
        step_attns = trace["step_attns"]  # list of [num_layers, cache_len]

        # full precision LTC — ground truth
        ltc_fp32 = compute_ltc_from_attns(step_attns, prompt_len)

        for bits in BIT_WIDTHS:
            if bits == 32:
                ltc_quant = ltc_fp32
            else:
                # apply quantization to every step's attention tensor
                quant_attns = [quantize(s, bits) for s in step_attns]
                ltc_quant   = compute_ltc_from_attns(quant_attns, prompt_len)

            a = ltc_fp32.numpy()
            b = ltc_quant.numpy()

            if len(a) < 3 or a.std() < 1e-6 or b.std() < 1e-6:
                continue

            r, _   = pearsonr(a, b)
            agree  = topk_agreement(
                torch.tensor(a), torch.tensor(b), k=max(1, prompt_len // 4)
            )
            results[bits].append((r, agree))

        print(f"  [{idx}] prompt_len={prompt_len}  "
              f"INT8 r={results[8][-1][0]:.3f}  "
              f"INT4 r={results[4][-1][0]:.3f}  "
              f"INT3 r={results[3][-1][0]:.3f}")

    print()
    print("=" * 62)
    print(f"{'Precision':<14} {'LTC correlation':>16} {'top-k evict agree':>20}")
    print("-" * 62)
    for bits in BIT_WIDTHS:
        rows = results[bits]
        if not rows:
            continue
        avg_r     = float(np.mean([r for r, _ in rows]))
        avg_agree = float(np.mean([a for _, a in rows]))
        label     = "FP32 (baseline)" if bits == 32 else f"INT{bits}"
        print(f"{label:<14} {avg_r:>16.3f} {avg_agree:>20.3f}")
    print("=" * 62)
    print()
    print("LTC correlation  — how similar are importance rankings vs full precision")
    print("top-k evict agree — do the same tokens get kept/evicted as full precision")
    print()
    print("INT3 row = direct answer to whether TurboQuant+ 3-bit compression")
    print("degrades H2O eviction signal enough to matter for TIU design.")


if __name__ == "__main__":
    main()
