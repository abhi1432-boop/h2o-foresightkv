"""
Compute Long-Term Contribution (LTC) labels from saved traces.

LTC for prompt token p = total attention received from ALL decode steps (t >= 1),
summed across every layer. The prefill step (t=0) is excluded — that attention is
concurrent, not future. Only post-prefill attention counts as "future."

LTC is then normalized to [0, 1] within each prompt.

Output: labels.pt — list of dicts, one per prompt:
  prompt_idx  int
  prompt_len  int
  input_ids   LongTensor[prompt_len]
  ltc         FloatTensor[prompt_len]   normalized
  ltc_raw     FloatTensor[prompt_len]   raw sums
"""

import os
import torch
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS, LONG_PROMPTS

TRACE_DIR = "traces"
OUT_PATH = "labels.pt"


def compute_ltc(trace):
    prompt_len = trace["prompt_len"]
    step_attns = trace["step_attns"]  # list of [num_layers, cache_len]

    ltc_raw = torch.zeros(prompt_len)

    for t, stack in enumerate(step_attns):
        if t == 0:
            continue  # skip prefill — not "future"
        # stack: [num_layers, cache_len]  where cache_len = prompt_len + t
        per_token = stack.sum(dim=0)          # sum over layers → [cache_len]
        ltc_raw += per_token[:prompt_len]     # only original prompt positions

    lo, hi = ltc_raw.min(), ltc_raw.max()
    ltc = (ltc_raw - lo) / (hi - lo) if hi > lo else torch.zeros_like(ltc_raw)
    return ltc_raw, ltc


def main():
    all_prompts = TRAIN_PROMPTS + EVAL_PROMPTS + LONG_PROMPTS
    records = []
    missing = 0

    for idx in range(len(all_prompts)):
        path = os.path.join(TRACE_DIR, f"trace_{idx:04d}.pt")
        if not os.path.exists(path):
            print(f"  [{idx}] trace missing, skipping")
            missing += 1
            continue

        trace = torch.load(path, weights_only=False)
        ltc_raw, ltc = compute_ltc(trace)

        records.append({
            "prompt_idx": idx,
            "prompt_len": trace["prompt_len"],
            "input_ids": trace["input_ids"][: trace["prompt_len"]],
            "ltc": ltc,
            "ltc_raw": ltc_raw,
        })

    torch.save(records, OUT_PATH)
    total_tokens = sum(r["prompt_len"] for r in records)
    print(f"Saved {len(records)} records  ({total_tokens} tokens)  missing={missing}")


if __name__ == "__main__":
    main()
