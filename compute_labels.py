"""
Compute Long-Term Contribution (LTC) labels from saved traces.

LTC is computed at four decode horizons (50, 100, 150, 200 steps), normalized
within each horizon, then averaged. This gives a more robust label than a single
horizon — a token that matters at EVERY horizon is more reliably important than
one that only matters at step 200.

The prefill step (t=0) is excluded — that attention is concurrent, not future.

Output: labels.pt — list of dicts, one per prompt:
  prompt_idx  int
  prompt_len  int
  input_ids   LongTensor[prompt_len]
  ltc         FloatTensor[prompt_len]   multi-horizon average, normalized
  ltc_raw     FloatTensor[prompt_len]   raw sum over all decode steps
"""

import os
import torch
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS, LONG_PROMPTS

TRACE_DIR = "traces"
OUT_PATH = "labels.pt"

HORIZONS = [50, 100, 150, 200]
BLOCK_SIZE = 16


def pool_blocks(v):
    """Sum token scores into 16-token blocks. Last block may be partial."""
    n = v.shape[0]
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks = torch.zeros(num_blocks)
    for b in range(num_blocks):
        blocks[b] = v[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE].sum()
    return blocks


def compute_ltc(trace):
    prompt_len = trace["prompt_len"]
    step_attns = trace["step_attns"]  # list of [num_layers, cache_len]

    # accumulate attention incrementally so each horizon costs O(1) extra
    running = torch.zeros(prompt_len)
    snapshots = {}  # horizon → cumulative raw at that step

    for t, stack in enumerate(step_attns):
        if t == 0:
            continue  # skip prefill
        running += stack.sum(dim=0)[:prompt_len]
        if t in HORIZONS:
            snapshots[t] = running.clone()

    ltc_raw = running.clone()

    def normalize(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else torch.zeros_like(v)

    # per-token: average normalized LTC across all horizons
    normalized = [normalize(snapshots.get(h, ltc_raw)) for h in HORIZONS]
    ltc = torch.stack(normalized).mean(dim=0)

    # per-block: pool each horizon snapshot into blocks, normalize, then average
    block_snapshots = [normalize(pool_blocks(snapshots.get(h, ltc_raw))) for h in HORIZONS]
    ltc_blocks = torch.stack(block_snapshots).mean(dim=0)

    return ltc_raw, ltc, ltc_blocks


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
        ltc_raw, ltc, ltc_blocks = compute_ltc(trace)

        records.append({
            "prompt_idx": idx,
            "prompt_len": trace["prompt_len"],
            "input_ids": trace["input_ids"][: trace["prompt_len"]],
            "ltc": ltc,
            "ltc_raw": ltc_raw,
            "ltc_blocks": ltc_blocks,
        })

    torch.save(records, OUT_PATH)
    total_tokens = sum(r["prompt_len"] for r in records)
    print(f"Saved {len(records)} records  ({total_tokens} tokens)  missing={missing}")


if __name__ == "__main__":
    main()
