"""
Extract 5 prefill-only features for every prompt token.

All features use ONLY information available at prefill time — no future attention.

Feature index | Name              | Description
------------- | ----------------- | -------------------------------------------
0             | normalized_pos    | p / (prompt_len - 1)
1             | is_sink           | 1 if position < 5 (sink token signal)
2             | late_layer_attn   | attention from top 1/3 of layers at prefill, normalized. Later layers = semantic signal.
3             | early_layer_attn  | attention from bottom 1/3 of layers at prefill, normalized. Contrast with late-layer.
4             | layer_consistency | 1 / (1 + CV) where CV = std/mean across layers. High = token gets attention uniformly at all depths.

Output: features.pt — list of dicts, one per prompt:
  prompt_idx  int
  split       'train' or 'eval'
  features    FloatTensor[prompt_len, 5]
  ltc         FloatTensor[prompt_len]
"""

import os
import torch
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS, LONG_PROMPTS

TRACE_DIR = "traces"
LABELS_PATH = "labels.pt"
OUT_PATH = "features.pt"

BLOCK_SIZE = 16
NUM_TRAIN = len(TRAIN_PROMPTS)
NUM_SHORT = len(TRAIN_PROMPTS) + len(EVAL_PROMPTS)


def extract_features(trace, label_rec):
    prompt_len = trace["prompt_len"]
    prefill = trace["step_attns"][0]          # [num_layers, prompt_len]
    num_layers = prefill.shape[0]
    num_blocks = (prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    third = max(num_layers // 3, 1)
    early = prefill[:third].sum(dim=0)        # [prompt_len] raw attention, bottom third of layers
    late  = prefill[num_layers - third:].sum(dim=0)  # [prompt_len] raw attention, top third

    layer_mean = prefill.mean(dim=0).clamp(min=1e-9)
    layer_std  = prefill.std(dim=0)
    consistency = 1.0 / (1.0 + layer_std / layer_mean)  # [prompt_len]

    # pool raw attention into blocks, then normalize across blocks so features are comparable
    def pool_and_norm(v):
        blocks = torch.zeros(num_blocks)
        for b in range(num_blocks):
            blocks[b] = v[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE].sum()
        s = blocks.sum()
        return blocks / s if s > 0 else torch.ones(num_blocks) / num_blocks

    late_block  = pool_and_norm(late)    # [num_blocks]
    early_block = pool_and_norm(early)   # [num_blocks]

    # layer_consistency: mean token consistency within each block
    consist_block = torch.zeros(num_blocks)
    for b in range(num_blocks):
        consist_block[b] = consistency[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE].mean()

    feats = torch.zeros(num_blocks, 5)
    for b in range(num_blocks):
        center = (b * BLOCK_SIZE + min((b + 1) * BLOCK_SIZE, prompt_len) - 1) / 2
        feats[b, 0] = center / max(prompt_len - 1, 1)   # normalized block center position
        feats[b, 1] = 1.0 if b == 0 else 0.0            # block 0 contains the sink tokens
        feats[b, 2] = late_block[b].item()
        feats[b, 3] = early_block[b].item()
        feats[b, 4] = consist_block[b].item()

    return feats


def main():
    labels = torch.load(LABELS_PATH, weights_only=False)

    records = []
    missing = 0

    for rec in labels:
        idx = rec["prompt_idx"]
        path = os.path.join(TRACE_DIR, f"trace_{idx:04d}.pt")
        if not os.path.exists(path):
            missing += 1
            continue

        trace = torch.load(path, weights_only=False)
        feats = extract_features(trace, rec)
        split = "train" if idx < NUM_TRAIN else ("long" if idx >= NUM_SHORT else "eval")
        records.append({
            "prompt_idx": idx,
            "split": split,
            "features": feats,             # [num_blocks, 5]
            "ltc": rec["ltc_blocks"],      # [num_blocks] block-level label
        })

    torch.save(records, OUT_PATH)
    train_n = sum(1 for r in records if r["split"] == "train")
    eval_n  = sum(1 for r in records if r["split"] == "eval")
    print(f"Saved {len(records)} records  train={train_n}  eval={eval_n}  missing={missing}")


if __name__ == "__main__":
    main()
