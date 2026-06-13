"""
Extract 5 prefill-only features for every prompt token.

All features use ONLY information available at prefill time — no future attention.

Feature index | Name              | Description
------------- | ----------------- | -------------------------------------------
0             | normalized_pos    | p / (prompt_len - 1)
1             | is_sink           | 1 if position < 5 (sink token signal)
2             | token_freq        | log-normalized frequency in training corpus
3             | prefill_attn      | attention received during prefill, summed over all heads and layers, then normalized within prompt
4             | layer_depth       | weighted mean layer index of prefill attention, normalized to [0,1]. Tokens that are mostly attended to in later layers score high.

Output: features.pt — list of dicts, one per prompt:
  prompt_idx  int
  split       'train' or 'eval'
  features    FloatTensor[prompt_len, 5]
  ltc         FloatTensor[prompt_len]
"""

import os
import math
import torch
from collections import Counter
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS, LONG_PROMPTS

TRACE_DIR = "traces"
LABELS_PATH = "labels.pt"
OUT_PATH = "features.pt"

NUM_TRAIN = len(TRAIN_PROMPTS)
NUM_SHORT = len(TRAIN_PROMPTS) + len(EVAL_PROMPTS)


def build_freq_table(labels):
    counter = Counter()
    for rec in labels:
        if rec["prompt_idx"] < NUM_TRAIN:
            counter.update(rec["input_ids"].tolist())
    return counter


def extract_features(trace, label_rec, freq_table, max_count):
    prompt_len = trace["prompt_len"]
    input_ids = trace["input_ids"][:prompt_len]
    prefill = trace["step_attns"][0]          # [num_layers, prompt_len]
    num_layers = prefill.shape[0]

    # Feature 3: total prefill attention per token (summed over all layers)
    total_prefill = prefill.sum(dim=0)        # [prompt_len]
    pf_sum = total_prefill.sum()
    pf_norm = total_prefill / pf_sum if pf_sum > 0 else torch.ones(prompt_len) / prompt_len

    # Feature 4: weighted mean layer index → which layers attend to this token
    layer_idx_col = torch.arange(num_layers, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    layer_sum = prefill.sum(dim=0).clamp(min=1e-9)   # [prompt_len]
    weighted = (layer_idx_col * prefill).sum(dim=0) / layer_sum  # [prompt_len]
    layer_depth = weighted / max(num_layers - 1, 1)              # [0, 1]

    feats = torch.zeros(prompt_len, 5)
    for p in range(prompt_len):
        feats[p, 0] = p / max(prompt_len - 1, 1)
        feats[p, 1] = 1.0 if p < 5 else 0.0
        tid = input_ids[p].item()
        count = freq_table.get(tid, 0)
        feats[p, 2] = math.log1p(count) / math.log1p(max_count) if max_count > 0 else 0.0
        feats[p, 3] = pf_norm[p].item()
        feats[p, 4] = layer_depth[p].item()

    return feats


def main():
    labels = torch.load(LABELS_PATH, weights_only=False)
    freq_table = build_freq_table(labels)
    max_count = max(freq_table.values()) if freq_table else 1

    records = []
    missing = 0

    for rec in labels:
        idx = rec["prompt_idx"]
        path = os.path.join(TRACE_DIR, f"trace_{idx:04d}.pt")
        if not os.path.exists(path):
            missing += 1
            continue

        trace = torch.load(path, weights_only=False)
        feats = extract_features(trace, rec, freq_table, max_count)
        split = "train" if idx < NUM_TRAIN else ("long" if idx >= NUM_SHORT else "eval")
        records.append({
            "prompt_idx": idx,
            "split": split,
            "features": feats,
            "ltc": rec["ltc"],
        })

    torch.save(records, OUT_PATH)
    train_n = sum(1 for r in records if r["split"] == "train")
    eval_n  = sum(1 for r in records if r["split"] == "eval")
    print(f"Saved {len(records)} records  train={train_n}  eval={eval_n}  missing={missing}")


if __name__ == "__main__":
    main()
