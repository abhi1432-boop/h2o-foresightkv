"""
Compare baseline H2O vs H2O + ForesightKV on eval prompts.

Conditions:
  1. H2O baseline   — zero-initialized accumulators (standard H2O)
  2. H2O + Foresight — seeded with trained scorer predictions

Metrics (per condition, averaged over eval prompts):
  cold_errors    tokens with LTC > 0.7 evicted within first COLD_WINDOW decode steps
  total_errors   tokens with LTC > 0.7 evicted at any point during generation
  token_match    fraction of generated tokens that match the full-cache reference

Usage:
  python evaluate.py
"""

import os
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from prompts import TRAIN_PROMPTS, EVAL_PROMPTS
from h2o_cache import H2OCache
from patch import H2OCacheAdapter
from train_scorer import Scorer

LABELS_PATH   = "labels.pt"
FEATURES_PATH = "features.pt"
SCORER_PATH   = "scorer.pt"

MAX_NEW_TOKENS = 50
COLD_WINDOW    = 50          # "cold start" = first N decode steps
LTC_THRESHOLD  = 0.7        # tokens above this are considered important
EVAL_BUDGET    = 8           # aggressive cache budget to stress-test eviction
EVAL_WINDOW    = 2

NUM_TRAIN = len(TRAIN_PROMPTS)
MODEL_NAME = "microsoft/phi-2"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, attn_implementation="eager"
)
model.eval()
print("done loading")


def load_scorer():
    m = Scorer()
    ckpt = torch.load(SCORER_PATH, weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


def build_freq_table(labels):
    counter = Counter()
    for rec in labels:
        if rec["prompt_idx"] < NUM_TRAIN:
            counter.update(rec["input_ids"].tolist())
    return counter


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_full_cache(prompt):
    """Run with unlimited cache — this is the reference output."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids
    past = DynamicCache()
    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            model_inputs = generated if step == 0 else generated[:, -1:]
            outputs = model(model_inputs, past_key_values=past, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
    return generated[0]


def generate_h2o(prompt, use_foresight, freq_table, scorer):
    """Run generation with H2O eviction and return (token_ids, H2OCache)."""
    num_layers = len(model.model.layers)
    cache = H2OCache(EVAL_BUDGET, EVAL_WINDOW, num_layers, use_foresight=use_foresight)
    adapter = H2OCacheAdapter(cache)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids
    handles = []

    # Register hooks to feed attention weights into H2OCache after each layer.
    def make_hook(idx):
        def hook(module, inputs, output):
            _, attn_weights = output
            if attn_weights is not None:
                cache.update_scores(attn_weights, idx)
            return output
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            for step in range(MAX_NEW_TOKENS):
                model_inputs = generated if step == 0 else generated[:, -1:]
                outputs = model(model_inputs, past_key_values=adapter, use_cache=True)

                # After prefill, seed scores with the scorer (foresight only).
                if step == 0 and use_foresight:
                    cache.seed_from_prefill(input_ids[0], freq_table, scorer)

                next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)
    finally:
        for h in handles:
            h.remove()

    return generated[0], cache


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def measure(generated_ids, ref_ids, h2o_cache, ltc, prompt_len):
    """Compute the three metrics for one prompt."""
    # cold_errors: important tokens evicted within the first COLD_WINDOW decode steps
    # total_errors: important tokens evicted at any decode step
    cold_errors  = 0
    total_errors = 0
    for pos in range(prompt_len):
        if ltc[pos].item() < LTC_THRESHOLD:
            continue
        step_evicted = h2o_cache.evicted_at_step.get(pos)
        if step_evicted is None:
            continue                            # never evicted
        total_errors += 1
        if step_evicted < COLD_WINDOW:
            cold_errors += 1

    # token_match: how many generated tokens agree with the full-cache reference
    gen_new = generated_ids[prompt_len:].tolist()
    ref_new = ref_ids[prompt_len: prompt_len + len(gen_new)].tolist()
    matches = sum(g == r for g, r in zip(gen_new, ref_new))
    token_match = matches / max(len(gen_new), 1)

    return cold_errors, total_errors, token_match


def main():
    if not os.path.exists(LABELS_PATH):
        print("labels.pt not found — run compute_labels.py first")
        return
    if not os.path.exists(SCORER_PATH):
        print("scorer.pt not found — run train_scorer.py first")
        return

    labels = torch.load(LABELS_PATH, weights_only=False)
    freq_table = build_freq_table(labels)
    scorer = load_scorer()

    # Build a fast lookup: prompt_idx -> label record
    label_map = {r["prompt_idx"]: r for r in labels}

    # Eval prompts are indices NUM_TRAIN onward
    eval_indices = [r["prompt_idx"] for r in labels if r["prompt_idx"] >= NUM_TRAIN]
    if not eval_indices:
        print("No eval prompts found in labels.pt")
        return

    results = {"baseline": [], "foresight": []}

    for i, idx in enumerate(eval_indices):
        prompt = EVAL_PROMPTS[idx - NUM_TRAIN]
        rec = label_map[idx]
        ltc = rec["ltc"]
        prompt_len = rec["prompt_len"]

        print(f"[{i+1}/{len(eval_indices)}] prompt_idx={idx}  prompt_len={prompt_len}")

        # Reference generation (full cache)
        ref_ids = generate_full_cache(prompt)

        # Baseline H2O
        gen_base, cache_base = generate_h2o(prompt, use_foresight=False,
                                             freq_table=freq_table, scorer=scorer)
        cold_b, total_b, match_b = measure(gen_base, ref_ids, cache_base, ltc, prompt_len)
        results["baseline"].append((cold_b, total_b, match_b))

        # H2O + ForesightKV
        gen_fore, cache_fore = generate_h2o(prompt, use_foresight=True,
                                             freq_table=freq_table, scorer=scorer)
        cold_f, total_f, match_f = measure(gen_fore, ref_ids, cache_fore, ltc, prompt_len)
        results["foresight"].append((cold_f, total_f, match_f))

    # Aggregate
    def avg(rows, col):
        vals = [r[col] for r in rows]
        return sum(vals) / max(len(vals), 1)

    print("\n" + "=" * 62)
    print(f"{'Condition':<20} {'cold_errors':>12} {'total_errors':>13} {'token_match':>12}")
    print("-" * 62)
    for name, key in [("H2O baseline", "baseline"), ("H2O + ForesightKV", "foresight")]:
        rows = results[key]
        print(
            f"{name:<20} {avg(rows,0):>12.2f} {avg(rows,1):>13.2f} {avg(rows,2):>12.3f}"
        )
    print("=" * 62)
    print(f"\ncache budget={EVAL_BUDGET}  window={EVAL_WINDOW}  "
          f"LTC_threshold={LTC_THRESHOLD}  cold_window={COLD_WINDOW}  "
          f"prompts={len(eval_indices)}")


if __name__ == "__main__":
    main()
