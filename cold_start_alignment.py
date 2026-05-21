"""
Measures how well the ForesightKV scorer's prior aligns with the H2O
accumulator state after the cold-start window.

Procedure (per eval prompt):
  1. Run scorer on prefill features → predicted importance scores for every token.
  2. Run H2O baseline (zero-init) for SETTLE_STEPS decode steps.
  3. After SETTLE_STEPS, read the accumulated_scores from the H2O cache for
     every token still in the cache, and map them back to their original
     sequence positions using _layer0_positions.
  4. Compare scorer predictions vs settled accumulator values (both normalized
     to [0,1]) for the tokens that survived eviction.
  5. Report Pearson r — how correlated the prior is with where H2O ends up.

This is the direct answer to: "does pre-seeding help without hurting steady-state?"
  r close to 1  → prior is well-aligned; pre-seeding is beneficial
  r near 0      → prior is uninformative at steady-state
  r negative    → prior is misleading; pre-seeding would hurt
"""

import torch
import numpy as np
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForCausalLM

from h2o_cache import H2OCache
from patch import H2OCacheAdapter
from train_scorer import Scorer
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS

FEATURES_PATH = "features.pt"
SCORER_PATH   = "scorer.pt"
NUM_TRAIN     = len(TRAIN_PROMPTS)

# Use a moderate budget so enough tokens survive for a meaningful correlation.
# Smaller = more evictions = fewer surviving tokens to correlate.
SETTLE_BUDGET = 20
SETTLE_WINDOW = 4
SETTLE_STEPS  = 50   # the "cold-start window" length


# ---------------------------------------------------------------------------
# Load model + scorer
# ---------------------------------------------------------------------------
print("Loading model...")
MODEL_NAME = "microsoft/phi-2"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, attn_implementation="eager"
)
model.eval()
print("done")


def load_scorer():
    m = Scorer()
    ckpt = torch.load(SCORER_PATH, weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


def run_h2o_baseline(prompt, steps):
    """Run H2O with zero-init for `steps` decode steps. Return the cache."""
    num_layers = len(model.model.layers)
    cache   = H2OCache(SETTLE_BUDGET, SETTLE_WINDOW, num_layers, use_foresight=False)
    adapter = H2OCacheAdapter(cache)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids

    def make_hook(idx):
        def hook(module, inputs, output):
            _, attn_weights = output
            if attn_weights is not None:
                cache.update_scores(attn_weights, idx)
            return output
        return hook

    handles = [
        layer.self_attn.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        with torch.no_grad():
            for step in range(steps + 1):   # +1 for prefill
                model_inputs = generated if step == 0 else generated[:, -1:]
                outputs = model(model_inputs, past_key_values=adapter, use_cache=True)
                if step < steps:
                    next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_tok], dim=1)
    finally:
        for h in handles:
            h.remove()

    return cache


def normalize(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def main():
    features_all = torch.load(FEATURES_PATH, weights_only=False)
    scorer       = load_scorer()

    # Build lookup: prompt_idx -> feature record
    feat_map = {r["prompt_idx"]: r for r in features_all if r["split"] == "eval"}

    all_pred, all_settled = [], []
    results = []

    eval_indices = sorted(feat_map.keys())
    print(f"\nRunning on {len(eval_indices)} eval prompts  "
          f"(budget={SETTLE_BUDGET}, window={SETTLE_WINDOW}, steps={SETTLE_STEPS})\n")

    for i, idx in enumerate(eval_indices):
        prompt = EVAL_PROMPTS[idx - NUM_TRAIN]
        rec    = feat_map[idx]

        # Step 1: scorer predictions for all prompt tokens
        with torch.no_grad():
            pred = scorer(rec["features"]).squeeze(1).numpy()   # [prompt_len]

        # Step 2: run H2O baseline and let the accumulator settle
        cache = run_h2o_baseline(prompt, SETTLE_STEPS)

        # Step 3: read settled accumulator (layer 0) + surviving positions
        scores_tensor  = cache.accumulated_scores[0]
        surviving_pos  = cache._layer0_positions       # original positions still in cache

        if scores_tensor is None or len(surviving_pos) < 3:
            print(f"  [{i+1}] skipped — too few surviving tokens")
            continue

        settled = scores_tensor.numpy()               # [num_surviving]

        # Step 4: align scorer predictions to the surviving positions
        pred_aligned = np.array([pred[p] for p in surviving_pos
                                 if p < len(pred)])
        settled_aligned = settled[:len(pred_aligned)]

        if len(pred_aligned) < 3:
            continue

        pred_norm    = normalize(pred_aligned)
        settled_norm = normalize(settled_aligned)

        r, _ = pearsonr(pred_norm, settled_norm)
        results.append(r)
        all_pred.append(pred_norm)
        all_settled.append(settled_norm)

        print(f"  [{i+1:2d}/{len(eval_indices)}] prompt_idx={idx}  "
              f"surviving={len(surviving_pos)}  r={r:.3f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    if not results:
        print("No results.")
        return

    overall_r, _ = pearsonr(np.concatenate(all_pred), np.concatenate(all_settled))
    mean_r = float(np.mean(results))

    print()
    print("=" * 58)
    print(f"Scorer prior vs H2O accumulator after {SETTLE_STEPS} steps")
    print("=" * 58)
    print(f"  Per-prompt mean r : {mean_r:.3f}")
    print(f"  Pooled r          : {overall_r:.3f}")
    print(f"  Prompts evaluated : {len(results)}")
    print()
    if overall_r > 0.5:
        verdict = "pre-seeding is beneficial — prior aligns well with steady-state"
    elif overall_r > 0.2:
        verdict = "pre-seeding gives a partial advantage — some alignment with steady-state"
    elif overall_r > 0:
        verdict = "weak alignment — pre-seeding has marginal benefit at steady-state"
    else:
        verdict = "prior does not align with steady-state — pre-seeding with this scorer may hurt"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
