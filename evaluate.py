"""
Compare baseline H2O vs H2O + ForesightKV on eval prompts.

Runs two evaluations:
  SHORT — 30 short eval prompts (indices 110-139), budget=8, window=2
  LONG  — 10 long eval prompts  (indices 140-149), budget=64, window=8

Metrics (per condition, averaged over prompts):
  cold_errors    tokens with LTC > 0.7 evicted within first COLD_WINDOW decode steps
  total_errors   tokens with LTC > 0.7 evicted at any point during generation
  token_match    fraction of generated tokens that match the full-cache reference
"""

import os
import time
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from prompts import TRAIN_PROMPTS, EVAL_PROMPTS, LONG_PROMPTS
from h2o_cache import H2OCache
from patch import H2OCacheAdapter
from train_scorer import Scorer

LABELS_PATH   = "labels.pt"
SCORER_PATH   = "scorer.pt"
TRACE_DIR     = "traces"

MAX_NEW_TOKENS = 5
COLD_WINDOW    = 10
LTC_THRESHOLD  = 0.7

# Short prompt eval settings
SHORT_BUDGET = 8
SHORT_WINDOW = 2

# Long prompt eval settings — sequences are ~185 tokens so budget needs to be much bigger
LONG_BUDGET  = 64
LONG_WINDOW  = 8

# Number of prompts to evaluate from each set (keep small for speed)
NUM_SHORT_EVAL = 3
NUM_LONG_EVAL  = 0

NUM_TRAIN  = len(TRAIN_PROMPTS)
NUM_SHORT  = len(TRAIN_PROMPTS) + len(EVAL_PROMPTS)
MODEL_NAME = "microsoft/phi-2"


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager"
).to(device)
model.eval()
num_layers = len(model.model.layers)
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


def get_prompt(idx):
    if idx < NUM_TRAIN:
        return TRAIN_PROMPTS[idx]
    elif idx < NUM_SHORT:
        return EVAL_PROMPTS[idx - NUM_TRAIN]
    else:
        return LONG_PROMPTS[idx - NUM_SHORT]


def load_ref_ids(prompt_idx):
    # Reuse the full-cache generation already saved in traces/ — no forward passes needed.
    # collect_traces.py saved MAX_NEW_TOKENS=50 tokens per prompt; we only compare
    # up to MAX_NEW_TOKENS of them, so the longer trace is always sufficient.
    path = os.path.join(TRACE_DIR, f"trace_{prompt_idx:04d}.pt")
    trace = torch.load(path, weights_only=False)
    return trace["generated_ids"]  # CPU LongTensor[prompt_len + 50]


def generate_h2o(prompt, use_foresight, freq_table, scorer, budget, window):
    cache   = H2OCache(budget, window, num_layers, use_foresight=use_foresight)
    adapter = H2OCacheAdapter(cache)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            t0 = time.time()
            model_inputs = generated if step == 0 else generated[:, -1:]
            outputs = model(model_inputs, past_key_values=adapter, use_cache=True,
                            output_attentions=True, return_dict=True)
            # one sync so the GPU fully finishes before we touch output tensors —
            # without this, each .cpu() in the scores loop forces its own sync (32x slower)
            if device == "mps":
                torch.mps.synchronize()
            t1 = time.time()

            for i, attn in enumerate(outputs.attentions):
                cache.update_scores(attn, i)
            t2 = time.time()

            if step == 0 and use_foresight:
                cache.seed_from_prefill(input_ids[0].cpu(), freq_table, scorer)

            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            print(f"      step {step}: fwd={t1-t0:.2f}s  scores={t2-t1:.2f}s", flush=True)

    return generated[0].cpu(), cache


def measure(generated_ids, ref_ids, h2o_cache, ltc, prompt_len):
    cold_errors  = 0
    total_errors = 0
    for pos in range(prompt_len):
        if ltc[pos].item() < LTC_THRESHOLD:
            continue
        step_evicted = h2o_cache.evicted_at_step.get(pos)
        if step_evicted is None:
            continue
        total_errors += 1
        if step_evicted < COLD_WINDOW:
            cold_errors += 1

    gen_new = generated_ids[prompt_len:].tolist()
    ref_new = ref_ids[prompt_len: prompt_len + len(gen_new)].tolist()
    matches = sum(g == r for g, r in zip(gen_new, ref_new))
    token_match = matches / max(len(gen_new), 1)

    return cold_errors, total_errors, token_match


def run_eval(indices, label_map, freq_table, scorer, budget, window, label):
    results = {"baseline": [], "foresight": []}

    for i, idx in enumerate(indices):
        prompt = get_prompt(idx)
        rec    = label_map[idx]
        ltc    = rec["ltc"]
        prompt_len = rec["prompt_len"]

        print(f"  [{i+1}/{len(indices)}] prompt_idx={idx}  prompt_len={prompt_len}")

        ref_ids = load_ref_ids(idx)

        gen_base, cache_base = generate_h2o(prompt, False, freq_table, scorer, budget, window)
        cold_b, total_b, match_b = measure(gen_base, ref_ids, cache_base, ltc, prompt_len)
        results["baseline"].append((cold_b, total_b, match_b))

        gen_fore, cache_fore = generate_h2o(prompt, True, freq_table, scorer, budget, window)
        cold_f, total_f, match_f = measure(gen_fore, ref_ids, cache_fore, ltc, prompt_len)
        results["foresight"].append((cold_f, total_f, match_f))

    def avg(rows, col):
        vals = [r[col] for r in rows]
        return sum(vals) / max(len(vals), 1)

    print(f"\n{'=' * 62}")
    print(f"{label}  budget={budget}  window={window}  prompts={len(indices)}")
    print(f"{'Condition':<20} {'cold_errors':>12} {'total_errors':>13} {'token_match':>12}")
    print(f"{'-' * 62}")
    for name, key in [("H2O baseline", "baseline"), ("H2O + ForesightKV", "foresight")]:
        rows = results[key]
        print(f"{name:<20} {avg(rows,0):>12.2f} {avg(rows,1):>13.2f} {avg(rows,2):>12.3f}")
    print(f"{'=' * 62}\n")


def main():
    if not os.path.exists(LABELS_PATH):
        print("labels.pt not found — run compute_labels.py first")
        return
    if not os.path.exists(SCORER_PATH):
        print("scorer.pt not found — run train_scorer.py first")
        return

    labels     = torch.load(LABELS_PATH, weights_only=False)
    freq_table = build_freq_table(labels)
    scorer     = load_scorer()
    label_map  = {r["prompt_idx"]: r for r in labels}

    short_indices = [r["prompt_idx"] for r in labels
                     if NUM_TRAIN <= r["prompt_idx"] < NUM_SHORT][:NUM_SHORT_EVAL]
    long_indices  = [r["prompt_idx"] for r in labels
                     if r["prompt_idx"] >= NUM_SHORT][:NUM_LONG_EVAL]

    if short_indices:
        print(f"\nEvaluating SHORT prompts ({len(short_indices)} prompts)...")
        run_eval(short_indices, label_map, freq_table, scorer,
                 SHORT_BUDGET, SHORT_WINDOW, "SHORT EVAL")

    if long_indices:
        print(f"\nEvaluating LONG prompts ({len(long_indices)} prompts)...")
        run_eval(long_indices, label_map, freq_table, scorer,
                 LONG_BUDGET, LONG_WINDOW, "LONG EVAL")


if __name__ == "__main__":
    main()
