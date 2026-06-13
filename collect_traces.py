"""
Run full (no-eviction) generations on all prompts and record, at every decode
step, how much attention each cached token received from each layer.

Output: traces/trace_{idx:04d}.pt per prompt, each a dict:
  prompt       str
  input_ids    LongTensor[prompt_len]
  generated_ids LongTensor[prompt_len + steps]
  prompt_len   int
  step_attns   list[Tensor[num_layers, cache_len_at_that_step]]
               step 0 = prefill (cache_len == prompt_len)
               step t = decode  (cache_len == prompt_len + t)

Supports MPS (Apple Silicon GPU) for faster collection.
Uses output_attentions=True instead of hooks — hooks return None on MPS.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS, LONG_PROMPTS

MAX_NEW_TOKENS = 50
MODEL_NAME = "microsoft/phi-2"
TRACE_DIR  = "traces"

os.makedirs(TRACE_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, attn_implementation="eager"
).to(device)
model.eval()

num_layers = len(model.model.layers)
print(f"done loading  num_layers={num_layers}  device={device}")


def collect_one(prompt: str, prompt_idx: int):
    out_path = os.path.join(TRACE_DIR, f"trace_{prompt_idx:04d}.pt")
    if os.path.exists(out_path):
        return

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    step_attns = []
    past       = DynamicCache()
    generated  = input_ids

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS + 1):
            model_inputs = generated if step == 0 else generated[:, -1:]

            outputs = model(
                model_inputs,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

            # outputs.attentions: tuple of [batch, heads, query_len, key_len] per layer
            # sum over heads and queries → one score per key token per layer, then stack
            stacked = torch.stack([
                a[0].sum(dim=(0, 1)).detach().cpu()
                for a in outputs.attentions
            ])  # [num_layers, cache_len]
            step_attns.append(stacked)

            if step < MAX_NEW_TOKENS:
                next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)

    torch.save({
        "prompt":        prompt,
        "input_ids":     input_ids[0].cpu(),
        "generated_ids": generated[0].cpu(),
        "prompt_len":    prompt_len,
        "step_attns":    step_attns,
    }, out_path)
    print(f"  [{prompt_idx:03d}] saved  prompt_len={prompt_len}  steps={len(step_attns)}")


if __name__ == "__main__":
    # standard prompts (indices 0-139)
    all_prompts = TRAIN_PROMPTS + EVAL_PROMPTS
    # long prompts (indices 140-149)
    long_start  = len(all_prompts)

    print(f"Collecting standard traces for {len(all_prompts)} prompts...")
    for idx, prompt in enumerate(all_prompts):
        print(f"[{idx}/{len(all_prompts)}]", end="  ")
        collect_one(prompt, idx)

    print(f"\nCollecting long traces for {len(LONG_PROMPTS)} prompts...")
    for i, prompt in enumerate(LONG_PROMPTS):
        idx = long_start + i
        print(f"[{idx}]", end="  ")
        collect_one(prompt, idx)

    print("All done.")
