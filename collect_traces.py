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
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS

MAX_NEW_TOKENS = 50
MODEL_NAME = "microsoft/phi-2"
TRACE_DIR = "traces"

os.makedirs(TRACE_DIR, exist_ok=True)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, attn_implementation="eager"
)
model.eval()

num_layers = len(model.model.layers)
print(f"done loading  num_layers={num_layers}")


def collect_one(prompt: str, prompt_idx: int):
    out_path = os.path.join(TRACE_DIR, f"trace_{prompt_idx:04d}.pt")
    if os.path.exists(out_path):
        return

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    # Buffer that hooks write into each forward pass.
    # Indexed by layer; holds the sum-over-heads attention received by each key.
    current_step_buf = [None] * num_layers

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            _, attn_weights = output
            if attn_weights is not None:
                # sum over batch (0) heads (1) → [key_len]
                # query dimension is also summed so all queries contribute equally
                current_step_buf[layer_idx] = attn_weights[0].sum(dim=(0, 1)).detach()
            return output
        return hook

    handles = [
        layer.self_attn.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.model.layers)
    ]

    step_attns = []
    past = DynamicCache()
    generated = input_ids

    try:
        with torch.no_grad():
            for step in range(MAX_NEW_TOKENS + 1):
                for i in range(num_layers):
                    current_step_buf[i] = None

                model_inputs = generated if step == 0 else generated[:, -1:]
                outputs = model(model_inputs, past_key_values=past, use_cache=True)

                # Stack all layers into [num_layers, cache_len]
                stacked = torch.stack([current_step_buf[i] for i in range(num_layers)])
                step_attns.append(stacked)

                if step < MAX_NEW_TOKENS:
                    next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_tok], dim=1)
    finally:
        for h in handles:
            h.remove()

    torch.save({
        "prompt": prompt,
        "input_ids": input_ids[0],
        "generated_ids": generated[0],
        "prompt_len": prompt_len,
        "step_attns": step_attns,
    }, out_path)
    print(f"  [{prompt_idx:03d}] saved  prompt_len={prompt_len}  steps={len(step_attns)}")


if __name__ == "__main__":
    all_prompts = TRAIN_PROMPTS + EVAL_PROMPTS
    print(f"Collecting traces for {len(all_prompts)} prompts...")
    for idx, prompt in enumerate(all_prompts):
        print(f"[{idx}/{len(all_prompts)}]", end="  ")
        collect_one(prompt, idx)
    print("All done.")
