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

# how many new tokens to generate per prompt (50 decode steps after the prefill)
MAX_NEW_TOKENS = 50
MODEL_NAME = "microsoft/phi-2"
# folder where we save one .pt file per prompt
TRACE_DIR = "traces"

# create the traces/ folder if it doesn't already exist
os.makedirs(TRACE_DIR, exist_ok=True)

print("Loading model...")
# tokenizer converts text into integer token IDs that phi-2 understands
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# load phi-2's weights from disk
# attn_implementation="eager" forces plain Python attention so we can read
# the attention weight matrix — the faster "sdpa" mode hides it from us
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, attn_implementation="eager"
)
# eval mode turns off dropout and other training-only behavior
model.eval()

# phi-2 has 32 attention layers — we need to hook all of them
num_layers = len(model.model.layers)
print(f"done loading  num_layers={num_layers}")


def collect_one(prompt: str, prompt_idx: int):
    # where this prompt's trace file will be saved
    out_path = os.path.join(TRACE_DIR, f"trace_{prompt_idx:04d}.pt")

    # if this prompt was already collected (e.g. from a previous run that got
    # interrupted), skip it so we don't redo hours of work
    if os.path.exists(out_path):
        return

    # convert the prompt text into token IDs
    # shape: [1, prompt_len] — the 1 is batch size (we only have one prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]

    # this buffer holds the attention data captured by the hooks each step
    # one slot per layer — the hook writes into here after each layer runs
    current_step_buf = [None] * num_layers

    def make_hook(layer_idx):
        # we wrap the hook in make_hook() to "bake in" the correct layer_idx
        # without this wrapper all 32 hooks would share the same layer_idx
        # variable and by the time any hook fires layer_idx would be 31 for all
        def hook(module, inputs, output):
            # PyTorch calls every forward hook with (module, inputs, output)
            # output is a tuple: (attn_output, attn_weights)
            # we only care about attn_weights so we unpack and ignore the rest
            _, attn_weights = output

            if attn_weights is not None:
                # attn_weights shape: [batch, heads, query_len, key_len]
                # [0] removes the batch dimension (we only have one sequence)
                # .sum(dim=(0,1)) sums over heads AND queries
                # result: one number per key token = total attention that token received this step
                # .detach() makes sure we don't accidentally track gradients
                current_step_buf[layer_idx] = attn_weights[0].sum(dim=(0, 1)).detach()

            # must return output unchanged — phi-2 still needs it
            return output

        return hook

    # register a forward hook on every attention layer
    # PyTorch will automatically call our hook after each layer's forward() finishes
    handles = [
        layer.self_attn.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.model.layers)
    ]

    # step_attns will collect one tensor per step
    # each tensor is shape [num_layers, cache_len] — all 32 layers stacked together
    step_attns = []

    # DynamicCache is HuggingFace's standard unlimited cache — no eviction
    # we use this here because we want the full unmodified attention signal
    # for computing LTC labels later
    past = DynamicCache()

    # generated starts as just the prompt and grows by one token each step
    generated = input_ids

    try:
        # torch.no_grad() turns off gradient tracking — we're not training phi-2
        # just running it to collect data
        with torch.no_grad():
            # +1 because step 0 is the prefill (the whole prompt)
            # steps 1 through MAX_NEW_TOKENS are the decode steps
            for step in range(MAX_NEW_TOKENS + 1):

                # clear the buffer before each step so last step's data doesn't bleed in
                for i in range(num_layers):
                    current_step_buf[i] = None

                # step 0: feed the full prompt so the cache gets populated
                # step 1+: feed only the newest token — the cache has everything else
                model_inputs = generated if step == 0 else generated[:, -1:]

                # run phi-2's forward pass
                # this triggers all 32 attention layers, which triggers all 32 hooks
                # each hook writes into current_step_buf[layer_idx]
                outputs = model(model_inputs, past_key_values=past, use_cache=True)

                # after the forward pass all 32 slots in current_step_buf are filled
                # stack them into one tensor: [num_layers, cache_len]
                stacked = torch.stack([current_step_buf[i] for i in range(num_layers)])
                step_attns.append(stacked)

                # pick the next token (greedy: just take the highest scoring one)
                # logits shape: [batch, seq_len, vocab_size]
                # [:, -1, :] takes the last position's predictions
                # argmax picks the single highest scoring token
                # keepdim=True keeps the shape as [1, 1] so we can concatenate it
                if step < MAX_NEW_TOKENS:
                    next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    # append the new token to the running sequence
                    generated = torch.cat([generated, next_tok], dim=1)

    finally:
        # critical: remove all hooks when done
        # if we don't do this, the hooks stay attached to the model forever
        # and would fire on every future generation run
        for h in handles:
            h.remove()

    # save everything to disk as a .pt file
    # torch.save just serializes the dict to a file — torch.load reads it back
    torch.save({
        "prompt": prompt,
        "input_ids": input_ids[0],       # the prompt token IDs
        "generated_ids": generated[0],   # prompt + all 50 generated tokens
        "prompt_len": prompt_len,
        "step_attns": step_attns,         # list of 51 tensors, one per step
    }, out_path)
    print(f"  [{prompt_idx:03d}] saved  prompt_len={prompt_len}  steps={len(step_attns)}")


if __name__ == "__main__":
    all_prompts = TRAIN_PROMPTS + EVAL_PROMPTS
    print(f"Collecting traces for {len(all_prompts)} prompts...")
    for idx, prompt in enumerate(all_prompts):
        print(f"[{idx}/{len(all_prompts)}]", end="  ")
        collect_one(prompt, idx)
    print("All done.")
