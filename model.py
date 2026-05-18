from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch
from patch import patch_model

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# eager attention exposes attn_weights (sdpa hides them — H2O needs them)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    attn_implementation="eager",
)
model.eval()

print("done loading")


def generate(prompt, max_new_tokens=30, h2o_config=None, verbose=False):
    """Greedy generation. If h2o_config is None, run a normal cached baseline.

    h2o_config: dict with keys 'max_cache_size' and 'local_window_size'.
    """
    if h2o_config is not None:
        cache, past, unpatch = patch_model(
            model,
            max_cache_size=h2o_config["max_cache_size"],
            local_window_size=h2o_config["local_window_size"],
        )
    else:
        # baseline still needs a persistent cache across steps, otherwise the
        # model only sees one token per step after step 0 and produces garbage
        cache, past, unpatch = None, DynamicCache(), None

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                # step 0 feeds the whole prompt to fill the cache;
                # later steps feed only the newest token (cache holds the rest)
                model_inputs = generated if step == 0 else generated[:, -1:]
                outputs = model(model_inputs, past_key_values=past, use_cache=True)

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if verbose and cache is not None:
                    size = cache.key_cache[0].shape[2]
                    assert size <= h2o_config["max_cache_size"], "cache exceeded budget"
                    if step % 10 == 0 or step == max_new_tokens - 1:
                        print(f"  step {step:>3}: cache size = {size}")
    finally:
        # critical: remove hooks so the next run starts clean
        if unpatch is not None:
            unpatch()

    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    prompt = "Instruct: Write a haiku about a cat.\nOutput:"
    n = 30

    print("\n=== BASELINE (full cache) ===")
    baseline = generate(prompt, max_new_tokens=n)
    print(baseline)

    print("\n=== H2O budget=40 window=8 (should match baseline — no eviction) ===")
    big = generate(prompt, max_new_tokens=n, h2o_config={"max_cache_size": 40, "local_window_size": 8}, verbose=True)
    print(big)

    print("\n=== H2O budget=12 window=4 (heavy eviction — should diverge) ===")
    small = generate(prompt, max_new_tokens=n, h2o_config={"max_cache_size": 12, "local_window_size": 4}, verbose=True)
    print(small)

    print("\n=== verdict ===")
    print(f"big-budget H2O matches baseline: {big == baseline}")
    print(f"small-budget H2O differs from baseline: {small != baseline}")
