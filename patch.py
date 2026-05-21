# This file wires H2OCache into phi-2 without touching phi-2's weights or rewriting
# its forward method. Two interception points:
#   1. H2OCacheAdapter.update() — phi-2 calls this to store K and V.
#      We redirect it to H2OCache instead of the default storage.
#   2. forward hooks — PyTorch calls these automatically after each attention layer
#      finishes. We use them to capture attn_weights and update H2O scores.

from transformers import DynamicCache
from h2o_cache import H2OCache


# --- Interception point 1: K and V storage ---

class H2OCacheAdapter(DynamicCache):
    # Phi-2 expects its past_key_values to be a Cache object.
    # DynamicCache is HuggingFace's standard Cache class — it has ~10 methods
    # the model calls internally (get_seq_length, __len__, to_legacy_cache, etc.)
    # By inheriting from it, we get all those methods for free and only need to
    # override the two that H2O actually changes: update() and get_seq_length().
    #
    # Think of it like: we're building on top of a working tool, changing only
    # the parts that need to behave differently. The rest stays untouched.

    def __init__(self, h2o_cache):
        # initialize the parent DynamicCache so all its internal state is set up
        super().__init__()
        # store a reference to our H2OCache so update() can delegate to it
        self.h2o_cache = h2o_cache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # This is the critical interception point for K and V.
        # Phi-2's attention layer calls this line inside its own forward():
        #   key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        # Because past_key_values IS our adapter, this method runs instead of
        # DynamicCache's built-in one. We redirect to H2OCache, which handles
        # the actual append-and-maybe-evict logic.
        del cache_kwargs  # phi-2 passes this but H2O doesn't use it
        return self.h2o_cache.update(key_states, value_states, layer_idx)

    def get_seq_length(self, layer_idx=0):
        # Phi-2 calls this to figure out the position ID of the next token.
        # Position IDs tell the model "this token is at position N in the sequence."
        # Without this, the model can't apply rotary embeddings correctly and
        # position-aware attention breaks.
        # We return the current number of tokens stored in layer 0's cache.
        cache = self.h2o_cache.key_cache[layer_idx]
        return 0 if cache is None else cache.shape[2]


# --- Interception point 2: attention score capture ---

def patch_model(model, max_cache_size, local_window_size):
    # Creates an H2OCache and attaches forward hooks to all 32 attention layers.
    # Returns three things:
    #   cache   — the H2OCache object (useful for inspection during generation)
    #   adapter — the H2OCacheAdapter to pass as past_key_values
    #   unpatch — call this when done to remove all hooks cleanly

    num_layers = len(model.model.layers)

    # One H2OCache shared across all layers — each layer gets its own slot
    # (key_cache[layer_idx], value_cache[layer_idx], accumulated_scores[layer_idx])
    cache = H2OCache(max_cache_size, local_window_size, num_layers)

    # We'll collect the hook handles here so unpatch() can remove them later.
    # A handle is what register_forward_hook returns — it's a reference to the
    # specific hook instance, used to remove that hook and only that hook.
    handles = []

    for layer_idx, layer in enumerate(model.model.layers):

        # We wrap the hook in make_hook() to "bake in" the correct layer_idx.
        # Without this wrapper, all 32 hooks would close over the same `layer_idx`
        # variable from the loop. By the time any hook fires, the loop has already
        # finished, so layer_idx would always equal 31 (the last value).
        # make_hook(layer_idx) captures the current value immediately, freezing it.
        def make_hook(idx):

            def hook(*args):
                # PyTorch calls every forward hook with exactly three arguments:
                #   args[0] — the module that just ran (the PhiAttention layer)
                #   args[1] — the inputs that were passed to it
                #   args[2] — the output it returned
                # We only care about the output, which is (attn_output, attn_weights).
                output = args[2]

                # Unpack the output tuple.
                # attn_output — the final result of attention (goes to the next layer)
                # attn_weights — the softmax scores matrix we need for H2O
                # The _ discards attn_output because we're not modifying it.
                _, attn_weights = output

                if attn_weights is not None:
                    # Pass the attention scores to H2OCache.
                    # update_scores() adds them to the running totals and calls
                    # _evict() if the cache has grown past the budget.
                    cache.update_scores(attn_weights, idx)

                # Hooks must return the output unchanged — phi-2 still needs it.
                # We're observers here, not modifiers.
                return output

            return hook

        # Attach the hook to this attention layer.
        # register_forward_hook tells PyTorch: "after this module's forward()
        # returns, automatically call our hook function."
        # The layer runs completely normally — the hook fires silently afterward.
        handle = layer.self_attn.register_forward_hook(make_hook(layer_idx))

        # Save the handle so we can remove this hook later via unpatch()
        handles.append(handle)

    def unpatch():
        # Remove every hook we registered in this call to patch_model.
        # This is critical: hooks persist on the module until explicitly removed.
        # If we don't call unpatch(), the next generate() call (even baseline)
        # will still fire these hooks and write into a stale H2OCache.
        for h in handles:
            h.remove()

    # Wrap the H2OCache in the adapter so it looks like a DynamicCache to phi-2
    adapter = H2OCacheAdapter(cache)
    return cache, adapter, unpatch
