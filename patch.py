from transformers import DynamicCache
from h2o_cache import H2OCache


class H2OCacheAdapter(DynamicCache):
    """Routes phi-2's cache calls through our H2OCache.

    Inherits from DynamicCache so the model's helper methods (get_seq_length,
    __len__, etc.) keep working out of the box.
    """

    def __init__(self, h2o_cache):
        super().__init__()
        self.h2o_cache = h2o_cache

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        del cache_kwargs  # parent passes this but H2O doesn't need it
        return self.h2o_cache.update(key_states, value_states, layer_idx)

    def get_seq_length(self, layer_idx=0):
        # model calls this to compute position IDs for new tokens
        cache = self.h2o_cache.key_cache[layer_idx]
        return 0 if cache is None else cache.shape[2]


def patch_model(model, max_cache_size, local_window_size):
    """Patch phi-2 with H2O eviction. Returns (cache, adapter, unpatch).

    Call unpatch() when done — otherwise hooks accumulate across calls and
    leak state from the old cache into new runs.
    """
    num_layers = len(model.model.layers)
    cache = H2OCache(max_cache_size, local_window_size, num_layers)
    handles = []

    for layer_idx, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook(*args):
                output = args[2]  # PyTorch calls hook(module, inputs, output)
                _, attn_weights = output
                if attn_weights is not None:
                    cache.update_scores(attn_weights, idx)
                return output
            return hook

        handle = layer.self_attn.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    def unpatch():
        for h in handles:
            h.remove()

    adapter = H2OCacheAdapter(cache)
    return cache, adapter, unpatch
