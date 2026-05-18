import torch


# The KV cache is just past K and V vectors saved between generation steps.
# Why save K and V?
#   - Every new token's Q has to compare itself against every past token's K
#     to decide where to look.
#   - It then mixes past V vectors using those comparison scores.
#   - Without the cache we'd recompute K and V for every past token, every step.
# Why NOT save Q?
#   - Q is only used once: by the token that just got generated.
#   - The next token will compute its own fresh Q. Past Q's are useless.
#
# H2O changes the cache from "keep everything" to "keep what matters":
#   - matters = tokens that received lots of attention across past steps
#   - plus a small window of recent tokens (so the model doesn't lose local context)
class H2OCache:
    def __init__(self, max_cache_size, local_window_size, num_layers):
        # safety check — if window > budget, math below breaks
        assert local_window_size <= max_cache_size, "window can't exceed total budget"

        # total number of tokens we're allowed to remember
        self.max_cache_size = max_cache_size
        # how many of those slots are reserved for the most recent tokens
        self.local_window_size = local_window_size
        # one cache per transformer layer (phi-2 has 32)
        self.num_layers = num_layers

        # K vectors for every remembered token, per layer
        self.key_cache = [None] * num_layers
        # V vectors for every remembered token, per layer
        self.value_cache = [None] * num_layers
        # running total: how much attention has each remembered token received so far?
        # this is the "heavy hitter score" — higher means more important to keep
        self.accumulated_scores = [None] * num_layers

    def update(self, key_states, value_states, layer_idx):
        # called every forward pass. Job: add the new token's K and V to the cache.
        # key_states / value_states shape: [batch, heads, new_tokens, head_dim]
        #   new_tokens = 1 during normal generation (one token at a time)
        #   new_tokens = prompt length on the very first pass (we feed all prompt tokens at once)

        if self.key_cache[layer_idx] is None:
            # first call — cache is empty, just store what we got
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # cache already has past tokens — append new ones on the end
            # dim=2 is the sequence dimension: we're growing the "list of remembered tokens"
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        # return the full cache so attention can run over all remembered tokens
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_scores(self, attn_weights, layer_idx):
        # called AFTER attention runs. Job: update each token's "importance score"
        # using how much it just got attended to, then evict if we're over budget.
        # attn_weights shape: [batch, heads, query_len, key_len]
        #   - query_len = how many new queries asked this step
        #   - key_len   = how many tokens are currently in the cache
        # For each key, we want one number: total attention received this step.
        # We sum across heads (all 32 of them) AND across queries (every asker).
        # .detach() makes sure we're not accidentally tracking gradients.
        new_scores = attn_weights[0].sum(dim=(0, 1)).detach()  # shape: [key_len]

        if self.accumulated_scores[layer_idx] is None:
            # first call — no running totals yet, just store what we got
            self.accumulated_scores[layer_idx] = new_scores
        else:
            # the cache grew this step. The scores tensor needs to grow too.
            # Compare lengths to find how many brand-new positions exist:
            old_len = self.accumulated_scores[layer_idx].shape[0]
            num_new = new_scores.shape[0] - old_len
            if num_new > 0:
                # pad the running totals with zeros for the new positions
                # (new tokens start with score 0 — they haven't been attended to before)
                self.accumulated_scores[layer_idx] = torch.cat([
                    self.accumulated_scores[layer_idx],
                    torch.zeros(num_new, device=new_scores.device),
                ])
            # add this step's attention onto the running totals
            self.accumulated_scores[layer_idx] = self.accumulated_scores[layer_idx] + new_scores

        # if the cache is now bigger than our budget, drop the least-attended tokens
        if self.key_cache[layer_idx].shape[2] > self.max_cache_size:
            self._evict(layer_idx)

    def _evict(self, layer_idx):
        # called when the cache is over budget. Job: pick who to keep, drop the rest.
        # Rule: keep the heavy hitters (high score) + the local window (most recent tokens).

        cache_size = self.key_cache[layer_idx].shape[2]
        scores = self.accumulated_scores[layer_idx]

        # split the cache into "older tokens" and "local window"
        # local window = the last N positions, always safe from eviction
        local_start = cache_size - self.local_window_size

        # how many slots are left for heavy hitters (after reserving the window)
        num_heavy_hitters = self.max_cache_size - self.local_window_size

        # look only at the older (non-window) tokens — these are eviction candidates
        non_window_scores = scores[:local_start]

        # pick the top scorers among the older tokens
        # if we have fewer older tokens than slots, just keep them all
        k = min(num_heavy_hitters, non_window_scores.shape[0])

        if k > 0:
            # topk returns indices in order of score, not position
            _, top_indices = non_window_scores.topk(k)
            # sort by position so the cache stays in original left-to-right order
            # (mixing up the order would confuse the model — attention is positional)
            top_indices = top_indices.sort().values
        else:
            # edge case: no heavy hitter slots (budget == window). keep only the window.
            top_indices = torch.empty(0, dtype=torch.long, device=scores.device)

        # the local window indices are simply the last N positions
        local_indices = torch.arange(local_start, cache_size, device=scores.device)

        # final keep list = heavy hitters + local window (already in position order)
        keep = torch.cat([top_indices, local_indices])

        # slice the cache and scores down to only the kept positions
        # everything not in `keep` is silently dropped — its K, V, and score are gone
        self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep, :]
        self.accumulated_scores[layer_idx] = scores[keep]
