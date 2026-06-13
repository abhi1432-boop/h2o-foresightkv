import math
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

#for phi-2 heres the input tensor[1,32,seq, 80] so from left to right
#[batch size, attention heads, tokens processed(depends on where we are), head dimension]
#head dimension = full embedding size / num heads = 2560 / 32 = 80 (each head gets its own slice)
class H2OCache:
    def __init__(self, max_cache_size, local_window_size, num_layers, use_foresight=False):
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

        # ForesightKV: when True, seed initial scores from a learned scorer instead
        # of from prefill attention. Everything after that (accumulation, eviction)
        # is identical — only the starting value changes.
        self.use_foresight = use_foresight
        # Stores per-layer prefill attention sums — populated during step 0,
        # consumed by seed_from_prefill(), then no longer needed.
        self._prefill_per_layer = [None] * num_layers

        # Beta decay: controls how much the scorer's prior influences the score
        # over time. Beta starts high so early eviction decisions trust the prior.
        # Each decode step beta multiplies by BETA_DECAY, shrinking toward zero.
        # By step 50 beta ≈ 0.08 — the prior has nearly faded out and the real
        # attention data is driving eviction decisions on its own.
        # Only active in foresight mode — standard H2O has no prior to decay.
        self.beta = 0.9          # starting trust in the scorer's prediction
        self.beta_decay = 0.95   # multiply beta by this each decode step
        # store the original predicted scores so we can re-blend each step
        self._prior_scores = [None] * num_layers

        # In foresight mode we defer eviction until after seed_from_prefill() runs.
        # That way the first eviction uses the scorer's predictions, not zeros.
        # For standard H2O this is always True (evict immediately as usual).
        self._prefill_done = not use_foresight

        # Eviction tracking — maps original sequence position to the decode step
        # at which it was first evicted. Used by evaluate.py to measure cold-start
        # errors. We track layer 0 as a proxy (all layers evict in near-lockstep
        # because they accumulate similar attention patterns).
        self._layer0_positions = None   # list: cache_index -> original seq position
        self.evicted_at_step = {}       # original_pos -> decode_step
        self._decode_step = 0           # incremented each time layer 0 gets a new token

    def update(self, key_states, value_states, layer_idx):
        # called every forward pass. Job: add the new token's K and V to the cache.
        # key_states / value_states shape: [batch, heads, new_tokens, head_dim]
        #   new_tokens = 1 during normal generation (one token at a time)
        #   new_tokens = prompt length on the very first pass (we feed all prompt tokens at once)

        if self.key_cache[layer_idx] is None:
            # first call — cache is empty, just store what we got
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            # Initialize position tracking on the first (prefill) call for layer 0.
            # _layer0_positions[i] = original sequence index of the token at cache slot i.
            if layer_idx == 0:
                self._layer0_positions = list(range(key_states.shape[2]))
        else:
            # cache already has past tokens — append new ones on the end
            # dim=2 is the sequence dimension: we're growing the "list of remembered tokens"
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            if layer_idx == 0 and self._layer0_positions is not None:
                next_pos = self._layer0_positions[-1] + 1
                self._layer0_positions.append(next_pos)
                self._decode_step += 1

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
        # move to CPU immediately: score math is on tiny tensors (budget-sized),
        # CPU is faster than MPS for these and avoids per-layer GPU sync overhead
        new_scores = attn_weights[0].sum(dim=(0, 1)).detach().cpu()  # shape: [key_len]

        if self.accumulated_scores[layer_idx] is None:
            if self.use_foresight:
                # Save the prefill attention per layer so seed_from_prefill() can
                # use it as feature 3 and 4 when building the scorer input.
                self._prefill_per_layer[layer_idx] = new_scores
                # Initialize to zeros — seed_from_prefill() will replace these
                # with the scorer's predicted importance scores after step 0.
                self.accumulated_scores[layer_idx] = torch.zeros_like(new_scores)
            else:
                # Standard H2O: start the accumulator from the prefill attention.
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

            # Beta decay (foresight mode only):
            # blend the accumulated real attention with the scorer's original prior.
            # formula: score = β * prior + (1 - β) * accumulated_attention
            # early on β is high so the prior dominates.
            # each step β shrinks by 0.95 so real attention gradually takes over.
            # by step 50 β ≈ 0.08 — the prior has almost completely faded out.
            if self.use_foresight and self._prior_scores[layer_idx] is not None:
                prior = self._prior_scores[layer_idx]
                # prior may be shorter than current cache if tokens were added after seeding
                # pad it with zeros for any new tokens (they have no prior prediction)
                if prior.shape[0] < self.accumulated_scores[layer_idx].shape[0]:
                    pad = torch.zeros(
                        self.accumulated_scores[layer_idx].shape[0] - prior.shape[0],
                        device=prior.device
                    )
                    prior = torch.cat([prior, pad])
                    self._prior_scores[layer_idx] = prior
                self.accumulated_scores[layer_idx] = (
                    self.beta * prior + (1 - self.beta) * self.accumulated_scores[layer_idx]
                )
                # only decay beta on layer 0 to avoid decaying 32 times per step
                if layer_idx == 0:
                    self.beta *= self.beta_decay

        # if the cache is now bigger than our budget, drop the least-attended tokens
        # (skipped during prefill in foresight mode — see seed_from_prefill)
        if self._prefill_done and self.key_cache[layer_idx].shape[2] > self.max_cache_size:
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

        # Record which original sequence positions are being dropped (layer 0 only).
        # This lets evaluate.py ask "was token p evicted, and at what decode step?"
        if layer_idx == 0 and self._layer0_positions is not None:
            keep_set = set(keep.tolist())
            for ci in range(cache_size):
                if ci not in keep_set:
                    orig = self._layer0_positions[ci]
                    if orig not in self.evicted_at_step:
                        self.evicted_at_step[orig] = self._decode_step
            self._layer0_positions = [self._layer0_positions[i] for i in keep.tolist()]

        # slice the cache and scores down to only the kept positions
        # scores/prior live on CPU; key_cache/value_cache live on the model device (MPS/CPU)
        # move keep to the KV device only for the KV indexing, keep it on CPU for scores
        kv_device = self.key_cache[layer_idx].device
        keep_kv = keep if kv_device.type == "cpu" else keep.to(kv_device)
        self.key_cache[layer_idx] = torch.index_select(self.key_cache[layer_idx], 2, keep_kv)
        self.value_cache[layer_idx] = torch.index_select(self.value_cache[layer_idx], 2, keep_kv)
        self.accumulated_scores[layer_idx] = scores[keep]   # CPU indexing, no GPU sync
        if self._prior_scores[layer_idx] is not None:
            self._prior_scores[layer_idx] = self._prior_scores[layer_idx][keep]  # CPU

    def seed_from_prefill(self, input_ids, freq_table, scorer_model):
        """Replace zero-initialized scores with scorer predictions.

        Call this exactly once, immediately after the step-0 (prefill) forward
        pass. After this point, normal accumulation continues on top of the
        predicted scores instead of on top of prefill attention.

        input_ids   — 1-D or 2-D LongTensor of the prompt token ids
        freq_table  — dict mapping token_id -> count in training corpus
        scorer_model — trained Scorer instance (nn.Module, eval mode)
        """
        if not self.use_foresight:
            return

        ids = input_ids.squeeze(0) if input_ids.dim() == 2 else input_ids
        prompt_len = ids.shape[0]
        num_layers  = self.num_layers

        # Build [num_layers, prompt_len] prefill attention matrix from the stored
        # per-layer sums that update_scores() saved during step 0.
        # move to CPU: scorer and feature math run on CPU regardless of device
        prefill = torch.stack([
            self._prefill_per_layer[l].cpu() if self._prefill_per_layer[l] is not None
            else torch.zeros(prompt_len)
            for l in range(num_layers)
        ])  # [L, prompt_len]

        # Feature 3: total prefill attention normalized within prompt
        total = prefill.sum(dim=0)                                   # [prompt_len]
        pf_sum = total.sum()
        pf_norm = total / pf_sum if pf_sum > 0 else torch.ones(prompt_len) / prompt_len

        # Feature 4: weighted mean layer index — captures where in the network
        # this token receives most attention (early layers vs late layers)
        layer_col  = torch.arange(num_layers, dtype=torch.float32).unsqueeze(1)
        layer_denom = prefill.sum(dim=0).clamp(min=1e-9)
        weighted    = (layer_col * prefill).sum(dim=0) / layer_denom
        layer_depth = weighted / max(num_layers - 1, 1)              # [0, 1]

        max_count = max(freq_table.values()) if freq_table else 1

        feats = torch.zeros(prompt_len, 5)
        for p in range(prompt_len):
            feats[p, 0] = p / max(prompt_len - 1, 1)
            feats[p, 1] = 1.0 if p < 5 else 0.0
            count = freq_table.get(ids[p].item(), 0)
            feats[p, 2] = math.log1p(count) / math.log1p(max_count) if max_count > 0 else 0.0
            feats[p, 3] = pf_norm[p].item()
            feats[p, 4] = layer_depth[p].item()

        scorer_model.eval()
        with torch.no_grad():
            predicted = scorer_model(feats).squeeze(1)  # [prompt_len]

        # Push predicted scores to every layer.  All 32 layers now start from the
        # same importance estimate rather than from zero.
        # Also save a copy in _prior_scores so beta decay can re-blend each step.
        for layer_idx in range(self.num_layers):
            if self.accumulated_scores[layer_idx] is not None:
                device = self.accumulated_scores[layer_idx].device
                self.accumulated_scores[layer_idx] = predicted.to(device).clone()
                self._prior_scores[layer_idx] = predicted.to(device).clone()

        # Now that scores are meaningful, allow eviction going forward and run
        # the first eviction pass (catches any prompt longer than the budget).
        self._prefill_done = True
        for layer_idx in range(self.num_layers):
            if (self.key_cache[layer_idx] is not None
                    and self.key_cache[layer_idx].shape[2] > self.max_cache_size):
                self._evict(layer_idx)
