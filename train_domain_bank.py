"""
train_domain_bank.py — the per-domain "register file" experiment, Part 1.

Why this exists
---------------
A single general scorer averages every domain's importance pattern into one set
of 97 weights. The original cross-domain run showed that can go NEGATIVE
(r = -0.507): a Code prior actively mis-ranks Creative text. Chaithu's fix is a
PROGRAMMABLE REGISTER FILE — instead of one scorer baked into silicon, hold a
BANK of per-domain weight sets and load the matching one per workload.

This script builds that bank. It trains one Scorer per prompt domain, each with
its own feat_mean/feat_std signature (reused later for routing + OOD gating).
Within each domain we hold out the last ~20% of prompts as a test set, so the
reported correlation is within-domain quality on UNSEEN prompts of the same
domain — exactly what the bank is supposed to deliver on every workload.

We have 7 natural domains (Chaithu's "10 scorers" is loose for "a register
file's worth"); the bank is sized to whatever domains have traces.

Prerequisite
------------
Needs the FULL features.pt covering all 320 prompts. The local features.pt only
covers prompt_idx 0-149 — regenerate on Colab via:
    collect_traces.py -> extract_features.py -> compute_labels.py

Outputs
-------
bank/scorer_<domain>.pt for each domain (model_state + routing stats +
train/test split), plus a per-domain correlation summary table.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Reuse the exact architecture, hyperparameters, and loader-based eval that the
# single general scorer uses — the bank must be an apples-to-apples comparison.
from train_scorer import Scorer, EPOCHS, LR, RANK_LOSS_WEIGHT, evaluate
from prompts import (_QA, _REASONING, _CONVERSATIONAL, _CODE, _CREATIVE,
                     _FACTUAL_LONG, _INSTRUCTIONS)

FEATURES_PATH = "features.pt"
BANK_DIR = "bank"
TEST_FRAC = 0.2     # last 20% of each domain's prompts held out for testing
BATCH_SIZE = 64     # domains are small (~40 prompts), so a small batch

# Order MUST match ALL_PROMPTS in prompts.py (_ALL is these lists concatenated),
# so cumulative lengths give each prompt_idx its domain.
DOMAIN_LISTS = [
    ("QA",             _QA),
    ("Reasoning",      _REASONING),
    ("Conversational", _CONVERSATIONAL),
    ("Code",           _CODE),
    ("Creative",       _CREATIVE),
    ("Factual-Long",   _FACTUAL_LONG),
    ("Instructions",   _INSTRUCTIONS),
]


def domain_index_ranges():
    """Map domain name -> list of prompt_idx, matching ALL_PROMPTS ordering."""
    ranges, start = {}, 0
    for name, lst in DOMAIN_LISTS:
        ranges[name] = list(range(start, start + len(lst)))
        start += len(lst)
    return ranges


def split_indices(idxs):
    """Deterministic within-domain train/test split: last TEST_FRAC as test.
    Deterministic so Part 3's routing eval can reproduce the identical test set."""
    idxs = sorted(idxs)
    n_test = max(1, int(round(len(idxs) * TEST_FRAC)))
    return idxs[:-n_test], idxs[-n_test:]


class IndexDataset(Dataset):
    """All tokens belonging to a given set of prompt indices."""
    def __init__(self, records, idx_set):
        xs, ys = [], []
        for rec in records:
            if rec["prompt_idx"] in idx_set:
                xs.append(rec["features"])           # [prompt_len, 5]
                ys.append(rec["ltc"].unsqueeze(1))   # [prompt_len, 1]
        self.x = torch.cat(xs, dim=0) if xs else torch.empty(0, 5)
        self.y = torch.cat(ys, dim=0) if ys else torch.empty(0, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def fit(model, train_loader, epochs=EPOCHS):
    """Same training recipe as the general scorer: MSE + pairwise ranking loss,
    Adam, cosine-annealed LR. No per-epoch logging (the bank is many quick runs)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    mse_fn  = nn.MSELoss()
    rank_fn = nn.MarginRankingLoss(margin=0.05)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = mse_fn(pred, y)
            if x.shape[0] >= 4:  # pairwise ranking: penalise wrong orderings
                mid = x.shape[0] // 2
                pa, pb = pred[:mid], pred[mid:2*mid]
                ya, yb = y[:mid],    y[mid:2*mid]
                loss = loss + RANK_LOSS_WEIGHT * rank_fn(pa, pb, (ya - yb).sign())
            loss.backward()
            optimizer.step()
        scheduler.step()


def main():
    records = torch.load(FEATURES_PATH, weights_only=False)
    traced = set(r["prompt_idx"] for r in records)
    os.makedirs(BANK_DIR, exist_ok=True)
    ranges = domain_index_ranges()

    print(f"Training per-domain bank from {len(traced)} traced prompts "
          f"(idx {min(traced)}-{max(traced)})\n")
    print(f"{'domain':16s}{'train':>7}{'test':>6}{'tok_tr':>8}{'tok_te':>8}"
          f"{'corr':>8}{'top50':>8}")
    print("-" * 61)

    results = {}
    for name, _ in DOMAIN_LISTS:
        idxs = [i for i in ranges[name] if i in traced]
        if len(idxs) < 4:
            print(f"{name:16s}  -- skipped: only {len(idxs)} traced prompts --")
            continue

        tr_idx, te_idx = split_indices(idxs)
        tr_ds = IndexDataset(records, set(tr_idx))
        te_ds = IndexDataset(records, set(te_idx))
        if len(tr_ds) == 0 or len(te_ds) == 0:
            print(f"{name:16s}  -- skipped: empty token split --")
            continue

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE)

        model = Scorer()
        # Routing signature: what "in-distribution" looks like for THIS domain.
        # The router (Part 2) z-scores a new prompt against every bank entry's
        # feat_mean/feat_std and loads the closest; OOD gate fires if none match.
        model.feat_mean.copy_(tr_ds.x.mean(dim=0))
        model.feat_std.copy_(tr_ds.x.std(dim=0).clamp(min=1e-6))

        fit(model, tr_loader)
        mse, corr, overlap = evaluate(model, te_loader)
        results[name] = corr

        path = os.path.join(BANK_DIR, f"scorer_{name.replace('-', '_')}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "domain": name,
            "train_idx": tr_idx,
            "test_idx": te_idx,
        }, path)

        print(f"{name:16s}{len(tr_idx):>7}{len(te_idx):>6}{len(tr_ds):>8}"
              f"{len(te_ds):>8}{corr:>8.3f}{overlap:>8.3f}")

    if results:
        avg = sum(results.values()) / len(results)
        print("-" * 61)
        print(f"{'MEAN per-domain corr':37s}{avg:>8.3f}")
        print(f"\nBank written to {BANK_DIR}/ ({len(results)} scorers).")
        print("Compare this mean to the single general scorer's cross-domain "
              "r = 0.791:\nif the bank's per-domain mean is higher, "
              "specialization beats the generalist\n(the empirical case for the "
              "programmable register file). Routing eval = Part 3.")
    else:
        print("\nNo scorers trained — features.pt likely missing most domains. "
              "Regenerate the FULL features.pt on Colab first.")


if __name__ == "__main__":
    main()
