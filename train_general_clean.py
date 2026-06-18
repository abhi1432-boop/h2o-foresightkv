"""
train_general_clean.py — the leakage-free general-vs-oracle comparison.

Why: the general scorer in train_scorer.py trains on idx 0-255, which CONTAINS
the bank test prompts for 5 of 7 domains — so its per-domain eval there is
inflated (scored on prompts it trained on). This retrains a general scorer with
EVERY bank test prompt excluded, then compares it head-to-head against each
domain's oracle specialist on those same (now truly held-out) test sets.

Both sides are now clean on the test prompts:
  - oracle        trained only on its domain's train split
  - general_clean trained on everything EXCEPT all bank test prompts

Outputs scorer_general_clean.pt + a leakage-free general-vs-oracle table.
Requires: features.pt and bank/scorer_<domain>.pt (from train_domain_bank.py).
"""

import os
import glob
import torch
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from train_scorer import Scorer
from train_domain_bank import IndexDataset, fit

FEATURES_PATH = "features.pt"
BANK_DIR = "bank"


def load_bank():
    bank = {}
    for path in sorted(glob.glob(os.path.join(BANK_DIR, "scorer_*.pt"))):
        ckpt = torch.load(path, weights_only=False)
        model = Scorer()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        bank[ckpt["domain"]] = (model, ckpt)
    return bank


def domain_corr(model, rec_by_idx, idxs):
    preds, trues = [], []
    for i in idxs:
        rec = rec_by_idx[i]
        with torch.no_grad():
            preds.append(model(rec["features"]).squeeze(-1).numpy())
        trues.append(rec["ltc"].numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    if pred.std() < 1e-9 or true.std() < 1e-9:
        return float("nan")
    return pearsonr(pred, true)[0]


def main():
    records = torch.load(FEATURES_PATH, weights_only=False)
    rec_by_idx = {r["prompt_idx"]: r for r in records}
    bank = load_bank()
    if not bank:
        print(f"No scorers in {BANK_DIR}/ — run train_domain_bank.py first.")
        return

    # Holdout = union of every bank test prompt. The general scorer must not see any.
    holdout = set()
    for _, ckpt in bank.values():
        holdout.update(ckpt["test_idx"])

    # Clean training pool: all short prompts (train+eval splits) minus the holdout.
    pool = [r["prompt_idx"] for r in records if r["split"] in ("train", "eval")]
    clean_idx = set(i for i in pool if i not in holdout)

    tr_ds = IndexDataset(records, clean_idx)
    print(f"Clean general scorer: training on {len(clean_idx)} prompts "
          f"({len(tr_ds)} tokens), excluding {len(holdout)} bank-test prompts.")
    tr_loader = DataLoader(tr_ds, batch_size=512, shuffle=True)

    model = Scorer()
    model.feat_mean.copy_(tr_ds.x.mean(dim=0))
    model.feat_std.copy_(tr_ds.x.std(dim=0).clamp(min=1e-6))
    fit(model, tr_loader)
    model.eval()
    torch.save({"model_state": model.state_dict()}, "scorer_general_clean.pt")

    print("\nLEAKAGE-FREE general vs oracle (each domain's held-out test set)")
    print(f"{'domain':16s}{'oracle':>9}{'general':>10}{'winner':>10}")
    print("-" * 45)
    oracles, generals = [], []
    for d, (om, ckpt) in bank.items():
        te = ckpt["test_idx"]
        oracle = domain_corr(om, rec_by_idx, te)
        gen = domain_corr(model, rec_by_idx, te)
        oracles.append(oracle)
        generals.append(gen)
        print(f"{d:16s}{oracle:>9.3f}{gen:>10.3f}"
              f"{('oracle' if oracle > gen else 'general'):>10}")
    print("-" * 45)
    print(f"{'MEAN':16s}{np.mean(oracles):>9.3f}{np.mean(generals):>10.3f}")

    gen_wins = sum(1 for o, g in zip(oracles, generals) if g > o)
    print(f"\nGeneral wins {gen_wins}/{len(oracles)} domains (leakage-free).")
    print("general >= oracle overall -> one general scorer is best "
          "(register file not needed).\noracle > general -> specialists win "
          "once the comparison is fair.")


if __name__ == "__main__":
    main()
