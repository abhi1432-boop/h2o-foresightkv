"""
eval_domain_bank.py — Parts 2 & 3 of the register-file experiment.

Part 2 (router): given a prompt's prefill features, pick which bank scorer to
load. We reuse the SAME signal as the OOD gate in h2o_cache.py — z-score the
prompt's mean feature vector against each scorer's training distribution
(feat_mean/feat_std) and route to the closest (lowest max-z). If even the
closest exceeds OOD_Z, the gate fires (beta -> 0, fall back to cold-start H2O).

Part 3 (eval): on each domain's OWN held-out test set (the same split the bank
was trained with), compare four routing policies, all apples-to-apples:

  oracle    — use that domain's own scorer (upper bound)
  learned   — let the z-score router pick the scorer
  general   — the single general scorer (scorer.pt), scored on this domain's
              test set (the fair version of the 0.747 comparison)
  wrong     — use a MISMATCHED domain's scorer; reported as the worst case
              (min) and the average over all other domains. This is the
              inversion check: does a wrong prior actively hurt (go negative)?

Plus two diagnostics:
  - routing accuracy: how often the learned router picks the true domain
  - held-out gate: remove a domain's own scorer, and check whether its prompts
    trip the OOD gate (z>OOD_Z). Tests whether the fallback would catch a
    genuinely novel workload.

Requires: bank/scorer_<domain>.pt (from train_domain_bank.py), features.pt,
and scorer.pt (the general scorer from train_scorer.py).
"""

import os
import glob
import torch
import numpy as np
from scipy.stats import pearsonr

from train_scorer import Scorer

FEATURES_PATH = "features.pt"
BANK_DIR = "bank"
GENERAL_PATH = "scorer.pt"
OOD_Z = 3.0     # same threshold as h2o_cache.py's domain-confidence gate


def load_scorer(path):
    ckpt = torch.load(path, weights_only=False)
    model = Scorer()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def pred_tokens(model, feats):
    with torch.no_grad():
        return model(feats).squeeze(-1).numpy()


def corr_over(pairs):
    """One Pearson r over all (pred, true) token pairs concatenated."""
    if not pairs:
        return float("nan")
    pred = np.concatenate([p for p, _ in pairs])
    true = np.concatenate([t for _, t in pairs])
    if pred.std() < 1e-9 or true.std() < 1e-9:
        return float("nan")
    return pearsonr(pred, true)[0]


def max_z(feat_vec, model):
    """How far OOD this feature vector is from a scorer's training dist."""
    z = (feat_vec - model.feat_mean).abs() / model.feat_std
    return z.max().item()


def main():
    # ── load the bank, the general scorer, and the features ──────────────
    bank = {}  # domain -> (model, ckpt)
    for path in sorted(glob.glob(os.path.join(BANK_DIR, "scorer_*.pt"))):
        model, ckpt = load_scorer(path)
        bank[ckpt["domain"]] = (model, ckpt)
    if not bank:
        print(f"No scorers in {BANK_DIR}/ — run train_domain_bank.py first.")
        return
    general, _ = load_scorer(GENERAL_PATH)

    records = torch.load(FEATURES_PATH, weights_only=False)
    rec_by_idx = {r["prompt_idx"]: r for r in records}
    test_sets = {d: bank[d][1]["test_idx"] for d in bank}
    domains = list(bank.keys())

    def route(feat_vec, exclude=None):
        best_d, best_z = None, float("inf")
        for d in domains:
            if d == exclude:
                continue
            z = max_z(feat_vec, bank[d][0])
            if z < best_z:
                best_z, best_d = z, d
        return best_d, best_z

    def domain_corr(d, model):
        pairs = []
        for idx in test_sets[d]:
            rec = rec_by_idx[idx]
            pairs.append((pred_tokens(model, rec["features"]), rec["ltc"].numpy()))
        return corr_over(pairs)

    # ── Part 3: four-policy comparison, per domain ───────────────────────
    print("POLICY COMPARISON — correlation on each domain's OWN held-out test set")
    print(f"{'domain':16s}{'oracle':>9}{'learned':>9}{'general':>9}"
          f"{'wrong_min':>11}{'wrong_avg':>11}")
    print("-" * 76)

    agg = {k: [] for k in ["oracle", "learned", "general", "wrong_min", "wrong_avg"]}
    route_correct = route_total = 0
    gate_fired = gate_total = 0

    for d in domains:
        # oracle / general (whole-domain concatenation)
        oracle = domain_corr(d, bank[d][0])
        general_r = domain_corr(d, general)

        # wrong routing: every OTHER domain's scorer on this domain's test set
        wrongs = [domain_corr(d, bank[od][0]) for od in domains if od != d]
        wrong_min = min(wrongs)
        wrong_avg = float(np.mean(wrongs))

        # learned routing: pick a scorer per prompt via the z-score router
        learned_pairs = []
        for idx in test_sets[d]:
            rec = rec_by_idx[idx]
            fv = rec["features"].mean(dim=0)          # prompt summary [5]
            rd, _ = route(fv)
            learned_pairs.append((pred_tokens(bank[rd][0], rec["features"]),
                                  rec["ltc"].numpy()))
            route_total += 1
            route_correct += int(rd == d)
            # held-out gate: would a novel workload (own scorer removed) be caught?
            _, bz = route(fv, exclude=d)
            gate_total += 1
            gate_fired += int(bz > OOD_Z)
        learned = corr_over(learned_pairs)

        for k, v in [("oracle", oracle), ("learned", learned), ("general", general_r),
                     ("wrong_min", wrong_min), ("wrong_avg", wrong_avg)]:
            agg[k].append(v)
        print(f"{d:16s}{oracle:>9.3f}{learned:>9.3f}{general_r:>9.3f}"
              f"{wrong_min:>11.3f}{wrong_avg:>11.3f}")

    print("-" * 76)
    print(f"{'MEAN':16s}"
          f"{np.mean(agg['oracle']):>9.3f}{np.mean(agg['learned']):>9.3f}"
          f"{np.mean(agg['general']):>9.3f}{np.mean(agg['wrong_min']):>11.3f}"
          f"{np.mean(agg['wrong_avg']):>11.3f}")

    # ── diagnostics ──────────────────────────────────────────────────────
    print(f"\nRouting accuracy (learned router picks true domain): "
          f"{route_correct}/{route_total} = {route_correct/route_total:.1%}")
    print(f"Held-out gate (own scorer removed, prompts tripping OOD z>{OOD_Z}): "
          f"{gate_fired}/{gate_total} = {gate_fired/gate_total:.1%}")

    # ── interpretation hints ─────────────────────────────────────────────
    print("\nHow to read this:")
    print("  oracle > general  -> specialists beat the generalist (register file wins)")
    print("  learned ~ oracle  -> the cheap z-score router actually works")
    print("  wrong_min < 0     -> a mismatched prior INVERTS eviction (justifies")
    print("                       both the register file and the OOD gate)")
    print("  gate% high        -> the fallback catches novel workloads;")
    print("  gate% low         -> these 7 domains aren't OOD from each other")
    print("                       (the gate is for genuinely alien inputs, not")
    print("                        for distinguishing similar in-domain prompts)")


if __name__ == "__main__":
    main()
