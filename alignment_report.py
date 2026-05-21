"""
Answers Chaithu's two specific questions:

  Q1: How well-aligned is the scorer's prior with the eventual H2O accumulator
      state after the cold-start window?
      → Pearson correlation between scorer predictions and LTC on the eval set,
        broken down by prompt domain.

  Q2: Training-cost-vs-generalization tradeoff — does the scorer generalize
      across domains or only within the domain it was trained on?
      → Cross-domain correlation: train on 5 domains, test on the held-out ones.

Run after train_scorer.py has produced scorer.pt and features.pt.
"""

import torch
import numpy as np
from scipy.stats import pearsonr
from train_scorer import Scorer
from prompts import TRAIN_PROMPTS, EVAL_PROMPTS

FEATURES_PATH = "features.pt"
SCORER_PATH   = "scorer.pt"
NUM_TRAIN     = len(TRAIN_PROMPTS)

# Domain labels match the order in prompts.py:
# _QA(20) _REASONING(20) _CONVERSATIONAL(20) _CODE(20) _CREATIVE(20)
# _FACTUAL_LONG(20) _INSTRUCTIONS(20)
DOMAIN_NAMES = ["QA", "Reasoning", "Conversational", "Code",
                "Creative", "Factual-Long", "Instructions"]
DOMAIN_SIZE  = 20


def domain_of(prompt_idx):
    return DOMAIN_NAMES[min(prompt_idx // DOMAIN_SIZE, len(DOMAIN_NAMES) - 1)]


def topk_overlap(pred, true, k=10):
    k = min(k, len(pred))
    return len(set(np.argsort(pred)[-k:]) & set(np.argsort(true)[-k:])) / k


def main():
    records = torch.load(FEATURES_PATH, weights_only=False)
    scorer  = Scorer()
    ckpt    = torch.load(SCORER_PATH, weights_only=False)
    scorer.load_state_dict(ckpt["model_state"])
    scorer.eval()

    # -----------------------------------------------------------------------
    # Q1: Overall alignment on eval set
    # -----------------------------------------------------------------------
    eval_records = [r for r in records if r["split"] == "eval"]

    all_pred, all_true = [], []
    per_domain = {}

    for rec in eval_records:
        with torch.no_grad():
            pred = scorer(rec["features"]).squeeze(1).numpy()
        true = rec["ltc"].numpy()

        all_pred.append(pred)
        all_true.append(true)

        domain = domain_of(rec["prompt_idx"])
        if domain not in per_domain:
            per_domain[domain] = ([], [])
        per_domain[domain][0].append(pred)
        per_domain[domain][1].append(true)

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    overall_corr, _ = pearsonr(all_pred, all_true)
    overall_overlap = topk_overlap(all_pred, all_true, k=50)

    print("=" * 58)
    print("Q1: Scorer prior vs eventual accumulator state (eval set)")
    print("=" * 58)
    print(f"  Overall Pearson r  : {overall_corr:.3f}")
    print(f"  Top-50 overlap     : {overall_overlap:.3f}")
    print()
    print(f"  {'Domain':<18} {'r':>6}  {'top10_overlap':>14}")
    print(f"  {'-'*42}")
    for domain, (preds, trues) in per_domain.items():
        p = np.concatenate(preds)
        t = np.concatenate(trues)
        if len(p) < 2:
            continue
        r, _ = pearsonr(p, t)
        ov = topk_overlap(p, t, k=10)
        print(f"  {domain:<18} {r:>6.3f}  {ov:>14.3f}")

    # -----------------------------------------------------------------------
    # Q2: Generalization — train on some domains, test on others
    # The eval set covers Factual-Long + Instructions (indices 110-139).
    # The scorer was trained on QA / Reasoning / Conversational / Code / Creative.
    # So Q2 is already answered: the eval set is out-of-distribution by design.
    # -----------------------------------------------------------------------
    print()
    print("=" * 58)
    print("Q2: Generalization (train domains -> eval domains)")
    print("=" * 58)
    train_domains = set(domain_of(r["prompt_idx"]) for r in records if r["split"] == "train")
    eval_domains  = set(domain_of(r["prompt_idx"]) for r in records if r["split"] == "eval")
    print(f"  Train domains : {', '.join(sorted(train_domains))}")
    print(f"  Eval domains  : {', '.join(sorted(eval_domains))}")
    print(f"  Cross-domain r: {overall_corr:.3f}  (same as Q1 — eval is already OOD)")
    print()
    print("  Interpretation:")
    if overall_corr > 0.5:
        msg = "strong cross-domain generalization — scorer learned domain-agnostic importance signals"
    elif overall_corr > 0.3:
        msg = "moderate cross-domain generalization — scorer captures some universal patterns but is workload-sensitive"
    else:
        msg = "weak cross-domain generalization — scorer is mostly domain-specific, needs per-workload retraining"
    print(f"    r={overall_corr:.3f}: {msg}")
    print()
    print("  Hardware implication:")
    if overall_corr > 0.5:
        print("    Fixed weights at tape-out likely sufficient.")
    else:
        print("    Programmable weights from register file recommended —")
        print("    swap scorer weights per workload to recover alignment.")


if __name__ == "__main__":
    main()
