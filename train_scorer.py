"""
Train a small MLP to predict token importance (LTC) from 5 prefill-only features.

Architecture: Linear(5, 16) -> ReLU -> Linear(16, 1) -> Sigmoid
Loss: MSELoss    Optimizer: Adam

Outputs:
  scorer.pt         full checkpoint (model_state + config)
  scorer_weights.py fixed-point integer constants for hardware export
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

FEATURES_PATH = "features.pt"
MODEL_OUT = "scorer.pt"
WEIGHTS_OUT = "scorer_weights.py"

EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 512
FIXED_POINT_BITS = 8
RANK_LOSS_WEIGHT = 0.3  # fraction of loss from pairwise ranking


class TokenDataset(Dataset):
    def __init__(self, records, split):
        xs, ys = [], []
        for rec in records:
            if rec["split"] != split:
                continue
            xs.append(rec["features"])           # [prompt_len, 5]
            ys.append(rec["ltc"].unsqueeze(1))   # [prompt_len, 1]
        self.x = torch.cat(xs, dim=0)
        self.y = torch.cat(ys, dim=0)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def topk_overlap(pred, true, k=50):
    k = min(k, pred.shape[0])
    pred_top = set(pred.topk(k).indices.tolist())
    true_top = set(true.topk(k).indices.tolist())
    return len(pred_top & true_top) / k


def evaluate(model, loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            all_pred.append(model(x).squeeze(-1))
            all_true.append(y.squeeze(-1))
    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    import numpy as np
    mse = float(((pred - true) ** 2).mean())
    corr, _ = pearsonr(pred, true)
    overlap = topk_overlap(torch.tensor(pred), torch.tensor(true), k=50)
    return mse, corr, overlap


def export_fixed_point(model, path, bits=8):
    scale = 2 ** (bits - 1) - 1
    lines = [f"# Fixed-point weights (int{bits}, scale factor per tensor)", ""]
    for name, param in model.named_parameters():
        arr = param.detach()
        max_val = arr.abs().max().item()
        if max_val == 0:
            max_val = 1.0
        quantized = (arr / max_val * scale).round().to(torch.int32)
        safe = name.replace(".", "_")
        lines += [
            f"{safe}_fp_scale = {max_val:.6f}",
            f"{safe}_fp = {quantized.tolist()}",
            "",
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Fixed-point weights written to {path}")


def main():
    records = torch.load(FEATURES_PATH, weights_only=False)

    train_ds = TokenDataset(records, "train")
    eval_ds  = TokenDataset(records, "eval")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=BATCH_SIZE)
    print(f"Train tokens: {len(train_ds)}  Eval tokens: {len(eval_ds)}")

    model = Scorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    mse_fn  = nn.MSELoss()
    rank_fn = nn.MarginRankingLoss(margin=0.05)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = mse_fn(pred, y)
            # pairwise ranking loss: split batch in half, penalise wrong orderings
            if x.shape[0] >= 4:
                mid = x.shape[0] // 2
                pa, pb = pred[:mid], pred[mid:2*mid]
                ya, yb = y[:mid],    y[mid:2*mid]
                target = (ya - yb).sign()
                loss = loss + RANK_LOSS_WEIGHT * rank_fn(pa, pb, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
        scheduler.step()

        if epoch % 20 == 0 or epoch == EPOCHS:
            mse, corr, overlap = evaluate(model, eval_loader)
            print(
                f"epoch {epoch:3d}  train_loss={total_loss/len(train_ds):.4f}  "
                f"val_mse={mse:.4f}  corr={corr:.3f}  top50_overlap={overlap:.3f}"
            )

    torch.save({"model_state": model.state_dict()}, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")
    export_fixed_point(model, WEIGHTS_OUT, bits=FIXED_POINT_BITS)
    return corr


def train_within_domain(cross_domain_r):
    """Train and test on Factual-Long only (indices 100-119).

    Training: prompts 100-109 (in the original train split)
    Testing:  prompts 110-119 (in the eval split, same domain)

    If the correlation here is positive while cross-domain is negative,
    it proves the architecture is sound and the issue is purely training data.
    Saves the within-domain scorer to scorer_indomain.pt.
    """
    FACTUAL_TRAIN = set(range(100, 110))
    FACTUAL_EVAL  = set(range(110, 120))

    records = torch.load(FEATURES_PATH, weights_only=False)

    def make_dataset(idx_set):
        xs, ys = [], []
        for rec in records:
            if rec["prompt_idx"] not in idx_set:
                continue
            xs.append(rec["features"])
            ys.append(rec["ltc"].unsqueeze(1))
        if not xs:
            return None
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)

        class _DS(Dataset):
            def __len__(self): return x.shape[0]
            def __getitem__(self, i): return x[i], y[i]
        return _DS()

    train_ds = make_dataset(FACTUAL_TRAIN)
    eval_ds  = make_dataset(FACTUAL_EVAL)

    if train_ds is None or eval_ds is None:
        print("Within-domain: missing traces for Factual-Long prompts — run collect_traces.py first")
        return

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=64)
    print(f"\n=== Within-domain experiment (Factual-Long only) ===")
    print(f"Train tokens: {len(train_ds)}  Eval tokens: {len(eval_ds)}")

    model     = Scorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    mse_fn  = nn.MSELoss()
    rank_fn = nn.MarginRankingLoss(margin=0.05)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = mse_fn(pred, y)
            if x.shape[0] >= 4:
                mid = x.shape[0] // 2
                pa, pb = pred[:mid], pred[mid:2*mid]
                ya, yb = y[:mid],    y[mid:2*mid]
                target = (ya - yb).sign()
                loss = loss + RANK_LOSS_WEIGHT * rank_fn(pa, pb, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
        scheduler.step()

        if epoch % 20 == 0 or epoch == EPOCHS:
            mse, corr, overlap = evaluate(model, eval_loader)
            print(
                f"epoch {epoch:3d}  train_loss={total_loss/len(train_ds):.4f}  "
                f"val_mse={mse:.4f}  corr={corr:.3f}  top50_overlap={overlap:.3f}"
            )

    torch.save({"model_state": model.state_dict()}, "scorer_indomain.pt")
    print(f"Within-domain scorer saved to scorer_indomain.pt")
    print(f"\nCross-domain r (from main run): {cross_domain_r:.3f}")
    print(f"Within-domain r (Factual-Long): {corr:.3f}")
    if corr > 0:
        print("→ Positive within-domain: architecture works, data is the bottleneck")
    else:
        print("→ Still negative: features may need revision too")


if __name__ == "__main__":
    cross_r = main()
    train_within_domain(cross_r)
