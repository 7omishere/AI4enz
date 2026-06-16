#!/usr/bin/env python3
"""Evaluate gate model on full test set with proper random sampling (includes negatives)."""
import sys, json, warnings
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/domi/AI4enz/dataset_building/models')
from train import OxidoreductaseDataset, collate_fn, NORM_PARAMS
from ranking_model import Trenzition

SCRIPT_DIR = Path('/home/domi/AI4enz/dataset_building/models')
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
BEST_CKPT = CHECKPOINT_DIR / "best.ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Train a quick fine-tuned model with gate
model = Trenzition(hidden_dim=256, use_gate=True)
ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
model.load_state_dict(state, strict=False)
model.to(device)

# Fine-tune on balanced negative data (2000 samples, 5 epochs)
train_ds = OxidoreductaseDataset(
    str(SCRIPT_DIR.parent / "processed/metadata_with_negatives.parquet"),
    str(SCRIPT_DIR.parent / "processed/proteins.h5"),
    str(SCRIPT_DIR.parent / "processed/ligands"),
    split="train", max_samples=None,
)
# Random sample balanced
sampled = train_ds.df.sample(n=2000, random_state=42)
train_ds.df = sampled.reset_index(drop=True)

train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True,
    collate_fn=collate_fn, num_workers=0,
)

from ranking_model import create_trenzition_optimizer
opt = create_trenzition_optimizer(model, lr=5e-5)

model.train()
for epoch in range(5):
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
        bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bg["ligand_data"] = bg["ligand_data"].to(device)
        opt.zero_grad()
        out = model(bg["ligand_data"], bg["seq_embed"], bg["cofactor_strs"])
        loss, losses = model.compute_loss(out, {
            "pkd_target": bg["pkd_target"], "pkd_target_mask": bg["pkd_target_mask"],
            "log_kcat_target": bg["log_kcat_target"], "kcat_target_mask": bg["kcat_target_mask"],
            "kcat_weights": bg["kcat_weights"], "quality_weight": bg["quality_weight"],
        })
        # Gate reg
        gate = out["gate_profile"].mean(dim=0)
        is_neg = bg.get("is_negative", torch.zeros_like(gate, dtype=torch.bool))
        if is_neg.any():
            l_g = 0.02 * (((1 - gate[~is_neg])**2).mean() + (gate[is_neg]**2).mean())
            loss = loss + l_g
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

model.eval()

# 2. Evaluate on FULL test set with random sampling
test_ds = OxidoreductaseDataset(
    str(SCRIPT_DIR.parent / "processed/metadata_with_negatives.parquet"),
    str(SCRIPT_DIR.parent / "processed/proteins.h5"),
    str(SCRIPT_DIR.parent / "processed/ligands"),
    split="test", max_samples=None,
)
# Random subsample to keep it fast (5000)
test_sampled = test_ds.df.sample(n=5000, random_state=42)
test_ds.df = test_sampled.reset_index(drop=True)
test_loader = DataLoader(
    test_ds, batch_size=128, shuffle=False,
    collate_fn=collate_fn, num_workers=0,
)

all_gates, all_is_neg, all_pkd_pred, all_pkd_true = [], [], [], []
all_kcat_pred, all_kcat_true = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating on full test set"):
        bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bg["ligand_data"] = bg["ligand_data"].to(device)
        out = model(bg["ligand_data"], bg["seq_embed"], bg["cofactor_strs"])

        gate_mean = out["gate_profile"].mean(dim=0).cpu()
        all_gates.append(gate_mean.numpy())
        is_neg = bg.get("is_negative", torch.zeros(len(gate_mean), dtype=torch.bool)).cpu().numpy()
        all_is_neg.append(is_neg)

        # Only collect metrics for non-negative samples with labels
        pkd_mask = bg["pkd_target_mask"].cpu().numpy() & ~is_neg
        kcat_mask = bg["kcat_target_mask"].cpu().numpy() & ~is_neg

        pkd_pred = (out["ts_stability"].cpu() * 12.0)
        pkd_true = (bg["pkd_target"].cpu() * 12.0)
        all_pkd_pred.append(pkd_pred[pkd_mask])
        all_pkd_true.append(pkd_true[pkd_mask])

        kcat_pred = (out["catalysis_rate"].cpu() * 15.0 - 7.0)
        kcat_true = (bg["log_kcat_target"].cpu() * 15.0 - 7.0)
        all_kcat_pred.append(kcat_pred[kcat_mask])
        all_kcat_true.append(kcat_true[kcat_mask])

gates = np.concatenate(all_gates)
is_neg = np.concatenate(all_is_neg)

pos_gates = gates[~is_neg]
neg_gates = gates[is_neg]
print(f"\n{'='*60}")
print(f"Gate Distribution on Full Test Set (n={len(gates)})")
print(f"{'='*60}")
print(f"  Positive (n={len(pos_gates)}): mean={pos_gates.mean():.4f}, median={np.median(pos_gates):.4f}, std={pos_gates.std():.4f}")
print(f"  Negative (n={len(neg_gates)}): mean={neg_gates.mean():.4f}, median={np.median(neg_gates):.4f}, std={neg_gates.std():.4f}")
print(f"  Separation (pos - neg): {pos_gates.mean() - neg_gates.mean():.4f}")

# Distribution bins
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    pos_above = (pos_gates > t).mean() * 100
    neg_below = (neg_gates <= t).mean() * 100
    print(f"    Gate > {t:.1f}: pos {pos_above:.1f}% | Gate ≤ {t:.1f}: neg {neg_below:.1f}%")

# Gate histogram
print(f"\n  Positive Gate histogram (0-1, 10 bins):")
hist_p, _ = np.histogram(pos_gates, bins=10, range=(0,1))
print(f"    {hist_p}")
print(f"  Negative Gate histogram (0-1, 10 bins):")
hist_n, _ = np.histogram(neg_gates, bins=10, range=(0,1))
print(f"    {hist_n}")

# pKd/kcat metrics
pkd_pred = np.concatenate(all_pkd_pred)
pkd_true = np.concatenate(all_pkd_true)
kcat_pred = np.concatenate(all_kcat_pred)
kcat_true = np.concatenate(all_kcat_true)

print(f"\n{'='*60}")
print(f"Regression Metrics (positive samples only)")
print(f"{'='*60}")
for name, pred, true in [("pKd", pkd_pred, pkd_true), ("kcat", kcat_pred, kcat_true)]:
    mse = np.mean((pred - true)**2)
    mae = np.mean(np.abs(pred - true))
    r2 = 1 - mse / np.var(true)
    sr, _ = spearmanr(pred, true)
    pr, _ = pearsonr(pred, true)
    print(f"  {name}: N={len(pred)}, MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, ρ={sr:.4f}, r={pr:.4f}")

# Comparison with original best.ckpt
print(f"\n{'='*60}")
print(f"Comparison with Original best.ckpt (from benchmark_enhanced_results.json)")
print(f"{'='*60}")
print(f"  Original pKd: R²=0.703, ρ=0.895")
print(f"  Original kcat: R²=0.458, ρ=0.671")
print(f"  (Note: original was trained on 97k samples × 81 epochs; this is 2k × 5 epochs)")

# Save results
results = {
    "gate": {
        "pos_mean": float(pos_gates.mean()),
        "neg_mean": float(neg_gates.mean()),
        "pos_median": float(np.median(pos_gates)),
        "neg_median": float(np.median(neg_gates)),
        "pos_std": float(pos_gates.std()),
        "neg_std": float(neg_gates.std()),
        "diff": float(pos_gates.mean() - neg_gates.mean()),
    },
    "metrics": {
        "pkd": {"N": len(pkd_pred), "MSE": float(np.mean((pkd_pred-pkd_true)**2)),
                "MAE": float(np.mean(np.abs(pkd_pred-pkd_true))),
                "R2": float(1 - np.mean((pkd_pred-pkd_true)**2) / np.var(pkd_true)),
                "Spearman": float(spearmanr(pkd_pred, pkd_true)[0]),
                "Pearson": float(pearsonr(pkd_pred, pkd_true)[0])},
        "kcat": {"N": len(kcat_pred), "MSE": float(np.mean((kcat_pred-kcat_true)**2)),
                 "MAE": float(np.mean(np.abs(kcat_pred-kcat_true))),
                 "R2": float(1 - np.mean((kcat_pred-kcat_true)**2) / np.var(kcat_true)),
                 "Spearman": float(spearmanr(kcat_pred, kcat_true)[0]),
                 "Pearson": float(pearsonr(kcat_pred, kcat_true)[0])},
    }
}

with open("/home/domi/AI4enz/dataset_building/models/checkpoints/gate_full_eval.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to gate_full_eval.json")
