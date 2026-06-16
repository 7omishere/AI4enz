#!/usr/bin/env python3
"""
Gate 消融实验 — 在含负样本的数据上训练，对比有 Gate vs 无 Gate。

策略：
1. 从 best.ckpt 加载基础权重（共享编码器已经在大量数据上训练好）
2. 在 metadata_with_negatives.parquet 上进行短微调
3. Ablation A: 有 Gate (use_gate=True, gate_weight=0.02)
4. Ablation B: 无 Gate (use_gate=False, gate_weight=0)

这样比从零训练快得多，且 Gate 的贡献不会被 encoder 没训好掩盖。
"""
import sys, json, logging, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gate_ablation")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train import OxidoreductaseDataset, collate_fn, NORM_PARAMS
from ranking_model import Trenzition, create_trenzition_optimizer

PROCESSED_DIR = SCRIPT_DIR.parent / "processed"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
BEST_CKPT = CHECKPOINT_DIR / "best.ckpt"
UNIFIED_METADATA = str(PROCESSED_DIR / "metadata_with_negatives.parquet")
PROTEINS_H5 = str(PROCESSED_DIR / "proteins.h5")
LIGAND_DIR = str(PROCESSED_DIR / "ligands")

def evaluate(model, loader, device) -> dict:
    """Evaluate on a dataloader, return metrics."""
    model.eval()
    all_pkd_pred, all_pkd_true = [], []
    all_kcat_pred, all_kcat_true = [], []
    all_gates = []
    all_is_neg = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(device)

            outputs = model(
                batch_gpu["ligand_data"],
                batch_gpu["seq_embed"],
                batch_gpu["cofactor_strs"],
            )

            # 反归一化
            pkd_pred = outputs["ts_stability"].cpu() * (NORM_PARAMS['pkd_max'] - NORM_PARAMS['pkd_min']) + NORM_PARAMS['pkd_min']
            kcat_pred = outputs["catalysis_rate"].cpu() * (NORM_PARAMS['kcat_max'] - NORM_PARAMS['kcat_min']) + NORM_PARAMS['kcat_min']

            pkd_mask = batch_gpu["pkd_target_mask"].cpu()
            kcat_mask = batch_gpu["kcat_target_mask"].cpu()

            pkd_true = batch_gpu["pkd_target"].cpu() * (NORM_PARAMS['pkd_max'] - NORM_PARAMS['pkd_min']) + NORM_PARAMS['pkd_min']

            kcat_true = batch_gpu["log_kcat_target"].cpu() * (NORM_PARAMS['kcat_max'] - NORM_PARAMS['kcat_min']) + NORM_PARAMS['kcat_min']

            all_pkd_pred.append(pkd_pred[pkd_mask])
            all_pkd_true.append(pkd_true[pkd_mask])
            all_kcat_pred.append(kcat_pred[kcat_mask])
            all_kcat_true.append(kcat_true[kcat_mask])

            if outputs.get("gate_profile") is not None:
                gate_mean = outputs["gate_profile"].mean(dim=0).cpu()
                all_gates.append(gate_mean.numpy())
                all_is_neg.append(batch_gpu.get("is_negative", torch.zeros(len(gate_mean), dtype=torch.bool)).cpu().numpy())

    metrics = {}
    for name, pred_list, true_list in [
        ("pkd", all_pkd_pred, all_pkd_true),
        ("kcat", all_kcat_pred, all_kcat_true),
    ]:
        pred = torch.cat(pred_list).numpy()
        true = torch.cat(true_list).numpy()
        if len(pred) == 0:
            metrics[name] = {"N": 0}
            continue

        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        r2 = 1 - mse / np.var(true)
        pr, _ = pearsonr(pred, true) if np.std(pred) > 0 and np.std(true) > 0 else (float('nan'), 1.0)
        sr, _ = spearmanr(pred, true) if np.std(pred) > 0 and np.std(true) > 0 else (float('nan'), 1.0)

        metrics[name] = {
            "N": int(len(pred)),
            "MSE": float(mse),
            "MAE": float(mae),
            "R2": float(r2),
            "Pearson_r": float(pr),
            "Spearman_rho": float(sr),
        }

    if all_gates:
        gates = np.concatenate(all_gates)
        is_neg = np.concatenate(all_is_neg)
        pos_gates = gates[~is_neg]
        neg_gates = gates[is_neg]
        metrics["gate"] = {
            "pos_mean": float(pos_gates.mean()) if len(pos_gates) > 0 else float('nan'),
            "neg_mean": float(neg_gates.mean()) if len(neg_gates) > 0 else float('nan'),
            "pos_median": float(np.median(pos_gates)) if len(pos_gates) > 0 else float('nan'),
            "neg_median": float(np.median(neg_gates)) if len(neg_gates) > 0 else float('nan'),
            "pos_std": float(pos_gates.std()) if len(pos_gates) > 0 else float('nan'),
            "neg_std": float(neg_gates.std()) if len(neg_gates) > 0 else float('nan'),
            "gate_diff": float(pos_gates.mean() - neg_gates.mean()) if len(pos_gates) > 0 and len(neg_gates) > 0 else float('nan'),
        }

    return metrics


def create_datasets(max_samples=5000, batch_size=64):
    """Create train/val/test datasets."""
    train_ds = OxidoreductaseDataset(
        UNIFIED_METADATA, PROTEINS_H5, LIGAND_DIR,
        split="train", max_samples=max_samples,
    )
    val_ds = OxidoreductaseDataset(
        UNIFIED_METADATA, PROTEINS_H5, LIGAND_DIR,
        split="val", max_samples=max(1, max_samples // 4),
    )
    test_ds = OxidoreductaseDataset(
        UNIFIED_METADATA, PROTEINS_H5, LIGAND_DIR,
        split="test", max_samples=max(1, max_samples // 4),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    return train_loader, val_loader, test_loader


def create_model(use_gate=True, gate_weight=0.02):
    """Create Trenzition with or without gate, load best.ckpt weights."""
    model = Trenzition(hidden_dim=256, use_gate=use_gate)

    # 加载 best.ckpt 权重（严格匹配，gate 参数可能缺失）
    ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}

    # 如果模型没有 gate 但 checkpoint 有，跳过 gate 参数
    if not use_gate:
        state = {k: v for k, v in state.items() if "gate" not in k}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.info(f"  Missing keys: {missing}")
    if unexpected:
        log.info(f"  Unexpected keys: {unexpected}")

    return model, gate_weight


def train_epoch(model, loader, optimizer, device, gate_weight, desc="Train"):
    """One epoch of training."""
    model.train()
    epoch_losses = {}
    pbar = tqdm(loader, desc=desc)

    for batch in pbar:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(device)

        optimizer.zero_grad()

        outputs = model(
            batch_gpu["ligand_data"],
            batch_gpu["seq_embed"],
            batch_gpu["cofactor_strs"],
        )

        total_loss, losses = model.compute_loss(
            outputs,
            {
                "pkd_target": batch_gpu["pkd_target"],
                "pkd_target_mask": batch_gpu["pkd_target_mask"],
                "log_kcat_target": batch_gpu["log_kcat_target"],
                "kcat_target_mask": batch_gpu["kcat_target_mask"],
                "kcat_weights": batch_gpu["kcat_weights"],
                "quality_weight": batch_gpu["quality_weight"],
            },
        )

        # Gate 正则化 (仅 use_gate=True 时)
        if gate_weight > 0 and outputs.get("gate_profile") is not None:
            gate_profile = outputs["gate_profile"]
            gate_mean = gate_profile.mean(dim=0)
            is_neg = batch_gpu.get("is_negative", torch.zeros_like(gate_mean, dtype=torch.bool))

            if is_neg.any():
                l_gate_pos = ((1.0 - gate_mean[~is_neg]) ** 2).mean() if (~is_neg).any() else 0.0
                l_gate_neg = (gate_mean[is_neg] ** 2).mean() if is_neg.any() else 0.0
                l_gate = gate_weight * (l_gate_pos + l_gate_neg)
                total_loss = total_loss + l_gate
                losses["L_gate"] = l_gate.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in losses.items():
            if isinstance(v, dict):
                continue  # skip nested dicts like 'weights'
            epoch_losses.setdefault(k, 0.0)
            epoch_losses[k] += v.item() if hasattr(v, 'item') else float(v)

        pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

    return {k: v / len(loader) for k, v in epoch_losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Gate Ablation Experiment")
    parser.add_argument("--max-samples", type=int, default=5000, help="Train samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gate-weight", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    log.info(f"Device: {device}")
    log.info(f"Max samples: {args.max_samples}, epochs: {args.epochs}, lr: {args.lr}")

    # ── 数据集 ──
    train_loader, val_loader, test_loader = create_datasets(
        max_samples=args.max_samples, batch_size=args.batch_size
    )

    # 统计正负样本
    neg_count = sum(batch.get("is_negative", torch.zeros(0)).sum().item() for batch in train_loader)
    total = sum(len(batch.get("is_negative", torch.zeros(0))) for batch in train_loader)
    log.info(f"Train: {total} samples, {neg_count} negatives ({100*neg_count/total:.1f}%)")

    if neg_count == 0:
        log.warning("⚠️  No negative samples in training! Need to fix sampling. "
                    "Negatives start at index 97351 in metadata file.")
        log.warning("Creating balanced dataset with random sampling (including negatives)...")

        # Fix: load with randomized max_samples via pandas sampling
        train_ds = OxidoreductaseDataset(
            UNIFIED_METADATA, PROTEINS_H5, LIGAND_DIR,
            split="train", max_samples=None,  # load all
        )
        # Random sample including both pos and neg
        sampled = train_ds.df.sample(n=min(args.max_samples, len(train_ds.df)), random_state=42)
        train_ds.df = sampled.reset_index(drop=True)
        # Also ensure val has neg samples
        val_ds = OxidoreductaseDataset(
            UNIFIED_METADATA, PROTEINS_H5, LIGAND_DIR,
            split="val", max_samples=None,
        )
        sampled_val = val_ds.df.sample(n=min(max(1, args.max_samples // 4), len(val_ds.df)), random_state=42)
        val_ds.df = sampled_val.reset_index(drop=True)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        neg_count = sum(batch.get("is_negative", torch.zeros(0)).sum().item() for batch in train_loader)
        total = sum(len(batch.get("is_negative", torch.zeros(0))) for batch in train_loader)
        log.info(f"Fixed Train: {total} samples, {neg_count} negatives ({100*neg_count/total:.1f}%)")

    # ──────────────────────────────────────────────────────────
    # Ablation A: With Gate
    # ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Ablation A: With Gate (use_gate=True, gate_weight={})".format(args.gate_weight))
    log.info("=" * 60)

    model_gate, gw = create_model(use_gate=True, gate_weight=args.gate_weight)
    model_gate = model_gate.to(device)
    optimizer = create_trenzition_optimizer(model_gate, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model_gate, train_loader, optimizer, device, gw,
                            desc=f"Gate Epoch {epoch}/{args.epochs}")

    log.info("Evaluating Gate model on test set...")
    test_metrics_gate = evaluate(model_gate, test_loader, device)
    log.info(f"  pKd: R²={test_metrics_gate['pkd']['R2']:.4f}, ρ={test_metrics_gate['pkd']['Spearman_rho']:.4f}")
    log.info(f"  kcat: R²={test_metrics_gate['kcat']['R2']:.4f}, ρ={test_metrics_gate['kcat']['Spearman_rho']:.4f}")
    if "gate" in test_metrics_gate:
        g = test_metrics_gate["gate"]
        log.info(f"  Gate: pos_mean={g['pos_mean']:.4f}, neg_mean={g['neg_mean']:.4f}, diff={g['gate_diff']:.4f}")

    # ──────────────────────────────────────────────────────────
    # Ablation B: Without Gate
    # ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Ablation B: Without Gate (use_gate=False, gate_weight=0)")
    log.info("=" * 60)

    model_no_gate, _ = create_model(use_gate=False, gate_weight=0)
    model_no_gate = model_no_gate.to(device)
    optimizer2 = create_trenzition_optimizer(model_no_gate, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model_no_gate, train_loader, optimizer2, device, 0,
                            desc=f"NoGate Epoch {epoch}/{args.epochs}")

    log.info("Evaluating No-Gate model on test set...")
    test_metrics_no_gate = evaluate(model_no_gate, test_loader, device)
    log.info(f"  pKd: R²={test_metrics_no_gate['pkd']['R2']:.4f}, ρ={test_metrics_no_gate['pkd']['Spearman_rho']:.4f}")
    log.info(f"  kcat: R²={test_metrics_no_gate['kcat']['R2']:.4f}, ρ={test_metrics_no_gate['kcat']['Spearman_rho']:.4f}")

    # ──────────────────────────────────────────────────────────
    # 对比总结
    # ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Gate 消融实验 — 对比总结")
    log.info("=" * 60)

    comparison = {
        "config": {
            "max_samples": args.max_samples,
            "epochs": args.epochs,
            "lr": args.lr,
            "gate_weight": args.gate_weight,
            "note": "Finetuned from best.ckpt on metadata_with_negatives.parquet",
        },
        "With_Gate": test_metrics_gate,
        "Without_Gate": test_metrics_no_gate,
    }

    # 计算差异
    for key in ["pkd", "kcat"]:
        w = test_metrics_gate[key]
        wo = test_metrics_no_gate[key]
        if "R2" in w and "R2" in wo:
            delta_r2 = w["R2"] - wo["R2"]
            delta_rho = w["Spearman_rho"] - wo["Spearman_rho"]
            log.info(f"  {key}: ΔR²={delta_r2:+.4f}, Δρ={delta_rho:+.4f}")

    # 保存
    output_path = CHECKPOINT_DIR / "gate_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    log.info(f"\nResults saved → {output_path}")

    # 打印最终 gate 分布
    if "gate" in test_metrics_gate:
        g = test_metrics_gate["gate"]
        log.info(f"\nFinal Gate distribution (With_Gate model, test set):")
        log.info(f"  Positive: mean={g['pos_mean']:.4f}, median={g['pos_median']:.4f}, std={g['pos_std']:.4f}")
        log.info(f"  Negative: mean={g['neg_mean']:.4f}, median={g['neg_median']:.4f}, std={g['neg_std']:.4f}")
        log.info(f"  Separation (diff): {g['gate_diff']:.4f}")


if __name__ == "__main__":
    main()
