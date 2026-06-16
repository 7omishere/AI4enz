#!/usr/bin/env python3
"""
全面模型评估：Trenzition vs 基线模型

基线模型：
- MeanPredictor: 用训练集均值预测（最弱基线）
- Linear Ridge: ESM-2 1280-dim 均值池化 → Ridge 回归
- MLP (sklearn): ESM-2 → MLPRegressor
- RandomForest: ESM-2 → RandomForestRegressor
- XGBoost: ESM-2 → XGBRegressor

评估指标：MSE, MAE, Spearman ρ, R²
"""

import sys
from pathlib import Path

# 路径设置
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import argparse
import json
import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# sklearn 基线
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 忽略 sklearn 警告
warnings.filterwarnings("ignore")

# 项目导入
from train import OxidoreductaseDataset, collate_fn
from ranking_model import Trenzition

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")

# 路径
PROCESSED_DIR = SCRIPT_DIR.parent / "processed"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"
METADATA_PATH = PROCESSED_DIR / "metadata.parquet"
LIGAND_DIR = PROCESSED_DIR / "ligands"
BEST_CKPT = CHECKPOINT_DIR / "best.ckpt"
OUTPUT_DIR = CHECKPOINT_DIR


def evaluate_trenzition(model, loader, device) -> dict:
    """用 Trenzition 模型在给定 loader 上做推理评估."""
    model.eval()
    all_preds_pkd = []
    all_targets_pkd = []
    all_preds_kcat = []
    all_targets_kcat = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Trenzition eval", leave=False):
            ligand_data = batch["ligand_data"].to(device)
            seq_embed = batch["seq_embed"].to(device)
            cofactor_strs = batch["cofactor_strs"]

            outputs = model(ligand_data, seq_embed, cofactor_strs)

            # pKd predictions (normalized [0,1])
            pkd_pred = outputs["ts_stability"].cpu().numpy().flatten()
            pkd_target = batch["pkd_target"].cpu().numpy().flatten()
            pkd_mask = batch["pkd_target_mask"].cpu().numpy().flatten().astype(bool)

            all_preds_pkd.extend(pkd_pred[pkd_mask].tolist())
            all_targets_pkd.extend(pkd_target[pkd_mask].tolist())

            # kcat predictions (normalized [0,1])
            kcat_pred = outputs["catalysis_rate"].cpu().numpy().flatten()
            kcat_target = batch["log_kcat_target"].cpu().numpy().flatten()
            kcat_mask = batch["kcat_target_mask"].cpu().numpy().flatten().astype(bool)

            all_preds_kcat.extend(kcat_pred[kcat_mask].tolist())
            all_targets_kcat.extend(kcat_target[kcat_mask].tolist())

    # Denormalize: pKd ∈ [0,12], kcat ∈ [-7,8]
    pkd_preds_raw = np.array(all_preds_pkd) * 12.0
    pkd_targets_raw = np.array(all_targets_pkd) * 12.0
    kcat_preds_raw = np.array(all_preds_kcat) * 15.0 - 7.0
    kcat_targets_raw = np.array(all_targets_kcat) * 15.0 - 7.0

    return {
        "pkd": {
            "preds": pkd_preds_raw,
            "targets": pkd_targets_raw,
        },
        "kcat": {
            "preds": kcat_preds_raw,
            "targets": kcat_targets_raw,
        },
    }


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """计算回归指标."""
    from scipy.stats import spearmanr

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else np.nan
    rho, _ = spearmanr(preds, targets) if len(preds) > 1 else (np.nan, 0)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"MSE": mse, "MAE": mae, "Pearson_r": corr, "Spearman_rho": rho, "R2": r2}


def extract_esm2_features(dataset, max_samples: int = None) -> tuple:
    """从数据集中提取 ESM-2 嵌入均值 + 原始标签值."""
    features_pkd = []
    targets_pkd = []
    features_kcat = []
    targets_kcat = []

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    np.random.seed(42)
    indices = np.random.choice(len(dataset), n, replace=False) if max_samples else np.arange(n)

    log.info(f"Extracting ESM-2 features from {n} samples...")

    for idx in tqdm(indices, desc="Extract features", leave=False):
        sample = dataset[idx]
        seq_embed = sample["seq_embed"].numpy()  # (1280,) ESM-2 or (6,) AA-props

        # Pad AA-property samples to 1280-dim for uniform feature space
        if seq_embed.shape[0] < 1280:
            padded = np.zeros(1280, dtype=np.float32)
            padded[:seq_embed.shape[0]] = seq_embed
            seq_embed = padded

        # pKd — use raw values (not normalized)
        if sample["has_pkd"]:
            features_pkd.append(seq_embed)
            targets_pkd.append(sample["pkd_raw"].item())

        # kcat — use raw values (not normalized)
        if sample["has_kcat"]:
            features_kcat.append(seq_embed)
            targets_kcat.append(sample["log_kcat_raw"].item())

    return (
        np.array(features_pkd), np.array(targets_pkd),
        np.array(features_kcat), np.array(targets_kcat),
    )


def train_sklearn_baselines(X_train, y_train, X_test, y_test, task_name: str) -> dict:
    """训练多种 sklearn 基线模型并评估."""
    results = {}

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Mean baseline
    y_mean = np.mean(y_train)
    mean_preds = np.full_like(y_test, y_mean)
    results["Mean"] = compute_metrics(mean_preds, y_test)

    # 2. Ridge
    log.info(f"  [{task_name}] Training Ridge...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    results["Ridge"] = compute_metrics(ridge.predict(X_test_scaled), y_test)

    # 3. MLP
    log.info(f"  [{task_name}] Training MLP (sklearn)...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    mlp.fit(X_train_scaled, y_train)
    results["MLP"] = compute_metrics(mlp.predict(X_test_scaled), y_test)

    # 4. Random Forest
    log.info(f"  [{task_name}] Training RandomForest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train_scaled, y_train)
    results["RandomForest"] = compute_metrics(rf.predict(X_test_scaled), y_test)

    # 5. XGBoost
    try:
        from xgboost import XGBRegressor
        log.info(f"  [{task_name}] Training XGBoost...")
        xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                           random_state=42, verbosity=0)
        xgb.fit(X_train_scaled, y_train)
        results["XGBoost"] = compute_metrics(xgb.predict(X_test_scaled), y_test)
    except ImportError:
        log.warning("  XGBoost not available, skipping")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=30000,
                       help="Max training samples for sklearn baselines (memory limit)")
    parser.add_argument("--no-trenzition", action="store_true", help="Skip Trenzition eval")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info(f"Device: {device}")
    log.info(f"Metadata: {METADATA_PATH}")

    # ── 1. 加载数据 ──
    log.info("Loading datasets...")
    train_ds = OxidoreductaseDataset(
        METADATA_PATH, PROTEINS_H5, LIGAND_DIR, split="train"
    )
    test_ds = OxidoreductaseDataset(
        METADATA_PATH, PROTEINS_H5, LIGAND_DIR, split="test"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    log.info(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # ── 2. 评估 Trenzition ──
    trenzition_results = {}
    if not args.no_trenzition and BEST_CKPT.exists():
        log.info(f"Loading Trenzition from {BEST_CKPT}...")
        model = Trenzition(hidden_dim=256, gnn_layers=3, n_ode_steps=5,
                          use_classification=False).to(device)
        ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=False)
        log.info(f"  Checkpoint best val: {ckpt.get('best_val_loss', 'N/A'):.6f}")
        log.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

        log.info("Evaluating Trenzition on test set...")
        trenzition_raw = evaluate_trenzition(model, test_loader, device)

        for task in ["pkd", "kcat"]:
            preds = trenzition_raw[task]["preds"]
            targets = trenzition_raw[task]["targets"]
            trenzition_results[task] = compute_metrics(preds, targets)
            log.info(f"  Trenzition {task}: MSE={trenzition_results[task]['MSE']:.4f}, "
                     f"MAE={trenzition_results[task]['MAE']:.4f}, "
                     f"R²={trenzition_results[task]['R2']:.4f}, "
                     f"ρ={trenzition_results[task]['Spearman_rho']:.4f}")
    else:
        log.warning("No best.ckpt found or --no-trenzition set, skipping Trenzition eval")

    # ── 3. 训练基线模型 ──
    log.info(f"Extracting ESM-2 features for baselines (max {args.max_train} train samples)...")
    (X_train_pkd, y_train_pkd, X_train_kcat, y_train_kcat) = \
        extract_esm2_features(train_ds, max_samples=args.max_train)

    log.info("Extracting ESM-2 features for test set...")
    (X_test_pkd, y_test_pkd, X_test_kcat, y_test_kcat) = \
        extract_esm2_features(test_ds, max_samples=None)

    log.info(f"pKd: train {X_train_pkd.shape}, test {X_test_pkd.shape}")
    log.info(f"kcat: train {X_train_kcat.shape}, test {X_test_kcat.shape}")

    baseline_results = {}

    log.info("\n" + "="*60)
    log.info("Training baselines for pKd prediction")
    log.info("="*60)
    baseline_results["pkd"] = train_sklearn_baselines(
        X_train_pkd, y_train_pkd, X_test_pkd, y_test_pkd, "pKd"
    )

    log.info("\n" + "="*60)
    log.info("Training baselines for kcat prediction")
    log.info("="*60)
    baseline_results["kcat"] = train_sklearn_baselines(
        X_train_kcat, y_train_kcat, X_test_kcat, y_test_kcat, "kcat"
    )

    # ── 4. 汇总 ──
    all_results = {
        "Trenzition": trenzition_results,
        "Baselines": baseline_results,
    }

    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved → {results_path}")

    # ── 5. 打印汇总表 ──
    for task in ["pkd", "kcat"]:
        print(f"\n{'='*70}")
        print(f"  {task.upper()} — Model Comparison")
        print(f"{'='*70}")
        print(f"{'Model':<16} {'MSE':>10} {'MAE':>10} {'Pearson r':>10} {'Spearman ρ':>10} {'R²':>10}")
        print("-" * 70)

        all_models = {}
        if task in trenzition_results:
            all_models["Trenzition"] = trenzition_results[task]
        for name, metrics in baseline_results.get(task, {}).items():
            all_models[name] = metrics

        # 按 R² 排序
        sorted_models = sorted(all_models.items(), key=lambda x: x[1].get("R2", -999), reverse=True)
        for name, m in sorted_models:
            print(f"{name:<16} {m['MSE']:>10.4f} {m['MAE']:>10.4f} "
                  f"{m['Pearson_r']:>10.4f} {m['Spearman_rho']:>10.4f} {m['R2']:>10.4f}")


if __name__ == "__main__":
    main()
