#!/usr/bin/env python3
"""
增强版 Benchmark：Trenzition vs 公平基线 + 排序指标

新增基线（与 Trenzition 公平对比，都用蛋白+配体信息）：
- MorganFP (2048-bit) + ESM-2 concat → XGBoost
- MorganFP (2048-bit) + ESM-2 concat → MLP (sklearn)
- MorganFP only → XGBoost (配体-only 基线)

新增指标：
- Spearman ρ, Pearson r（已有）
- NDCG@K, MRR, Top-K Recall（排序指标 — 酶挖掘场景核心）

运行: python benchmark_enhanced.py --max-train 30000 --device cpu
"""

import sys, json, logging, warnings, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr, pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train import OxidoreductaseDataset, collate_fn
from ranking_model import Trenzition

BASE_DIR = SCRIPT_DIR.parent
DATASET_DIR = BASE_DIR / "dataset_building"
PROCESSED_DIR = DATASET_DIR / "processed"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
BEST_CKPT = CHECKPOINT_DIR / "best.ckpt"
OUTPUT_DIR = CHECKPOINT_DIR


# ═══════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════

def regression_metrics(preds, targets):
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    r, _ = pearsonr(preds, targets) if len(preds) > 1 else (np.nan, None)
    rho, _ = spearmanr(preds, targets) if len(preds) > 1 else (np.nan, None)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"MSE": mse, "MAE": mae, "Pearson_r": r, "Spearman_rho": rho, "R2": r2}


def ranking_metrics(preds, targets, k_vals=[5, 10, 20, 50]):
    """
    排序指标：模拟「给定底物，推荐最优酶」场景。
    对每个样本单独计算无意义（只有一个 label），
    需要在 EC class 内排序：同 EC 号下，预测值最高的酶 = 最推荐。
    """
    return {}  # 在 aggregate 阶段按 EC 分组计算


def ndcg_at_k(y_true, y_score, k):
    """NDCG@K：排序质量"""
    order = np.argsort(y_score)[::-1][:k]
    y_rel = y_true[order]
    if len(y_rel) == 0:
        return 0.0
    dcg = np.sum((2 ** y_rel - 1) / np.log2(np.arange(2, len(y_rel) + 2)))
    ideal_order = np.argsort(y_true)[::-1][:k]
    y_ideal = y_true[ideal_order]
    idcg = np.sum((2 ** y_ideal - 1) / np.log2(np.arange(2, len(y_ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr(y_true, y_score):
    """MRR: 第一个 top-1 真实值的倒数排名"""
    order = np.argsort(y_score)[::-1]
    best_idx = np.argmax(y_true)
    rank = np.where(order == best_idx)[0][0] + 1
    return 1.0 / rank


def topk_recall(y_true, y_score, k):
    """Top-K Recall：前 K 个预测中是否包含真实最优"""
    order = np.argsort(y_score)[::-1][:k]
    best_idx = np.argmax(y_true)
    return 1.0 if best_idx in order else 0.0


# ═══════════════════════════════════════════════════════════════
# 特征提取
# ═══════════════════════════════════════════════════════════════

def load_inchikey_smiles():
    """加载 inchikey → SMILES 映射"""
    import pickle
    path = PROCESSED_DIR / "inchikey_smiles_map.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_morgan_fp(smiles, radius=2, nbits=2048):
    """计算 Morgan 指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def extract_features(dataset, inchikey_smiles, max_samples=None, use_morgan=True):
    """
    提取 ESM-2 + MorganFP 特征 + 标签。
    返回 (X_pkd, y_pkd, X_kcat, y_kcat)
    """
    X_pkd, y_pkd = [], []
    X_kcat, y_kcat = [], []

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), n, replace=False) if max_samples else np.arange(n)

    inchikeys = dataset.df['ligand_inchikey'].values

    for idx in tqdm(indices, desc="Extract features"):
        sample = dataset[idx]
        seq_embed = sample["seq_embed"].numpy()

        # Pad AA-property samples
        if seq_embed.shape[0] < 1280:
            padded = np.zeros(1280, dtype=np.float32)
            padded[:seq_embed.shape[0]] = seq_embed
            seq_embed = padded

        # Morgan FP
        inchikey = inchikeys[idx]
        smiles = inchikey_smiles.get(inchikey, "")
        morgan = compute_morgan_fp(smiles) if smiles else np.zeros(2048, dtype=np.float32)

        combined = np.concatenate([seq_embed, morgan]).astype(np.float32)

        if sample["has_pkd"]:
            X_pkd.append(combined)
            y_pkd.append(sample["pkd_raw"].item())
        if sample["has_kcat"]:
            X_kcat.append(combined)
            y_kcat.append(sample["log_kcat_raw"].item())

    return (
        np.array(X_pkd), np.array(y_pkd, dtype=np.float32),
        np.array(X_kcat), np.array(y_kcat, dtype=np.float32),
    )


# ═══════════════════════════════════════════════════════════════
# 模型评估
# ═══════════════════════════════════════════════════════════════

def evaluate_trenzition(model, loader, device):
    model.eval()
    all_preds_pkd, all_targets_pkd = [], []
    all_preds_kcat, all_targets_kcat = [], []
    all_preds_km, all_targets_km = [], []
    all_inchikeys = []
    all_ecs = []

    pkd_min, pkd_max = 0.0, 12.0
    kcat_min, kcat_max = -7.0, 8.0
    km_min, km_max = -13.0, 3.0

    for batch in tqdm(loader, desc="Trenzition eval", leave=False):
        with torch.no_grad():
            outputs = model(
                batch["ligand_data"].to(device),
                batch["seq_embed"].to(device),
                batch["cofactor_strs"],
                measurement_types=batch.get("measurement_type", None).to(device)
                if "measurement_type" in batch else None,
                temperature_K=batch.get("temperature_K", None).to(device)
                if "temperature_K" in batch else None,
            )

        # pKd — 三头模型输出归一化 kd_pred [0,1]
        pkd_pred_norm = outputs["kd_pred"].cpu().numpy().flatten()
        pkd_target_norm = batch["pkd_target"].cpu().numpy().flatten()
        pkd_mask = batch["pkd_target_mask"].cpu().numpy().flatten().astype(bool)
        all_preds_pkd.extend((pkd_pred_norm[pkd_mask] * (pkd_max - pkd_min) + pkd_min).tolist())
        all_targets_pkd.extend((pkd_target_norm[pkd_mask] * (pkd_max - pkd_min) + pkd_min).tolist())

        # kcat — 三头模型输出归一化 kcat_pred [0,1]
        kcat_pred_norm = outputs["kcat_pred"].cpu().numpy().flatten()
        kcat_target_norm = batch["log_kcat_target"].cpu().numpy().flatten()
        kcat_mask = batch["kcat_target_mask"].cpu().numpy().flatten().astype(bool)
        all_preds_kcat.extend((kcat_pred_norm[kcat_mask] * (kcat_max - kcat_min) + kcat_min).tolist())
        all_targets_kcat.extend((kcat_target_norm[kcat_mask] * (kcat_max - kcat_min) + kcat_min).tolist())

        # Km — 三头模型输出归一化 log_km_pred [0,1]
        km_pred_norm = outputs["log_km_pred"].cpu().numpy().flatten()
        km_mask = batch.get("km_target_mask", torch.zeros_like(batch.get("pkd_target_mask", torch.empty(0)))).cpu().numpy().flatten().astype(bool)
        if km_mask.any():
            km_target_raw = batch["log_km_target"].cpu().numpy().flatten()
            all_preds_km.extend((km_pred_norm[km_mask] * (km_max - km_min) + km_min).tolist())
            all_targets_km.extend(km_target_raw[km_mask].tolist())

    results = {
        "pkd": {
            "preds": np.array(all_preds_pkd),
            "targets": np.array(all_targets_pkd),
        },
        "kcat": {
            "preds": np.array(all_preds_kcat),
            "targets": np.array(all_targets_kcat),
        },
    }
    if all_preds_km:
        results["km"] = {
            "preds": np.array(all_preds_km),
            "targets": np.array(all_targets_km),
        }
    return results


def train_baselines(X_train, y_train, X_test, y_test, task_name):
    """训练多个基线。返回 (results_dict, predictions_dict)"""
    results = {}
    preds = {}
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xe = scaler.transform(X_test)
    esm_dim = 1280

    # Mean
    ym = np.mean(y_train)
    p = np.full_like(y_test, ym)
    results["Mean"] = regression_metrics(p, y_test)
    preds["Mean"] = p

    # Ridge (ESM-2 only)
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xt[:, :esm_dim], y_train)
    p = ridge.predict(Xe[:, :esm_dim])
    results["Ridge_ESM2only"] = regression_metrics(p, y_test)
    preds["Ridge_ESM2only"] = p

    # Morgan FP only → XGBoost
    log.info(f"  [{task_name}] MorganFP → XGBoost...")
    xgb_morgan = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, verbosity=0, n_jobs=-1)
    xgb_morgan.fit(Xt[:, esm_dim:], y_train)
    p = xgb_morgan.predict(Xe[:, esm_dim:])
    results["XGB_MorganOnly"] = regression_metrics(p, y_test)
    preds["XGB_MorganOnly"] = p

    # ESM-2 + MorganFP → XGBoost
    log.info(f"  [{task_name}] ESM-2 + MorganFP → XGBoost...")
    xgb_full = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                            random_state=42, verbosity=0, n_jobs=-1)
    xgb_full.fit(Xt, y_train)
    p = xgb_full.predict(Xe)
    results["XGB_ESM2+Morgan"] = regression_metrics(p, y_test)
    preds["XGB_ESM2+Morgan"] = p

    # ESM-2 + MorganFP → MLP
    log.info(f"  [{task_name}] ESM-2 + MorganFP → MLP...")
    mlp = MLPRegressor(hidden_layer_sizes=(512, 256, 128), activation='relu',
                       max_iter=200, early_stopping=True, validation_fraction=0.1,
                       random_state=42)
    mlp.fit(Xt, y_train)
    p = mlp.predict(Xe)
    results["MLP_ESM2+Morgan"] = regression_metrics(p, y_test)
    preds["MLP_ESM2+Morgan"] = p

    return results, preds


def compute_ec_ranking_metrics(df, preds, targets, task, k_vals=[5, 10, 20]):
    """
    按 EC 号分组计算排序指标。
    模拟场景：给定一个底物，在同 EC 号下找最优酶。
    """
    ec_list = df['ec_numbers'].values
    ec_groups = defaultdict(list)
    for i, ec in enumerate(ec_list):
        if isinstance(ec, str) and ec:
            ec_groups[ec].append(i)

    metrics = {f"NDCG@{k}": [] for k in k_vals}
    metrics["MRR"] = []
    metrics.update({f"Top{k}_Recall": [] for k in k_vals})

    for ec, indices in ec_groups.items():
        if len(indices) < 3:  # 需要至少 3 个样本
            continue
        idx_arr = np.array(indices)
        g_preds = preds[idx_arr]
        g_targets = targets[idx_arr]

        for k in k_vals:
            if len(g_preds) >= k:
                metrics[f"NDCG@{k}"].append(ndcg_at_k(g_targets, g_preds, k))
                metrics[f"Top{k}_Recall"].append(topk_recall(g_targets, g_preds, k))
        metrics["MRR"].append(mrr(g_targets, g_preds))

    return {k: float(np.mean(v)) if v else np.nan for k, v in metrics.items()}


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train", type=int, default=30000)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── 1. 加载数据 ──
    log.info("Loading data...")
    train_ds = OxidoreductaseDataset(
        str(PROCESSED_DIR / 'metadata.parquet'),
        str(PROCESSED_DIR / 'proteins.h5'),
        str(PROCESSED_DIR / 'ligands'),
        split='train',
    )
    test_ds = OxidoreductaseDataset(
        str(PROCESSED_DIR / 'metadata.parquet'),
        str(PROCESSED_DIR / 'proteins.h5'),
        str(PROCESSED_DIR / 'ligands'),
        split='test',
    )

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    inchikey_smiles = load_inchikey_smiles()

    # ── 2. Trenzition (三头模型) ──
    log.info("Evaluating Trenzition (three-head)...")
    model = Trenzition(hidden_dim=256, gnn_layers=3,
                       three_head=True, kcat_ode_steps=10).to(device)
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    trenz_raw = evaluate_trenzition(model, test_loader, device)
    trenz_results = {}
    for task in ["pkd", "kcat"]:
        if task in trenz_raw and len(trenz_raw[task]["preds"]) > 0:
            trenz_results[task] = regression_metrics(
                trenz_raw[task]["preds"], trenz_raw[task]["targets"]
            )
    if "km" in trenz_raw and len(trenz_raw["km"]["preds"]) > 0:
        trenz_results["km"] = regression_metrics(
            trenz_raw["km"]["preds"], trenz_raw["km"]["targets"]
        )

    # ── 3. 基线 ──
    log.info("Extracting features for baselines...")
    X_train_pkd, y_train_pkd, X_train_kcat, y_train_kcat = extract_features(
        train_ds, inchikey_smiles, max_samples=args.max_train)
    X_test_pkd, y_test_pkd, X_test_kcat, y_test_kcat = extract_features(
        test_ds, inchikey_smiles, max_samples=None)

    log.info(f"pKd: train {X_train_pkd.shape}, test {X_test_pkd.shape}")
    log.info(f"kcat: train {X_train_kcat.shape}, test {X_test_kcat.shape}")

    baseline_results = {}
    baseline_preds = {}
    for task, (Xt, yt, Xe, ye) in [
        ("pkd", (X_train_pkd, y_train_pkd, X_test_pkd, y_test_pkd)),
        ("kcat", (X_train_kcat, y_train_kcat, X_test_kcat, y_test_kcat)),
    ]:
        log.info(f"\n{'='*50}\n  Training baselines: {task}\n{'='*50}")
        baseline_results[task], baseline_preds[task] = train_baselines(Xt, yt, Xe, ye, task)

    # ── 4. 排序指标 ──
    log.info("\nComputing ranking metrics (EC-level)...")
    test_df = test_ds.df
    pkd_mask_all = test_df['pkd_raw'].notna() | test_df['pkd_aligned'].notna()
    kcat_mask_all = test_df['has_kcat'].astype(bool)
    km_mask_all = test_df.get('has_km', pd.Series(False, index=test_df.index)).astype(bool)

    ranking_all = {}
    # Trenzition
    for task, mask in [("pkd", pkd_mask_all.values), ("kcat", kcat_mask_all.values),
                       ("km", km_mask_all.values)]:
        if task not in trenz_raw:
            continue
        sub_df = test_df[mask].reset_index(drop=True)
        rm = compute_ec_ranking_metrics(
            sub_df, trenz_raw[task]["preds"], trenz_raw[task]["targets"], task)
        ranking_all[f"Trenzition_{task}"] = rm

    # XGBoost baseline
    for task, mask in [("pkd", pkd_mask_all.values), ("kcat", kcat_mask_all.values)]:
        sub_df = test_df[mask].reset_index(drop=True)
        preds_arr = baseline_preds[task]["XGB_ESM2+Morgan"]
        targets_arr = y_test_pkd if task == "pkd" else y_test_kcat
        rm = compute_ec_ranking_metrics(sub_df, preds_arr, targets_arr, task)
        ranking_all[f"XGB_ESM2+Morgan_{task}"] = rm

    # ── 5. 汇总输出 ──
    all_results = {
        "Trenzition": trenz_results,
        "Baselines": baseline_results,
        "Ranking_Metrics": ranking_all,
    }

    out_path = OUTPUT_DIR / "benchmark_enhanced_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved → {out_path}")

    # ── 打印 ──
    for task in ["pkd", "kcat", "km"]:
        if task not in trenz_results and task not in baseline_results:
            continue
        print(f"\n{'='*70}")
        print(f"  {task.upper()} — Enhanced Benchmark")
        print(f"{'='*70}")
        print(f"{'Model':<22} {'R²':>8} {'Spearman ρ':>10} {'MSE':>10} {'MAE':>10}")
        print("-" * 70)

        all_models = {}
        if task in trenz_results:
            all_models["Trenzition"] = trenz_results[task]
        if task in baseline_results:
            for name, m in baseline_results[task].items():
                all_models[name] = m

        for name in ["Trenzition", "XGB_ESM2+Morgan", "MLP_ESM2+Morgan",
                      "XGB_MorganOnly", "Ridge_ESM2only", "Mean"]:
            if name in all_models:
                m = all_models[name]
                print(f"{name:<22} {m.get('R2',0):>8.4f} {m.get('Spearman_rho',0):>10.4f} "
                      f"{m.get('MSE',0):>10.4f} {m.get('MAE',0):>10.4f}")

    # 排序指标
    print(f"\n{'='*70}")
    print(f"  Ranking Metrics (EC-level, enzyme mining scenario)")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'NDCG@10':>10} {'MRR':>10} {'Top10_Recall':>12}")
    print("-" * 70)
    for name, metrics in ranking_all.items():
        print(f"{name:<30} {metrics.get('NDCG@10',0):>10.4f} "
              f"{metrics.get('MRR',0):>10.4f} {metrics.get('Top10_Recall',0):>12.4f}")


if __name__ == "__main__":
    main()
