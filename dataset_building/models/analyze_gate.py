#!/usr/bin/env python3
"""
Gate 行为分析：从训练好的 best.ckpt 加载模型，分析 gate_profile 分布。
看 Gate 是否真正区分了正样本和负样本，以及在各类别上的表现。

用法: python analyze_gate.py
"""
import sys, json, logging, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gate_analysis")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train import OxidoreductaseDataset, collate_fn
from ranking_model import Trenzition

PROCESSED_DIR = SCRIPT_DIR.parent / "processed"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
BEST_CKPT = CHECKPOINT_DIR / "best.ckpt"
METADATA_PATH = PROCESSED_DIR / "metadata_with_negatives.parquet"
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"
LIGAND_DIR = PROCESSED_DIR / "ligands"

def main():
    log.info("=" * 60)
    log.info("Gate 行为分析")
    log.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── 加载模型 ──
    model = Trenzition(hidden_dim=256, use_gate=True)
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    log.info(f"Checkpoint loaded: {BEST_CKPT}")

    # ── 数据集 (test split) ──
    # 先看 metadata 中有多少负样本
    df_check = pd.read_parquet(METADATA_PATH)
    log.info(f"Metadata: {len(df_check)} rows, neg={df_check.get('is_negative', pd.Series([False])).sum()}")

    test_dataset = OxidoreductaseDataset(
        str(METADATA_PATH), str(PROTEINS_H5), str(LIGAND_DIR),
        split="test", max_samples=None,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    log.info(f"Test set: {len(test_dataset)} samples")

    # 也需要 train 来获取 neg samples
    train_dataset = OxidoreductaseDataset(
        str(METADATA_PATH), str(PROTEINS_H5), str(LIGAND_DIR),
        split="train", max_samples=10000,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── 收集 gate 值 ──
    def collect_gates(loader, desc="Eval") -> dict:
        all_gates = []
        all_is_neg = []
        all_has_pkd = []
        all_has_kcat = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                batch_gpu = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                batch_gpu["ligand_data"] = batch_gpu["ligand_data"].to(device)

                outputs = model(
                    batch_gpu["ligand_data"],
                    batch_gpu["seq_embed"],
                    batch_gpu["cofactor_strs"],
                )

                if outputs.get("gate_profile") is not None:
                    # gate_profile: (n_steps, B) → 平均门控值 (B,)
                    gate_mean = outputs["gate_profile"].mean(dim=0).cpu()
                    all_gates.append(gate_mean.numpy())
                    if "is_negative" in batch_gpu:
                        all_is_neg.append(batch_gpu["is_negative"].cpu().numpy())
                    else:
                        all_is_neg.append(np.zeros(len(gate_mean), dtype=bool))
                    all_has_pkd.append(batch_gpu["pkd_target_mask"].cpu().numpy())
                    all_has_kcat.append(batch_gpu["kcat_target_mask"].cpu().numpy())

        gates = np.concatenate(all_gates)
        is_neg = np.concatenate(all_is_neg)
        has_pkd = np.concatenate(all_has_pkd)
        has_kcat = np.concatenate(all_has_kcat)
        return {"gates": gates, "is_neg": is_neg, "has_pkd": has_pkd, "has_kcat": has_kcat}

    log.info("\n── Collecting gates from train set (10k samples) ──")
    train_gates = collect_gates(train_loader, "train")
    log.info(f"\n── Collecting gates from test set ──")
    test_gates = collect_gates(test_loader, "test")

    # ── 分析 ──
    def analyze(gates_dict, name: str):
        gates = gates_dict["gates"]
        is_neg = gates_dict["is_neg"]

        pos_gates = gates[~is_neg]
        neg_gates = gates[is_neg]

        print(f"\n{'='*60}")
        print(f"  {name}: Gate 分布统计")
        print(f"{'='*60}")
        print(f"  总样本:       {len(gates)}")
        print(f"  正样本:       {len(pos_gates)} (gate均值={pos_gates.mean():.4f}, 中位数={np.median(pos_gates):.4f}, "
              f"std={pos_gates.std():.4f})")
        print(f"  负样本:       {len(neg_gates)} (gate均值={neg_gates.mean():.4f}, 中位数={np.median(neg_gates):.4f}, "
              f"std={neg_gates.std():.4f})")

        # 分布分段统计
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pos_above = (pos_gates > threshold).mean() * 100 if len(pos_gates) > 0 else 0
            neg_below = (neg_gates <= threshold).mean() * 100 if len(neg_gates) > 0 else 0
            print(f"    Gate > {threshold:.1f}: 正样本 {pos_above:.1f}% | "
                  f"Gate ≤ {threshold:.1f}: 负样本 {neg_below:.1f}%")

        # Histrogram bins
        hist_pos, _ = np.histogram(pos_gates, bins=10, range=(0, 1))
        hist_neg, _ = np.histogram(neg_gates, bins=10, range=(0, 1))
        print(f"\n  正样本 Gate 直方图 (0-1, 10 bins):")
        print(f"    {hist_pos}")
        print(f"  负样本 Gate 直方图 (0-1, 10 bins):")
        print(f"    {hist_neg}")

        # 分离度指标
        if len(pos_gates) > 0 and len(neg_gates) > 0:
            from scipy.stats import ks_2samp
            ks_stat, ks_p = ks_2samp(pos_gates, neg_gates)
            print(f"\n  KS 检验: stat={ks_stat:.4f}, p={ks_p:.2e}")
            print(f"  正样本均值 - 负样本均值 = {pos_gates.mean() - neg_gates.mean():.4f}")

    analyze(train_gates, "Train (10k)")
    analyze(test_gates, "Test (all)")

    # ── Gate vs 预测误差的关系 ──
    print(f"\n{'='*60}")
    print(f"  Gate vs 预测误差分析 (仅正样本)")
    print(f"{'='*60}")

    # 只分析 test set 正样本
    test_pos_mask = ~test_gates["is_neg"]
    test_pos_gates = test_gates["gates"][test_pos_mask]
    test_pos_pkd = test_gates["has_pkd"][test_pos_mask]
    test_pos_kcat = test_gates["has_kcat"][test_pos_mask]

    # 按 gate 值分桶，看低 gate 样本的预测误差
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        thresh = np.quantile(test_pos_gates, q)
        print(f"  Gate {q*100:.0f}%分位数 = {thresh:.4f}")

    log.info("\n✓ Gate 分析完成")

if __name__ == "__main__":
    main()
