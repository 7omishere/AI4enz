#!/usr/bin/env python3
"""
负样本生成：为 Trenzition 训练构造"酶-底物不匹配"样本。

三种策略：
  1. cross_ec:   酶和底物来自不同 EC 大类（1.x vs 2.x vs ...） → 易负样本
  2. random:     随机打乱配体分配 → 多数是负样本（有小概率是巧合正样本）
  3. hard:       同 EC 大类内交叉配对 → 难负样本（底物结构相似但不对应）

标签赋值：
  - pKd = 0.0        (无结合)
  - log_kcat = -7.0  (无催化，当前数据下限)
  - quality_weight = 0.25 (合成数据，低权重)
  - measurement_type = 'negative'
  - has_kcat = True

输出：在每个 split 内独立生成负样本，附加到原数据后。

用法：
  python generate_negatives.py --ratio 1.0 --strategy cross_ec
  python generate_negatives.py --ratio 0.5 --strategy random
"""

import argparse, logging, random
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR.parent / "processed"
METADATA = PROCESSED_DIR / "metadata.parquet"


def generate_cross_ec_negatives(df, n_negatives):
    """
    跨 EC 大类负采样：酶和底物必须来自不同 EC 大类。

    方法：
    1. 按 EC 大类分组
    2. 对于每个正样本 (EC_A, S_A)，从 EC_B (≠ EC_A) 中随机选一个配体
    3. 配体替换，保留原酶和辅因子
    """
    # 提取 EC 大类 (1-7)
    df = df.copy()
    df['ec_class'] = df['ec_numbers'].apply(lambda x: str(x)[:1] if isinstance(x, str) and x else '?')

    ec_groups = df.groupby('ec_class')
    ec_ligands = {}
    for ec, grp in ec_groups:
        ligands = grp['ligand_inchikey'].dropna().unique().tolist()
        if len(ligands) > 0:
            ec_ligands[ec] = ligands

    ec_classes = sorted(ec_ligands.keys())
    log.info(f"  EC 大类: {ec_classes}, 配体数: {[len(ec_ligands.get(e,[])) for e in ec_classes]}")

    negatives = []
    n_generated = 0
    rng = random.Random(42)

    for _, row in df.iterrows():
        if n_generated >= n_negatives:
            break
        ec_src = row['ec_class']
        # 选一个不同的 EC 大类
        other_ecs = [e for e in ec_classes if e != ec_src and len(ec_ligands.get(e,[])) > 0]
        if not other_ecs:
            continue
        target_ec = rng.choice(other_ecs)
        neg_ligand = rng.choice(ec_ligands[target_ec])

        neg = row.to_dict()
        neg['ligand_inchikey'] = neg_ligand
        neg['pkd_raw'] = 0.0
        neg['pkd_aligned'] = 0.0
        neg['log_kcat_median'] = -7.0
        neg['has_kcat'] = True
        neg['quality_weight'] = 0.25
        neg['measurement_type'] = 'negative'
        neg['kcat_source'] = 'negative_synthetic'
        neg['sample_id'] = f"neg_cross_ec_{n_generated}"
        neg['is_negative'] = True

        negatives.append(neg)
        n_generated += 1

    return pd.DataFrame(negatives)


def generate_random_negatives(df, n_negatives):
    """
    随机负采样：随机打乱配体分配。

    确保新配体 ≠ 原配体（避免正样本）。
    不保证新配体和原酶完全不反应（有小概率巧合）。
    """
    ligands = df['ligand_inchikey'].dropna().unique().tolist()
    rng = random.Random(42)

    negatives = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= n_negatives:
            break
        # 随机选 ≠ 原配体的配体
        candidates = [l for l in ligands if l != row['ligand_inchikey']]
        neg_ligand = rng.choice(candidates)

        neg = row.to_dict()
        neg['ligand_inchikey'] = neg_ligand
        neg['pkd_raw'] = 0.0
        neg['pkd_aligned'] = 0.0
        neg['log_kcat_median'] = -7.0
        neg['has_kcat'] = True
        neg['quality_weight'] = 0.25
        neg['measurement_type'] = 'negative'
        neg['kcat_source'] = 'negative_synthetic'
        neg['sample_id'] = f"neg_random_{i}"
        neg['is_negative'] = True

        negatives.append(neg)

    return pd.DataFrame(negatives)


def generate_hard_negatives(df, n_negatives):
    """
    难负样本：同 EC 大类内交叉配对。

    酶和底物属同一 EC 大类但不同反应。
    更难区分，逼模型学习精细的底物特异性。
    """
    df = df.copy()
    df['ec_class'] = df['ec_numbers'].apply(lambda x: str(x)[:1] if isinstance(x, str) and x else '?')

    ec_groups = df.groupby('ec_class')

    negatives = []
    n_generated = 0
    rng = random.Random(42)

    for ec, grp in ec_groups.items():
        if len(grp) < 2:
            continue
        ligands = grp['ligand_inchikey'].dropna().unique().tolist()
        if len(ligands) < 2:
            continue
        for _, row in grp.iterrows():
            if n_generated >= n_negatives:
                break
            candidates = [l for l in ligands if l != row['ligand_inchikey']]
            if not candidates:
                continue
            neg_ligand = rng.choice(candidates)

            neg = row.to_dict()
            neg['ligand_inchikey'] = neg_ligand
            neg['pkd_raw'] = 0.0
            neg['pkd_aligned'] = 0.0
            neg['log_kcat_median'] = -7.0
            neg['has_kcat'] = True
            neg['quality_weight'] = 0.25
            neg['measurement_type'] = 'negative'
            neg['kcat_source'] = 'negative_synthetic'
            neg['sample_id'] = f"neg_hard_{n_generated}"
            neg['is_negative'] = True

            negatives.append(neg)
            n_generated += 1
        if n_generated >= n_negatives:
            break

    return pd.DataFrame(negatives)


def validate_negatives(df, neg_df, split_name):
    """检查负样本质量"""
    # 负样本的配体不应该出现在正样本的 (protein, ligand) 对中
    pos_pairs = set(zip(df['protein_seq_hash'], df['ligand_inchikey']))
    neg_pairs = set(zip(neg_df['protein_seq_hash'], neg_df['ligand_inchikey']))
    collisions = neg_pairs & pos_pairs

    log.info(f"  [{split_name}] 正样本: {len(df):,}, 负样本: {len(neg_df):,}")
    log.info(f"  [{split_name}] 误碰正样本: {len(collisions)}/{len(neg_df)} ({len(collisions)/max(len(neg_df),1)*100:.2f}%)")

    if len(collisions) > 0:
        log.warning(f"  [{split_name}] 移除 {len(collisions)} 个碰撞")
        neg_df = neg_df[~neg_df.apply(
            lambda r: (r['protein_seq_hash'], r['ligand_inchikey']) in collisions, axis=1)]

    return neg_df


def main():
    parser = argparse.ArgumentParser(description="生成酶-底物负样本")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="负:正比例 (1.0 = 等量负样本)")
    parser.add_argument("--strategy", choices=['cross_ec', 'random', 'hard', 'all'],
                        default='cross_ec', help="负采样策略")
    parser.add_argument("--output", default="metadata_with_negatives.parquet")
    parser.add_argument("--no-val-negatives", action="store_true",
                        help="不对 val/test 加负样本 (保持评估纯净)")
    args = parser.parse_args()

    df = pd.read_parquet(METADATA)
    log.info(f"原始数据: {len(df):,} 样本")

    strategy_fn = {
        'cross_ec': generate_cross_ec_negatives,
        'random': generate_random_negatives,
        'hard': generate_hard_negatives,
    }

    all_parts = [df.copy()]

    for split_name, split_mask in [
        ('train', df['split'] == 'train'),
        ('val', df['split'] == 'val'),
        ('test', df['split'] == 'test'),
    ]:
        split_df = df[split_mask]
        n_pos = len(split_df)

        if args.no_val_negatives and split_name != 'train':
            log.info(f"  [{split_name}] 跳过 (--no-val-negatives)")
            continue

        if args.strategy == 'all':
            # 三种策略各 1/3
            n_each = int(n_pos * args.ratio / 3)
            neg_dfs = []
            for strat_name, fn in strategy_fn.items():
                ndf = fn(split_df, n_each)
                ndf = validate_negatives(split_df, ndf, f"{split_name}/{strat_name}")
                neg_dfs.append(ndf)
            neg_df = pd.concat(neg_dfs, ignore_index=True)
        else:
            fn = strategy_fn[args.strategy]
            n_neg = int(n_pos * args.ratio)
            neg_df = fn(split_df, n_neg)
            neg_df = validate_negatives(split_df, neg_df, split_name)

        # 标记 split
        neg_df['split'] = split_name
        all_parts.append(neg_df)

    result = pd.concat(all_parts, ignore_index=True)

    # 确保 is_negative 列
    result['is_negative'] = result.get('is_negative', False).fillna(False).astype(bool)

    out_path = PROCESSED_DIR / args.output
    result.to_parquet(out_path)

    log.info(f"\n{'='*50}")
    log.info(f"输出: {out_path}")
    log.info(f"总样本: {len(result):,}")
    for split in ['train', 'val', 'test']:
        sub = result[result['split'] == split]
        n_pos = (sub['is_negative'] == False).sum()
        n_neg = sub['is_negative'].sum()
        log.info(f"  {split}: {len(sub):,} (正:{n_pos:,}  负:{n_neg:,})")


if __name__ == "__main__":
    main()
