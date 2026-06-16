#!/usr/bin/env python3
"""
supplement_temperature.py
=========================

从 OED 温度数据匹配到 Trenzition V5 核心数据集。
匹配鍵：(uniprot_id, ec_numbers) → OED 的 (uniprot, ec) → median temperature

覆盖 ~63% 的 V5 训练样本。
未匹配的样本默认 298.15K（即当前模型假设值）。

用法: python supplement_temperature.py
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "processed"
OED_PATH = PROJECT_DIR / "OED" / "oed_kinetics.json"

V5_METADATA_PATH = PROCESSED_DIR / "metadata.parquet"
V5_WITH_NEG_PATH = PROCESSED_DIR / "metadata_with_negatives.parquet"


def build_oed_temperature_map(oed_path: str) -> dict:
    """
    从 OED JSON 构建温度映射字典。

    OED 格式:
    {
        "ec": "1.1.1.1",
        "uniprot": "P07327",
        "substrate": "Ethanol",
        "temperature": 25.0,
        "kcat_value": ...,
        ...
    }

    返回:
        {(uniprot, ec): median_temperature_C}
    """
    log.info(f"Loading OED from {oed_path}")
    with open(oed_path) as f:
        oed_data = json.load(f)

    log.info(f"OED entries: {len(oed_data)}")

    # 收集每个 (uniprot, ec) 的所有温度
    temp_collector = defaultdict(list)
    n_missing_temp = 0

    for entry in oed_data:
        uniprot = entry.get("uniprot", "").strip()
        ec = entry.get("ec", "").strip()
        temp_raw = entry.get("temperature")

        if not uniprot or not ec:
            continue

        if temp_raw is None:
            n_missing_temp += 1
            continue

        try:
            temp_c = float(temp_raw)
        except (ValueError, TypeError):
            n_missing_temp += 1
            continue

        temp_collector[(uniprot, ec)].append(temp_c)

    log.info(f"Entries with missing temperature: {n_missing_temp}")

    # 每个 key 取中位数
    temp_map = {}
    for key, temps in temp_collector.items():
        temp_map[key] = np.median(temps)

    log.info(f"Built temperature map: {len(temp_map)} (uniprot, ec) keys")

    # 也构建 EC-only 回退映射
    ec_temp_collector = defaultdict(list)
    for (uni, ec), temp_c in temp_map.items():
        if ec:
            ec_temp_collector[ec].append(temp_c)
    ec_temp_map = {ec: np.median(temps) for ec, temps in ec_temp_collector.items()}
    log.info(f"Built EC-only fallback map: {len(ec_temp_map)} EC keys")

    return temp_map, ec_temp_map


def supplement_metadata(
    metadata_path: str,
    output_path: str,
    temp_map: dict,
    ec_temp_map: dict,
    default_temp: float = 298.15,
) -> pd.DataFrame:
    """
    为 metadata 补充温度列。

    匹配策略（优先级递减）:
    1. OED (uniprot_id, ec_numbers) → 中位数温度
    2. OED (ec_numbers) → 中位数温度 (EC-only 回退)
    3. 默认 298.15K

    OED 温度是 °C，需要转为 Kelvin。
    """
    log.info(f"Loading metadata from {metadata_path}")
    df = pd.read_parquet(metadata_path)
    log.info(f"Metadata entries: {len(df)}")

    # 初始化为默认温度 (K)
    df["temperature_K"] = default_temp

    # ── 策略 1: (uniprot, ec) 精确匹配 ──
    uniprot_ids = df["uniprot_id"].fillna("")
    ec_numbers = df["ec_numbers"].fillna("")
    keys = list(zip(uniprot_ids, ec_numbers))

    # 统计匹配
    matched_mask = pd.Series([k in temp_map for k in keys], index=df.index)
    n_matched = matched_mask.sum()
    log.info(f"Matched: {n_matched}/{len(df)} ({100 * n_matched / len(df):.1f}%)")

    # 应用匹配的温度（OED 是 °C → +273.15 转为 Kelvin）
    matched_temps_c = pd.Series(
        [temp_map[k] for k, is_match in zip(keys, matched_mask) if is_match],
        index=df.index[matched_mask],
    )
    df.loc[matched_mask, "temperature_K"] = matched_temps_c + 273.15

    # ── 策略 2: EC-only 回退 ──
    still_default = df["temperature_K"] == default_temp
    ec_only_keys = df.loc[still_default, "ec_numbers"].fillna("")
    ec_fallback_mask = ec_only_keys.isin(ec_temp_map.keys())
    n_ec_fallback = ec_fallback_mask.sum()
    log.info(f"\nEC-only fallback: +{n_ec_fallback}/{len(df)} "
             f"({100 * n_ec_fallback / len(df):.1f}%)")

    # 应用 EC-only 回退温度
    ec_fallback_indices = still_default.index[still_default][ec_fallback_mask.values]
    ec_fallback_temps_c = ec_only_keys[ec_fallback_mask.values].map(ec_temp_map)
    df.loc[ec_fallback_indices, "temperature_K"] = ec_fallback_temps_c.values + 273.15

    # 最终统计
    log.info(f"\nFinal coverage:")
    n_with_any_temp = (df["temperature_K"] != default_temp).sum()
    log.info(f"  with non-default temperature: {n_with_any_temp}/{len(df)} "
             f"({100 * n_with_any_temp / len(df):.1f}%)")

    # 温度统计
    temps = df["temperature_K"]
    log.info(f"Temperature stats (K):")
    log.info(f"  min={temps.min():.1f}, max={temps.max():.1f}")
    log.info(f"  mean={temps.mean():.1f}, median={temps.median():.1f}")
    log.info(f"  default (298.15K) used for: {(temps == 298.15).sum()} entries")

    # 按 split 统计覆盖
    log.info(f"\nCoverage by split:")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        split_covered = (split_df["temperature_K"] != default_temp).sum()
        split_total = len(split_df)
        log.info(
            f"  {split_name}: {split_covered}/{split_total} "
            f"({100 * split_covered / split_total:.1f}%)"
        )

    # 保存
    df.to_parquet(output_path, index=False)
    log.info(f"Saved to {output_path}")

    return df


def main():
    # 1. 构建 OED 温度映射（含 EC-only 回退）
    temp_map, ec_temp_map = build_oed_temperature_map(str(OED_PATH))

    # 2. 补充 V5 核心 metadata
    log.info("\n" + "=" * 60)
    log.info("Supplementing V5 core metadata (metadata.parquet)")
    log.info("=" * 60)
    df_v5 = supplement_metadata(
        metadata_path=str(V5_METADATA_PATH),
        output_path=str(V5_METADATA_PATH),  # 覆盖写入
        temp_map=temp_map,
        ec_temp_map=ec_temp_map,
    )

    # 3. 补充含负样本的 metadata (负样本也写入默认温度，方便统一加载)
    log.info("\n" + "=" * 60)
    log.info("Supplementing metadata with negatives (metadata_with_negatives.parquet)")
    log.info("=" * 60)
    df_wn = supplement_metadata(
        metadata_path=str(V5_WITH_NEG_PATH),
        output_path=str(V5_WITH_NEG_PATH),  # 覆盖写入
        temp_map=temp_map,
        ec_temp_map=ec_temp_map,
    )

    # 4. 验证: 温度分布可视化
    log.info("\n" + "=" * 60)
    log.info("Temperature distribution (V5 core)")
    log.info("=" * 60)
    hist, bins = np.histogram(df_v5["temperature_K"], bins=10)
    for i in range(len(hist)):
        pct = 100 * hist[i] / len(df_v5)
        # 标记 298.15K 区间的默认值占比
        if bins[i] <= 298.15 < bins[i + 1]:
            n_default = (df_v5["temperature_K"] == 298.15).sum()
            log.info(f"  {bins[i]:.0f}-{bins[i+1]:.0f}K: {hist[i]:6d} ({pct:5.1f}%) "
                     f"[includes {n_default} default (298.15K)]")
        else:
            log.info(f"  {bins[i]:.0f}-{bins[i+1]:.0f}K: {hist[i]:6d} ({pct:5.1f}%)")

    log.info("\n✓ Temperature supplementation complete!")


if __name__ == "__main__":
    main()
