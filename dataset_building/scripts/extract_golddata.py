#!/usr/bin/env python3
"""
extract_golddata.py
===================
从 V5 核心数据集中提取同时满足以下条件的金标数据：
  1. 有 pKd 标签 (pkd_raw 非空)
  2. 有 kcat 标签 (has_kcat=True)
  3. 有辅因子信息 (cofactors 非空)
  4. 有温度数据 (temperature_K ≠ 298.15K，即来自 OED/BRENDA 匹配)

输出: dataset_building/golddata/gold.parquet
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
V5_PATH = PROJECT_DIR / "processed" / "metadata.parquet"
OUTPUT_DIR = PROJECT_DIR / "golddata"
OUTPUT_PATH = OUTPUT_DIR / "gold.parquet"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(V5_PATH)
    log.info(f"V5 total: {len(df)}")

    # ── 过滤条件 ──
    has_pkd = df["pkd_raw"].notna()
    has_kcat = df["has_kcat"]
    has_cofactor = df["cofactors"].notna() & (df["cofactors"] != "")
    has_temp = df["temperature_K"] != 298.15

    gold = df[has_pkd & has_kcat & has_cofactor & has_temp].copy()

    log.info(f"Gold data: {len(gold)} ({100 * len(gold) / len(df):.1f}%)")

    # ── 统计 ──
    log.info(f"Unique proteins: {gold['protein_seq_hash'].nunique()}")
    log.info(f"Unique ligands:  {gold['ligand_inchikey'].nunique()}")

    for s in ["train", "val", "test"]:
        n = (gold["split"] == s).sum()
        log.info(f"  {s}: {n}")

    pkd = gold["pkd_raw"]
    kcat = gold["log_kcat_median"]
    temp = gold["temperature_K"]
    log.info(f"pKd:  {pkd.min():.2f} - {pkd.max():.2f} (mean={pkd.mean():.2f})")
    log.info(f"kcat: {kcat.min():.2f} - {kcat.max():.2f} (mean={kcat.mean():.2f})")
    log.info(f"Temp: {temp.min():.1f} - {temp.max():.1f}K (mean={temp.mean():.1f})")

    gold["ec_class"] = gold["ec_numbers"].str.split(".").str[0]
    log.info(f"EC classes:\n{gold['ec_class'].value_counts().sort_index().to_string()}")

    # ── 保存 ──
    gold.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
