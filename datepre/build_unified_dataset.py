"""
build_unified_dataset.py
========================
组装统一的氧化还原酶数据集，修复所有已知数据加载 bug，重新划分 train/val/test。

问题修复：
  1. 用 (protein_seq_hash, ligand_inchikey) 替代错误的 record_idx 合并
  2. 从主 metadata.parquet 传播 w_multiplier
  3. 合并多源动力学：BRENDA + SABIO-RK + UniProt → consensus kcat
  4. 重新按 protein_seq_hash 划分 80/10/10，确保无蛋白泄露
  5. 添加 kcat_source 跟踪列

输出：
  processed/oxidoreductase/unified_metadata.parquet   （训练就绪的元数据）
  processed/oxidoreductase/unified_records.pkl         （完整记录，可选）

用法：
  python datepre/build_unified_dataset.py
  python datepre/build_unified_dataset.py --no-sabio   # SABIO-RK 数据不可用时
"""

import os
import pickle
import hashlib
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
OUT_DIR = PROCESSED_DIR / "oxidoreductase"


# ═══════════════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════════════

def load_main_metadata() -> pd.DataFrame:
    """加载主 metadata.parquet（完整数据集 2,192,378 条）。"""
    path = PROCESSED_DIR / "metadata.parquet"
    log.info(f"Loading main metadata: {path}")
    df = pd.read_parquet(path)
    log.info(f"  {len(df):,} rows, {len(df.columns)} columns: {list(df.columns)}")
    return df


def load_oxid_metadata() -> pd.DataFrame:
    """加载氧化还原酶子集 metadata（78,118 条）。"""
    path = OUT_DIR / "metadata.parquet"
    log.info(f"Loading oxidoreductase metadata: {path}")
    df = pd.read_parquet(path)
    log.info(f"  {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_brenda_aligned() -> pd.DataFrame:
    """加载 BRENDA 对齐数据。"""
    path = OUT_DIR / "brenda_aligned.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        log.info(f"Loaded BRENDA aligned: {len(df):,} rows")
        return df
    log.warning(f"BRENDA aligned not found: {path}")
    return pd.DataFrame()


def load_sabio_aligned() -> pd.DataFrame:
    """加载 SABIO-RK 对齐数据（如果存在）。"""
    path = OUT_DIR / "sabio_aligned.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        log.info(f"Loaded SABIO-RK aligned: {len(df):,} rows")
        return df
    log.warning(f"SABIO-RK aligned not found: {path}")
    return pd.DataFrame()


def load_uniprot_kinetics() -> pd.DataFrame:
    """加载 UniProt 动力学数据（如果存在）。"""
    path = OUT_DIR / "kinetics.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        log.info(f"Loaded UniProt kinetics: {len(df):,} rows")
        return df
    log.warning(f"UniProt kinetics not found: {path}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 2. 合并逻辑（修复 bug）
# ═══════════════════════════════════════════════════════════════

def merge_oxid_to_main(
    oxid_meta: pd.DataFrame,
    main_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    用 (protein_seq_hash, ligand_inchikey) 作为复合键合并，
    替代错误的 record_idx 合并。

    策略：
      - 从 main_meta 获取：w_multiplier, pkd_aligned, split, sample_id 等
      - 从 oxid_meta 获取：cofactors, ec_numbers, protein_name, reviewed, domain 列等
      - 合并键：(protein_seq_hash, ligand_inchikey)
    """
    log.info("Merging oxidoreductase metadata with main metadata...")

    # main_meta 的关键列
    main_cols = [
        "protein_seq_hash", "ligand_inchikey",
        "sample_id", "pkd_aligned", "pkd_raw",
        "quality_weight", "w_multiplier",
        "measurement_type", "is_censored",
        "has_structure", "has_binding_site",
        "n_measurements", "pkd_std",
        "source_db", "pdb_id", "uniprot_id", "split",
    ]
    # 只取 main_meta 中存在的列
    main_cols_avail = [c for c in main_cols if c in main_meta.columns]
    main_sub = main_meta[main_cols_avail].copy()

    # oxid_meta 中独有的列（不在 main_cols 中）
    oxid_only_cols = [
        c for c in oxid_meta.columns
        if c not in main_cols_avail and c != "record_idx"
    ]
    oxid_sub = oxid_meta[oxid_only_cols + ["protein_seq_hash", "ligand_inchikey"]].copy()

    # inner join on (protein_seq_hash, ligand_inchikey)
    merged = oxid_sub.merge(
        main_sub,
        on=["protein_seq_hash", "ligand_inchikey"],
        how="inner",
        suffixes=("_oxid", "_main"),
    )

    # 去重：同一 (protein_seq_hash, ligand_inchikey) 可能有多个条目
    # （来自同一蛋白质-配体对的多次测量），保留第一个
    n_before = len(merged)
    merged = merged.drop_duplicates(
        subset=["protein_seq_hash", "ligand_inchikey"]
    )
    if n_before > len(merged):
        log.info(f"  Deduplicated: {n_before} → {len(merged)} rows")

    log.info(f"  Merged: {len(merged):,} rows from {len(oxid_meta):,} oxid × "
             f"{len(main_sub):,} main")

    # 修复：无结构蛋白不应该有 binding_site
    if "has_structure" in merged.columns and "has_binding_site" in merged.columns:
        n_before = int(merged["has_binding_site"].sum())
        merged.loc[~merged["has_structure"], "has_binding_site"] = False
        n_after = int(merged["has_binding_site"].sum())
        log.info(f"  Fixed has_binding_site: {n_before} → {n_after} "
                 f"(removed {n_before - n_after} false positives on non-structure proteins)")

    return merged


# ═══════════════════════════════════════════════════════════════
# 3. 合并多源动力学
# ═══════════════════════════════════════════════════════════════

def merge_kinetics(
    merged: pd.DataFrame,
    brenda_df: pd.DataFrame,
    sabio_df: pd.DataFrame,
    uniprot_kinetics_df: pd.DataFrame,
) -> pd.DataFrame:
    """合并 BRENDA + SABIO-RK + UniProt 动力学数据。"""
    log.info("Merging multi-source kinetics...")

    # ── BRENDA ──
    if len(brenda_df) > 0:
        brenda_cols = [
            "uniprot_id",
            "n_km_bdb", "n_kcat_bdb", "n_kcatkm_bdb",
            "km_median_uM", "kcat_median_s", "kcatkm_median_M1s1",
        ]
        brenda_sub = brenda_df[[c for c in brenda_cols if c in brenda_df.columns]].copy()
        # 按 uniprot_id 去重取中位数
        if "uniprot_id" in brenda_sub.columns:
            brenda_sub = brenda_sub.groupby("uniprot_id", as_index=False).agg({
                c: "median" if c not in ["uniprot_id"] else "first"
                for c in brenda_sub.columns if c != "uniprot_id"
            })
            # 重命名避免冲突
            brenda_sub = brenda_sub.rename(columns={
                "n_km_bdb": "bdb_n_km",
                "n_kcat_bdb": "bdb_n_kcat",
                "n_kcatkm_bdb": "bdb_n_kcatkm",
                "km_median_uM": "bdb_km_median_uM",
                "kcat_median_s": "bdb_kcat_median_s",
                "kcatkm_median_M1s1": "bdb_kcatkm_median_M1s1",
            })
            merged = merged.merge(brenda_sub, on="uniprot_id", how="left")
            n_bdb = int(merged["bdb_kcat_median_s"].notna().sum())
            log.info(f"  BRENDA: {len(brenda_sub)} UniProt IDs, "
                     f"{n_bdb} records with kcat")
    else:
        for col in ["bdb_n_km", "bdb_n_kcat", "bdb_n_kcatkm",
                     "bdb_km_median_uM", "bdb_kcat_median_s",
                     "bdb_kcatkm_median_M1s1"]:
            merged[col] = np.nan

    # ── SABIO-RK ──
    if len(sabio_df) > 0:
        sabio_cols = [
            "uniprot_id", "n_km_sabio", "n_kcat_sabio", "n_kcatkm_sabio",
            "km_median_uM_sabio", "kcat_median_s_sabio",
            "kcatkm_median_M1s1_sabio",
        ]
        sabio_sub = sabio_df[[c for c in sabio_cols if c in sabio_df.columns]].copy()
        if "uniprot_id" in sabio_sub.columns:
            sabio_sub = sabio_sub.groupby("uniprot_id", as_index=False).agg({
                c: "median" if c not in ["uniprot_id"] else "first"
                for c in sabio_sub.columns if c != "uniprot_id"
            })
            merged = merged.merge(sabio_sub, on="uniprot_id", how="left")
            n_sabio = int(merged["kcat_median_s_sabio"].notna().sum()) if "kcat_median_s_sabio" in merged.columns else 0
            log.info(f"  SABIO-RK: {n_sabio} records with kcat")
    else:
        for col in ["n_km_sabio", "n_kcat_sabio", "n_kcatkm_sabio",
                     "km_median_uM_sabio", "kcat_median_s_sabio",
                     "kcatkm_median_M1s1_sabio"]:
            merged[col] = np.nan

    # ── UniProt kinetics (existing field in oxid_meta) ──
    # Already included as has_kinetics, n_km_entries, n_kcat_entries
    merged["up_has_kinetics"] = merged.get("has_kinetics", False)
    merged["up_n_kcat"] = merged.get("n_kcat_entries", 0)

    # ── 计算 consensus kcat ──
    log.info("  Computing consensus kcat...")
    kcat_sources = []
    consensus_kcat = []
    for _, row in merged.iterrows():
        vals = []
        srcs = []
        # BRENDA
        bdb_val = row.get("bdb_kcat_median_s")
        if pd.notna(bdb_val) and bdb_val > 0:
            vals.append(float(bdb_val))
            srcs.append("bdb")
        # SABIO-RK
        sabio_val = row.get("kcat_median_s_sabio")
        if pd.notna(sabio_val) and sabio_val > 0:
            vals.append(float(sabio_val))
            srcs.append("sabio")
        # UniProt (we don't have a consolidated kcat per-uniprot, skip for now)

        if vals:
            kcat_sources.append("|".join(srcs))
            consensus_kcat.append(np.median(vals))
        else:
            kcat_sources.append("")
            consensus_kcat.append(np.nan)

    merged["kcat_source"] = kcat_sources
    merged["kcat_median_s"] = consensus_kcat
    merged["has_kcat"] = merged["kcat_median_s"].notna()
    merged["log_kcat_median"] = np.log10(merged["kcat_median_s"].where(
        merged["kcat_median_s"].notna() & (merged["kcat_median_s"] > 0)))

    # 标记 kcat 异常值（可能为注释错误或非催化事件）
    merged["kcat_outlier"] = (
        merged["kcat_median_s"].notna() &
        ((merged["kcat_median_s"] > 1e3) | (merged["kcat_median_s"] < 1e-3))
    )
    n_outliers = int(merged["kcat_outlier"].sum())
    if n_outliers > 0:
        log.info(f"  kcat outliers (>1e3 or <1e-3 s⁻¹): {n_outliers}")

    n_with_kcat = int(merged["has_kcat"].sum())
    log.info(f"  Records with kcat (any source): {n_with_kcat}")

    # kcat source breakdown
    if n_with_kcat > 0:
        for src in ["bdb", "sabio"]:
            n = int(merged["kcat_source"].str.contains(src, na=False).sum())
            log.info(f"    from {src}: {n}")

    return merged


# ═══════════════════════════════════════════════════════════════
# 4. 重新划分 train/val/test
# ═══════════════════════════════════════════════════════════════

def make_splits(df: pd.DataFrame,
                train_frac: float = 0.8,
                val_frac: float = 0.1,
                test_frac: float = 0.1,
                seed: int = 42,
                ) -> pd.DataFrame:
    """按 protein_seq_hash 哈希划分 train/val/test，确保辅因子类型和 kcat 覆盖分层。

    分层策略：
      1. 蛋白按是否有 kcat 数据分为两层
      2. 每层内独立哈希分割 → 各 split 的 kcat 覆盖率保持一致
      3. 辅因子类型覆盖检查 + 修复
    """
    log.info(f"Re-splitting dataset: {train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%} "
             f"(stratified by kcat availability)")

    unique_hashes = df["protein_seq_hash"].unique()
    n_proteins = len(unique_hashes)
    log.info(f"  Unique proteins: {n_proteins}")

    # ── 分层：按蛋白是否有 kcat 数据 ──
    protein_has_kcat = df.groupby("protein_seq_hash")["has_kcat"].any()
    hashes_with_kcat = sorted(protein_has_kcat[protein_has_kcat].index)
    hashes_without_kcat = sorted(protein_has_kcat[~protein_has_kcat].index)
    log.info(f"  With kcat: {len(hashes_with_kcat)}, "
             f"Without kcat: {len(hashes_without_kcat)}")

    rng = np.random.RandomState(seed)
    train_hashes: set[str] = set()
    val_hashes: set[str] = set()
    test_hashes: set[str] = set()

    # ── 预处理：每个蛋白的辅因子类型和 record 数 ──
    def _primary_cf(row):
        cf = str(row.get("cofactors", "") or "")
        return cf.split("|")[0].strip() if cf else "none"

    protein_cf_map = {}
    protein_n_records = {}
    for h in unique_hashes:
        sub = df[df["protein_seq_hash"] == h]
        protein_n_records[h] = len(sub)
        protein_cf_map[h] = _primary_cf(sub.iloc[0])

    # ── kcat 分层内全局 LPT：主目标 = record 平衡 (80/10/10) ──
    # 辅因子分布无法与 record+kcat 同时完美平衡（单个蛋白可占一种辅因子 90%+ 的 records）
    # 辅因子通过后续 fixup 保证覆盖，不在此处做硬约束
    for stratum_name, stratum_hashes in [
        ("kcat+", hashes_with_kcat),
        ("kcat-", hashes_without_kcat),
    ]:
        n = len(stratum_hashes)
        if n == 0:
            continue

        stratum_records = sum(protein_n_records[h] for h in stratum_hashes)
        target_r_train = stratum_records * train_frac
        target_r_val = stratum_records * val_frac
        target_r_test = stratum_records * test_frac

        s_train, s_val, s_test = set(), set(), set()
        r_train, r_val, r_test = 0, 0, 0

        # 按 record 降序排列（LPT：大蛋白优先分配）
        shuffled = list(stratum_hashes)
        rng.shuffle(shuffled)
        sorted_hashes = sorted(shuffled, key=lambda h: protein_n_records[h], reverse=True)

        for h in sorted_hashes:
            n_rec = protein_n_records[h]
            deficit_train = target_r_train - r_train
            deficit_val = target_r_val - r_val
            deficit_test = target_r_test - r_test

            candidates = [
                ("train", deficit_train),
                ("val", deficit_val),
                ("test", deficit_test),
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            pick = candidates[0][0]

            if pick == "train":
                s_train.add(h); r_train += n_rec
            elif pick == "val":
                s_val.add(h); r_val += n_rec
            else:
                s_test.add(h); r_test += n_rec

        train_hashes.update(s_train)
        val_hashes.update(s_val)
        test_hashes.update(s_test)

        log.info(f"  {stratum_name} ({n} proteins, {stratum_records} records): "
                 f"train={len(s_train)}p/{r_train}r ({r_train/stratum_records*100:.0f}%), "
                 f"val={len(s_val)}p/{r_val}r ({r_val/stratum_records*100:.0f}%), "
                 f"test={len(s_test)}p/{r_test}r ({r_test/stratum_records*100:.0f}%)")

    # 验证分层后各 split 的 kcat 覆盖率
    for name, hashes in [("train", train_hashes), ("val", val_hashes), ("test", test_hashes)]:
        n_kcat = sum(1 for h in hashes if h in hashes_with_kcat)
        n_total = len(hashes)
        log.info(f"  {name} kcat coverage: {n_kcat}/{n_total} proteins ({n_kcat/n_total*100:.0f}%)")

    # ── 辅因子类型覆盖检查 + 修复 ──
    all_cf = set(protein_cf_map.values())
    for split_hashes, split_name in [(train_hashes, "train"), (val_hashes, "val"), (test_hashes, "test")]:
        split_cf = set(protein_cf_map[h] for h in split_hashes)
        missing = all_cf - split_cf
        if missing:
            log.warning(f"  {split_name} missing cofactor types: {missing}")

    # 修复：确保所有 split 都有每种辅因子类型
    for target_set, target_name in [(val_hashes, "val"), (test_hashes, "test"), (train_hashes, "train")]:
        target_cf = set(protein_cf_map[h] for h in target_set)
        missing_cf = all_cf - target_cf
        if missing_cf:
            donor_pool = (train_hashes | val_hashes | test_hashes) - target_set
            for cf_type in missing_cf:
                candidates = [h for h in donor_pool if protein_cf_map.get(h) == cf_type]
                if candidates:
                    h_move = candidates[0]
                    for pool_set in [train_hashes, val_hashes, test_hashes]:
                        pool_set.discard(h_move)
                    target_set.add(h_move)
            log.info(f"  Moved proteins to {target_name} for cofactor coverage: {missing_cf}")

    def _assign(h):
        if h in train_hashes:
            return "train"
        elif h in val_hashes:
            return "val"
        else:
            return "test"

    df["split"] = df["protein_seq_hash"].apply(_assign)

    # ── 验证 ──
    counts = df["split"].value_counts()
    for s in ["train", "val", "test"]:
        pct = counts.get(s, 0) / len(df) * 100
        log.info(f"    {s}: {counts.get(s, 0):,} records ({pct:.1f}%)")

    # 蛋白隔离检查
    train_proteins = set(df[df["split"] == "train"]["protein_seq_hash"])
    val_proteins = set(df[df["split"] == "val"]["protein_seq_hash"])
    test_proteins = set(df[df["split"] == "test"]["protein_seq_hash"])

    overlap_tv = len(train_proteins & val_proteins)
    overlap_tt = len(train_proteins & test_proteins)
    overlap_vt = len(val_proteins & test_proteins)
    if overlap_tv or overlap_tt or overlap_vt:
        log.error(f"  PROTEIN LEAKAGE DETECTED! TV={overlap_tv}, "
                  f"TT={overlap_tt}, VT={overlap_vt}")
    else:
        log.info("  No protein leakage across splits ✓")

    # 最终覆盖率报告
    if "cofactors" in df.columns:
        log.info("  Cofactor coverage by split:")
        for split_val in ["train", "val", "test"]:
            sub = df[df["split"] == split_val]
            cf_counts = sub[sub["cofactors"] != ""]["cofactors"].str.split(
                "|").str[0].value_counts().head(5)
            cf_str = ", ".join(f"{k}:{v}" for k, v in cf_counts.items())
            log.info(f"    {split_val}: {cf_str}")

    # kcat 覆盖率最终报告
    log.info("  kcat record coverage by split:")
    for split_val in ["train", "val", "test"]:
        sub = df[df["split"] == split_val]
        n_kcat = int(sub["has_kcat"].sum())
        n_total = len(sub)
        log.info(f"    {split_val}: {n_kcat}/{n_total} records ({n_kcat/n_total*100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════
# 5. 最终组装 + 输出
# ═══════════════════════════════════════════════════════════════

def finalize_and_save(df: pd.DataFrame, out_dir: Path):
    """整理列顺序、清理类型、保存。"""
    log.info("Finalizing unified metadata...")

    # 确保关键列存在且类型正确
    bool_cols = ["is_censored", "has_structure", "has_binding_site",
                 "reviewed", "has_kcat", "has_domain_annotation",
                 "up_has_kinetics", "kcat_outlier"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # 整数字段
    int_cols = ["n_measurements", "n_domains",
                "bdb_n_km", "bdb_n_kcat", "bdb_n_kcatkm",
                "n_km_sabio", "n_kcat_sabio", "n_kcatkm_sabio",
                "up_n_kcat"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # 浮点字段
    float_cols = ["pkd_raw", "pkd_aligned", "pkd_std",
                  "quality_weight", "w_multiplier",
                  "bdb_km_median_uM", "bdb_kcat_median_s",
                  "bdb_kcatkm_median_M1s1",
                  "km_median_uM_sabio", "kcat_median_s_sabio",
                  "kcatkm_median_M1s1_sabio",
                  "kcat_median_s", "log_kcat_median"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # 字符串填充
    str_cols = ["cofactors", "ec_numbers", "protein_name",
                "cofactor_domain_types", "domains_json",
                "kcat_source", "source_db", "measurement_type",
                "split", "sample_id"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # 定义列顺序（关键列在前，辅助列在后）
    column_order = [
        # 标识
        "sample_id", "protein_seq_hash", "ligand_inchikey",
        "uniprot_id", "pdb_id", "source_db",
        # 氧化还原酶特定
        "ec_numbers", "cofactors", "protein_name", "reviewed",
        # 标签
        "pkd_aligned", "pkd_raw", "measurement_type",
        "quality_weight", "w_multiplier", "is_censored",
        "n_measurements", "pkd_std",
        # 结构
        "has_structure", "has_binding_site",
        # 域注释
        "has_domain_annotation", "n_domains",
        "cofactor_domain_types", "domains_json",
        # 动力学 — BRENDA
        "bdb_n_km", "bdb_n_kcat", "bdb_n_kcatkm",
        "bdb_km_median_uM", "bdb_kcat_median_s",
        "bdb_kcatkm_median_M1s1",
        # 动力学 — SABIO-RK
        "n_km_sabio", "n_kcat_sabio", "n_kcatkm_sabio",
        "km_median_uM_sabio", "kcat_median_s_sabio",
        "kcatkm_median_M1s1_sabio",
        # 动力学 — UniProt
        "up_has_kinetics", "up_n_kcat",
        # Consensus
        "has_kcat", "kcat_source", "kcat_median_s", "log_kcat_median", "kcat_outlier",
        # Split
        "split",
    ]
    # 只保留实际存在的列
    column_order = [c for c in column_order if c in df.columns]
    # 追加其他未列出的列
    remaining = [c for c in df.columns if c not in column_order]
    column_order.extend(remaining)

    df = df[column_order]

    # 保存
    path = out_dir / "unified_metadata.parquet"
    df.to_parquet(path, index=False)
    log.info(f"Unified metadata saved → {path}")
    log.info(f"  {len(df):,} rows × {len(column_order)} columns")

    return df


# ═══════════════════════════════════════════════════════════════
# 6. 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="构建统一氧化还原酶数据集")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--no-sabio", action="store_true",
                        help="SABIO-RK 数据不可用时跳过")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # ── Step 1: 加载数据 ──
    log.info("=" * 50)
    log.info("Step 1: Loading data sources")
    log.info("=" * 50)

    main_meta = load_main_metadata()
    oxid_meta = load_oxid_metadata()
    brenda_df = load_brenda_aligned()
    sabio_df = load_sabio_aligned() if not args.no_sabio else pd.DataFrame()

    # ── Step 2: 合并 oxid → main（修复 bug） ──
    log.info("=" * 50)
    log.info("Step 2: Merging oxidoreductase with main metadata")
    log.info("=" * 50)

    merged = merge_oxid_to_main(oxid_meta, main_meta)

    # 验证：不应该丢记录
    n_lost = len(oxid_meta) - len(merged)
    if n_lost > 0:
        log.warning(f"  {n_lost} oxidoreductase records lost in merge (no matching "
                     "main metadata). These may be from different processing runs.")
    else:
        log.info("  All oxidoreductase records matched ✓")

    # ── Step 3: 合并多源动力学 ──
    log.info("=" * 50)
    log.info("Step 3: Merging multi-source kinetics")
    log.info("=" * 50)

    merged = merge_kinetics(merged, brenda_df, sabio_df, pd.DataFrame())

    # ── Step 4: 重新划分 train/val/test ──
    log.info("=" * 50)
    log.info("Step 4: Re-splitting dataset")
    log.info("=" * 50)

    merged = make_splits(merged, seed=args.seed)

    # ── Step 5: 最终化 + 保存 ──
    log.info("=" * 50)
    log.info("Step 5: Finalizing and saving")
    log.info("=" * 50)

    final_df = finalize_and_save(merged, out_dir)

    # ── 报告 ──
    print("\n" + "=" * 60)
    print("  统一数据集构建 — 完成报告")
    print("=" * 60)
    print(f"  总记录数:               {len(final_df):,}")
    print(f"  唯一蛋白质:             {final_df['protein_seq_hash'].nunique():,}")
    print(f"  唯一配体:               {final_df['ligand_inchikey'].nunique():,}")
    print(f"  有结构特征:             {final_df['has_structure'].sum():,}")
    print(f"  有辅因子注释:           {(final_df['cofactors'] != '').sum():,}")
    print(f"  有域注释:               {final_df['has_domain_annotation'].sum():,}")
    print(f"  有 kcat (任意源):       {final_df['has_kcat'].sum():,}")

    # 测量类型分布
    print("\n  测量类型分布:")
    for mtype, count in final_df["measurement_type"].value_counts().items():
        print(f"    {mtype:10s}: {count:,}")

    # Split 分布
    print("\n  Split 分布:")
    for split_val in ["train", "val", "test"]:
        count = (final_df["split"] == split_val).sum()
        pct = count / len(final_df) * 100
        print(f"    {split_val}: {count:,} ({pct:.1f}%)")

    # kcat 来源
    if final_df["has_kcat"].sum() > 0:
        print("\n  kcat 来源分布:")
        for src in ["bdb", "sabio"]:
            n = final_df["kcat_source"].str.contains(src, na=False).sum()
            print(f"    {src}: {n:,}")

    print(f"\n  输出文件: {out_dir / 'unified_metadata.parquet'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
