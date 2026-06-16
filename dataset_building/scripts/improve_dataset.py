#!/usr/bin/env python3
"""
improve_dataset.py
==================
Trenzition V5 数据集质量提升脚本：

1. ✅ 修复 measurement_type bug — 无 pKd 的行不再被错误标记为 "Kd"
2. ✅ 过滤 kcat_source="unknown" (log_kcat=-10 占位符)
3. ✅ 过滤极弱 Ki 数据 (pKd < 0.3, Ki > 500mM，非特异性结合)
4. ✅ 扩充 BindingDB 数据 — 为 V5 中已有蛋白添加更多的 Kd/Ki 结合数据
5. ✅ 为新增配体构建 GNN 图文件
6. ✅ 输出改良版 V5 数据集

用法:
  source /home/domi/BINN/.venv/bin/activate
  python scripts/improve_dataset.py

输出:
  - release/trenzition_full_v5.parquet (原地更新)
  - processed/ligands/*.pt (新增配体图文件)
  - 统计报告打印到 stdout
"""

import logging
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

# ── 引入配体图构建函数 ──
# 不能直接 import 04_build_ligand_graphs（文件名以数字开头）
# 使用 importlib 动态加载
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ligand_graphs",
    str(Path(__file__).resolve().parent.parent / "pipeline" / "04_build_ligand_graphs.py")
)
lg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lg_module)
smiles_to_graph = lg_module.smiles_to_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RELEASE_DIR = BASE_DIR / "release"
PROCESSED_DIR = BASE_DIR / "processed"
LIGAND_DIR = PROCESSED_DIR / "ligands"

V5_PATH = RELEASE_DIR / "trenzition_full_v5.parquet"
IK2SMILES_PATH = PROCESSED_DIR / "inchikey_smiles_map.pkl"
BDB_INDEX_PATH = PROCESSED_DIR / "bindingdb_kd_ki_index.pkl"

KCAT_SOURCE_FILTER = {"unknown"}   # 需过滤的 kcat_source
PKI_MIN = 0.3                       # Ki→pKd 最低阈值（Ki < 500mM）


def load_v5() -> pd.DataFrame:
    log.info("Loading V5...")
    df = pd.read_parquet(V5_PATH)
    log.info(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
    return df


def fix_measurement_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复 measurement_type_pkd:
    - 有 pKd 值 (pkd_value notna) 但 measurement_type_pkd 为 NaN → 标记为 "Kd"
      （这类数据来自 PDBbind/辅因子匹配，是有结合数据的）
    - 无 pKd 值 → measurement_type_pkd = ""（之前被 fillna("Kd") 错误标记）
    """
    log.info("Fixing measurement_type_pkd...")
    n_before = len(df)

    # 记录改动
    n_fix_has_pkd = ((df["pkd_value"].notna()) & (df["measurement_type_pkd"].isna())).sum()
    n_fix_no_pkd = ((df["pkd_value"].isna()) & (df["measurement_type_pkd"].notna())).sum()

    # 有 pKd 但 measurement_type 缺失 → 设为 "Kd"
    mask_has_pkd = df["pkd_value"].notna() & df["measurement_type_pkd"].isna()
    df.loc[mask_has_pkd, "measurement_type_pkd"] = "Kd"

    # 无 pKd 但之前被设为 "Kd" → 清空
    mask_no_pkd = df["pkd_value"].isna() & df["measurement_type_pkd"].notna()
    df.loc[mask_no_pkd, "measurement_type_pkd"] = ""

    log.info(f"  Fixed measurement_type_pkd:")
    log.info(f"    has_pkd + missing_type → Kd: {n_fix_has_pkd:,}")
    log.info(f"    no_pkd + had_fake_Kd → '': {n_fix_no_pkd:,}")

    # 打印更新后的分布
    log.info(f"  Updated distribution:")
    for val, cnt in df["measurement_type_pkd"].value_counts(dropna=False).items():
        label = repr(val) if val == "" else str(val)
        log.info(f"    {label}: {cnt:,}")

    return df


def filter_invalid_kcat(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 kcat_source="unknown" 的行。
    这 4,608 行的 log_kcat=-10，是占位符非真实测量。
    不删除整行（它们有可用的 pKd 数据），而是将 kcat 设为 NaN，
    模型训练时会通过 mask 机制跳过 kcat 损失。
    """
    log.info("Fixing invalid kcat (source=unknown)...")

    invalid_mask = df["kcat_per_s_source"].isin(KCAT_SOURCE_FILTER)
    n_fix = invalid_mask.sum()
    log.info(f"  kcat_source=unknown: {n_fix:,} rows")

    # 将 kcat 置为 NaN（mask 会跳过 kcat 损失）
    df.loc[invalid_mask, "kcat_per_s"] = np.nan
    df.loc[invalid_mask, "kcat_per_s_source"] = None
    # 保留 measurement_type_pkd 和 pkd_value 不变

    log.info(f"  Set kcat→NaN for {n_fix:,} rows (pKd preserved)")
    return df


def filter_weak_ki(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理极弱的 Ki 数据。
    Ki 测量 pKd < 0.3 意味着 Ki > 500 mM，这更像非特异性结合而非真实抑制。
    但 kcat 数据可能仍然有效，所以只 mask pKd 不删除整行。
    BindingDB 的 Kd 数据不做此过滤（它们已经有正常的范围）。
    """
    log.info("Filtering weak Ki data (pKd < 0.3)...")

    weak_ki = (
        df["measurement_type_pkd"].isin(["Ki", "ki"]) &
        (df["pkd_value"].notna()) &
        (df["pkd_value"] < PKI_MIN)
    )
    n_mask = weak_ki.sum()
    log.info(f"  Weak Ki (pKd < {PKI_MIN}): {n_mask:,} rows")

    # 只 mask pKd，保留 kcat
    df.loc[weak_ki, "pkd_value"] = np.nan
    df.loc[weak_ki, "measurement_type_pkd"] = ""
    df.loc[weak_ki, "has_pkd"] = False
    df.loc[weak_ki, "pkd_valid"] = False

    log.info(f"  Masked pKd→NaN for {n_mask:,} rows (kcat preserved)")
    return df


def filter_extreme_pkd_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤极端异常的 pKd 值：pKd > 14 或 pKd < 0
    """
    n_before = len(df)
    extreme = (df["pkd_value"].notna()) & ((df["pkd_value"] < -1) | (df["pkd_value"] > 14))
    df = df[~extreme].reset_index(drop=True)
    log.info(f"  Removed extreme pKd: {n_before - len(df):,}")
    return df


def add_bindingdb_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 V5 中已有的蛋白添加更多的 BindingDB Kd/Ki 数据。
    前提条件:
    1. BindingDB 的 UniProt ID 在 V5 中存在
    2. 该 (UniProt, InChIKey) 对尚未在 V5 中
    3. pKd >= 1.0 (已有质量过滤)

    对于新增条目，会检查配体图文件是否已存在。
    不存在的会由 build_ligand_graphs_for_new_ligands() 补充。
    """
    log.info("Adding BindingDB expansion data...")

    # 加载 BindingDB index
    log.info("  Loading BindingDB index...")
    with open(BDB_INDEX_PATH, "rb") as f:
        bdb_idx = pickle.load(f)
    log.info(f"  BindingDB entries: {len(bdb_idx):,}")

    # V5 已有的 UniProt 和 (UniProt, InChIKey) 对
    v5_uniprots = set(df["uniprot_id"].dropna().unique())
    existing_pairs = set()
    for _, row in df.iterrows():
        up = row.get("uniprot_id")
        ik = row.get("ligand_inchikey")
        if pd.notna(up) and pd.notna(ik):
            existing_pairs.add((up, ik))

    log.info(f"  V5 unique UniProt: {len(v5_uniprots):,}")
    log.info(f"  V5 existing (UniProt, InChIKey) pairs: {len(existing_pairs):,}")

    # 找可新增的 BindingDB 条目
    new_rows_data = []
    skipped_no_smiles = 0

    for (up, ik), pkd in bdb_idx.items():
        if up not in v5_uniprots:
            continue
        if (up, ik) in existing_pairs:
            continue
        if pkd < 1.0:  # 低质量过滤
            continue

        # 检查是否有 SMILES（建配体图需要）
        ik2smiles_cache = getattr(add_bindingdb_expansion, "_ik2smiles", None)
        if ik2smiles_cache is None:
            with open(IK2SMILES_PATH, "rb") as f:
                ik2smiles_cache = pickle.load(f)
            add_bindingdb_expansion._ik2smiles = ik2smiles_cache

        smiles = ik2smiles_cache.get(ik, "")
        if not smiles:
            skipped_no_smiles += 1
            continue

        new_rows_data.append({
            "uniprot_id": up,
            "ligand_inchikey": ik,
            "pkd_value": pkd,
            "canonical_smiles": smiles,
            "measurement_type_pkd": "Kd",  # BindingDB 直接测量值
        })

    log.info(f"  Found {len(new_rows_data):,} new BindingDB entries to add")
    log.info(f"  Skipped (no SMILES): {skipped_no_smiles:,}")

    if not new_rows_data:
        return df

    new_df = pd.DataFrame(new_rows_data)

    # 需要从 V5 模板中继承更多信息
    uniprot_info = df[["uniprot_id", "protein_seq_hash", "sequence", "ec_number",
                       "enzyme_type", "organism_name", "smiles"]].drop_duplicates("uniprot_id")

    # 合并到新数据（使用 left merge 以保留所有 new_df 的行）
    merged = new_df.merge(uniprot_info, on="uniprot_id", how="left",
                          suffixes=("", "_v5"))
    matched = merged[merged["protein_seq_hash"].notna()].copy()
    log.info(f"  Successfully matched proteins: {len(matched):,}")

    if len(matched) == 0:
        return df

    # 用新配体的 canonical_smiles 覆盖（来自 BindingDB）
    # canonical_smiles_x 是 BindingDB 的，canonical_smiles_y 是 V5 的
    if "canonical_smiles_v5" in matched.columns:
        # 保留新配体自己的 SMILES
        pass
    elif "canonical_smiles_x" in matched.columns:
        matched.rename(columns={"canonical_smiles_x": "canonical_smiles"}, inplace=True)

    # 删除 merge 产生的多余列
    drop_cols = [c for c in matched.columns if c.endswith("_y") or c.endswith("_v5")]
    matched.drop(columns=drop_cols, inplace=True, errors="ignore")

    # 设置来源标记 — 先全部置空，再逐一填充
    for col in df.columns:
        if col not in matched.columns:
            matched[col] = None

    matched["data_source"] = "BindingDB"
    matched["pkd_value_source"] = "bindingdb_expanded"
    matched["kcat_per_s"] = np.nan
    matched["kcat_per_s_source"] = None
    matched["km_M"] = np.nan
    matched["km_M_source"] = None
    matched["has_pkd"] = True
    matched["pkd_valid"] = True
    matched["n_pkd_measurements"] = 1
    matched["pkd_std"] = 0.0
    matched["pkd_mean"] = matched["pkd_value"]  # 只有单测量值
    matched["pkd_min"] = matched["pkd_value"]
    matched["pkd_max"] = matched["pkd_value"]
    matched["quality_tier"] = 1  # Kd = tier 1
    matched["quality_weight_pkd"] = 1.0
    matched["match_level"] = "L1_exact"
    matched["pkd_confidence"] = "high"

    # 确保列顺序一致
    matched = matched[df.columns]

    # 追加
    combined = pd.concat([df, matched], ignore_index=True)
    log.info(f"  Total after BindingDB expansion: {len(combined):,} rows")

    return combined


def build_ligand_graphs_for_new_ligands(df: pd.DataFrame, existing_graphs: set):
    """
    为新增的配体（尚无 .pt 图文件）构建 GNN 图。
    使用 pipeline/build_ligand_graphs.py 中的 smiles_to_graph()。
    """
    log.info("Building ligand graphs for new ligands...")

    # 找到需要建图的配体
    all_iks = set(df["ligand_inchikey"].dropna().unique())
    need_build = all_iks - existing_graphs

    # 加载 SMILES 映射
    with open(IK2SMILES_PATH, "rb") as f:
        ik2smiles = pickle.load(f)

    log.info(f"  Total unique ligands: {len(all_iks):,}")
    log.info(f"  Already have graphs: {len(all_iks - need_build):,}")
    log.info(f"  Need to build: {len(need_build):,}")

    built = 0
    failed = 0
    skipped_no_smiles = 0
    start = time.time()

    for ik in sorted(need_build):
        smiles = ik2smiles.get(ik, "")
        if not smiles:
            skipped_no_smiles += 1
            continue

        graph = smiles_to_graph(smiles)
        if graph is None:
            failed += 1
            continue

        out_path = LIGAND_DIR / f"{ik}.pt"
        torch.save(graph, out_path)
        built += 1

        if built % 500 == 0:
            elapsed = time.time() - start
            rate = built / elapsed if elapsed > 0 else 0
            eta = (len(need_build) - built) / rate if rate > 0 else 0
            log.info(f"    Built {built}/{len(need_build):,} graphs "
                     f"({rate:.0f}/s, ETA {eta/60:.0f}min)")

    elapsed = time.time() - start
    log.info(f"  Done: {built:,} built, {failed:,} failed, "
             f"{skipped_no_smiles:,} skipped (no SMILES)")
    log.info(f"  Time: {elapsed:.1f}s ({built/elapsed:.0f} graphs/s)")


def print_stats_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame):
    """打印改良前后的数据统计对比"""
    log.info("\n" + "=" * 60)
    log.info("STATISTICS COMPARISON")
    log.info("=" * 60)

    def stats(d, label):
        n = len(d)
        n_prot = d["protein_seq_hash"].nunique() if "protein_seq_hash" in d else "?"
        n_lig = d["ligand_inchikey"].nunique() if "ligand_inchikey" in d else "?"
        n_ec = d["ec_number"].nunique() if "ec_number" in d else "?"
        n_pkd = d["pkd_value"].notna().sum() if "pkd_value" in d else 0
        n_kcat = d["kcat_per_s"].notna().sum() if "kcat_per_s" in d else 0
        n_both = (d["pkd_value"].notna() & d["kcat_per_s"].notna()).sum() if "pkd_value" in d and "kcat_per_s" in d else 0

        mt_dist = ""
        if "measurement_type_pkd" in d:
            mt = d["measurement_type_pkd"].value_counts(dropna=False).to_dict()
            mt_dist = f" | mt={mt}"

        return f"n={n:,} | prot={n_prot:,} | lig={n_lig:,} | EC4={n_ec:,} | pKd={n_pkd:,} | kcat={n_kcat:,} | both={n_both:,}{mt_dist}"

    log.info(f"  BEFORE: {stats(df_before, 'before')}")
    log.info(f"  AFTER:  {stats(df_after, 'after')}")

    # BindingDB 特有统计
    bdb_before = df_before[df_before["data_source"] == "BindingDB"]
    bdb_after = df_after[df_after["data_source"] == "BindingDB"]
    log.info(f"\n  BindingDB contribution:")
    log.info(f"    Before: {len(bdb_before):,} rows, {bdb_before['uniprot_id'].nunique():,} proteins")
    log.info(f"    After:  {len(bdb_after):,} rows, {bdb_after['uniprot_id'].nunique():,} proteins")


def main():
    log.info("=" * 60)
    log.info("TRENZITION V5 DATASET IMPROVEMENT")
    log.info("=" * 60)

    # ── Step 0: 检查已有配体图文件 ──
    LIGAND_DIR.mkdir(parents=True, exist_ok=True)
    existing_graphs = set(f.stem for f in LIGAND_DIR.glob("*.pt"))
    log.info(f"Existing ligand graphs: {len(existing_graphs):,}")

    # ── Step 1: 加载 V5 ──
    df = load_v5()
    df_before = df.copy()

    # ── Step 2: 修复 measurement_type ──
    df = fix_measurement_type(df)

    # ── Step 3: 过滤无效数据 ──
    df = filter_invalid_kcat(df)
    df = filter_weak_ki(df)
    df = filter_extreme_pkd_values(df)

    # ── Step 4: 扩充 BindingDB ──
    df = add_bindingdb_expansion(df)

    # ── Step 5: 建配体图（新增配体） ──
    # 重新检查现有配体图（可能已被上面新增的覆盖）
    updated_existing = set(f.stem for f in LIGAND_DIR.glob("*.pt"))
    build_ligand_graphs_for_new_ligands(df, updated_existing)

    # ── Step 6: 保存 ──
    log.info("\nSaving improved V5...")
    df.to_parquet(V5_PATH, index=False)
    log.info(f"  → {V5_PATH} ({len(df):,} rows × {len(df.columns)} cols)")

    # ── Step 7: 统计对比 ──
    print_stats_comparison(df_before, df_after=df)

    log.info("\n✅ Dataset improvement complete!")
    log.info("Next steps:")
    log.info("  python scripts/encode_trenzition_v5.py [--skip-esm]")
    log.info("  python scripts/extract_golddata.py")


if __name__ == "__main__":
    main()
