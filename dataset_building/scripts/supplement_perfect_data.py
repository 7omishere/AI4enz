"""
从原AI4enz数据集中提取"完美数据"（同时含kcat+Km+pKd），
补充到 Trenzition V4 中，生成 V5。

完美数据定义：has_kcat=True AND bdb_km_median_uM not null AND pkd_aligned not null
来源：BindingDB（5,307条），全部含蛋白+配体+动力学+亲和力完整标注。
"""

import pandas as pd
import numpy as np
import h5py
import pickle
import os
from pathlib import Path

PROCESSED_DIR = Path("/home/domi/AI4enz/dataset_building/processed")
RELEASE_DIR = Path("/home/domi/AI4enz/dataset_building/release")

# ── Step 1: Load datasets ──────────────────────────────────────────
print("=" * 60)
print("Step 1: 加载数据集")
print("=" * 60)

orig = pd.read_parquet(RELEASE_DIR / "recommended_training_set_with_pdbbind.parquet")
v4 = pd.read_parquet(RELEASE_DIR / "trenzition_full_v4.parquet")
print(f"原始数据集: {len(orig):,} 条")
print(f"Trenzition V4: {len(v4):,} 条")

# ── Step 2: Extract perfect records from original ───────────────────
print("\n" + "=" * 60)
print("Step 2: 提取完美数据")
print("=" * 60)

perf_mask = (
    orig["has_kcat"].fillna(False)
    & orig["bdb_km_median_uM"].notna()
    & orig["pkd_aligned"].notna()
)
perf = orig[perf_mask].copy()
print(f"完美数据: {len(perf):,} 条")
print(f"  Ki: {(perf['measurement_type'] == 'Ki').sum():,}")
print(f"  Kd: {(perf['measurement_type'] == 'Kd').sum():,}")
print(f"  唯一蛋白: {perf['uniprot_id'].nunique()}")
print(f"  唯一配体: {perf['ligand_inchikey'].nunique()}")

# ── Step 3: Remove duplicates already in V4 ─────────────────────────
print("\n" + "=" * 60)
print("Step 3: 去重 - 移除已在 V4 中的记录")
print("=" * 60)

v4_pairs = set(
    zip(
        v4["protein_seq_hash"].fillna("__na__"),
        v4["ligand_inchikey"].fillna("__na__"),
    )
)

perf["_pair_key"] = list(
    zip(
        perf["protein_seq_hash"].fillna("__na__"),
        perf["ligand_inchikey"].fillna("__na__"),
    )
)

# 标记已存在的
perf["_in_v4"] = perf["_pair_key"].isin(v4_pairs)
n_overlap = perf["_in_v4"].sum()
print(f"已存在于 V4: {n_overlap} 条")
print(f"需补充: {len(perf) - n_overlap:,} 条")

new_records = perf[~perf["_in_v4"]].copy()
print(f"  唯一蛋白: {new_records['uniprot_id'].nunique()}")
print(f"  唯一配体: {new_records['ligand_inchikey'].nunique()}")

# ── Step 4: Look up sequences from proteins.h5 ──────────────────────
print("\n" + "=" * 60)
print("Step 4: 查找蛋白序列")
print("=" * 60)

n_found_seq = 0
protein_seqs = {}

with h5py.File(PROCESSED_DIR / "proteins.h5", "r") as f:
    protein_hashes = new_records["protein_seq_hash"].unique()
    for ph in protein_hashes:
        if ph in f and "sequence" in f[ph]:
            seq_bytes = f[ph]["sequence"][()]
            if isinstance(seq_bytes, bytes):
                protein_seqs[ph] = seq_bytes.decode("utf-8")
            else:
                protein_seqs[ph] = str(seq_bytes)
            n_found_seq += 1

print(f"从 proteins.h5 找到序列: {n_found_seq}/{len(protein_hashes)}")

# ── Step 5: Look up SMILES from inchikey map ────────────────────────
print("\n" + "=" * 60)
print("Step 5: 查找配体 SMILES")
print("=" * 60)

with open(PROCESSED_DIR / "inchikey_smiles_map.pkl", "rb") as f:
    inchikey_smiles = pickle.load(f)

n_found_smiles = 0
ligand_smiles = {}
ligand_inchikeys = new_records["ligand_inchikey"].unique()
for ik in ligand_inchikeys:
    if ik in inchikey_smiles:
        ligand_smiles[ik] = inchikey_smiles[ik]
        n_found_smiles += 1

print(f"从 inchikey_smiles_map 找到 SMILES: {n_found_smiles}/{len(ligand_inchikeys)}")

# ── Step 6: Build V4-compatible records ─────────────────────────────
print("\n" + "=" * 60)
print("Step 6: 构建 V4 兼容记录")
print("=" * 60)

# 映射列
new_df = pd.DataFrame()

# 标识列
new_df["ec_number"] = new_records["ec_numbers"].values
new_df["enzyme_type"] = new_records["enzymetype"].values if "enzymetype" in new_records.columns else None
new_df["organism_name"] = new_records["organism_name"].values if "organism_name" in new_records.columns else None
new_df["sequence"] = new_records["protein_seq_hash"].map(protein_seqs).values
new_df["substrate_name"] = new_records["substrate_name"].values if "substrate_name" in new_records.columns else None
new_df["smiles"] = new_records["ligand_inchikey"].map(ligand_smiles).values
new_df["uniprot_id"] = new_records["uniprot_id"].values

# 动力学列
new_df["kcat_per_s"] = new_records["kcat_median_s"].values  # 蛋白级 kcat 中位数
new_df["cv_fold"] = np.nan
new_df["data_type"] = "perfect"  # 含 kcat+Km+pKd 三标签
new_df["km_M"] = new_records["bdb_km_median_uM"].values / 1_000_000  # uM → M

# 蛋白/配体哈希
new_df["protein_seq_hash"] = new_records["protein_seq_hash"].values
new_df["ligand_inchikey"] = new_records["ligand_inchikey"].values
new_df["canonical_smiles"] = new_records["ligand_inchikey"].map(ligand_smiles).values

# pKd 相关列
new_df["pkd_value"] = new_records["pkd_aligned"].values
new_df["pkd_mean"] = new_records["pkd_aligned"].values  # 单值=均值
new_df["pkd_std"] = new_records["pkd_std"].fillna(0).values
new_df["pkd_min"] = new_records["pkd_aligned"].values
new_df["pkd_max"] = new_records["pkd_aligned"].values
new_df["n_pkd_measurements"] = new_records["n_measurements"].fillna(1).values
new_df["measurement_type_pkd"] = new_records["measurement_type"].values
new_df["quality_tier"] = None
new_df["quality_weight_pkd"] = new_records["quality_weight"].fillna(1.0).values
new_df["match_level"] = "L0_bindingdb_perfect"
new_df["pkd_confidence"] = new_records["measurement_type"].map(
    {"Kd": "high", "Ki": "medium"}
).fillna("medium").values
new_df["source_db_pkd"] = new_records["source_db"].values
new_df["matched_inchikey"] = new_records["ligand_inchikey"].values
new_df["pkd_valid"] = True  # 混用 bool
new_df["data_source"] = "BindingDB"
new_df["has_pkd"] = True

# 来源追踪
new_df["kcat_per_s_source"] = "BindingDB_protein_level"
new_df["km_M_source"] = "BindingDB_protein_level"
new_df["pkd_value_source"] = "BindingDB"

print(f"新记录数: {len(new_df):,}")
print(f"有蛋白序列: {new_df['sequence'].notna().sum():,}")
print(f"有配体SMILES: {new_df['smiles'].notna().sum():,}")
print(f"有EC号: {new_df['ec_number'].notna().sum():,}")

# ── Step 7: Merge with V4 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7: 合并到 V4 → V5")
print("=" * 60)

# 确保列对齐
v4_cols = v4.columns.tolist()
for col in v4_cols:
    if col not in new_df.columns:
        new_df[col] = None

# 只保留 V4 的列（按顺序）
new_df = new_df[v4_cols]

# 类型对齐
for col in new_df.columns:
    if col in v4.columns:
        v4_dtype = v4[col].dtype
        new_dtype = new_df[col].dtype
        if v4_dtype != new_dtype:
            try:
                # 尝试转换到V4类型
                if str(v4_dtype) == "Int64":
                    new_df[col] = pd.to_numeric(new_df[col], errors="coerce").astype("Int64")
                elif str(v4_dtype) == "boolean" or str(v4_dtype) == "bool":
                    new_df[col] = new_df[col].astype(bool)
                elif str(v4_dtype) == "string":
                    new_df[col] = new_df[col].astype(str).replace({"nan": None, "None": None})
                else:
                    new_df[col] = new_df[col].astype(v4_dtype)
            except Exception as e:
                print(f"  警告: 列 '{col}' 类型转换失败 ({new_dtype} → {v4_dtype}): {e}")

# 合并
v5 = pd.concat([v4, new_df], ignore_index=True)
print(f"合并前 V4: {len(v4):,} 条")
print(f"新增完美数据: {len(new_df):,} 条")
print(f"合并后 V5: {len(v5):,} 条")

# ── Step 8: 验证 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 8: 验证")
print("=" * 60)

# 完美数据计数
perf_v5 = v5[
    v5["kcat_per_s"].notna() & v5["km_M"].notna() & v5["pkd_value"].notna()
]
print(f"V5 中完美数据: {len(perf_v5):,} 条 (V4原有: 61,991)")

# 检查无重复
dup_pairs = v5.groupby(["protein_seq_hash", "ligand_inchikey"]).size()
dup_count = (dup_pairs > 1).sum()
print(f"重复 (protein, ligand) 对: {dup_count}")

# 数据来源分布
print(f"\n数据来源分布:")
print(v5["data_source"].value_counts().to_string())

print(f"\n完美数据来源:")
perf_sources = perf_v5["data_source"].value_counts()
print(perf_sources.to_string())

# ── Step 9: Save ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 9: 保存")
print("=" * 60)

output_path = RELEASE_DIR / "trenzition_full_v5.parquet"
v5.to_parquet(output_path, index=False)
print(f"已保存: {output_path}")
print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# ── Step 10: Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("补充摘要")
print("=" * 60)
print(f"""
原始完美数据:                5,307 条
已在 V4 中:                  {n_overlap} 条
新增到 V5:                   {len(new_df):,} 条
  - 有蛋白序列:              {new_df['sequence'].notna().sum():,}
  - 有配体 SMILES:           {new_df['smiles'].notna().sum():,}
  - 有 EC 号:                {new_df['ec_number'].notna().sum():,}
  - Ki 测量:                 {(new_df['measurement_type_pkd'] == 'Ki').sum():,}
  - Kd 测量:                 {(new_df['measurement_type_pkd'] == 'Kd').sum():,}

V5 总记录:                   {len(v5):,} 条
V5 完美数据 (三标签):         {len(perf_v5):,} 条
""")

# 生成补充统计
supplement_info = {
    "supplement_version": "v4_to_v5",
    "date": "2026-06-09",
    "description": "从BindingDB原始数据集补充含kcat+Km+pKd的完美数据到Trenzition V4",
    "original_perfect_count": 5307,
    "already_in_v4": int(n_overlap),
    "newly_added": len(new_df),
    "new_unique_proteins": int(new_records["uniprot_id"].nunique()),
    "new_unique_ligands": int(new_records["ligand_inchikey"].nunique()),
    "source": "BindingDB (protein-level kinetics)",
    "v5_total": len(v5),
    "v5_perfect_count": len(perf_v5),
    "column_mapping": {
        "kcat_median_s": "kcat_per_s",
        "bdb_km_median_uM": "km_M (converted uM→M)",
        "pkd_aligned": "pkd_value",
        "measurement_type": "measurement_type_pkd",
    },
}

with open(RELEASE_DIR / "trenzition_supplement_stats_v5.json", "w") as f:
    import json

    json.dump(supplement_info, f, indent=2, ensure_ascii=False)
print(f"统计已保存: trenzition_supplement_stats_v5.json")
