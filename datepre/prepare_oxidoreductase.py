"""
prepare_oxidoreductase.py
=========================
从已有的 PDBbind + BindingDB 已处理数据中，筛选氧化还原酶 (EC 1.x.x.x)，
提取辅因子信息，为 Marcus 理论约束的 GNN 模型准备训练数据。

工作流程：
  1. 加载 processed/*.pkl → 获取所有唯一 UniProt ID & PDB ID
  2. 批量查询 UniProt API → EC number + 辅因子注释
  3. 筛选 EC 1.x.x.x 氧化还原酶
  4. 组装特征：蛋白序列、配体、辅因子类型、结合位点、接触图
  5. 输出：oxidoreductase_dataset.parquet + 统计报告

输出目录：processed/oxidoreductase/
"""

import os
import re
import json
import time
import hashlib
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# 已知氧化还原酶辅因子（UniProt 注释名称 → 编码）
# 匹配时使用小写关键词，匹配对象为 UniProt COFACTOR name 字段和 reaction 字符串
COFACTOR_MAP = {
    # 吡啶核苷酸
    "nad": "NAD",
    "nadh": "NAD",
    "nadph": "NADP",
    "nadp": "NADP",
    # 黄素
    "fad": "FAD",
    "fmn": "FMN",
    "flavin": "FAD",
    "riboflavin": "FMN",
    # 血红素
    "heme": "HEME",
    "protoporphyrin": "HEME",
    "cytochrome": "HEME",
    # 铁硫簇
    "2fe-2s": "FES",
    "[2fe-2s]": "FES",
    "4fe-4s": "FES",
    "[4fe-4s]": "FES",
    "3fe-4s": "FES",
    "iron-sulfur": "FES",
    "fe-s": "FES",
    "fes": "FES",
    # 铜
    "copper": "CU",
    "cu2+": "CU",
    "cu+": "CU",
    "cu(ii)": "CU",
    "cu(i)": "CU",
    # 醌
    "ubiquinone": "COQ",
    "coenzyme q": "COQ",
    "menaquinone": "COQ",
    "quinone": "COQ",
    "plastoquinone": "COQ",
    # 钼
    "molybdopterin": "MPT",
    "molybdenum": "MPT",
    "mo-molybdopterin": "MPT",
    # 吡咯喹啉醌
    "pqq": "PQQ",
    "pyrroloquinoline": "PQQ",
    # 辅酶A
    "coenzyme a": "COA",
    "coa": "COA",
    # 硫胺素
    "thiamine": "TPP",
    "tpp": "TPP",
    # 吡哆醛
    "pyridoxal": "PLP",
    "plp": "PLP",
    # 生物素
    "biotin": "BIO",
    # 钴胺素
    "cobalamin": "B12",
    "b12": "B12",
    # 四氢叶酸
    "tetrahydrofolate": "THF",
    # 谷胱甘肽 (非经典辅因子但在氧化还原中很重要)
    "glutathione": "GSH",
    # 硫辛酸
    "lipoate": "LIP",
    "lipoamide": "LIP",
}

COFACTOR_CATEGORIES = sorted(set(COFACTOR_MAP.values()))
log.info(f"Cofactor categories: {COFACTOR_CATEGORIES}")

# UniProt API
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
UNIPROT_BATCH_SIZE = 500
UNIPROT_RETRIES = 3
UNIPROT_TIMEOUT = 30

# Output
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
OUT_DIR = PROCESSED_DIR / "oxidoreductase"


# ─────────────────────────────────────────────────────────────
# 1. 加载已处理数据 & 提取唯一蛋白质标识
# ─────────────────────────────────────────────────────────────

def load_records(source: str) -> list[dict]:
    """加载 processed/*.pkl"""
    path = PROCESSED_DIR / f"{source}_records.pkl"
    log.info(f"Loading {path} ...")
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_unique_proteins(
    bindingdb_records: list[dict],
    pdbbind_records: list[dict],
) -> tuple[dict, dict]:
    """
    从两个数据库中提取唯一蛋白质。

    Returns:
      (uniprot_id → info_dict, pdb_id → chain_info_dict)
      BindingDB: 主要用 UniProt ID
      PDBbind: 仅有 PDB ID，需额外映射到 UniProt
    """
    uniprot_proteins: dict[str, dict] = {}
    pdb_proteins: dict[str, dict] = {}

    # BindingDB: grouped by UniProt ID
    for r in bindingdb_records:
        uid = r.get("uniprot_id")
        if not uid:
            continue
        if uid not in uniprot_proteins:
            uniprot_proteins[uid] = {
                "source": "BindingDB",
                "seq_hash": r["protein_seq_hash"],
                "sequence": r["sequence"],
                "n_records": 0,
                "has_structure": False,
            }
        uniprot_proteins[uid]["n_records"] += 1

    # PDBbind: grouped by PDB ID
    for r in pdbbind_records:
        pid = r["pdb_id"]
        if not pid:
            continue
        if pid not in pdb_proteins:
            pdb_proteins[pid] = {
                "source": "PDBbind",
                "chain": r.get("selected_chain_id", "A"),
                "seq_hash": r["protein_seq_hash"],
                "sequence": r["sequence"],
                "n_records": 0,
                "has_structure": r.get("has_structure", False),
            }
        pdb_proteins[pid]["n_records"] += 1

    log.info(f"Unique UniProt IDs (BindingDB): {len(uniprot_proteins)}")
    log.info(f"Unique PDB IDs   (PDBbind):   {len(pdb_proteins)}")
    return uniprot_proteins, pdb_proteins


# ─────────────────────────────────────────────────────────────
# 2. UniProt API 批量查询 → EC number + 辅因子
# ─────────────────────────────────────────────────────────────

def batch_fetch_uniprot(
    uniprot_ids: list[str],
    cache_path: Optional[Path] = None,
) -> dict[str, dict]:
    """
    批量查询 UniProt REST API，获取 EC number 与辅因子注释。

    使用 POST /uniprotkb/accessions 接口，每次最多 UNIPROT_BATCH_SIZE 个。
    结果缓存到磁盘以便断点续跑。

    Returns: {uniprot_id: {"ec_numbers": [...], "cofactors": [...], "protein_name": str}}
    """
    # Try loading from cache
    cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        log.info(f"Loaded {len(cache)} cached UniProt annotations from {cache_path}")

    to_fetch = [uid for uid in uniprot_ids if uid not in cache]
    if not to_fetch:
        log.info("All UniProt IDs already cached.")
        return cache

    log.info(f"Fetching UniProt annotations for {len(to_fetch)} IDs "
             f"(batches of {UNIPROT_BATCH_SIZE})...")

    session = requests.Session()
    session.headers.update({"User-Agent": "AI4enz/1.0"})

    failures = 0
    with tqdm(total=len(to_fetch), desc="UniProt batch fetch") as pbar:
        for i in range(0, len(to_fetch), UNIPROT_BATCH_SIZE):
            batch = to_fetch[i : i + UNIPROT_BATCH_SIZE]
            url = f"{UNIPROT_BASE}/accessions"

            for attempt in range(UNIPROT_RETRIES):
                try:
                    resp = session.get(
                        url,
                        params={"accessions": ",".join(batch)},
                        timeout=UNIPROT_TIMEOUT,
                    )
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        for entry in results:
                            uid = entry.get("primaryAccession", "")
                            annotations = _parse_uniprot_entry(entry)
                            cache[uid] = annotations
                        break
                    elif resp.status_code == 429:
                        time.sleep(2 ** attempt)
                    else:
                        log.warning(f"HTTP {resp.status_code} for batch {i}")
                        time.sleep(2 ** attempt)
                except requests.exceptions.RequestException as e:
                    if attempt == UNIPROT_RETRIES - 1:
                        log.error(f"Failed batch {i}: {e}")
                        failures += len(batch)
                    time.sleep(2 ** attempt)

            pbar.update(len(batch))

            # Periodic cache save
            if cache_path and i % (UNIPROT_BATCH_SIZE * 10) == 0:
                os.makedirs(cache_path.parent, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache, f)

            # Rate limit
            time.sleep(0.15)

    if cache_path:
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        log.info(f"Saved UniProt cache → {cache_path}")

    log.info(f"UniProt fetch done: {len(cache)} total, {failures} failures")
    return cache


def _parse_uniprot_entry(entry: dict) -> dict:
    """解析单个 UniProt JSON 条目 → EC + cofactors."""
    result = {
        "ec_numbers": [],
        "cofactors": [],
        "protein_name": "",
        "reviewed": False,
    }

    # Reviewed vs unreviewed
    result["reviewed"] = entry.get("entryType", "").startswith("Swiss-Prot")

    # Protein name
    pd_ = entry.get("proteinDescription", {})
    rec = pd_.get("recommendedName", {}) or pd_.get("submissionNames", [{}])[0]
    result["protein_name"] = rec.get("fullName", {}).get("value", "")

    # EC numbers
    ec_list = rec.get("ecNumbers", [])
    result["ec_numbers"] = [e.get("value", "") for e in ec_list]

    # Also check comments for catalytic activity EC
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "CATALYTIC_ACTIVITY":
            reaction = comment.get("reaction", {})
            ecs = reaction.get("ecNumbers", [])
            for e in ecs:
                ec_val = e.get("value", "")
                if ec_val and ec_val not in result["ec_numbers"]:
                    result["ec_numbers"].append(ec_val)

    # Cofactors — source 1: structured COFACTOR comments
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "COFACTOR":
            for cof in comment.get("cofactors", []):
                name = cof.get("name", "")
                if name:
                    mapped = _classify_cofactor(name)
                    if mapped and mapped not in result["cofactors"]:
                        result["cofactors"].append(mapped)

    # Cofactors — source 2: CATALYTIC_ACTIVITY reaction strings
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "CATALYTIC_ACTIVITY":
            rxn_name = comment.get("reaction", {}).get("name", "")
            for keyword, category in COFACTOR_MAP.items():
                if keyword.lower() in rxn_name.lower():
                    if category not in result["cofactors"]:
                        result["cofactors"].append(category)

    # Cofactors — source 3: protein name (many NAD/NADP enzymes have cofactor in name)
    pn_lower = result["protein_name"].lower()
    if re.search(r'\[nad[ph]?', pn_lower) or re.search(r'\(nad[ph]?', pn_lower):
        if 'nadp' in pn_lower:
            if 'NADP' not in result["cofactors"]:
                result["cofactors"].append('NADP')
        else:
            if 'NAD' not in result["cofactors"]:
                result["cofactors"].append('NAD')
    if re.search(r'\bfad\b', pn_lower) and 'FAD' not in result["cofactors"]:
        result["cofactors"].append('FAD')
    if re.search(r'\bfmn\b', pn_lower) and 'FMN' not in result["cofactors"]:
        result["cofactors"].append('FMN')

    # Also check features for cofactor binding
    for feat in entry.get("features", []):
        if feat.get("type") in ("Binding site", "binding site"):
            desc = feat.get("description", "")
            mapped = _classify_cofactor(desc)
            if mapped and mapped not in result["cofactors"]:
                result["cofactors"].append(mapped)

    return result


def _classify_cofactor(description: str) -> Optional[str]:
    """根据 UniProt 注释文本匹配辅因子类型。"""
    desc_lower = description.lower()
    for keyword, category in COFACTOR_MAP.items():
        if keyword.lower() in desc_lower:
            return category
    return None


# ─────────────────────────────────────────────────────────────
# 3. PDB → UniProt 映射 (via PDBe SIFTS)
# ─────────────────────────────────────────────────────────────

def map_pdb_to_uniprot(
    pdb_ids: list[str],
    cache_path: Optional[Path] = None,
) -> dict[str, list[str]]:
    """
    通过 PDBe SIFTS 批量文件将 PDB ID 映射到 UniProt accession。

    下载 pdb_chain_uniprot.tsv.gz (~5.9 MB)，本地解析，高效处理大量 PDB ID。
    结果缓存为 JSON。

    Returns: {pdb_id: [uniprot_id, ...]}
    """
    cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        log.info(f"Loaded {len(cache)} cached PDB→UniProt mappings")

    to_fetch = [pid for pid in pdb_ids if pid not in cache]
    if not to_fetch:
        log.info("All PDB→UniProt already cached.")
        return cache

    log.info(f"Mapping {len(to_fetch)} PDB IDs → UniProt via SIFTS bulk file...")

    # Download SIFTS file (cached locally)
    sifts_path = cache_path.parent / "pdb_chain_uniprot.tsv.gz" if cache_path else None
    if sifts_path and not sifts_path.exists():
        log.info("  Downloading SIFTS pdb_chain_uniprot.tsv.gz (~5.9 MB)...")
        sifts_url = (
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/"
            "flatfiles/tsv/pdb_chain_uniprot.tsv.gz"
        )
        resp = requests.get(sifts_url, timeout=60)
        resp.raise_for_status()
        sifts_path.write_bytes(resp.content)
        log.info(f"  Saved → {sifts_path}")

    # Parse SIFTS file: build PDB→UniProt mapping
    import gzip

    target_set = set(to_fetch)
    pdb_to_uniprots: dict[str, set] = {}
    with gzip.open(sifts_path, "rt") as fh:
        for line in tqdm(fh, desc="  Parsing SIFTS", total=978_000):
            if line.startswith("#") or line.startswith("PDB\t"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pdb_id = parts[0].lower()
            if pdb_id in target_set:
                uniprot_id = parts[2]  # SP_PRIMARY
                if uniprot_id and uniprot_id != "-":
                    pdb_to_uniprots.setdefault(pdb_id, set()).add(uniprot_id)

    for pid in to_fetch:
        cache[pid] = sorted(pdb_to_uniprots.get(pid, set()))

    if cache_path:
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        log.info(f"Saved PDB→UniProt cache → {cache_path}")

    mapped = sum(1 for v in cache.values() if v)
    log.info(f"PDB→UniProt done: {mapped}/{len(to_fetch)} mapped to at least one UniProt")
    return cache


# ─────────────────────────────────────────────────────────────
# 4. 筛选氧化还原酶
# ─────────────────────────────────────────────────────────────

def is_oxidoreductase(ec_numbers: list[str]) -> bool:
    """检查是否有 EC 1.x.x.x (氧化还原酶)"""
    for ec in ec_numbers:
        ec_clean = ec.strip().replace(" ", "")
        if re.match(r"^1\.\d+\.\d+\.\d+$", ec_clean) or re.match(
            r"^1\.\d+\.\d+\.-?$", ec_clean
        ):
            return True
    return False


def filter_oxidoreductase_records(
    bindingdb_records: list[dict],
    pdbbind_records: list[dict],
    uniprot_annotations: dict[str, dict],
    pdb_uniprot_map: dict[str, list[str]],
) -> tuple[list[dict], pd.DataFrame]:
    """
    筛选 EC 1.x.x.x 氧化还原酶记录，附上辅因子信息。

    Returns:
      - 筛选后的记录列表 (与原始 record 格式兼容，增加 ec_numbers, cofactors 字段)
      - 统计 DataFrame
    """
    filtered = []

    # BindingDB
    for r in bindingdb_records:
        uid = r.get("uniprot_id")
        if not uid:
            continue
        anno = uniprot_annotations.get(uid, {})
        if is_oxidoreductase(anno.get("ec_numbers", [])):
            r_new = r.copy()
            r_new["ec_numbers"] = anno.get("ec_numbers", [])
            r_new["cofactors"] = anno.get("cofactors", [])
            r_new["protein_name"] = anno.get("protein_name", "")
            r_new["reviewed"] = anno.get("reviewed", False)
            filtered.append(r_new)

    log.info(f"BindingDB oxidoreductases: {len(filtered)}")

    # PDBbind
    pdbbind_filtered = []
    for r in pdbbind_records:
        pid = r["pdb_id"]
        if not pid:
            continue
        uniprots_for_pdb = pdb_uniprot_map.get(pid, [])
        ecs = []
        cofs = []
        for uid in uniprots_for_pdb:
            anno = uniprot_annotations.get(uid, {})
            ecs.extend(anno.get("ec_numbers", []))
            cofs.extend(anno.get("cofactors", []))
        # 去重
        ecs = list(dict.fromkeys(ecs))
        cofs = list(dict.fromkeys(cofs))

        if is_oxidoreductase(ecs):
            r_new = r.copy()
            r_new["ec_numbers"] = ecs
            r_new["cofactors"] = cofs
            r_new["uniprot_id"] = uniprots_for_pdb[0] if uniprots_for_pdb else None
            r_new["protein_name"] = ""
            r_new["reviewed"] = False
            pdbbind_filtered.append(r_new)

    log.info(f"PDBbind oxidoreductases:   {len(pdbbind_filtered)}")
    filtered.extend(pdbbind_filtered)
    log.info(f"Total oxidoreductase records: {len(filtered)}")

    # ── Statistics ──
    stats = _compute_statistics(filtered, uniprot_annotations)
    return filtered, stats


def _compute_statistics(
    records: list[dict],
    uniprot_annotations: dict,
) -> pd.DataFrame:
    """生成统计报告 DataFrame"""
    unique_seqs = set(r["protein_seq_hash"] for r in records)
    unique_ligands = set(r["ligand_inchikey"] for r in records if r.get("ligand_inchikey"))

    # EC distribution
    ec_count: dict[str, int] = defaultdict(int)
    cofactor_count: dict[str, int] = defaultdict(int)
    source_count = {"PDBbind": 0, "BindingDB": 0}
    mtype_count: dict[str, int] = defaultdict(int)

    for r in records:
        source_count[r.get("source_db", "Unknown")] += 1
        mtype_count[r.get("measurement_type", "Unknown")] += 1
        for ec in r.get("ec_numbers", []):
            # Aggregate to EC 1.x level (subclass)
            parts = ec.replace(" ", "").split(".")
            if len(parts) >= 2:
                ec_count[f"1.{parts[1]}"] += 1
        for cof in r.get("cofactors", []):
            cofactor_count[cof] += 1

    stats_rows = [
        {"category": "overview", "key": "total_records", "value": len(records)},
        {
            "category": "overview",
            "key": "unique_proteins",
            "value": len(unique_seqs),
        },
        {
            "category": "overview",
            "key": "unique_ligands",
            "value": len(unique_ligands),
        },
        {
            "category": "overview",
            "key": "with_structure",
            "value": sum(1 for r in records if r.get("has_structure")),
        },
        {
            "category": "overview",
            "key": "with_cofactor_annotation",
            "value": sum(1 for r in records if r.get("cofactors")),
        },
        {
            "category": "overview",
            "key": "reviewed_swissprot",
            "value": sum(1 for r in records if r.get("reviewed")),
        },
    ]

    for source, count in sorted(source_count.items()):
        stats_rows.append({"category": "source", "key": source, "value": count})

    for mtype, count in sorted(mtype_count.items()):
        stats_rows.append({"category": "measurement_type", "key": mtype, "value": count})

    for ec, count in sorted(ec_count.items(), key=lambda x: -x[1]):
        stats_rows.append({"category": "ec_subclass", "key": ec, "value": count})

    for cof, count in sorted(cofactor_count.items(), key=lambda x: -x[1]):
        stats_rows.append({"category": "cofactor", "key": cof, "value": count})

    stats_df = pd.DataFrame(stats_rows)
    return stats_df


# ─────────────────────────────────────────────────────────────
# 5. 保存最终数据集
# ─────────────────────────────────────────────────────────────

def save_dataset(
    records: list[dict],
    out_dir: Path,
    stats_df: pd.DataFrame,
):
    """保存筛选后的数据集成 Parquet + 统计报告。"""
    os.makedirs(out_dir, exist_ok=True)

    # ── 元数据 DataFrame ──
    meta_rows = []
    for i, r in enumerate(records):
        meta_rows.append(
            {
                "record_idx": i,
                "source_db": r.get("source_db", ""),
                "uniprot_id": r.get("uniprot_id"),
                "pdb_id": r.get("pdb_id"),
                "protein_seq_hash": r.get("protein_seq_hash", ""),
                "ligand_inchikey": r.get("ligand_inchikey"),
                "pkd_raw": r.get("pkd_raw"),
                "pkd_aligned": r.get("pkd_aligned", r.get("pkd_raw")),
                "measurement_type": r.get("measurement_type", ""),
                "quality_weight": r.get("quality_weight", 1.0),
                "is_censored": r.get("is_censored", False),
                "has_binding_site": r.get("has_binding_site", False),
                "has_structure": r.get("has_structure", False),
                "ec_numbers": "|".join(r.get("ec_numbers", [])),
                "cofactors": "|".join(r.get("cofactors", [])),
                "protein_name": r.get("protein_name", ""),
                "reviewed": r.get("reviewed", False),
                "n_measurements": r.get("n_measurements", 1),
                "pkd_std": r.get("pkd_std", 0.0),
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    meta_path = out_dir / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)
    log.info(f"Metadata saved → {meta_path} ({len(meta_df)} rows)")

    # ── 完整记录 Pickle（兼容现有格式） ──
    pkl_path = out_dir / "oxidoreductase_records.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    log.info(f"Records saved → {pkl_path}")

    # ── 统计报告 ──
    stats_path = out_dir / "statistics.parquet"
    stats_df.to_parquet(stats_path, index=False)
    log.info(f"Statistics saved → {stats_path}")

    # ── 文本摘要 ──
    summary = _generate_summary(stats_df, records)
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    log.info(f"Summary saved → {summary_path}")
    print("\n" + summary)


def _generate_summary(stats_df: pd.DataFrame, records: list[dict]) -> str:
    """生成可读的文本摘要。"""

    def _val(cat, key):
        row = stats_df[(stats_df["category"] == cat) & (stats_df["key"] == key)]
        return int(row["value"].values[0]) if len(row) else 0

    total = _val("overview", "total_records")
    n_prot = _val("overview", "unique_proteins")
    n_lig = _val("overview", "unique_ligands")
    n_struct = _val("overview", "with_structure")
    n_cof = _val("overview", "with_cofactor_annotation")
    n_reviewed = _val("overview", "reviewed_swissprot")

    lines = [
        "=" * 60,
        "  氧化还原酶数据集 — 数据准备报告",
        "=" * 60,
        "",
        f"  总记录数:          {total:>12,}",
        f"  唯一蛋白质:        {n_prot:>12,}",
        f"  唯一配体:          {n_lig:>12,}",
        f"  有结构特征 (PDB):  {n_struct:>12,}",
        f"  有辅因子注释:      {n_cof:>12,}",
        f"  Swiss-Prot 已审阅: {n_reviewed:>12,}",
        "",
        "─" * 40,
        "  数据来源分布",
        "─" * 40,
    ]

    src = stats_df[stats_df["category"] == "source"]
    for _, row in src.iterrows():
        lines.append(f"    {row['key']:<20s} {int(row['value']):>10,}")

    lines += [
        "",
        "─" * 40,
        "  测量类型分布",
        "─" * 40,
    ]

    mt = stats_df[stats_df["category"] == "measurement_type"]
    for _, row in mt.iterrows():
        lines.append(f"    {row['key']:<10s} {int(row['value']):>10,}")

    lines += [
        "",
        "─" * 40,
        "  EC 子类分布 (EC 1.x.x.x)",
        "─" * 40,
    ]

    ec = stats_df[stats_df["category"] == "ec_subclass"]
    for _, row in ec.iterrows():
        lines.append(f"    {row['key']:<12s} {int(row['value']):>10,}")

    lines += [
        "",
        "─" * 40,
        "  辅因子分布",
        "─" * 40,
    ]

    cf = stats_df[stats_df["category"] == "cofactor"]
    for _, row in cf.iterrows():
        lines.append(f"    {row['key']:<10s} {int(row['value']):>10,}")

    lines += [
        "",
        "─" * 40,
        "  关键设计说明",
        "─" * 40,
        "  1. EC 分类来自 UniProt REST API",
        "  2. 辅因子分类基于 UniProt COFACTOR 注释 + 关键词匹配",
        "  3. PDBbind PDB ID → UniProt 映射来自 PDBe SIFTS",
        "  4. 有结构的蛋白质 (has_structure=True) 可用于 Marcus 距离约束",
        "  5. 无辅因子注释不代表没有辅因子——可能是 UniProt 注释不完整",
        "  6. 该数据集可作为 GNN 模型训练输入，辅因子编码为类别特征",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 6. 主入口
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从已处理数据中筛选氧化还原酶并提取辅因子信息"
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip UniProt API calls (use cache only)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Clear cache and re-fetch all UniProt annotations",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--max-bindingdb",
        type=int,
        default=None,
        help="Max BindingDB records to load (for testing)",
    )
    parser.add_argument(
        "--max-pdbbind",
        type=int,
        default=None,
        help="Max PDBbind records to load (for testing)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    os.makedirs(cache_dir, exist_ok=True)

    uniprot_cache_path = cache_dir / "uniprot_annotations.json"
    pdb_cache_path = cache_dir / "pdb_uniprot_map.json"

    if args.force_refresh:
        for p in [uniprot_cache_path, pdb_cache_path]:
            if p.exists():
                p.unlink()

    # ── Step 1: Load records ──
    log.info("=" * 50)
    log.info("Step 1: Loading processed records")
    log.info("=" * 50)

    bindingdb_records = load_records("bindingdb")
    pdbbind_records = load_records("pdbbind")

    if args.max_bindingdb:
        bindingdb_records = bindingdb_records[: args.max_bindingdb]
        log.info(f"  Capped BindingDB to {args.max_bindingdb}")

    if args.max_pdbbind:
        pdbbind_records = pdbbind_records[: args.max_pdbbind]
        log.info(f"  Capped PDBbind to {args.max_pdbbind}")

    # ── Step 2: Extract unique proteins ──
    log.info("=" * 50)
    log.info("Step 2: Extracting unique protein identifiers")
    log.info("=" * 50)

    uniprot_proteins, pdb_proteins = extract_unique_proteins(
        bindingdb_records, pdbbind_records
    )

    all_bindingdb_uniprot_ids = sorted(set(uniprot_proteins.keys()))
    all_pdb_ids = sorted(set(pdb_proteins.keys()))

    # ── Step 3: PDB → UniProt mapping (must run BEFORE UniProt fetch) ──
    log.info("=" * 50)
    log.info("Step 3: Mapping PDB IDs to UniProt (PDBe SIFTS)")
    log.info("=" * 50)

    if not args.no_fetch:
        pdb_uniprot_map = map_pdb_to_uniprot(all_pdb_ids, cache_path=pdb_cache_path)
    else:
        if pdb_cache_path.exists():
            with open(pdb_cache_path) as f:
                pdb_uniprot_map = json.load(f)
            log.info(f"Loaded {len(pdb_uniprot_map)} cached PDB→UniProt (--no-fetch)")
        else:
            log.warning("No PDB cache found. PDBbind oxidoreductases may be missed.")
            pdb_uniprot_map = {}

    # Collect ALL UniProt IDs that need annotations
    # (BindingDB direct + PDB-mapped)
    pdb_uniprot_ids = set()
    for pid, uids in pdb_uniprot_map.items():
        pdb_uniprot_ids.update(uids)
    all_uniprot_ids = sorted(set(all_bindingdb_uniprot_ids) | pdb_uniprot_ids)
    log.info(
        f"Total unique UniProt IDs to annotate: {len(all_uniprot_ids)} "
        f"(BindingDB: {len(all_bindingdb_uniprot_ids)}, "
        f"PDB-mapped: {len(pdb_uniprot_ids)})"
    )

    # ── Step 4: Batch UniProt API query ──
    log.info("=" * 50)
    log.info("Step 4: Fetching UniProt annotations (EC + cofactors)")
    log.info("=" * 50)

    if not args.no_fetch:
        uniprot_annotations = batch_fetch_uniprot(
            all_uniprot_ids, cache_path=uniprot_cache_path
        )
    else:
        if uniprot_cache_path.exists():
            with open(uniprot_cache_path) as f:
                uniprot_annotations = json.load(f)
            log.info(
                f"Loaded {len(uniprot_annotations)} cached annotations (--no-fetch)"
            )
        else:
            log.error("No cache found and --no-fetch set. Exiting.")
            return

    # Log a few examples
    oxidoreductase_ids = [
        uid
        for uid, anno in uniprot_annotations.items()
        if is_oxidoreductase(anno.get("ec_numbers", []))
    ]
    log.info(
        f"  {len(oxidoreductase_ids)}/{len(all_uniprot_ids)} "
        f"UniProt IDs are oxidoreductases (EC 1.x.x.x)"
    )

    # Print some examples
    for uid in oxidoreductase_ids[:5]:
        anno = uniprot_annotations[uid]
        log.info(
            f"  Example: {uid} | EC: {anno['ec_numbers']} "
            f"| Cofactors: {anno['cofactors']} | {anno['protein_name'][:60]}"
        )

    # ── Step 5: Filter oxidoreductases ──
    log.info("=" * 50)
    log.info("Step 5: Filtering oxidoreductase records")
    log.info("=" * 50)

    filtered_records, stats_df = filter_oxidoreductase_records(
        bindingdb_records,
        pdbbind_records,
        uniprot_annotations,
        pdb_uniprot_map,
    )

    # ── Step 6: Save ──
    log.info("=" * 50)
    log.info("Step 6: Saving dataset")
    log.info("=" * 50)

    save_dataset(filtered_records, out_dir, stats_df)


if __name__ == "__main__":
    main()
