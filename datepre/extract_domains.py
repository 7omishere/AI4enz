"""
extract_domains.py
==================
从 UniProt 提取辅因子结合域的序列位置（start/end），构建 per-protein domain_masks。

数据来源：UniProt REST API 完整 JSON
  - features[type=Domain]:      域名称 + 起止位置
  - uniProtKBCrossReferences:   InterPro/Pfam ID + EntryName
  - 两者通过描述文本相似度匹配

工作流程：
  1. 加载氧化还原酶唯一 UniProt ID 列表
  2. 获取每个蛋白的完整 UniProt JSON（利用已有缓存 + 增量获取）
  3. 解析 domain features → 提取位置
  4. 匹配 InterPro/Pfam cross-references → 映射到辅因子类型
  5. 构建 domain_masks (n_cofactor_types, L) 存储到 proteins.h5
  6. 更新 oxidoreductase metadata 添加 domain 列

输出：
  - proteins.h5 每个 seq_hash group 新增 domain_masks, domain_positions
  - processed/oxidoreductase/metadata.parquet 更新：domains_json, n_domains, has_domain_annotation

用法：
  python datepre/extract_domains.py --max-proteins 20                # 测试
  python datepre/extract_domains.py --workers 8                       # 完整运行
  python datepre/extract_domains.py --no-fetch                        # 仅用缓存
"""

import os
import re
import json
import time
import h5py
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional, Union
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

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
UNIPROT_TIMEOUT = 30
UNIPROT_RETRIES = 3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
OUT_DIR = PROCESSED_DIR / "oxidoreductase"
H5_PATH = PROCESSED_DIR / "proteins.h5"

# ── InterPro/Pfam ID → 辅因子类型映射 ──
DOMAIN_COFACTOR_MAP: dict[str, str] = {
    # NAD(P)-binding Rossmann fold
    "IPR016040": "NAD",
    "IPR036291": "NAD",
    "PF13460": "NAD",
    "PF00175": "NAD",
    "PF03446": "NAD",
    "PF07992": "NAD",
    "PF14833": "NAD",
    # FAD-binding
    "IPR003097": "FAD",
    "IPR003953": "FAD",
    "IPR002938": "FAD",
    "IPR002346": "FAD",
    "IPR036188": "FAD",
    "PF00667": "FAD",
    "PF00890": "FAD",
    "PF01494": "FAD",
    "PF00941": "FAD",
    # FMN-binding
    "IPR008254": "FMN",
    "IPR001094": "FMN",
    "IPR029039": "FMN",
    "PF00258": "FMN",
    "PF03350": "FMN",
    "PF02441": "FMN",
    # Heme-binding (Cytochrome P450 + others)
    "IPR001128": "HEME",
    "IPR036396": "HEME",
    "IPR017972": "HEME",
    "IPR002016": "HEME",
    "IPR010255": "HEME",
    "IPR000883": "HEME",
    "PF00067": "HEME",
    # Iron-Sulfur (2Fe-2S)
    "IPR001041": "FES",
    "IPR036010": "FES",
    "PF00111": "FES",
    "PF13510": "FES",
    # Iron-Sulfur (4Fe-4S)
    "IPR017900": "FES",
    "IPR017896": "FES",
    "PF00037": "FES",
    "PF13247": "FES",
    "PF00355": "FES",
    # Copper
    "IPR001117": "CU",
    "IPR011707": "CU",
    "IPR011706": "CU",
    "IPR002355": "CU",
    "IPR008972": "CU",
    "PF00394": "CU",
    "PF07731": "CU",
    "PF07732": "CU",
    # Molybdopterin
    "IPR001453": "MPT",
    "IPR000572": "MPT",
    "IPR006656": "MPT",
    "IPR006657": "MPT",
    "PF00994": "MPT",
    "PF00174": "MPT",
    "PF00384": "MPT",
    "PF01568": "MPT",
    # Coenzyme Q / Quinone
    "IPR000440": "COQ",
    "IPR001135": "COQ",
    "PF00507": "COQ",
    "PF00346": "COQ",
    "PF21162": "COQ",
    # PQQ
    "IPR002372": "PQQ",
    "IPR018391": "PQQ",
    "IPR011047": "PQQ",
    "IPR019551": "PQQ",
    "IPR019556": "PQQ",
    "PF01011": "PQQ",
    "PF13360": "PQQ",
    "PF10527": "PQQ",
    "PF10535": "PQQ",
    # TPP
    "IPR012001": "TPP",
    "IPR012000": "TPP",
    "IPR011766": "TPP",
    "IPR029061": "TPP",
    "PF02776": "TPP",
    "PF00205": "TPP",
    "PF02775": "TPP",
    # PLP
    "IPR004839": "PLP",
    "IPR004838": "PLP",
    "IPR015424": "PLP",
    "IPR015421": "PLP",
    "IPR000192": "PLP",
    "PF00155": "PLP",
    "PF00202": "PLP",
    "PF00266": "PLP",
    # CoA
    "IPR003781": "COA",
    "IPR005811": "COA",
    "IPR004165": "COA",
    "PF02629": "COA",
    "PF13380": "COA",
    "PF00549": "COA",
    "PF01144": "COA",
}

# 辅因子类别排序（domain_masks 维度 0..14）
COFACTOR_INDEX: dict[str, int] = {
    "NAD": 0, "NADP": 1, "FAD": 2, "FMN": 3, "HEME": 4,
    "FES": 5, "CU": 6, "MPT": 7, "COQ": 8, "PQQ": 9,
    "TPP": 10, "PLP": 11, "COA": 12, "B12": 13, "THF": 14,
}
N_COFACTOR_TYPES = len(COFACTOR_INDEX)


# ═══════════════════════════════════════════════════════════════
# 1. UniProt 数据获取
# ═══════════════════════════════════════════════════════════════

def fetch_uniprot_full(uniprot_id: str) -> Optional[dict]:
    """获取完整 UniProt JSON 条目。"""
    url = f"{UNIPROT_BASE}/{uniprot_id}.json"
    session = requests.Session()
    session.headers.update({"User-Agent": "AI4enz/1.0"})
    for attempt in range(UNIPROT_RETRIES):
        try:
            resp = session.get(url, timeout=UNIPROT_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def batch_fetch_uniprot_full(
    uniprot_ids: list[str],
    cache_path: Path,
    n_workers: int = 8,
) -> dict[str, dict]:
    """批量获取完整 UniProt JSON，缓存到磁盘。"""
    cache: dict = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        log.info(f"Loaded {len(cache)} cached full UniProt entries")

    to_fetch = sorted(set(uniprot_ids) - set(cache.keys()))
    if not to_fetch:
        log.info("All UniProt entries already cached.")
        return cache

    log.info(f"Fetching {len(to_fetch)} full UniProt entries ({n_workers} workers)...")

    def _fetch_one(uid):
        result = fetch_uniprot_full(uid)
        return uid, result

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fetch_one, uid): uid for uid in to_fetch}
        with tqdm(total=len(to_fetch), desc="Fetching UniProt") as pbar:
            for future in as_completed(futures):
                uid, result = future.result()
                if result:
                    cache[uid] = result
                else:
                    cache[uid] = {}
                pbar.update(1)

                # 每 500 条保存
                if len(cache) % 500 == 0:
                    os.makedirs(cache_path.parent, exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump(cache, f)

    os.makedirs(cache_path.parent, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    n_success = sum(1 for v in cache.values() if v and v.get("primaryAccession"))
    log.info(f"Fetched: {n_success} valid / {len(to_fetch)} total")
    return cache


# ═══════════════════════════════════════════════════════════════
# 2. Domain 解析
# ═══════════════════════════════════════════════════════════════

def _normalize(text: Optional[str]) -> str:
    """标准化文本用于模糊匹配。"""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _parse_domain_entry(feature: dict, interpro_entries: dict[str, str],
                        pfam_entries: dict[str, str]) -> dict:
    """
    解析单个 Domain feature，匹配到 InterPro/Pfam ID 和辅因子类型。

    Args:
      feature: UniProt feature dict
      interpro_entries: {entry_name_normalized: interpro_id}
      pfam_entries: {entry_name_normalized: pfam_id}

    Returns:
      {cofactor_type, start, end, description, interpro_id, pfam_id}
    """
    desc = feature.get("description", "")
    loc = feature.get("location", {})
    start = loc.get("start", {}).get("value", 0)
    end = loc.get("end", {}).get("value", 0)

    result = {
        "cofactor_type": None,
        "start": int(start),
        "end": int(end),
        "description": desc,
        "interpro_id": "",
        "pfam_id": "",
    }

    if not desc:
        return result

    # 尝试从描述匹配 InterPro entry name
    desc_norm = _normalize(desc)

    # 首先尝试精确匹配
    for entry_name_norm, ipr_id in interpro_entries.items():
        if desc_norm == entry_name_norm or desc_norm in entry_name_norm or entry_name_norm in desc_norm:
            result["interpro_id"] = ipr_id
            if ipr_id in DOMAIN_COFACTOR_MAP:
                result["cofactor_type"] = DOMAIN_COFACTOR_MAP[ipr_id]
            break

    # 如果没找到 InterPro，尝试 Pfam
    if not result["interpro_id"]:
        for entry_name_norm, pf_id in pfam_entries.items():
            if desc_norm == entry_name_norm or desc_norm in entry_name_norm or entry_name_norm in desc_norm:
                result["pfam_id"] = pf_id
                if pf_id in DOMAIN_COFACTOR_MAP:
                    result["cofactor_type"] = DOMAIN_COFACTOR_MAP[pf_id]
                break

    # 如果仍然没有匹配，用关键词直接匹配描述
    if result["cofactor_type"] is None:
        result["cofactor_type"] = _classify_by_description(desc)

    return result


def _classify_by_description(description: str) -> Optional[str]:
    """基于域名描述的关键词匹配辅因子类型。"""
    desc_lower = description.lower()

    # 关键词规则
    rules = [
        (["nad", "nadp", "nadph", "rossmann", "nad_binding"], "NAD"),
        (["fad", "fad_binding", "fad-binding"], "FAD"),
        (["fmn", "flavodoxin", "flavoprotein"], "FMN"),
        (["heme", "cytochrome", "p450", "haem", "peroxidase"], "HEME"),
        (["2fe-2s", "4fe-4s", "ferredoxin", "iron-sulfur", "iron-sulphur",
          "rieske", "fes", "fe-s"], "FES"),
        (["copper", "cupredoxin", "cu-oxidase", "multicopper"], "CU"),
        (["molybdopterin", "molybdenum", "mpt"], "MPT"),
        (["quinone", "ubiquinone", "menaquinone", "coq", "nad(p)h dehydrogenase"], "COQ"),
        (["pqq", "pyrroloquinoline", "quinoprotein"], "PQQ"),
        (["thiamine", "thiamin", "tpp"], "TPP"),
        (["pyridoxal", "plp", "aminotransferase"], "PLP"),
        (["coenzyme a", "coa", "acetyl-coa", "succinyl-coa"], "COA"),
        (["cobalamin", "b12", "cobalt"], "B12"),
    ]

    for keywords, cofactor_type in rules:
        for kw in keywords:
            if kw in desc_lower:
                return cofactor_type

    return None


def extract_domains(uniprot_entry: dict) -> list[dict]:
    """
    从 UniProt 完整 JSON 条目提取域注释（类型 + 位置 + 辅因子映射）。
    """
    # 解析 InterPro/Pfam 交叉引用
    interpro_entries: dict[str, str] = {}  # normalized_name → ID
    pfam_entries: dict[str, str] = {}
    for xref in uniprot_entry.get("uniProtKBCrossReferences", []):
        db = xref.get("database", "")
        xref_id = xref.get("id", "")
        props = xref.get("properties", [])
        entry_name = ""
        for p in props:
            if p.get("key") == "EntryName":
                entry_name = p.get("value", "")
                break
        if db == "InterPro" and entry_name:
            interpro_entries[_normalize(entry_name)] = xref_id
        elif db == "Pfam" and entry_name:
            pfam_entries[_normalize(entry_name)] = xref_id

    # 解析 Domain features
    domains = []
    for feat in uniprot_entry.get("features", []):
        if feat.get("type") == "Domain":
            parsed = _parse_domain_entry(feat, interpro_entries, pfam_entries)
            domains.append(parsed)

    return domains


# ═══════════════════════════════════════════════════════════════
# 3. 构建 domain_masks
# ═══════════════════════════════════════════════════════════════

def build_domain_masks(
    domains: list[dict],
    sequence_length: int,
) -> np.ndarray:
    """
    将域注释列表转换为 (n_cofactor_types, L) 的二进制掩码。

    每个辅因子类型的域位置标记为 1.0，重叠区域叠加。
    """
    masks = np.zeros((N_COFACTOR_TYPES, sequence_length), dtype=np.float32)
    for d in domains:
        ct = d.get("cofactor_type")
        if ct is None:
            continue
        if ct not in COFACTOR_INDEX:
            continue
        idx = COFACTOR_INDEX[ct]
        start = max(0, d["start"] - 1)  # UniProt 是 1-based
        end = min(sequence_length, d["end"])  # end 包含
        if start < end:
            masks[idx, start:end] = 1.0
    return masks


def build_domain_positions_array(domains: list[dict]) -> Optional[np.ndarray]:
    """
    将域注释转换为结构化数组，用于存储到 HDF5。
    使用 h5py 兼容的变长字符串类型。
    """
    if not domains:
        return None

    str_dtype = h5py.string_dtype()

    dtype = np.dtype([
        ("cofactor_type", str_dtype),
        ("start", np.int32),
        ("end", np.int32),
        ("interpro_id", str_dtype),
        ("pfam_id", str_dtype),
        ("description", str_dtype),
    ])
    arr = np.array([
        (d.get("cofactor_type") or "", d["start"], d["end"],
         d.get("interpro_id") or "", d.get("pfam_id") or "",
         d.get("description") or "")
        for d in domains
    ], dtype=dtype)
    return arr


# ═══════════════════════════════════════════════════════════════
# 4. 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="InterPro/Pfam 辅因子结合域位置扫描")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--proteins-h5", default=str(H5_PATH))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-fetch", action="store_true",
                        help="仅从缓存加载 UniProt 数据")
    parser.add_argument("--force-refresh", action="store_true",
                        help="重新获取所有 UniProt 条目")
    parser.add_argument("--max-proteins", type=int, default=None,
                        help="限制处理的蛋白质数（测试用）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    os.makedirs(cache_dir, exist_ok=True)
    full_json_cache = cache_dir / "uniprot_full.json"

    if args.force_refresh and full_json_cache.exists():
        full_json_cache.unlink()
        log.info("Cleared full JSON cache")

    # ── Step 1: 获取蛋白质标识映射 ──
    log.info("=" * 50)
    log.info("Step 1: Building protein identifier mapping")
    log.info("=" * 50)

    recs_path = out_dir / "oxidoreductase_records.pkl"
    if not recs_path.exists():
        log.error(f"Records not found: {recs_path}")
        return
    records = pickle.load(open(recs_path, "rb"))

    # seq_hash → {sequence, uniprot_ids, length}
    protein_map: dict[str, dict] = {}
    for r in records:
        sh = r.get("protein_seq_hash")
        if not sh:
            continue
        if sh not in protein_map:
            seq = r.get("sequence", "")
            protein_map[sh] = {
                "sequence": seq,
                "uniprot_ids": set(),
                "length": len(seq) if seq else 0,
            }
        uid = r.get("uniprot_id")
        if uid:
            protein_map[sh]["uniprot_ids"].add(uid)

    log.info(f"  Unique proteins: {len(protein_map)}")

    # 收集所有 UniProt ID
    all_uniprot_ids = set()
    for pinfo in protein_map.values():
        all_uniprot_ids.update(pinfo["uniprot_ids"])

    log.info(f"  Unique UniProt IDs: {len(all_uniprot_ids)}")

    # 对于没有 UniProt ID 的蛋白质，尝试通过 h5 sequence 匹配
    proteins_without_uniprot = sum(
        1 for pinfo in protein_map.values() if not pinfo["uniprot_ids"]
    )
    log.info(f"  Proteins without UniProt ID: {proteins_without_uniprot}")

    if args.max_proteins:
        # 限制处理的蛋白数
        target_ids = sorted(all_uniprot_ids)[:args.max_proteins]
        log.info(f"  Limited to {len(target_ids)} UniProt IDs for testing")
    else:
        target_ids = sorted(all_uniprot_ids)

    # ── Step 2: 获取 UniProt 完整 JSON ──
    log.info("=" * 50)
    log.info("Step 2: Fetching full UniProt JSON entries")
    log.info("=" * 50)

    if args.no_fetch:
        if full_json_cache.exists():
            with open(full_json_cache) as f:
                uniprot_full = json.load(f)
            log.info(f"  Loaded {len(uniprot_full)} cached entries (--no-fetch)")
        else:
            log.error("No cache found with --no-fetch. Exiting.")
            return
    else:
        uniprot_full = batch_fetch_uniprot_full(
            target_ids,
            cache_path=full_json_cache,
            n_workers=args.workers,
        )

    # ── Step 3: 解析域注释 ──
    log.info("=" * 50)
    log.info("Step 3: Extracting domain annotations")
    log.info("=" * 50)

    # uid → domains list
    domain_annotations: dict[str, list[dict]] = {}
    n_with_domains = 0
    n_with_cofactor_domains = 0
    cofactor_domain_counts: dict[str, int] = defaultdict(int)

    for uid, entry in tqdm(uniprot_full.items(), desc="Extracting domains"):
        if not entry or not entry.get("primaryAccession"):
            domain_annotations[uid] = []
            continue
        domains = extract_domains(entry)
        domain_annotations[uid] = domains
        if domains:
            n_with_domains += 1
        has_cofactor = any(d["cofactor_type"] is not None for d in domains)
        if has_cofactor:
            n_with_cofactor_domains += 1
            for d in domains:
                if d["cofactor_type"]:
                    cofactor_domain_counts[d["cofactor_type"]] += 1

    log.info(f"  UniProt entries with domains:  {n_with_domains}")
    log.info(f"  With cofactor domains:         {n_with_cofactor_domains}")
    log.info(f"  Cofactor domain type counts:")
    for ct, count in sorted(cofactor_domain_counts.items(), key=lambda x: -x[1]):
        log.info(f"    {ct:8s}: {count}")

    # ── Step 4: 构建 domain_masks，写入 proteins.h5 ──
    log.info("=" * 50)
    log.info("Step 4: Building domain_masks and writing to proteins.h5")
    log.info("=" * 50)

    h5_path = Path(args.proteins_h5)
    if not h5_path.exists():
        log.error(f"proteins.h5 not found: {h5_path}")
        return

    with h5py.File(h5_path, "r+") as h5:
        # 按 seq_hash 聚合
        n_updated = 0
        n_skipped = 0

        for seq_hash, pinfo in tqdm(protein_map.items(), desc="Writing domain_masks"):
            if seq_hash not in h5:
                n_skipped += 1
                continue

            grp = h5[seq_hash]
            seq_len = pinfo["length"]

            # 从 HDF5 获取序列长度（如果 pinfo 中没有）
            if seq_len == 0 and "sequence" in grp:
                seq_bytes = grp["sequence"][()]
                seq_len = len(seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes))

            if seq_len == 0:
                n_skipped += 1
                continue

            # 合并该蛋白质所有 UniProt ID 的域注释
            all_domains = []
            for uid in pinfo["uniprot_ids"]:
                domains = domain_annotations.get(uid, [])
                all_domains.extend(domains)

            # 按位置去重（同一位置同一辅因子类型只保留一次）
            seen = set()
            unique_domains = []
            for d in all_domains:
                key = (d["cofactor_type"], d["start"], d["end"])
                if key not in seen:
                    seen.add(key)
                    unique_domains.append(d)

            # 构建 masks
            masks = build_domain_masks(unique_domains, seq_len)

            # 存储到 HDF5
            if "domain_masks" in grp:
                del grp["domain_masks"]
            grp.create_dataset("domain_masks", data=masks, compression="gzip",
                              compression_opts=4)

            # 存储结构化域位置
            pos_arr = build_domain_positions_array(unique_domains)
            if pos_arr is not None and len(pos_arr) > 0:
                if "domain_positions" in grp:
                    del grp["domain_positions"]
                grp.create_dataset("domain_positions", data=pos_arr)

            # 标记是否有辅因子域
            has_cofactor_domain = any(d["cofactor_type"] is not None for d in unique_domains)
            grp.attrs["has_cofactor_domain"] = has_cofactor_domain
            grp.attrs["n_domains"] = len(unique_domains)

            n_updated += 1

        log.info(f"  Updated proteins: {n_updated}")
        log.info(f"  Skipped:          {n_skipped}")

    # ── Step 5: 更新 oxidoreductase metadata ──
    log.info("=" * 50)
    log.info("Step 5: Updating oxidoreductase metadata")
    log.info("=" * 50)

    meta_path = out_dir / "metadata.parquet"
    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path)
        log.info(f"  Loaded existing metadata: {len(meta_df)} rows")

        # 为每条记录添加 domain 信息
        domains_json_list = []
        n_domains_list = []
        has_domain_ann_list = []
        cofactor_domain_types_list = []

        for _, row in meta_df.iterrows():
            sh = row.get("protein_seq_hash", "")
            pinfo = protein_map.get(sh, {})

            all_domains = []
            for uid in pinfo.get("uniprot_ids", set()):
                all_domains.extend(domain_annotations.get(uid, []))

            # 去重
            seen = set()
            unique_domains = []
            for d in all_domains:
                key = (d["cofactor_type"], d["start"], d["end"])
                if key not in seen:
                    seen.add(key)
                    unique_domains.append(d)

            domains_json_list.append(json.dumps(unique_domains))
            n_domains_list.append(len(unique_domains))
            has_domain_ann_list.append(len(unique_domains) > 0)
            cofactor_types = set(d["cofactor_type"] for d in unique_domains if d["cofactor_type"])
            cofactor_domain_types_list.append("|".join(sorted(cofactor_types)))

        meta_df["domains_json"] = domains_json_list
        meta_df["n_domains"] = n_domains_list
        meta_df["has_domain_annotation"] = has_domain_ann_list
        meta_df["cofactor_domain_types"] = cofactor_domain_types_list

        # 备份原文件
        bak_path = out_dir / "metadata.parquet.bak"
        if not bak_path.exists():
            import shutil
            shutil.copy(meta_path, bak_path)
            log.info(f"  Backup saved → {bak_path}")

        meta_df.to_parquet(meta_path, index=False)
        log.info(f"  Updated metadata saved → {meta_path}")
        log.info(f"    Records with domain annotation: {sum(has_domain_ann_list)}")

    # ── 报告 ──
    print("\n" + "=" * 60)
    print("  InterPro/Pfam 辅因子结合域扫描 — 完成报告")
    print("=" * 60)
    print(f"  UniProt 条目数:              {len(uniprot_full)}")
    print(f"  有域注释的条目:              {n_with_domains}")
    print(f"  有辅因子结合域的条目:        {n_with_cofactor_domains}")
    print(f"  proteins.h5 更新蛋白数:      {n_updated}")
    print(f"  辅因子域类型分布:")
    for ct, count in sorted(cofactor_domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {ct:8s}: {count}")
    print(f"  HDF5 新增数据集: domain_masks (float32, {N_COFACTOR_TYPES} × L)")
    print(f"                    domain_positions (结构化数组)")
    print("=" * 60)


if __name__ == "__main__":
    main()
