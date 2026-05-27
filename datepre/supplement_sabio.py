"""
supplement_sabio.py
===================
从 SABIO-RK 补充氧化还原酶的动力学标签（KM, kcat, kcat/KM）。

SABIO-RK REST API:
  - 搜索: POST /sabioRestWebServices/searchKineticLaws/entryIDs?q=ECNumber:"X.X.X.X"
  - 条目: GET /sabioRestWebServices/kineticLaws/{entryID} → SBML XML
  - SBML 中的 localParameter: kcat (s⁻¹), Km_* (M), kcat_Km_* (M⁻¹s⁻¹)

工作流程：
  1. 获取氧化还原酶数据集中所有唯一 EC number
  2. 按 EC 搜索 SABIO-RK → 收集 EntryID
  3. 逐 EntryID 获取 SBML → 解析动力学参数 + UniProt ID
  4. 按 UniProt ID 对齐到氧化还原酶记录
  5. 输出 sabio_kinetics.parquet + sabio_aligned.parquet

用法：
  python datepre/supplement_sabio.py --max-ec 5                  # 测试
  python datepre/supplement_sabio.py --workers 4                  # 完整运行
  python datepre/supplement_sabio.py --no-fetch                   # 仅用缓存
"""

import os
import re
import json
import pickle
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SABIO_BASE = "https://sabiork.h-its.org/sabioRestWebServices"
SABIO_TIMEOUT = 60
SABIO_RETRIES = 3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
OUT_DIR = PROJECT_DIR / "processed" / "oxidoreductase"
CACHE_DIR = OUT_DIR / "cache"

# SBML namespace
SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"

# 常见底物名称简写映射（用于清理底物名）
SUBSTRATE_CLEANUP = re.compile(r"\(.*?\)|\[.*?\]|<.*?>", re.DOTALL)


# ═══════════════════════════════════════════════════════════════
# 1. SABIO-RK API 查询
# ═══════════════════════════════════════════════════════════════

def _encode_query(query: str) -> str:
    """URL-encode a SABIO-RK query string."""
    import urllib.parse
    return urllib.parse.quote(query, safe="")


def search_entries_by_ec(ec_number: str, session: requests.Session = None) -> list[int]:
    """
    按 EC number 搜索 SABIO-RK，返回 EntryID 列表。
    """
    if session is None:
        session = requests.Session()
    query = f'ECNumber:"{ec_number}"'
    url = f"{SABIO_BASE}/searchKineticLaws/entryIDs?q={_encode_query(query)}"

    for attempt in range(SABIO_RETRIES):
        try:
            resp = session.post(url, timeout=SABIO_TIMEOUT)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                entry_ids = []
                for elem in root.iter():
                    if elem.tag == "SabioEntryID":
                        try:
                            entry_ids.append(int(elem.text.strip()))
                        except (ValueError, AttributeError):
                            pass
                return entry_ids
            elif resp.status_code == 404:
                return []
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == SABIO_RETRIES - 1:
                log.warning(f"  Failed searching EC {ec_number}: {e}")
            time.sleep(2 ** attempt)
    return []


def search_entries_by_uniprot(uniprot_id: str, session: requests.Session = None) -> list[int]:
    """按 UniProt accession 搜索 SABIO-RK。"""
    if session is None:
        session = requests.Session()
    query = f'UniProtKB_AC:"{uniprot_id}"'
    url = f"{SABIO_BASE}/searchKineticLaws/entryIDs?q={_encode_query(query)}"
    for attempt in range(SABIO_RETRIES):
        try:
            resp = session.post(url, timeout=SABIO_TIMEOUT)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                return [
                    int(elem.text.strip())
                    for elem in root.iter()
                    if elem.tag == "SabioEntryID"
                ]
            elif resp.status_code == 404:
                return []
            else:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return []


def fetch_entry_sbml(entry_id: int, session: requests.Session = None) -> Optional[str]:
    """获取单个 SABIO-RK 条目的 SBML XML 文本。"""
    if session is None:
        session = requests.Session()
    url = f"{SABIO_BASE}/kineticLaws/{entry_id}"
    for attempt in range(SABIO_RETRIES):
        try:
            resp = session.get(url, timeout=SABIO_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                return None
            else:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


# ═══════════════════════════════════════════════════════════════
# 2. SBML 解析
# ═══════════════════════════════════════════════════════════════

def _clean_substrate_name(name: str) -> str:
    """清理底物名称。"""
    name = SUBSTRATE_CLEANUP.sub("", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def parse_sbml_entry(sbml_text: str, entry_id: int) -> Optional[dict]:
    """
    解析 SABIO-RK SBML 条目，提取动力学参数。

    Returns dict:
      {entry_id, uniprot_ids, ec_numbers, organism, enzyme_name,
       km_entries: [{substrate, value_M, pmids}],
       kcat_entries: [{value_s, pmids}],
       kcatkm_entries: [{substrate, value_M1s1, pmids}],
       ki_entries: [{inhibitor, value_M}],
       pmid: str}
    """
    try:
        root = ET.fromstring(sbml_text)
    except ET.ParseError:
        return None

    ns = SBML_NS

    result = {
        "entry_id": entry_id,
        "uniprot_ids": [],
        "ec_numbers": [],
        "organism": "",
        "enzyme_name": "",
        "km_entries": [],
        "kcat_entries": [],
        "kcatkm_entries": [],
        "ki_entries": [],
        "pmid": "",
    }

    model = root.find(f"{{{ns}}}model")
    if model is None:
        return None

    # ── 提取酶信息 ──
    for species in model.findall(f".//{{{ns}}}species"):
        sid = species.attrib.get("id", "")
        if sid.startswith("ENZ_"):
            name = species.attrib.get("name", "")
            result["enzyme_name"] = name
            # 解析: "alcohol dehydrogenase(Enzyme) wildtype isoenzyme ADH-2"
            enzyme_match = re.match(r"^(.+?)\(Enzyme\)\s*(.*)$", name)
            if enzyme_match:
                result["enzyme_name"] = enzyme_match.group(1).strip()
                # 剩余部分可能包含 organism/wildtype 信息

    # ── 提取 UniProt ID (从 RDF annotation) ──
    for desc_elem in model.iter():
        if "Description" in desc_elem.tag and "rdf:about" in desc_elem.attrib:
            about = desc_elem.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
            about_lower = about.lower()
            if "uniprot" in about_lower or "purl.uniprot" in about_lower:
                # Extract UniProt ID from URI
                uid_match = re.search(r"([A-Z][0-9][A-Z0-9]{3,})", about)
                if uid_match:
                    uid = uid_match.group(1)
                    if uid not in result["uniprot_ids"]:
                        result["uniprot_ids"].append(uid)
        # Also check rdf:resource attributes
        for attr_key, attr_val in desc_elem.attrib.items():
            if "resource" in attr_key.lower() and "uniprot" in attr_val.lower():
                uid_match = re.search(r"([A-Z][0-9][A-Z0-9]{3,})", attr_val)
                if uid_match:
                    uid = uid_match.group(1)
                    if uid not in result["uniprot_ids"]:
                        result["uniprot_ids"].append(uid)

    # ── 提取 EC number ──
    notes = model.find(f"{{{ns}}}notes")
    if notes is not None:
        notes_text = ET.tostring(notes, encoding="unicode")
        ec_matches = re.findall(r"EC\s*Number:?\s*([\d\.]+)", notes_text, re.IGNORECASE)
        result["ec_numbers"] = list(dict.fromkeys(ec_matches))  # 去重保序

    # ── 提取底物名称映射 ──
    substrate_names: dict[str, str] = {}  # SPC_XXX → name
    for species in model.findall(f".//{{{ns}}}species"):
        sid = species.attrib.get("id", "")
        name = species.attrib.get("name", "")
        if sid.startswith("SPC_") and not sid.startswith("SPC_") == False:
            substrate_names[sid] = _clean_substrate_name(name)

    # ── 提取动力学参数 ──
    for lp in model.findall(f".//{{{ns}}}localParameter"):
        pid = lp.attrib.get("id", "")
        name = lp.attrib.get("name", "")
        value_str = lp.attrib.get("value", "")
        unit = lp.attrib.get("units", "")

        try:
            value = float(value_str)
        except (ValueError, TypeError):
            continue

        # 提取底物名称 (从参数 ID 如 Km_SPC_56_Cell)
        substrate = ""
        sub_match = re.match(r"K[im]\w*_(SPC_\d+)_", pid)
        if sub_match:
            spc_id = sub_match.group(1)
            substrate = substrate_names.get(spc_id, "")
        elif "_" in name:
            # 从参数名提取: Km_Ethanol
            parts = name.split("_", 1)
            if len(parts) > 1:
                substrate = parts[1]

        if pid.startswith("kcat") and not pid.startswith("kcat_Km"):
            # kcat (turnover number), unit: s⁻¹
            result["kcat_entries"].append({
                "value_s": value,
                "unit": "s⁻¹",
            })
        elif pid.startswith("Km_"):
            # KM, unit: M → uM
            result["km_entries"].append({
                "substrate": substrate,
                "value_M": value,
                "value_uM": value * 1e6,
            })
        elif pid.startswith("kcat_Km_"):
            # kcat/KM, unit: M⁻¹s⁻¹
            result["kcatkm_entries"].append({
                "substrate": substrate,
                "value_M1s1": value,
            })
        elif pid.startswith("Ki_"):
            result["ki_entries"].append({
                "inhibitor": substrate if substrate else name,
                "value_M": value,
            })

    # ── 提取 PubMed ID ──
    if notes is not None:
        notes_text = ET.tostring(notes, encoding="unicode")
        pmid_matches = re.findall(r"PubMed\s*ID:?\s*(\d+)", notes_text, re.IGNORECASE)
        if pmid_matches:
            result["pmid"] = ";".join(pmid_matches)

    return result


# ═══════════════════════════════════════════════════════════════
# 3. 批量获取 + 缓存
# ═══════════════════════════════════════════════════════════════

def collect_all_entry_ids(
    ec_numbers: list[str],
    uniprot_ids: list[str],
    cache_path: Optional[Path] = None,
    n_workers: int = 4,
) -> set[int]:
    """
    通过 EC number + UniProt ID 搜索 SABIO-RK，收集所有 EntryID。

    先按 EC 搜索（覆盖范围最广），再按 UniProt 补充遗漏的。
    """
    cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        log.info(f"Loaded {len(cache)} cached search results")

    all_entry_ids: set[int] = set()

    # ── 按 EC 搜索 ──
    ecs_to_search = [ec for ec in ec_numbers if ec not in cache]
    if ecs_to_search:
        log.info(f"Searching SABIO-RK by {len(ecs_to_search)} EC numbers...")
        session = requests.Session()
        for ec in tqdm(ecs_to_search, desc="EC search"):
            ids = search_entries_by_ec(ec, session)
            cache[ec] = ids
            all_entry_ids.update(ids)
            time.sleep(0.2)  # 礼貌速率

        # 保存缓存
        if cache_path:
            os.makedirs(cache_path.parent, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)
    else:
        log.info("All EC searches cached.")
        for ec, ids in cache.items():
            all_entry_ids.update(ids)

    log.info(f"  Collected {len(all_entry_ids)} unique EntryIDs from EC search")

    # ── 按 UniProt 搜索（补充） ──
    up_cache: dict = {}
    up_cache_path = cache_path.parent / "sabio_uniprot_search.json" if cache_path else None
    if up_cache_path and up_cache_path.exists():
        with open(up_cache_path) as f:
            up_cache = json.load(f)

    uids_to_search = [uid for uid in uniprot_ids if uid not in up_cache]
    if uids_to_search:
        log.info(f"Searching SABIO-RK by {len(uids_to_search)} UniProt IDs...")
        session = requests.Session()
        up_hits = 0
        for uid in tqdm(uids_to_search, desc="UniProt search"):
            ids = search_entries_by_uniprot(uid, session)
            up_cache[uid] = ids
            new_ids = set(ids) - all_entry_ids
            if new_ids:
                up_hits += len(new_ids)
                all_entry_ids.update(new_ids)
            time.sleep(0.15)

        if up_cache_path:
            with open(up_cache_path, "w") as f:
                json.dump(up_cache, f)
        log.info(f"  UniProt search added {up_hits} new EntryIDs")

    log.info(f"  Total unique EntryIDs: {len(all_entry_ids)}")
    return all_entry_ids


def fetch_all_entries(
    entry_ids: set[int],
    cache_path: Optional[Path] = None,
    n_workers: int = 4,
) -> dict[int, dict]:
    """并行获取所有 SBML 条目并解析。"""
    # 加载缓存
    cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
            # 转换键回 int
            cache = {int(k): v for k, v in raw.items()}
        log.info(f"Loaded {len(cache)} cached entries")

    to_fetch = sorted(set(entry_ids) - set(cache.keys()))
    if not to_fetch:
        log.info("All entries cached.")
        return cache

    log.info(f"Fetching {len(to_fetch)} SABIO-RK entries ({n_workers} workers)...")

    session = requests.Session()
    fetched = 0
    failed = 0

    def _fetch_one(eid):
        nonlocal fetched, failed
        xml_text = fetch_entry_sbml(eid, session)
        if xml_text:
            parsed = parse_sbml_entry(xml_text, eid)
            fetched += 1
            return eid, parsed
        else:
            failed += 1
            return eid, None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fetch_one, eid): eid for eid in to_fetch}
        with tqdm(total=len(to_fetch), desc="Fetching entries") as pbar:
            for future in as_completed(futures):
                eid, parsed = future.result()
                if parsed:
                    cache[eid] = parsed
                else:
                    cache[eid] = None
                pbar.update(1)

                # 周期性保存
                if fetched % 100 == 0 and cache_path:
                    os.makedirs(cache_path.parent, exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump({str(k): v for k, v in cache.items()}, f)

    # 最终保存
    if cache_path:
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in cache.items()}, f)

    n_valid = sum(1 for v in cache.values() if v is not None)
    log.info(f"Fetched {fetched} entries ({n_valid} valid, {failed} failed)")
    return cache


# ═══════════════════════════════════════════════════════════════
# 4. 汇总与对齐
# ═══════════════════════════════════════════════════════════════

def build_summary(entries_cache: dict[int, dict]) -> pd.DataFrame:
    """按 UniProt ID 聚合 SABIO-RK 动力学参数。"""
    # 按 UniProt ID 分组
    by_uniprot: dict[str, dict] = defaultdict(lambda: {
        "km_values_uM": [],
        "kcat_values_s": [],
        "kcatkm_values_M1s1": [],
        "substrates": set(),
        "n_entries": 0,
    })

    for eid, entry in entries_cache.items():
        if entry is None:
            continue
        for uid in entry.get("uniprot_ids", []):
            bucket = by_uniprot[uid]
            bucket["n_entries"] += 1
            for km in entry.get("km_entries", []):
                bucket["km_values_uM"].append(km["value_uM"])
                if km["substrate"]:
                    bucket["substrates"].add(km["substrate"])
            for kcat in entry.get("kcat_entries", []):
                bucket["kcat_values_s"].append(kcat["value_s"])
            for kkm in entry.get("kcatkm_entries", []):
                bucket["kcatkm_values_M1s1"].append(kkm["value_M1s1"])

    rows = []
    for uid, bucket in by_uniprot.items():
        rows.append({
            "uniprot_id": uid,
            "n_entries": bucket["n_entries"],
            "n_km": len(bucket["km_values_uM"]),
            "km_median_uM": float(np.median(bucket["km_values_uM"])) if bucket["km_values_uM"] else None,
            "km_min_uM": float(np.min(bucket["km_values_uM"])) if bucket["km_values_uM"] else None,
            "km_max_uM": float(np.max(bucket["km_values_uM"])) if bucket["km_values_uM"] else None,
            "n_kcat": len(bucket["kcat_values_s"]),
            "kcat_median_s": float(np.median(bucket["kcat_values_s"])) if bucket["kcat_values_s"] else None,
            "kcat_min_s": float(np.min(bucket["kcat_values_s"])) if bucket["kcat_values_s"] else None,
            "kcat_max_s": float(np.max(bucket["kcat_values_s"])) if bucket["kcat_values_s"] else None,
            "n_kcatkm": len(bucket["kcatkm_values_M1s1"]),
            "kcatkm_median_M1s1": float(np.median(bucket["kcatkm_values_M1s1"])) if bucket["kcatkm_values_M1s1"] else None,
            "n_unique_substrates": len(bucket["substrates"]),
        })

    df = pd.DataFrame(rows)
    columns_order = [
        "uniprot_id", "n_entries", "n_km", "km_median_uM", "km_min_uM", "km_max_uM",
        "n_kcat", "kcat_median_s", "kcat_min_s", "kcat_max_s",
        "n_kcatkm", "kcatkm_median_M1s1", "n_unique_substrates",
    ]
    # 只保留存在的列
    return df[[c for c in columns_order if c in df.columns]]


def align_to_records(
    summary_df: pd.DataFrame,
    records: list[dict],
) -> pd.DataFrame:
    """将 SABIO-RK 汇总对齐到氧化还原酶记录。"""
    aligned_rows = []

    for r in records:
        uid = r.get("uniprot_id")
        if not uid:
            continue

        row = summary_df[summary_df["uniprot_id"] == uid]
        if len(row) == 0:
            continue

        s = row.iloc[0]
        aligned_rows.append({
            "uniprot_id": uid,
            "pdb_id": r.get("pdb_id"),
            "source_db": r.get("source_db"),
            "measurement_type": r.get("measurement_type"),
            "pkd_raw": r.get("pkd_raw"),
            "n_km_sabio": int(s["n_km"]),
            "n_kcat_sabio": int(s["n_kcat"]),
            "n_kcatkm_sabio": int(s["n_kcatkm"]),
            "km_median_uM_sabio": s["km_median_uM"],
            "kcat_median_s_sabio": s["kcat_median_s"],
            "kcatkm_median_M1s1_sabio": s["kcatkm_median_M1s1"],
            "has_sabio": True,
        })

    df = pd.DataFrame(aligned_rows)
    return df


def align_sabio_to_metadata(
    summary_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    将 SABIO-RK 汇总按 UniProt ID 对齐到 metadata.parquet 的蛋白哈希。

    Returns DataFrame with columns:
      protein_seq_hash, uniprot_id, n_kcat_sabio, kcat_median_s_sabio, ...
    """
    up_to_sh = meta_df.groupby('uniprot_id')['protein_seq_hash'].first().to_dict()
    existing_kcat = meta_df.groupby('protein_seq_hash').agg(
        n_kcat_bindingdb=('n_measurements', 'sum'),
        kcat_median_bindingdb=('kcat_median_s', 'first'),
        log_kcat_bindingdb=('log_kcat_median', 'first'),
        kcat_source_existing=('kcat_source', 'first'),
    ).reset_index()

    rows = []
    for _, row in summary_df.iterrows():
        uid = row['uniprot_id']
        sh = up_to_sh.get(uid)
        if not sh:
            continue
        entry = {
            'protein_seq_hash': sh,
            'uniprot_id': uid,
            'n_entries_sabio': row['n_entries'],
            'n_km_sabio': row['n_km'],
            'km_median_uM_sabio': row['km_median_uM'],
            'n_kcat_sabio': row['n_kcat'],
            'kcat_median_s_sabio': row['kcat_median_s'],
            'n_kcatkm_sabio': row['n_kcatkm'],
            'kcatkm_median_M1s1_sabio': row['kcatkm_median_M1s1'],
            'n_unique_substrates_sabio': row['n_unique_substrates'],
            'has_sabio': True,
        }
        rows.append(entry)

    aligned = pd.DataFrame(rows)
    if len(aligned) == 0:
        return aligned

    # Merge with existing kcat data for cross-validation
    aligned = aligned.merge(existing_kcat, on='protein_seq_hash', how='left')

    # Compute log-space consensus if both sources exist
    mask_both = (
        aligned['n_kcat_sabio'].gt(0) &
        aligned['kcat_median_bindingdb'].notna() &
        aligned['kcat_median_bindingdb'].gt(0)
    )
    log_kcat_sabio = np.log10(aligned.loc[mask_both, 'kcat_median_s_sabio'].astype(float))
    log_kcat_bdb = aligned.loc[mask_both, 'log_kcat_bindingdb'].astype(float)
    consensus_log = (log_kcat_sabio + log_kcat_bdb) / 2.0
    aligned.loc[mask_both, 'kcat_consensus_s'] = 10.0 ** consensus_log
    aligned.loc[mask_both, 'log_kcat_consensus'] = consensus_log
    aligned.loc[mask_both, 'kcat_cross_validated'] = True

    # Only SABIO-RK
    mask_sabio_only = aligned['n_kcat_sabio'].gt(0) & ~mask_both
    aligned.loc[mask_sabio_only, 'kcat_consensus_s'] = aligned.loc[mask_sabio_only, 'kcat_median_s_sabio']
    aligned.loc[mask_sabio_only, 'log_kcat_consensus'] = np.log10(
        aligned.loc[mask_sabio_only, 'kcat_median_s_sabio'].astype(float)
    )
    aligned.loc[mask_sabio_only, 'kcat_cross_validated'] = False

    # Only BindingDB/BRENDA
    mask_bdb_only = ~mask_both & ~mask_sabio_only
    aligned.loc[mask_bdb_only, 'kcat_consensus_s'] = aligned.loc[mask_bdb_only, 'kcat_median_bindingdb']
    aligned.loc[mask_bdb_only, 'log_kcat_consensus'] = aligned.loc[mask_bdb_only, 'log_kcat_bindingdb']
    aligned.loc[mask_bdb_only, 'kcat_cross_validated'] = False

    return aligned


def build_detailed_kinetics(entries_cache: dict[int, dict]) -> pd.DataFrame:
    """展开每一条动力学参数为详细 DataFrame。"""
    rows = []
    for eid, entry in entries_cache.items():
        if entry is None:
            continue
        for uid in entry.get("uniprot_ids", []):
            for km in entry.get("km_entries", []):
                rows.append({
                    "entry_id": eid,
                    "uniprot_id": uid,
                    "ec_numbers": "|".join(entry.get("ec_numbers", [])),
                    "enzyme_name": entry.get("enzyme_name", ""),
                    "param_type": "KM",
                    "substrate": km.get("substrate", ""),
                    "value": km["value_uM"],
                    "unit": "uM",
                    "pmid": entry.get("pmid", ""),
                })
            for kcat in entry.get("kcat_entries", []):
                rows.append({
                    "entry_id": eid,
                    "uniprot_id": uid,
                    "ec_numbers": "|".join(entry.get("ec_numbers", [])),
                    "enzyme_name": entry.get("enzyme_name", ""),
                    "param_type": "kcat",
                    "substrate": "",
                    "value": kcat["value_s"],
                    "unit": "s⁻¹",
                    "pmid": entry.get("pmid", ""),
                })
            for kkm in entry.get("kcatkm_entries", []):
                rows.append({
                    "entry_id": eid,
                    "uniprot_id": uid,
                    "ec_numbers": "|".join(entry.get("ec_numbers", [])),
                    "enzyme_name": entry.get("enzyme_name", ""),
                    "param_type": "kcatKM",
                    "substrate": kkm.get("substrate", ""),
                    "value": kkm["value_M1s1"],
                    "unit": "M⁻¹s⁻¹",
                    "pmid": entry.get("pmid", ""),
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 5. 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SABIO-RK 动力学补充")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-fetch", action="store_true",
                        help="仅从缓存加载，不查询 API")
    parser.add_argument("--force-refresh", action="store_true",
                        help="清除缓存重新获取")
    parser.add_argument("--from-parquet", default=None,
                        help="从 unified_metadata.parquet 读取 (替代 pickle)")
    parser.add_argument("--max-ec", type=int, default=None,
                        help="限制 EC 搜索数量（测试用）")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="限制获取的 Entry 数量（测试用）")
    parser.add_argument("--skip-uniprot", action="store_true",
                        help="跳过 UniProt ID 搜索（仅用 EC 搜索）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    os.makedirs(cache_dir, exist_ok=True)

    ec_search_cache = cache_dir / "sabio_ec_search.json"
    entries_cache_path = cache_dir / "sabio_entries.json"

    if args.force_refresh:
        for p in [ec_search_cache, entries_cache_path,
                  cache_dir / "sabio_uniprot_search.json"]:
            if p.exists():
                p.unlink()
                log.info(f"Cleared cache: {p}")

    # ── Step 1: 获取氧化还原酶 EC / UniProt ID ──
    log.info("=" * 50)
    log.info("Step 1: Loading oxidoreductase identifiers")
    log.info("=" * 50)

    meta_df = None  # populated when --from-parquet
    if args.from_parquet:
        parquet_path = Path(args.from_parquet)
        if not parquet_path.exists():
            log.error(f"Parquet not found: {parquet_path}")
            return
        meta_df = pd.read_parquet(parquet_path)
        ec_set: set[str] = set()
        for ecs in meta_df['ec_numbers'].dropna():
            # EC numbers may be separated by | or ,
            for ec in re.split(r'[|,]', str(ecs).replace(' ', '')):
                if re.match(r"^1\.\d+\.\d+\.\d+", ec.strip()):
                    ec_set.add(ec.strip())
        uniprot_set = set(meta_df['uniprot_id'].dropna().unique())
        log.info(f"  Loaded from parquet: {len(meta_df)} records")
    else:
        recs_path = out_dir / "oxidoreductase_records.pkl"
        if not recs_path.exists():
            log.error(f"Records not found: {recs_path}")
            log.error("Run prepare_oxidoreductase.py first, or use --from-parquet.")
            return
        records = pickle.load(open(recs_path, "rb"))
        ec_set = set()
        uniprot_set = set()
        for r in records:
            for ec in r.get("ec_numbers", []):
                ec_clean = ec.strip().replace(" ", "")
                if re.match(r"^1\.\d+\.\d+\.\d+", ec_clean):
                    ec_set.add(ec_clean)
            uid = r.get("uniprot_id")
            if uid:
                uniprot_set.add(uid)

    ec_list = sorted(ec_set)
    uniprot_list = sorted(uniprot_set)

    log.info(f"  Unique EC numbers:   {len(ec_list)}")
    log.info(f"  Unique UniProt IDs:  {len(uniprot_list)}")

    if args.max_ec:
        ec_list = ec_list[:args.max_ec]
        log.info(f"  Limited to {args.max_ec} EC numbers for testing")

    # ── Step 2: 搜索 EntryID ──
    log.info("=" * 50)
    log.info("Step 2: Searching SABIO-RK by EC + UniProt")
    log.info("=" * 50)

    if args.no_fetch:
        all_entries: set[int] = set()
        if ec_search_cache.exists():
            with open(ec_search_cache) as f:
                for ec, ids in json.load(f).items():
                    all_entries.update(ids)
        log.info(f"  Loaded {len(all_entries)} EntryIDs from cache (--no-fetch)")
    elif args.skip_uniprot:
        uniprot_list = []  # Skip UniProt search
        all_entries = collect_all_entry_ids(
            ec_list, [],
            cache_path=ec_search_cache,
            n_workers=args.workers,
        )
    else:
        all_entries = collect_all_entry_ids(
            ec_list, uniprot_list,
            cache_path=ec_search_cache,
            n_workers=args.workers,
        )

    if not all_entries:
        log.warning("No SABIO-RK entries found. Check EC numbers or API connectivity.")
        return

    if args.max_entries:
        all_entries = set(sorted(all_entries)[:args.max_entries])
        log.info(f"  Limited to {args.max_entries} entries for testing")

    # ── Step 3: 获取 + 解析 SBML ──
    log.info("=" * 50)
    log.info("Step 3: Fetching SBML entries")
    log.info("=" * 50)

    if args.no_fetch:
        if entries_cache_path.exists():
            with open(entries_cache_path) as f:
                entries_cache = {int(k): v for k, v in json.load(f).items()}
            log.info(f"  Loaded {len(entries_cache)} entries from cache")
        else:
            log.error("No entry cache found with --no-fetch. Exiting.")
            return
    else:
        entries_cache = fetch_all_entries(
            all_entries,
            cache_path=entries_cache_path,
            n_workers=args.workers,
        )

    # ── Step 4: 汇总 ──
    log.info("=" * 50)
    log.info("Step 4: Building SABIO-RK summary")
    log.info("=" * 50)

    summary_df = build_summary(entries_cache)
    n_with_kcat = int((summary_df["n_kcat"] > 0).sum())
    n_with_km = int((summary_df["n_km"] > 0).sum())
    n_with_kkm = int((summary_df["n_kcatkm"] > 0).sum())

    log.info(f"  UniProt IDs with SABIO data: {len(summary_df)}")
    log.info(f"    With kcat:    {n_with_kcat}")
    log.info(f"    With KM:      {n_with_km}")
    log.info(f"    With kcat/KM: {n_with_kkm}")

    if n_with_kcat > 0:
        kcat_vals = summary_df.loc[summary_df["n_kcat"] > 0, "kcat_median_s"].dropna()
        if len(kcat_vals) > 0:
            log.info(f"    kcat range:    [{kcat_vals.min():.2e}, {kcat_vals.max():.2e}] s⁻¹")
    if n_with_km > 0:
        km_vals = summary_df.loc[summary_df["n_km"] > 0, "km_median_uM"].dropna()
        if len(km_vals) > 0:
            log.info(f"    KM range:      [{km_vals.min():.2f}, {km_vals.max():.2f}] uM")

    # ── Step 5: 对齐到氧化还原酶记录 ──
    log.info("=" * 50)
    log.info("Step 5: Aligning to oxidoreductase records")
    log.info("=" * 50)

    if meta_df is not None:
        # Use parquet-based alignment with cross-validation
        aligned_df = align_sabio_to_metadata(summary_df, meta_df)
        log.info(f"  Aligned proteins: {len(aligned_df)}")
        n_cv = int(aligned_df['kcat_cross_validated'].sum()) if 'kcat_cross_validated' in aligned_df.columns else 0
        log.info(f"  Cross-validated (BRENDA+SABIO): {n_cv}")
        log.info(f"  SABIO-only kcat: {int((aligned_df['n_kcat_sabio'].gt(0) & ~aligned_df['kcat_cross_validated']).sum())}")
        log.info(f"  BindingDB-only kcat: {int((aligned_df['kcat_median_bindingdb'].notna() & aligned_df['n_kcat_sabio'].eq(0)).sum())}")
    else:
        aligned_df = align_to_records(summary_df, records)
        log.info(f"  Aligned records: {len(aligned_df)}")
        n_kcat_aligned = int(aligned_df["n_kcat_sabio"].gt(0).sum())
        log.info(f"  Records with SABIO kcat: {n_kcat_aligned}")

    # ── Step 6: 保存 ──
    log.info("=" * 50)
    log.info("Step 6: Saving")
    log.info("=" * 50)

    summary_path = out_dir / "sabio_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)
    log.info(f"  Summary → {summary_path}")

    aligned_path = out_dir / "sabio_aligned.parquet"
    aligned_df.to_parquet(aligned_path, index=False)
    log.info(f"  Aligned → {aligned_path}")

    detailed_df = build_detailed_kinetics(entries_cache)
    detailed_path = out_dir / "sabio_kinetics.parquet"
    detailed_df.to_parquet(detailed_path, index=False)
    log.info(f"  Detailed ({len(detailed_df):,} entries) → {detailed_path}")

    # ── Step 7: 更新 unified_metadata.parquet ──
    if meta_df is not None and 'kcat_consensus_s' in aligned_df.columns:
        log.info("=" * 50)
        log.info("Step 7: Updating unified_metadata.parquet with SABIO-RK kcat")
        log.info("=" * 50)

        # Build lookup: protein_seq_hash → consensus kcat
        sh_to_consensus = aligned_df.set_index('protein_seq_hash')['log_kcat_consensus'].to_dict()
        sh_to_source = {}
        for _, row in aligned_df.iterrows():
            sh = row['protein_seq_hash']
            if row.get('kcat_cross_validated'):
                sh_to_source[sh] = 'multi_source_bindingdb_sabio'
            elif row.get('n_kcat_sabio', 0) > 0:
                sh_to_source[sh] = 'sabio_only'
            else:
                sh_to_source[sh] = None  # unchanged

        n_updated = 0
        n_cv = 0
        mask_updated = pd.Series(False, index=meta_df.index)
        for i, row in meta_df.iterrows():
            sh = row['protein_seq_hash']
            if sh in sh_to_consensus and pd.notna(sh_to_consensus[sh]):
                consensus_log = sh_to_consensus[sh]
                # Only update if significantly different or new source
                existing_log = row.get('log_kcat_median', np.nan)
                if pd.isna(existing_log) or abs(consensus_log - existing_log) > 0.01:
                    meta_df.at[i, 'log_kcat_median'] = consensus_log
                    meta_df.at[i, 'kcat_median_s'] = 10.0 ** consensus_log
                new_source = sh_to_source[sh]
                if new_source:
                    meta_df.at[i, 'kcat_source'] = new_source
                n_updated += 1
                if sh_to_source[sh] == 'multi_source_bindingdb_sabio':
                    n_cv += 1

        # Save
        meta_df.to_parquet(parquet_path, index=False)
        log.info(f"  Updated {n_updated} proteins in unified_metadata.parquet")
        log.info(f"    Cross-validated (multi-source): {n_cv}")
        log.info(f"    SABIO-only:                     {n_updated - n_cv}")
        log.info(f"  Saved: {parquet_path}")

    # ── 报告 ──
    print("\n" + "=" * 60)
    print("  SABIO-RK 动力学数据补充 — 完成报告")
    print("=" * 60)
    print(f"  搜索的 EC 数:           {len(ec_list)}")
    print(f"  SABIO-RK Entry 总数:    {len(all_entries)}")
    print(f"  有数据的 UniProt ID:    {len(summary_df)}")
    print(f"  有 kcat 的 UniProt ID:  {n_with_kcat}")
    print(f"  有 KM 的 UniProt ID:    {n_with_km}")
    print(f"  有 kcat/KM 的 UniProt:  {n_with_kkm}")
    print(f"  对齐到记录:             {len(aligned_df)}")
    if meta_df is not None and 'kcat_cross_validated' in aligned_df.columns:
        print(f"  BRENDA+SABIO 交叉验证:  {int(aligned_df['kcat_cross_validated'].sum())}")
    print(f"  输出目录:               {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
