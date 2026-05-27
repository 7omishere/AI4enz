"""
supplement_kinetics.py
======================
从 UniProt 补充氧化还原酶的动力学标签（KM, Vmax, kcat, kcat/KM）。

工作流程：
  1. 加载 oxidoreductase_records.pkl → 获取所有唯一 UniProt ID（EC 1.x.x.x）
  2. 逐个查询 UniProt JSON → 提取 BIOPHYSICOCHEMICAL PROPERTIES
  3. 解析 KM、Vmax、turnover numbers、pH/温度依赖
  4. 计算 kcat/K_M（催化效率，当 kcat 和 KM 均可获取时）
  5. 输出 kinetics.parquet + 统计报告

输出：processed/oxidoreductase/kinetics.parquet
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
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
UNIPROT_RETRIES = 5

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
PROCESSED_DIR = PROJECT_DIR / "processed"
OUT_DIR = PROCESSED_DIR / "oxidoreductase"


# ─────────────────────────────────────────────────────────────
# 1. 从 UniProt 提取动力学参数
# ─────────────────────────────────────────────────────────────

def fetch_kinetic_params(uniprot_id: str) -> Optional[dict]:
    """
    查询单个 UniProt 条目，提取所有动力学参数。

    Returns dict with:
      - uniprot_id
      - ec_numbers
      - protein_name
      - km_entries: [{substrate, value_uM, ph, temp_C, pmid}]
      - vmax_entries: [{substrate, value (in original units), unit, ph, temp_C, pmid}]
      - kcat_entries: [{substrate, value_s, ph, temp_C, pmid}]
      - has_kinetics: bool
    """
    url = f"{UNIPROT_BASE}/{uniprot_id}.json"
    session = requests.Session()
    session.headers.update({"User-Agent": "AI4enz/1.0"})

    for attempt in range(UNIPROT_RETRIES):
        try:
            resp = session.get(url, timeout=UNIPROT_TIMEOUT)
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                time.sleep(2**attempt)
                continue
            return _parse_kinetics(resp.json())
        except Exception:
            time.sleep(2**attempt)
    return None


def _parse_unit_to_uM(value: float, unit: str) -> float:
    """归一化到 uM。"""
    unit_lower = (unit or "").strip().lower()
    conversions = {
        "m": 1e9,
        "mm": 1e6,
        "um": 1.0,
        "µm": 1.0,
        "nm": 0.001,
        "pm": 1e-6,
        "fm": 1e-9,
    }
    # Strip trailing content
    for key, factor in conversions.items():
        if unit_lower.startswith(key):
            return value * factor
    return value  # unknown unit, return as-is


def _parse_kinetics(data: dict) -> dict:
    """解析 UniProt JSON → 动力学参数。"""
    result = {
        "uniprot_id": data.get("primaryAccession", ""),
        "ec_numbers": [],
        "protein_name": "",
        "reviewed": data.get("entryType", "").startswith("Swiss-Prot"),
        "km_entries": [],
        "vmax_entries": [],
        "kcat_entries": [],
        "has_kinetics": False,
        "ph_dependence": [],
        "temp_dependence": [],
    }

    # EC + protein name
    pd_ = data.get("proteinDescription", {})
    rec = pd_.get("recommendedName", {}) or pd_.get("submissionNames", [{}])[0]
    result["ec_numbers"] = [e.get("value", "") for e in rec.get("ecNumbers", [])]
    result["protein_name"] = rec.get("fullName", {}).get("value", "")

    # Also check catalytic activity for ECs
    for comment in data.get("comments", []):
        if comment.get("commentType") == "CATALYTIC_ACTIVITY":
            reaction = comment.get("reaction", {})
            for e in reaction.get("ecNumbers", []):
                ec_val = e.get("value", "")
                if ec_val and ec_val not in result["ec_numbers"]:
                    result["ec_numbers"].append(ec_val)

    # Kinetics from BIOPHYSICOCHEMICAL PROPERTIES
    for comment in data.get("comments", []):
        if comment.get("commentType") != "BIOPHYSICOCHEMICAL PROPERTIES":
            continue

        kp = comment.get("kineticParameters", {})
        if not kp:
            continue

        # Get pH / temperature context for this entry
        ph_range = _extract_ph_range(comment)
        temp_range = _extract_temp_range(comment)

        # Michaelis constants (KM)
        for km in kp.get("michaelisConstants", []):
            entry = {
                "substrate": km.get("substrate", ""),
                "value_uM": _parse_unit_to_uM(
                    float(km["constant"]), km.get("unit", "uM")
                ),
                "original_unit": km.get("unit", ""),
                "ph": ph_range,
                "temp_C": temp_range,
                "pmid": _extract_pmid(km.get("evidences", [])),
            }
            result["km_entries"].append(entry)
            result["has_kinetics"] = True

        # Maximum velocities (Vmax)
        for vmax in kp.get("maximumVelocities", []):
            entry = {
                "substrate": vmax.get("enzyme", ""),
                "value": float(vmax["velocity"]),
                "unit": vmax.get("unit", ""),
                "ph": ph_range,
                "temp_C": temp_range,
                "pmid": _extract_pmid(vmax.get("evidences", [])),
            }
            result["vmax_entries"].append(entry)
            result["has_kinetics"] = True

        # Turnover numbers (kcat)
        for tn in kp.get("turnoverNumbers", []):
            entry = {
                "substrate": tn.get("substrate", ""),
                "value_s": float(tn["constant"]),
                "unit": tn.get("unit", "s⁻¹"),
                "ph": ph_range,
                "temp_C": temp_range,
                "pmid": _extract_pmid(tn.get("evidences", [])),
            }
            result["kcat_entries"].append(entry)
            result["has_kinetics"] = True

        # pH dependence
        result["ph_dependence"] = comment.get("phDependence", [])
        result["temp_dependence"] = comment.get("tempDependence", [])

    return result


def _extract_ph_range(comment: dict) -> Optional[str]:
    """Extract pH range as a string like '7.0-8.5'."""
    ph_data = comment.get("phDependence")
    if not ph_data:
        return None
    if isinstance(ph_data, dict):
        ph_data = ph_data.get("texts", [])
    if isinstance(ph_data, list):
        values = [t.get("value", "") for t in ph_data if isinstance(t, dict)]
        return "; ".join(values) if values else None
    return None


def _extract_temp_range(comment: dict) -> Optional[str]:
    """Extract temperature range."""
    temp_data = comment.get("tempDependence")
    if not temp_data:
        return None
    if isinstance(temp_data, dict):
        temp_data = temp_data.get("texts", [])
    if isinstance(temp_data, list):
        values = [t.get("value", "") for t in temp_data if isinstance(t, dict)]
        return "; ".join(values) if values else None
    return None


def _extract_pmid(evidences: list) -> Optional[str]:
    """Extract PubMed ID from evidence list."""
    for ev in evidences:
        if ev.get("source") == "PubMed" and ev.get("id"):
            return ev["id"]
    return None


# ─────────────────────────────────────────────────────────────
# 2. 批量获取 + 缓存
# ─────────────────────────────────────────────────────────────

def fetch_all_kinetics(
    uniprot_ids: list[str],
    cache_path: Optional[Path] = None,
    n_workers: int = 8,
) -> dict[str, dict]:
    """并行获取所有氧化还原酶的动力学参数。"""
    # Load cache
    cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        log.info(f"Loaded {len(cache)} cached kinetics entries")

    to_fetch = [uid for uid in uniprot_ids if uid not in cache]
    if not to_fetch:
        log.info("All kinetics already cached.")
        return cache

    log.info(f"Fetching kinetic data for {len(to_fetch)} UniProt IDs "
             f"({n_workers} workers)...")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(fetch_kinetic_params, uid): uid
                   for uid in to_fetch}
        with tqdm(total=len(to_fetch), desc="Fetching kinetics") as pbar:
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    result = future.result()
                    cache[uid] = result if result else {}
                except Exception:
                    cache[uid] = {}
                pbar.update(1)

            # Periodic save
            if cache_path:
                os.makedirs(cache_path.parent, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache, f, default=str)

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(cache, f, default=str)
        log.info(f"Saved kinetics cache → {cache_path}")

    n_with = sum(1 for v in cache.values() if v.get("has_kinetics"))
    log.info(f"Kinetics fetch done: {n_with}/{len(cache)} have kinetic data")
    return cache


# ─────────────────────────────────────────────────────────────
# 3. 组装最终数据集
# ─────────────────────────────────────────────────────────────

def build_kinetics_dataset(
    kinetics_cache: dict[str, dict],
    oxidoreductase_uniprot_ids: set[str],
) -> pd.DataFrame:
    """将动力学缓存展平为 DataFrame，按 UniProt ID 对齐。"""
    rows = []

    for uid in oxidoreductase_uniprot_ids:
        kdata = kinetics_cache.get(uid, {})
        if not kdata or not kdata.get("has_kinetics"):
            continue

        ecs = kdata.get("ec_numbers", [])
        name = kdata.get("protein_name", "")
        reviewed = kdata.get("reviewed", False)

        # KM entries
        for km in kdata.get("km_entries", []):
            rows.append({
                "uniprot_id": uid,
                "protein_name": name,
                "ec_numbers": "|".join(ecs),
                "reviewed": reviewed,
                "param_type": "KM",
                "substrate": km["substrate"],
                "value": km["value_uM"],
                "unit": "uM",
                "ph": km["ph"],
                "temp_C": km["temp_C"],
                "pmid": km["pmid"],
            })

        # Vmax entries
        for vmax in kdata.get("vmax_entries", []):
            rows.append({
                "uniprot_id": uid,
                "protein_name": name,
                "ec_numbers": "|".join(ecs),
                "reviewed": reviewed,
                "param_type": "Vmax",
                "substrate": vmax["substrate"],
                "value": vmax["value"],
                "unit": vmax["unit"],
                "ph": vmax["ph"],
                "temp_C": vmax["temp_C"],
                "pmid": vmax["pmid"],
            })

        # kcat entries
        for kcat in kdata.get("kcat_entries", []):
            rows.append({
                "uniprot_id": uid,
                "protein_name": name,
                "ec_numbers": "|".join(ecs),
                "reviewed": reviewed,
                "param_type": "kcat",
                "substrate": kcat["substrate"],
                "value": kcat["value_s"],
                "unit": "s⁻¹",
                "ph": kcat["ph"],
                "temp_C": kcat["temp_C"],
                "pmid": kcat["pmid"],
            })

    df = pd.DataFrame(rows)
    return df


def compute_catalytic_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    对同一 UniProt 同一底物，当 KM 和 kcat 同时存在时，
    计算 kcat/KM (M⁻¹s⁻¹) —— 催化效率/有效 ΔG‡ 降低。
    """
    # Group by (uniprot_id, substrate) and check for KM+kcat pairs
    km_df = df[df["param_type"] == "KM"].copy()
    kcat_df = df[df["param_type"] == "kcat"].copy()

    efficiency_rows = []
    for (uid, sub), km_grp in km_df.groupby(["uniprot_id", "substrate"]):
        if (uid, sub) in kcat_df.groupby(["uniprot_id", "substrate"]).groups:
            kcat_grp = kcat_df.groupby(["uniprot_id", "substrate"]).get_group((uid, sub))
            for _, km_row in km_grp.iterrows():
                for _, kcat_row in kcat_grp.iterrows():
                    km_uM = km_row["value"]
                    kcat_s = kcat_row["value"]
                    # kcat/KM in M⁻¹s⁻¹
                    km_M = km_uM * 1e-6
                    if km_M > 0:
                        eff = kcat_s / km_M
                        efficiency_rows.append({
                            "uniprot_id": uid,
                            "substrate": sub,
                            "kcat_per_KM_M1s1": eff,
                            "log_kcat_per_KM": np.log10(eff) if eff > 0 else np.nan,
                            "KM_uM": km_uM,
                            "kcat_s": kcat_s,
                            "pmid_km": km_row["pmid"],
                            "pmid_kcat": kcat_row["pmid"],
                        })

    eff_df = pd.DataFrame(efficiency_rows)
    if len(eff_df) > 0:
        eff_df = eff_df.drop_duplicates(
            subset=["uniprot_id", "substrate", "KM_uM", "kcat_s"]
        )
    return eff_df


# ─────────────────────────────────────────────────────────────
# 4. 主入口
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 UniProt 补充氧化还原酶动力学标签"
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip UniProt API calls (use cache only)",
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Clear cache and re-fetch all",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel workers for UniProt API",
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = out_dir / "cache" / "kinetics_uniprot.json"

    if args.force_refresh and cache_path.exists():
        cache_path.unlink()

    # ── Step 1: Get oxidoreductase UniProt IDs ──
    log.info("=" * 50)
    log.info("Step 1: Loading oxidoreductase UniProt IDs")
    log.info("=" * 50)

    import pickle
    recs_path = out_dir / "oxidoreductase_records.pkl"
    if recs_path.exists():
        recs = pickle.load(open(recs_path, "rb"))
    else:
        log.warning("oxidoreductase_records.pkl not found. "
                     "Using UniProt annotation cache instead.")
        ann_cache_path = out_dir / "cache" / "uniprot_annotations.json"
        with open(ann_cache_path) as f:
            ann_cache = json.load(f)
        oxid_ids = set(
            uid for uid, a in ann_cache.items()
            if a.get("ec_numbers") and any(e.startswith("1.") for e in a["ec_numbers"])
        )
        recs = []  # No records to align

    oxid_uniprot_ids = set()
    for r in recs:
        uid = r.get("uniprot_id")
        if uid:
            oxid_uniprot_ids.add(uid)

    log.info(f"  {len(oxid_uniprot_ids)} unique oxidoreductase UniProt IDs")

    # ── Step 2: Fetch kinetic data ──
    log.info("=" * 50)
    log.info("Step 2: Fetching kinetic parameters from UniProt")
    log.info("=" * 50)

    if not args.no_fetch:
        kinetics_cache = fetch_all_kinetics(
            sorted(oxid_uniprot_ids),
            cache_path=cache_path,
            n_workers=args.workers,
        )
    else:
        if cache_path.exists():
            with open(cache_path) as f:
                kinetics_cache = json.load(f)
            log.info(f"Loaded {len(kinetics_cache)} cached (--no-fetch)")
        else:
            log.error("No cache found and --no-fetch. Exiting.")
            return

    # ── Step 3: Build dataset ──
    log.info("=" * 50)
    log.info("Step 3: Building kinetics dataset")
    log.info("=" * 50)

    df = build_kinetics_dataset(kinetics_cache, oxid_uniprot_ids)
    log.info(f"  Total kinetic entries: {len(df)}")

    # Summary statistics
    if len(df) > 0:
        n_proteins = df["uniprot_id"].nunique()
        n_km = len(df[df["param_type"] == "KM"])
        n_vmax = len(df[df["param_type"] == "Vmax"])
        n_kcat = len(df[df["param_type"] == "kcat"])

        log.info(f"  Unique proteins with kinetics: {n_proteins}")
        log.info(f"  KM entries:   {n_km}")
        log.info(f"  Vmax entries: {n_vmax}")
        log.info(f"  kcat entries: {n_kcat}")

        # Substrate distribution
        top_substrates = (
            df[df["param_type"] == "KM"]["substrate"]
            .value_counts().head(10)
        )
        log.info(f"  Top substrates (KM):")
        for sub, count in top_substrates.items():
            log.info(f"    {sub}: {count}")

    # ── Step 4: Compute catalytic efficiency ──
    log.info("=" * 50)
    log.info("Step 4: Computing kcat/KM (catalytic efficiency)")
    log.info("=" * 50)

    eff_df = compute_catalytic_efficiency(df)
    log.info(f"  kcat/KM entries: {len(eff_df)}")
    if len(eff_df) > 0:
        log.info(f"  log10(kcat/KM) range: "
                 f"[{eff_df['log_kcat_per_KM'].min():.2f}, "
                 f"{eff_df['log_kcat_per_KM'].max():.2f}]")
        log.info(f"  Diffusion limit (10^8-10^9 M⁻¹s⁻¹): "
                 f"{sum(eff_df['kcat_per_KM1s1'] > 1e8)} entries exceed 10^8")

    # ── Step 5: Save ──
    log.info("=" * 50)
    log.info("Step 5: Saving")
    log.info("=" * 50)

    kinetics_path = out_dir / "kinetics.parquet"
    df.to_parquet(kinetics_path, index=False)
    log.info(f"Kinetics saved → {kinetics_path}")

    if len(eff_df) > 0:
        eff_path = out_dir / "catalytic_efficiency.parquet"
        eff_df.to_parquet(eff_path, index=False)
        log.info(f"Catalytic efficiency saved → {eff_path}")

    # ── Align with existing records ──
    if recs:
        log.info("Aligning kinetics with oxidoreductase records...")
        # Build per-uniprot summary: median KM for main cofactor substrate
        aligned = 0
        for r in recs:
            uid = r.get("uniprot_id")
            if uid and uid in kinetics_cache:
                kdata = kinetics_cache[uid]
                if kdata.get("has_kinetics"):
                    r["has_kinetics"] = True
                    r["n_km_entries"] = len(kdata.get("km_entries", []))
                    r["n_kcat_entries"] = len(kdata.get("kcat_entries", []))
                    # Store first KM as representative
                    if kdata.get("km_entries"):
                        r["km_first_uM"] = kdata["km_entries"][0]["value_uM"]
                        r["km_first_substrate"] = kdata["km_entries"][0]["substrate"]
                    aligned += 1

        log.info(f"  Aligned kinetics to {aligned} records")

        # Re-save records with kinetics annotations
        aligned_path = out_dir / "oxidoreductase_records_with_kinetics.pkl"
        with open(aligned_path, "wb") as f:
            pickle.dump(recs, f)
        log.info(f"Records with kinetics saved → {aligned_path}")

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("  动力学标签补充 — 完成报告")
    print("=" * 60)
    print(f"  总蛋白质数:       {len(oxid_uniprot_ids)}")
    if len(df) > 0:
        print(f"  有动力学数据:     {df['uniprot_id'].nunique()}")
        print(f"  KM 条目数:        {len(df[df['param_type'] == 'KM'])}")
        print(f"  Vmax 条目数:      {len(df[df['param_type'] == 'Vmax'])}")
        print(f"  kcat 条目数:      {len(df[df['param_type'] == 'kcat'])}")
        print(f"  kcat/KM 计算值:   {len(eff_df)}")
    print(f"  输出目录:         {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
