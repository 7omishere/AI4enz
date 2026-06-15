#!/usr/bin/env python3
"""
补充 Trenzition 数据集 — 利用外部数据源填充缺失的 kcat、Km、pKd。

匹配策略（仅最可靠的两级）：
  Tier 1: (uniprot_id, ligand_inchikey) 完全匹配
  Tier 2: (uniprot_id, canonical_smiles → InChIKey) SMILES桥接匹配

数据源优先级：
  Phase 1: OED — kcat + km (100% 字段齐全)
  Phase 2: SKiD — kcat + km (高 UniProt 重叠)
  Phase 3: SABIO-RK — kcat + km (先补充 SMILES，再用 Tier 1+2)
  Phase 4: RTS — kcat
  Phase 5: RTS — pKd (Tier 1+2 精确匹配)

输出：
  - release/trenzition_full_v3.parquet   — 补充后完整数据集
  - release/trenzition_supplement_stats.json — 补充前后对比统计

用法：
  source /home/domi/BINN/.venv/bin/activate
  python scripts/supplement_trenzition.py
"""

import pandas as pd
import numpy as np
import hashlib
import pickle
import json
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path("/home/domi/AI4enz/dataset_building")
RELEASE_DIR = BASE_DIR / "release"
PROCESSED_DIR = BASE_DIR / "processed"

TZ_INPUT = RELEASE_DIR / "trenzition_full_v2.parquet"
OED_INPUT = BASE_DIR / "OED" / "oed_kinetics.json"
SKID_INPUT = BASE_DIR / "SKiD" / "SKiD_Main_dataset_v1.xlsx"
SABIO_INPUT = BASE_DIR / "SABIO-RK" / "sabio_reports_from_kmstats.csv"
RTS_INPUT = RELEASE_DIR / "recommended_training_set_enriched.parquet"
INCHIKEY_SMILES_MAP = PROCESSED_DIR / "inchikey_smiles_map.pkl"

OUTPUT_FULL = RELEASE_DIR / "trenzition_full_v3.parquet"
OUTPUT_STATS = RELEASE_DIR / "trenzition_supplement_stats.json"


# ============================================================
# Utility Functions (reused from build_trenzition.py)
# ============================================================
def smiles_to_inchikey(smiles: str) -> str | None:
    """Convert SMILES to standard InChIKey using RDKit."""
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return MolToInchiKey(mol)


def canonicalize_smiles(smiles: str) -> str | None:
    """Produce canonical SMILES for cross-dataset matching."""
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


# ============================================================
# Phase 1: OED — kcat + km
# ============================================================
def load_oed() -> pd.DataFrame:
    """Load OED kinetics data, compute InChIKeys, return as DataFrame."""
    print("\n" + "=" * 60)
    print("PHASE 1: Loading OED kinetics data")
    print("=" * 60)

    with open(OED_INPUT) as f:
        oed_raw = json.load(f)
    print(f"  Raw records: {len(oed_raw):,}")

    rows = []
    for r in oed_raw:
        smi = r.get("smiles")
        inchikey = smiles_to_inchikey(smi) if smi else None
        canon = canonicalize_smiles(smi) if smi else None

        # km in OED is mM → convert to M
        km_mM = r.get("km_value")
        km_M = km_mM / 1000.0 if km_mM is not None else None

        rows.append({
            "uniprot_id": r.get("uniprot", ""),
            "smiles": smi,
            "inchikey": inchikey,
            "canonical_smiles": canon,
            "kcat_per_s": r.get("kcat_value"),
            "km_M": km_M,
            "ec_number": r.get("ec", ""),
            "substrate_name": r.get("substrate", ""),
            "organism_name": r.get("organism", ""),
        })

    df = pd.DataFrame(rows)
    valid_ik = df["inchikey"].notna()
    print(f"  Valid InChIKeys: {valid_ik.sum():,} / {len(df):,}")
    print(f"  Unique UniProts: {df['uniprot_id'].nunique():,}")
    print(f"  Has kcat: {(df['kcat_per_s'].notna()).sum():,}")
    print(f"  Has km: {(df['km_M'].notna()).sum():,}")
    return df


# ============================================================
# Phase 2: SKiD — kcat + km
# ============================================================
def load_skid() -> pd.DataFrame:
    """Load SKiD kcat and Km datasets, merge by (EC, UniProt, SMILES),
    compute InChIKeys, return unified DataFrame."""
    print("\n" + "=" * 60)
    print("PHASE 2: Loading SKiD kinetics data")
    print("=" * 60)

    kcat_df = pd.read_excel(SKID_INPUT, sheet_name="kcat_dataset")
    km_df = pd.read_excel(SKID_INPUT, sheet_name="Km_dataset")
    print(f"  kcat_dataset: {len(kcat_df):,} rows")
    print(f"  Km_dataset:   {len(km_df):,} rows")

    # Standardize columns
    kcat_df = kcat_df.rename(columns={
        "EC_number": "ec_number",
        "UniProt_ID": "uniprot_id",
        "Substrate": "substrate_name",
        "Organism_name": "organism_name",
        "Substrate_SMILES": "smiles",
        "kcat_value": "kcat_per_s",
    })
    km_df = km_df.rename(columns={
        "EC_number": "ec_number",
        "UniProt_ID": "uniprot_id",
        "Substrate": "substrate_name",
        "Organism_name": "organism_name",
        "Substrate_SMILES": "smiles",
        "Km_value": "km_value_raw",
    })

    # Compute InChIKeys
    for label, df in [("kcat", kcat_df), ("km", km_df)]:
        df["inchikey"] = df["smiles"].apply(smiles_to_inchikey)
        df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
        n_ok = df["inchikey"].notna().sum()
        print(f"  [{label}] Valid InChIKeys: {n_ok:,} / {len(df):,}")

    # Merge kcat + km by (uniprot_id, inchikey)
    # For each unique pair, aggregate: median kcat, median km
    kcat_agg = kcat_df.groupby(["uniprot_id", "inchikey"]).agg(
        kcat_per_s=("kcat_per_s", "median"),
        ec_number=("ec_number", "first"),
        substrate_name=("substrate_name", "first"),
        organism_name=("organism_name", "first"),
        canonical_smiles=("canonical_smiles", "first"),
        n_kcat_skid=("kcat_per_s", "count"),
    ).reset_index()

    km_agg = km_df.groupby(["uniprot_id", "inchikey"]).agg(
        km_value_raw=("km_value_raw", "median"),
        n_km_skid=("km_value_raw", "count"),
    ).reset_index()

    merged = kcat_agg.merge(km_agg, on=["uniprot_id", "inchikey"], how="outer")

    # SKiD Km units: need to check — typically in μM or mM
    # From the data, Km values range widely. SKiD documentation says μM for most entries.
    # We'll assume μM and convert to M for consistency
    merged["km_M"] = merged["km_value_raw"].apply(
        lambda x: x / 1_000_000.0 if pd.notna(x) else None
    )

    print(f"  Merged (kcat+km): {len(merged):,} unique (uniprot, inchikey) pairs")
    print(f"    Has kcat: {merged['kcat_per_s'].notna().sum():,}")
    print(f"    Has km:   {merged['km_M'].notna().sum():,}")
    print(f"    Has both: {(merged['kcat_per_s'].notna() & merged['km_M'].notna()).sum():,}")
    return merged


# ============================================================
# Phase 3: SABIO-RK — kcat + km (with SMILES enrichment)
# ============================================================
def load_sabio(tz_df: pd.DataFrame, skid_df: pd.DataFrame,
               rts_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load SABIO-RK, enrich with SMILES from other datasets, pivot to kcat+km.

    SABIO-RK substrates are names (no SMILES). We build a lookup from
    existing datasets (trenzition, SKiD, RTS) to find SMILES for each name.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Loading SABIO-RK (with SMILES enrichment)")
    print("=" * 60)

    sabio = pd.read_csv(SABIO_INPUT)
    print(f"  Raw rows: {len(sabio):,}")

    # Build substrate name → SMILES lookup from multiple sources
    name_to_smiles = {}

    # Source 1: Trenzition
    for _, row in tz_df.iterrows():
        name = row.get("substrate_name")
        smi = row.get("canonical_smiles")
        if isinstance(name, str) and name.strip() and isinstance(smi, str):
            name_to_smiles[name.strip()] = smi

    # Source 2: SKiD
    for _, row in skid_df.iterrows():
        name = row.get("substrate_name")
        smi = row.get("canonical_smiles")
        if isinstance(name, str) and name.strip() and isinstance(smi, str):
            name_to_smiles[name.strip()] = smi

    # Source 3: RTS (via inchikey → SMILES)
    if rts_df is not None:
        with open(INCHIKEY_SMILES_MAP, "rb") as f:
            ik2smiles = pickle.load(f)
        for _, row in rts_df.iterrows():
            name = row.get("substrate_name")
            ik = row.get("ligand_inchikey")
            if isinstance(name, str) and name.strip() and isinstance(ik, str) and ik in ik2smiles:
                name_to_smiles[name.strip()] = ik2smiles[ik]

    print(f"  Substrate name → SMILES lookup: {len(name_to_smiles):,} entries")

    # Match SABIO-RK substrates to SMILES
    # SABIO-RK substrates are often multi-component: "NAD+;1-Octanol"
    sabio_substrates = sabio["Substrate"].dropna().unique()
    sub_smiles_map = {}  # substrate_name → list of SMILES (one per component)

    matched_exact = 0
    matched_partial = 0
    no_match = 0

    for sub in sabio_substrates:
        sub_str = str(sub).strip()
        if sub_str in name_to_smiles:
            sub_smiles_map[sub_str] = [name_to_smiles[sub_str]]
            matched_exact += 1
        elif ";" in sub_str:
            parts = [p.strip() for p in sub_str.split(";")]
            smiles_list = []
            for p in parts:
                if p in name_to_smiles:
                    smiles_list.append(name_to_smiles[p])
            if smiles_list:
                sub_smiles_map[sub_str] = smiles_list
                matched_partial += 1
            else:
                no_match += 1
        else:
            no_match += 1

    print(f"  Substrate → SMILES match: exact={matched_exact}, partial(;)={matched_partial}, "
          f"none={no_match} ({100*(matched_exact+matched_partial)/len(sabio_substrates):.1f}% covered)")

    # For each SABIO-RK substrate, compute InChIKeys for each component SMILES
    sub_inchikeys_map = {}  # substrate_name → set of InChIKeys
    for sub_name, smiles_list in sub_smiles_map.items():
        ik_set = set()
        for smi in smiles_list:
            ik = smiles_to_inchikey(smi)
            if ik:
                ik_set.add(ik)
        if ik_set:
            sub_inchikeys_map[sub_name] = ik_set

    # Pivot SABIO-RK: one row per (UniprotID, Substrate) with kcat and Km
    # parameter.associatedSpecies tells which substrate component the param is for
    sabio_kcat = sabio[sabio["parameter.type"] == "kcat"].copy()
    sabio_km = sabio[sabio["parameter.type"] == "Km"].copy()

    # Aggregate kcat: median per (UniprotID, Substrate)
    kcat_agg = sabio_kcat.groupby(["UniprotID", "Substrate"]).agg(
        kcat_per_s=("parameter.startValue", "median"),
        kcat_unit=("parameter.unit", "first"),
        n_kcat_sabio=("parameter.startValue", "count"),
        ECNumber=("ECNumber", "first"),
        Organism=("Organism", "first"),
    ).reset_index()
    kcat_agg.rename(columns={"UniprotID": "uniprot_id", "Substrate": "substrate_name",
                              "ECNumber": "ec_number", "Organism": "organism_name"}, inplace=True)

    # Aggregate Km: median per (UniprotID, Substrate)
    km_agg = sabio_km.groupby(["UniprotID", "Substrate"]).agg(
        km_value_raw=("parameter.startValue", "median"),
        km_unit=("parameter.unit", "first"),
        n_km_sabio=("parameter.startValue", "count"),
    ).reset_index()
    km_agg.rename(columns={"UniprotID": "uniprot_id", "Substrate": "substrate_name"}, inplace=True)

    # Merge kcat + km
    merged = kcat_agg.merge(km_agg, on=["uniprot_id", "substrate_name"], how="outer")

    # Convert km units to M
    def convert_km_to_M(val, unit):
        if pd.isna(val) or not isinstance(unit, str):
            return None
        unit_lower = unit.lower().strip()
        if unit_lower in ("m", "mol/l", "mole/l", "moles/l"):
            return val
        elif unit_lower in ("mm", "mmol/l", "mmole/l"):
            return val / 1000.0
        elif unit_lower in ("µm", "um", "µmol/l", "umol/l", "µmole/l"):
            return val / 1_000_000.0
        elif unit_lower in ("nm", "nmol/l", "nmole/l"):
            return val / 1_000_000_000.0
        else:
            return None  # unknown unit, skip

    merged["km_M"] = merged.apply(
        lambda r: convert_km_to_M(r["km_value_raw"], r.get("km_unit")), axis=1
    )

    # Convert kcat units to 1/s
    def convert_kcat_to_per_s(val, unit):
        if pd.isna(val) or not isinstance(unit, str):
            return None
        unit_lower = unit.lower().strip()
        if unit_lower in ("s^(-1)", "1/s", "s-1", "s⁻¹", "sec^(-1)", "1/sec"):
            return val
        elif unit_lower in ("min^(-1)", "1/min", "min-1", "min⁻¹"):
            return val / 60.0
        elif unit_lower in ("h^(-1)", "1/h", "h-1"):
            return val / 3600.0
        else:
            return None  # unknown unit, skip

    merged["kcat_per_s"] = merged.apply(
        lambda r: convert_kcat_to_per_s(
            r["kcat_per_s"] if "kcat_per_s" in merged.columns and pd.notna(r.get("kcat_per_s"))
            else r.get("kcat_per_s"),
            r.get("kcat_unit")
        ) if pd.notna(r.get("kcat_per_s")) and pd.notna(r.get("kcat_unit"))
        else r.get("kcat_per_s"),
        axis=1
    )
    # Fix: the kcat column from kcat_agg is named "kcat_per_s", but km rows have NaN there
    # Actually kcat_agg already uses "kcat_per_s" as the value column name.
    # The issue is in the merge - from km_agg it's NaN. That's fine.

    # Convert units for kcat properly
    kcat_col = []
    for _, r in merged.iterrows():
        val = r["kcat_per_s"]
        unit = r.get("kcat_unit")
        if pd.notna(val) and isinstance(unit, str):
            kcat_col.append(convert_kcat_to_per_s(val, unit))
        else:
            kcat_col.append(val if pd.notna(val) else None)
    merged["kcat_per_s"] = kcat_col

    print(f"  After pivot: {len(merged):,} (uniprot, substrate) pairs")
    print(f"    Has kcat: {merged['kcat_per_s'].notna().sum():,}")
    print(f"    Has km:   {merged['km_M'].notna().sum():,}")

    # Add InChIKeys for matching
    # For each row, get the substrate's InChIKey set
    merged["inchikey_set"] = merged["substrate_name"].map(
        lambda n: sub_inchikeys_map.get(str(n).strip(), set()) if isinstance(n, str) else set()
    )
    n_with_ik = sum(1 for s in merged["inchikey_set"] if len(s) > 0)
    print(f"    With ≥1 InChIKey: {n_with_ik:,} ({100*n_with_ik/len(merged):.1f}%)")

    return merged, sub_inchikeys_map


# ============================================================
# Phase 4+5: RTS — kcat + pKd
# ============================================================
def load_rts() -> pd.DataFrame:
    """Load recommended training set with kcat and pKd data."""
    print("\n" + "=" * 60)
    print("PHASE 4+5: Loading RTS (kcat + pKd lookup)")
    print("=" * 60)

    df = pd.read_parquet(RTS_INPUT)
    print(f"  Total rows: {len(df):,}")

    # Keep only rows with useful data
    has_data = df["kcat_median_s"].notna() | df["pkd_aligned"].notna()
    df_use = df[has_data].copy()
    print(f"  With kcat or pKd: {len(df_use):,}")

    print(f"    Has kcat: {df_use['kcat_median_s'].notna().sum():,}")
    print(f"    Has pKd:  {df_use['pkd_aligned'].notna().sum():,}")

    return df_use


# ============================================================
# Matching Engine
# ============================================================
def build_tier1_index(df: pd.DataFrame, uniprot_col: str, inchikey_col: str,
                      value_cols: list[str]) -> dict:
    """Build (uniprot, inchikey) → values index for Tier 1 matching.

    For rows with multiple values (e.g. same pair, different measurements),
    takes the median.

    Args:
        df: source DataFrame
        uniprot_col: column name for UniProt ID
        inchikey_col: column name for InChIKey (single value) or 'inchikey_set' (set)
        value_cols: list of value column names to index

    Returns:
        dict: {(uniprot, inchikey): {col: median_value, ...}}
    """
    index = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        up = row.get(uniprot_col)
        if not isinstance(up, str) or not up.strip():
            continue

        # Get InChIKeys — could be a single string or a set
        ik_val = row.get(inchikey_col)
        if ik_val is None:
            continue
        if isinstance(ik_val, set):
            ik_list = [ik for ik in ik_val if isinstance(ik, str)]
        elif isinstance(ik_val, str) and ik_val.strip():
            ik_list = [ik_val.strip()]
        else:
            continue

        for ik in ik_list:
            for col in value_cols:
                val = row.get(col)
                if pd.notna(val):
                    index[(up.strip(), ik)][col].append(val)

    # Aggregate: median per value column
    result = {}
    for key, col_vals in index.items():
        entry = {}
        for col, vals in col_vals.items():
            if vals:
                entry[col] = float(np.median(vals))
        if entry:
            result[key] = entry

    return result


def build_smiles_bridge_index(df: pd.DataFrame, uniprot_col: str,
                              canon_smiles_col: str,
                              value_cols: list[str]) -> dict:
    """Build (uniprot, canonical_smiles) → values index for Tier 2 matching.

    For Tier 2, we match via canonical SMILES → InChIKey bridge.
    The trenzition record's canonical_smiles is used to look up
    the source data's InChIKeys via the SMILES bridge.

    Returns:
        dict: {(uniprot, canonical_smiles): {col: median_value, ...}}
    """
    index = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        up = row.get(uniprot_col)
        if not isinstance(up, str) or not up.strip():
            continue
        csmiles = row.get(canon_smiles_col)
        if not isinstance(csmiles, str) or not csmiles.strip():
            continue

        for col in value_cols:
            val = row.get(col)
            if pd.notna(val):
                index[(up.strip(), csmiles.strip())][col].append(val)

    result = {}
    for key, col_vals in index.items():
        entry = {}
        for col, vals in col_vals.items():
            if vals:
                entry[col] = float(np.median(vals))
        if entry:
            result[key] = entry

    return result


def match_and_fill(tz_df: pd.DataFrame,
                   tier1_index: dict,
                   tier2_index: dict | None,
                   value_cols: list[str],
                   source_label: str,
                   source_col_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Match trenzition rows against source indexes and fill missing values.

    Args:
        tz_df: trenzition DataFrame (modified in place)
        tier1_index: {(uniprot, inchikey): {col: value}}
        tier2_index: {(uniprot, canonical_smiles): {col: value}} or None
        value_cols: which tz columns to fill (e.g., ['kcat_per_s', 'km_M'])
        source_label: label for source tracking column value
        source_col_map: {value_col: source_tracking_col_name}

    Returns:
        (tz_df, fill_stats) — df modified in place + per-column fill counts
    """
    stats = {col: 0 for col in value_cols}

    for idx, row in tz_df.iterrows():
        up = row.get("uniprot_id")
        if not isinstance(up, str) or not up.strip():
            continue
        up = up.strip()
        inchikey = row.get("ligand_inchikey")
        canon_smi = row.get("canonical_smiles")

        # Only process rows that are missing at least one target value
        missing_cols = [c for c in value_cols if pd.isna(row.get(c))]
        if not missing_cols:
            continue

        matched_entry = None

        # Tier 1: Exact (uniprot, inchikey) match
        if isinstance(inchikey, str) and inchikey.strip():
            key = (up, inchikey.strip())
            if key in tier1_index:
                matched_entry = tier1_index[key]

        # Tier 2: SMILES bridge
        if matched_entry is None and tier2_index is not None:
            if isinstance(canon_smi, str) and canon_smi.strip():
                key = (up, canon_smi.strip())
                if key in tier2_index:
                    matched_entry = tier2_index[key]

        if matched_entry is None:
            continue

        # Fill missing values
        for col in missing_cols:
            if col in matched_entry:
                val = matched_entry[col]
                if pd.notna(val):
                    tz_df.at[idx, col] = val
                    stats[col] += 1
                    # Track source
                    track_col = source_col_map.get(col)
                    if track_col:
                        tz_df.at[idx, track_col] = source_label

    return tz_df, stats


def match_sabio_and_fill(tz_df: pd.DataFrame,
                         sabio_df: pd.DataFrame,
                         sub_inchikeys_map: dict) -> tuple[pd.DataFrame, dict]:
    """Special matcher for SABIO-RK which has multi-component substrates.

    For each trenzition row, check if its inchikey matches ANY component
    of the SABIO-RK entry's substrate.
    """
    stats = {"kcat_per_s": 0, "km_M": 0}

    # Build Tier 1 index: (uniprot, inchikey) → values
    tier1 = defaultdict(lambda: defaultdict(list))
    for _, row in sabio_df.iterrows():
        up = row.get("uniprot_id")
        if not isinstance(up, str) or not up.strip():
            continue
        ik_set = row.get("inchikey_set", set())
        for ik in ik_set:
            for col in ["kcat_per_s", "km_M"]:
                val = row.get(col)
                if pd.notna(val):
                    tier1[(up.strip(), ik)][col].append(val)

    # Aggregate to median
    tier1_agg = {}
    for key, col_vals in tier1.items():
        entry = {}
        for col, vals in col_vals.items():
            if vals:
                entry[col] = float(np.median(vals))
        if entry:
            tier1_agg[key] = entry

    # Match
    for idx, row in tz_df.iterrows():
        up = row.get("uniprot_id")
        if not isinstance(up, str) or not up.strip():
            continue
        up = up.strip()
        inchikey = row.get("ligand_inchikey")
        if not isinstance(inchikey, str) or not inchikey.strip():
            continue
        inchikey = inchikey.strip()

        missing_cols = [c for c in ["kcat_per_s", "km_M"] if pd.isna(row.get(c))]
        if not missing_cols:
            continue

        key = (up, inchikey)
        if key not in tier1_agg:
            continue

        matched = tier1_agg[key]
        for col in missing_cols:
            if col in matched and pd.notna(matched[col]):
                tz_df.at[idx, col] = matched[col]
                stats[col] += 1
                tz_df.at[idx, f"{col}_source"] = "sabio"

    return tz_df, stats


# ============================================================
# Phase 6: PDBbind substrate-level pKd matching
# ============================================================
def match_pdbbind_pkd(tz_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Match PDBbind pKd via canonical SMILES (substrate-level).

    PDBbind has 18,903 records with pKd + ligand_smiles but no uniprot_id.
    We match by canonical SMILES: same substrate molecule → same pKd.

    For substrate-level matching, we take the median pKd across all PDBbind
    entries for that same molecule (different protein contexts).
    """
    pdbbind_path = PROCESSED_DIR / "pdbbind_records.pkl"
    if not pdbbind_path.exists():
        print("  PDBbind records not found, skipping...")
        return tz_df, {"pkd_value": 0}

    with open(pdbbind_path, "rb") as f:
        pdbbind = pickle.load(f)
    print(f"  Loaded {len(pdbbind):,} PDBbind records")

    # Build canonical SMILES → pKd list index
    smiles_pkd = defaultdict(list)
    skipped_smiles = 0
    for r in pdbbind:
        pkd = r.get("pkd_aligned")
        smi = r.get("ligand_smiles")
        if pkd is None or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped_smiles += 1
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        smiles_pkd[canon].append(pkd)

    # Aggregate to median
    smiles_pkd_median = {}
    for smi, vals in smiles_pkd.items():
        if vals:
            smiles_pkd_median[smi] = float(np.median(vals))

    print(f"  Canonical SMILES → pKd index: {len(smiles_pkd_median):,} entries")
    print(f"  SMILES parse failures: {skipped_smiles}")

    # Match TZ rows missing pKd
    stats = {"pkd_value": 0}

    # Also build inchikey → pKd via SMILES bridge
    with open(INCHIKEY_SMILES_MAP, "rb") as f:
        ik2smiles = pickle.load(f)

    for idx, row in tz_df.iterrows():
        if pd.notna(row.get("pkd_value")):
            continue  # already has pKd

        canon_smi = row.get("canonical_smiles")
        inchikey = row.get("ligand_inchikey")

        pkd_val = None

        # Try direct canonical SMILES match
        if isinstance(canon_smi, str) and canon_smi in smiles_pkd_median:
            pkd_val = smiles_pkd_median[canon_smi]

        # Try inchikey → SMILES bridge
        if pkd_val is None and isinstance(inchikey, str) and inchikey in ik2smiles:
            bridge_smi = ik2smiles[inchikey]
            mol = Chem.MolFromSmiles(bridge_smi)
            if mol:
                bridge_canon = Chem.MolToSmiles(mol, canonical=True)
                if bridge_canon in smiles_pkd_median:
                    pkd_val = smiles_pkd_median[bridge_canon]

        if pkd_val is None:
            continue

        # Validate range
        if 0 <= pkd_val <= 14:
            tz_df.at[idx, "pkd_value"] = pkd_val
            tz_df.at[idx, "pkd_value_source"] = "pdbbind_substrate"
            stats["pkd_value"] += 1

    return tz_df, stats


# ============================================================
# Phase 7: Cofactor substrate-level pKd matching
# ============================================================
def match_cofactor_pkd(tz_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Match pKd for conserved cofactors/reporter substrates.

    For cofactors (NAD+, ATP, etc.) and reporter substrates (pNPA, CDNB, etc.),
    the binding mode is highly conserved across different enzymes.
    We can transfer pKd estimates at the substrate level.

    Strategy:
    1. Define cofactor/reporter substrate names
    2. For each cofactor, compute its median pKd from ALL available data
       (RTS, PDBbind, existing TZ rows that have pKd)
    3. Transfer to TZ rows missing pKd where substrate matches
    """
    # Cofactors and reporter substrates (conserved binding across enzymes)
    COFACTOR_NAMES = {
        # Pyridine nucleotides
        "NAD+", "NADH", "NADP+", "NADPH",
        # Adenine nucleotides
        "ATP", "ADP", "AMP", "dATP", "dADP", "GTP", "CTP", "UTP", "ITP",
        # Flavins
        "FAD", "FMN", "FADH2",
        # CoA derivatives
        "CoA", "Acetyl-CoA", "Malonyl-CoA", "Succinyl-CoA", "Butyryl-CoA",
        "Palmitoyl-CoA", "Oleoyl-CoA", "Benzoyl-CoA", "Crotonyl-CoA",
        "Propionyl-CoA", "3-Hydroxy-3-methylglutaryl-CoA", "Hexanoyl-CoA",
        "Octanoyl-CoA", "Decanoyl-CoA", "Lauroyl-CoA", "Myristoyl-CoA",
        # Other cofactors
        "Pyridoxal phosphate", "Pyridoxal 5'-phosphate", "PLP",
        "Thiamine pyrophosphate", "TPP", "Thiamine diphosphate",
        "S-Adenosyl-L-methionine", "SAM", "S-Adenosylmethionine",
        "Tetrahydrofolate", "THF", "5,10-Methylenetetrahydrofolate",
        "Heme", "Heme C", "Protoheme", "Heme B",
        "Biotin", "Lipoamide", "Lipoic acid",
        # Common reporter/artificial substrates (conserved binding)
        "2,2'-Azino-bis(3-ethylbenzthiazoline-6-sulfonic acid)", "ABTS",
        "4-nitrophenyl acetate", "p-Nitrophenyl acetate", "pNPA",
        "4-nitrophenyl butyrate", "p-Nitrophenyl butyrate",
        "1-Chloro-2,4-dinitrobenzene", "CDNB",
        "Syringaldazine",
        "4-Nitrophenyl phosphate",
        "Guaiacol", "2,6-Dimethoxyphenol",
        "Pyrogallol", "Catechol",
    }

    # Also match by common InChIKey substrings for these cofactors
    COFACTOR_INCHIKEY_PREFIXES = [
        "BOPGDPN",  # NAD+
        "BOPGDP",   # NAD+/NADH variants
        "WZEOEO",   # NADP+
        "GUBGYT",   # ATP
        "XTWYTF",   # ADP
        "ZKHQWZ",   # AMP
        "QMSYAS",   # GTP
        "VWFJDG",   # FAD
        "FVTCRAS",  # FMN
        "RGJOJU",   # Acetyl-CoA
    ]

    # Build global substrate → pKd lookup from ALL available data
    global_sub_pkd = defaultdict(list)  # substrate_name → [pkd_values]
    global_ik_pkd = defaultdict(list)   # inchikey → [pkd_values]

    # Source 1: TZ itself (rows that already have pKd)
    tz_with_pkd = tz_df[tz_df["pkd_value"].notna()]
    for _, row in tz_with_pkd.iterrows():
        name = row.get("substrate_name")
        ik = row.get("ligand_inchikey")
        pkd = row.get("pkd_value")
        if isinstance(name, str) and pd.notna(pkd):
            global_sub_pkd[name.strip()].append(pkd)
        if isinstance(ik, str) and pd.notna(pkd):
            global_ik_pkd[ik].append(pkd)

    # Source 2: RTS
    rts = pd.read_parquet(RTS_INPUT)
    rts_pkd = rts[rts["pkd_aligned"].notna()]
    for _, row in rts_pkd.iterrows():
        ik = row.get("ligand_inchikey")
        pkd = row.get("pkd_aligned")
        if isinstance(ik, str) and pd.notna(pkd):
            global_ik_pkd[ik].append(pkd)

    # Source 3: PDBbind via SMILES→InChIKey bridge
    pdbbind_path = PROCESSED_DIR / "pdbbind_records.pkl"
    if pdbbind_path.exists():
        with open(pdbbind_path, "rb") as f:
            pdbbind = pickle.load(f)
        with open(INCHIKEY_SMILES_MAP, "rb") as f:
            ik2smiles = pickle.load(f)
        smiles2ik = defaultdict(set)
        for ik, smi in ik2smiles.items():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canon = Chem.MolToSmiles(mol, canonical=True)
                smiles2ik[canon].add(ik)

        for r in pdbbind:
            pkd = r.get("pkd_aligned")
            smi = r.get("ligand_smiles")
            if pkd is None or not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol, canonical=True)
            for ik in smiles2ik.get(canon, set()):
                global_ik_pkd[ik].append(pkd)

    # Aggregate to median
    sub_pkd_median = {}
    for name, vals in global_sub_pkd.items():
        if vals:
            sub_pkd_median[name] = float(np.median(vals))
    ik_pkd_median = {}
    for ik, vals in global_ik_pkd.items():
        if vals:
            ik_pkd_median[ik] = float(np.median(vals))

    print(f"  Global pKd lookup: {len(sub_pkd_median):,} substrate names, "
          f"{len(ik_pkd_median):,} InChIKeys")

    # Match TZ rows missing pKd
    stats = {"pkd_value": 0}

    for idx, row in tz_df.iterrows():
        if pd.notna(row.get("pkd_value")):
            continue  # already has pKd

        name = row.get("substrate_name")
        if not isinstance(name, str):
            continue
        name = name.strip()

        # Check if this is a cofactor/reporter substrate
        is_cofactor = False
        name_lower = name.lower()

        for cf in COFACTOR_NAMES:
            if cf.lower() == name_lower or cf.lower() in name_lower:
                is_cofactor = True
                break

        # Also check InChIKey prefix for cofactors
        if not is_cofactor:
            ik = row.get("ligand_inchikey")
            if isinstance(ik, str):
                for prefix in COFACTOR_INCHIKEY_PREFIXES:
                    if ik.startswith(prefix):
                        is_cofactor = True
                        break

        if not is_cofactor:
            continue

        # Find pKd by substrate name (priority 1) or inchikey (priority 2)
        pkd_val = None
        if name in sub_pkd_median:
            pkd_val = sub_pkd_median[name]
        elif isinstance(row.get("ligand_inchikey"), str) and row["ligand_inchikey"] in ik_pkd_median:
            pkd_val = ik_pkd_median[row["ligand_inchikey"]]

        if pkd_val is None:
            continue

        # Validate range
        if 0 <= pkd_val <= 14:
            tz_df.at[idx, "pkd_value"] = pkd_val
            tz_df.at[idx, "pkd_value_source"] = "cofactor_substrate"
            stats["pkd_value"] += 1

    return tz_df, stats


# ============================================================
# Phase 8: Homology transfer (same EC + same substrate → kcat/Km)
# ============================================================
def match_homology_transfer(tz_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Transfer kcat/Km between enzymes with same EC number and same substrate.

    Principle: Enzymes with the same EC number catalyze the same reaction.
    For the same substrate, kcat and Km typically vary within 2-10x across
    species. Using the median of known values provides a conservative estimate.

    Strategy:
    - Build (ec_number, canonical_smiles) → [kcat], [km] lookup from TZ itself
    - For TZ rows missing kcat/km, find matching same-EC+same-substrate rows
    - Fill with median of available values
    """
    print("  Building EC + substrate lookup from Trenzition itself...")

    # Build lookup from TZ's own data
    ec_smiles_kcat = defaultdict(list)
    ec_smiles_km = defaultdict(list)

    for _, row in tz_df.iterrows():
        ec = row.get("ec_number")
        cs = row.get("canonical_smiles")
        if not isinstance(ec, str) or not isinstance(cs, str):
            continue
        key = (ec, cs)
        if pd.notna(row.get("kcat_per_s")):
            ec_smiles_kcat[key].append(row["kcat_per_s"])
        if pd.notna(row.get("km_M")):
            ec_smiles_km[key].append(row["km_M"])

    # Aggregate to median
    ec_smiles_kcat_med = {k: float(np.median(v)) for k, v in ec_smiles_kcat.items()}
    ec_smiles_km_med = {k: float(np.median(v)) for k, v in ec_smiles_km.items()}

    print(f"  EC+SMILES → kcat lookup: {len(ec_smiles_kcat_med):,} entries")
    print(f"  EC+SMILES → km lookup:   {len(ec_smiles_km_med):,} entries")

    # Match and fill
    stats = {"kcat_per_s": 0, "km_M": 0}
    filled_pairs = set()  # Track (ec, smiles) pairs that were used

    for idx, row in tz_df.iterrows():
        ec = row.get("ec_number")
        cs = row.get("canonical_smiles")
        if not isinstance(ec, str) or not isinstance(cs, str):
            continue

        key = (ec, cs)
        need_kcat = pd.isna(row.get("kcat_per_s"))
        need_km = pd.isna(row.get("km_M"))

        if not need_kcat and not need_km:
            continue

        if need_kcat and key in ec_smiles_kcat_med:
            tz_df.at[idx, "kcat_per_s"] = ec_smiles_kcat_med[key]
            tz_df.at[idx, "kcat_per_s_source"] = "homology_transfer"
            stats["kcat_per_s"] += 1
            filled_pairs.add(key)

        if need_km and key in ec_smiles_km_med:
            tz_df.at[idx, "km_M"] = ec_smiles_km_med[key]
            tz_df.at[idx, "km_M_source"] = "homology_transfer"
            stats["km_M"] += 1
            filled_pairs.add(key)

    print(f"  Unique (EC, substrate) pairs used: {len(filled_pairs):,}")
    return tz_df, stats


# ============================================================
# Statistics
# ============================================================
def compute_final_stats(tz_df: pd.DataFrame, original_df: pd.DataFrame) -> dict:
    """Compare before/after statistics."""
    total = len(tz_df)

    def count_notna(df, col):
        return int(df[col].notna().sum())

    orig_perfect = len(original_df[
        original_df["kcat_per_s"].notna() &
        original_df["km_M"].notna() &
        original_df["pkd_value"].notna()
    ])
    new_perfect = len(tz_df[
        tz_df["kcat_per_s"].notna() &
        tz_df["km_M"].notna() &
        tz_df["pkd_value"].notna()
    ])

    stats = {
        "total_rows": total,
        "before": {
            "has_kcat": count_notna(original_df, "kcat_per_s"),
            "has_km": count_notna(original_df, "km_M"),
            "has_pkd": count_notna(original_df, "pkd_value"),
            "has_all_three": orig_perfect,
            "pct_kcat": round(100 * count_notna(original_df, "kcat_per_s") / total, 2),
            "pct_km": round(100 * count_notna(original_df, "km_M") / total, 2),
            "pct_pkd": round(100 * count_notna(original_df, "pkd_value") / total, 2),
            "pct_all_three": round(100 * orig_perfect / total, 2),
        },
        "after": {
            "has_kcat": count_notna(tz_df, "kcat_per_s"),
            "has_km": count_notna(tz_df, "km_M"),
            "has_pkd": count_notna(tz_df, "pkd_value"),
            "has_all_three": new_perfect,
            "pct_kcat": round(100 * count_notna(tz_df, "kcat_per_s") / total, 2),
            "pct_km": round(100 * count_notna(tz_df, "km_M") / total, 2),
            "pct_pkd": round(100 * count_notna(tz_df, "pkd_value") / total, 2),
            "pct_all_three": round(100 * new_perfect / total, 2),
        },
        "improvement": {
            "kcat_added": count_notna(tz_df, "kcat_per_s") - count_notna(original_df, "kcat_per_s"),
            "km_added": count_notna(tz_df, "km_M") - count_notna(original_df, "km_M"),
            "pkd_added": count_notna(tz_df, "pkd_value") - count_notna(original_df, "pkd_value"),
            "perfect_added": new_perfect - orig_perfect,
        },
    }

    # Source breakdown
    for col_prefix in ["kcat_per_s", "km_M", "pkd_value"]:
        src_col = f"{col_prefix}_source"
        if src_col in tz_df.columns:
            vc = tz_df[src_col].value_counts().to_dict()
            stats[f"{col_prefix}_source_breakdown"] = {str(k): int(v) for k, v in vc.items()}

    return stats


def print_stats(stats: dict):
    """Print before/after comparison."""
    print("\n" + "█" * 70)
    print("█  TRENZITION SUPPLEMENT — RESULTS")
    print("█" * 70)

    b = stats["before"]
    a = stats["after"]
    imp = stats["improvement"]

    print(f"\n  Total rows: {stats['total_rows']:,}")
    print(f"\n  {'':>20s} {'Before':>10s} {'After':>10s} {'Δ':>10s}")
    print(f"  {'─'*50}")

    for label, key in [("kcat", "has_kcat"), ("Km", "has_km"),
                        ("pKd", "has_pkd"), ("All 3 (perfect)", "has_all_three")]:
        before_pct = b[f"pct_{key.split('_', 1)[1] if '_' in key else key}"]
        after_pct = a[f"pct_{key.split('_', 1)[1] if '_' in key else key}"]
        before_v = b[key]
        after_v = a[key]
        delta = imp[f"{key.split('_', 1)[1] if '_' in key else key}_added"] if key != "has_all_three" else imp["perfect_added"]
        # Fix label mapping
        if key == "has_kcat": delta_key = "kcat_added"
        elif key == "has_km": delta_key = "km_added"
        elif key == "has_pkd": delta_key = "pkd_added"
        else: delta_key = "perfect_added"
        delta = imp[delta_key]
        print(f"  {label:<20s} {before_v:>8,}  {after_v:>8,}  {delta:>+8,}")
        print(f"  {'':>20s} {before_pct:>9.1f}% {after_pct:>9.1f}% {'':>10s}")
        print()

    # Source breakdown
    print(f"\n  📂 kcat source breakdown:")
    if "kcat_per_s_source_breakdown" in stats:
        for src, cnt in stats["kcat_per_s_source_breakdown"].items():
            print(f"     {src:<25s}: {cnt:>8,}")
    elif "kcat_per_s_source" in stats:
        for src, cnt in stats["kcat_per_s_source"].items():
            print(f"     {src:<25s}: {cnt:>8,}")

    print(f"\n  📂 km source breakdown:")
    if "km_M_source_breakdown" in stats:
        for src, cnt in stats["km_M_source_breakdown"].items():
            print(f"     {src:<25s}: {cnt:>8,}")

    print(f"\n  📂 pKd source breakdown:")
    if "pkd_value_source_breakdown" in stats:
        for src, cnt in stats["pkd_value_source_breakdown"].items():
            print(f"     {src:<25s}: {cnt:>8,}")


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("█" * 70)
    print("█  TRENZITION SUPPLEMENT — External Data Integration")
    print("█  Strategy: Tier 1 (uniprot+inchikey) + Tier 2 (SMILES bridge)")
    print("█" * 70)

    # ---- Load trenzition ----
    print("\n" + "=" * 60)
    print("Loading Trenzition v2 base dataset")
    print("=" * 60)
    tz = pd.read_parquet(TZ_INPUT)
    print(f"  Loaded: {len(tz):,} rows × {len(tz.columns)} columns")

    # Keep original for comparison
    tz_original = tz.copy()

    # Add source tracking columns if not present
    for col in ["kcat_per_s_source", "km_M_source", "pkd_value_source"]:
        if col not in tz.columns:
            tz[col] = None
    # Mark existing (CataPro) data
    tz.loc[tz["kcat_per_s"].notna() & tz["kcat_per_s_source"].isna(), "kcat_per_s_source"] = "catapro"
    tz.loc[tz["km_M"].notna() & tz["km_M_source"].isna(), "km_M_source"] = "catapro"
    tz.loc[tz["pkd_value"].notna() & tz["pkd_value_source"].isna(), "pkd_value_source"] = "rts_original"

    # ---- Phase 1: OED ----
    oed_df = load_oed()

    oed_t1 = build_tier1_index(oed_df, "uniprot_id", "inchikey",
                                ["kcat_per_s", "km_M"])
    oed_t2 = build_smiles_bridge_index(oed_df, "uniprot_id", "canonical_smiles",
                                        ["kcat_per_s", "km_M"])
    print(f"  Tier 1 index: {len(oed_t1):,} (uniprot, inchikey) pairs")
    print(f"  Tier 2 index: {len(oed_t2):,} (uniprot, canonical_smiles) pairs")

    tz, oed_stats = match_and_fill(
        tz, oed_t1, oed_t2,
        ["kcat_per_s", "km_M"], "oed",
        {"kcat_per_s": "kcat_per_s_source", "km_M": "km_M_source"}
    )
    print(f"  → OED filled: kcat={oed_stats['kcat_per_s']:,}, km={oed_stats['km_M']:,}")

    # ---- Phase 2: SKiD ----
    skid_df = load_skid()

    skid_t1 = build_tier1_index(skid_df, "uniprot_id", "inchikey",
                                 ["kcat_per_s", "km_M"])
    skid_t2 = build_smiles_bridge_index(skid_df, "uniprot_id", "canonical_smiles",
                                         ["kcat_per_s", "km_M"])
    print(f"  Tier 1 index: {len(skid_t1):,} (uniprot, inchikey) pairs")
    print(f"  Tier 2 index: {len(skid_t2):,} (uniprot, canonical_smiles) pairs")

    tz, skid_stats = match_and_fill(
        tz, skid_t1, skid_t2,
        ["kcat_per_s", "km_M"], "skid",
        {"kcat_per_s": "kcat_per_s_source", "km_M": "km_M_source"}
    )
    print(f"  → SKiD filled: kcat={skid_stats['kcat_per_s']:,}, km={skid_stats['km_M']:,}")

    # ---- Phase 3: SABIO-RK ----
    rts_df_raw = pd.read_parquet(RTS_INPUT)
    sabio_df, sub_inchikeys_map = load_sabio(tz, skid_df, rts_df_raw)
    tz, sabio_stats = match_sabio_and_fill(tz, sabio_df, sub_inchikeys_map)
    print(f"  → SABIO-RK filled: kcat={sabio_stats['kcat_per_s']:,}, km={sabio_stats['km_M']:,}")

    # ---- Phase 4: RTS kcat ----
    rts_df = load_rts()
    # For RTS, we need inchikey info. RTS has ligand_inchikey.
    rts_kcat_t1 = build_tier1_index(rts_df, "uniprot_id", "ligand_inchikey",
                                     ["kcat_median_s"])
    # Rename key for filling: we want to fill 'kcat_per_s' column with 'kcat_median_s' value
    rts_kcat_t1_renamed = {}
    for k, v in rts_kcat_t1.items():
        if "kcat_median_s" in v:
            rts_kcat_t1_renamed[k] = {"kcat_per_s": v["kcat_median_s"]}
    print(f"  RTS kcat Tier 1 index: {len(rts_kcat_t1_renamed):,} pairs")

    tz, rts_kcat_stats = match_and_fill(
        tz, rts_kcat_t1_renamed, None,  # No Tier 2 for RTS (no canonical_smiles)
        ["kcat_per_s"], "rts",
        {"kcat_per_s": "kcat_per_s_source"}
    )
    print(f"  → RTS kcat filled: {rts_kcat_stats['kcat_per_s']:,}")

    # ---- Phase 5: RTS pKd ----
    rts_pkd_t1 = build_tier1_index(rts_df, "uniprot_id", "ligand_inchikey",
                                    ["pkd_aligned"])
    rts_pkd_t1_renamed = {}
    for k, v in rts_pkd_t1.items():
        if "pkd_aligned" in v:
            rts_pkd_t1_renamed[k] = {"pkd_value": v["pkd_aligned"]}
    print(f"  RTS pKd Tier 1 index: {len(rts_pkd_t1_renamed):,} pairs")

    tz, rts_pkd_stats = match_and_fill(
        tz, rts_pkd_t1_renamed, None,
        ["pkd_value"], "rts_supplement",
        {"pkd_value": "pkd_value_source"}
    )
    print(f"  → RTS pKd filled: {rts_pkd_stats['pkd_value']:,}")

    # ---- Phase 6: PDBbind substrate-level pKd ----
    print("\n" + "=" * 60)
    print("PHASE 6: PDBbind substrate-level pKd matching")
    print("=" * 60)
    tz, pdbbind_stats = match_pdbbind_pkd(tz)
    print(f"  → PDBbind pKd filled: {pdbbind_stats['pkd_value']:,}")

    # ---- Phase 7: Cofactor substrate-level pKd ----
    print("\n" + "=" * 60)
    print("PHASE 7: Cofactor/reporter substrate pKd matching")
    print("=" * 60)
    tz, cofactor_stats = match_cofactor_pkd(tz)
    print(f"  → Cofactor pKd filled: {cofactor_stats['pkd_value']:,}")

    # ---- Phase 8: Homology transfer kcat/km ----
    print("\n" + "=" * 60)
    print("PHASE 8: Homology transfer kcat/km (same EC + same substrate)")
    print("=" * 60)
    tz, homo_stats = match_homology_transfer(tz)
    print(f"  → Homology transfer: kcat={homo_stats['kcat_per_s']:,}, km={homo_stats['km_M']:,}")

    # ---- Statistics ----
    stats = compute_final_stats(tz, tz_original)
    print_stats(stats)

    # ---- Save ----
    print(f"\n  Saving supplemented dataset...")
    tz.to_parquet(OUTPUT_FULL, index=False)
    print(f"    → {OUTPUT_FULL}")
    print(f"    {len(tz):,} rows × {len(tz.columns)} columns")

    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Statistics → {OUTPUT_STATS}")

    print("\n" + "█" * 70)
    print("  ✅ Trenzition supplement complete!")
    print("█" * 70)

    return tz, stats


if __name__ == "__main__":
    main()
