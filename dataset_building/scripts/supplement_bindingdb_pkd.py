#!/usr/bin/env python3
"""
从 BindingDB 原始 TSV 中提取 Kd/Ki 数据，补充 Trenzition 缺失的 pKd。

策略（仅高置信度）：
  - 只取 Kd 和 Ki 测量值（排除 IC50、EC50 等）
  - pKd = -log10(value_nM * 1e-9)
  - Tier 1: (uniprot_id, inchikey) 精确匹配

用法：
  source /home/domi/BINN/.venv/bin/activate
  python scripts/supplement_bindingdb_pkd.py
"""

import pandas as pd
import numpy as np
import hashlib
import pickle
import json
import zipfile
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path("/home/domi/AI4enz/dataset_building")
BINDINGDB_ZIP = BASE_DIR / "BindingDB" / "BindingDB_All_202605_tsv.zip"
TZ_INPUT = BASE_DIR / "release" / "trenzition_full_v3.parquet"
TZ_OUTPUT = BASE_DIR / "release" / "trenzition_full_v3.parquet"  # update in place
CACHE_PATH = BASE_DIR / "processed" / "bindingdb_kd_ki_index.pkl"


def extract_bindingdb_pkd() -> dict:
    """Extract Kd/Ki from BindingDB TSV, build (uniprot, inchikey) → pKd index.

    Returns:
        dict: {(uniprot, inchikey): [pKd_values]}
    """
    if CACHE_PATH.exists():
        print(f"  Loading cached index from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("  Streaming BindingDB TSV (2.3 GB)...")
    index = defaultdict(list)

    with zipfile.ZipFile(BINDINGDB_ZIP) as zf:
        with zf.open("BindingDB_All.tsv") as f:
            header = f.readline().decode("utf-8", errors="replace").strip().split("\t")

            # Column indices (verified from header scan)
            smiles_idx = 1
            inchikey_idx = 3  # pre-computed InChIKey in BindingDB
            ki_idx = 8        # Ki (nM)
            kd_idx = 10       # Kd (nM)
            uniprot_swiss_idx = 44
            uniprot_trembl_idx = 49

            row_count = 0
            kept_kd = 0
            kept_ki = 0
            kept_total = 0  # deduplicated

            for line_bytes in f:
                row_count += 1
                if row_count % 500000 == 0:
                    print(f"    Processed {row_count:,} rows... (kept {kept_total:,} unique pairs)")

                try:
                    row = line_bytes.decode("utf-8", errors="replace").strip().split("\t")
                except Exception:
                    continue

                if len(row) <= max(ki_idx, kd_idx, uniprot_swiss_idx):
                    continue

                ki_str = row[ki_idx] if len(row) > ki_idx else ""
                kd_str = row[kd_idx] if len(row) > kd_idx else ""

                # Skip if no Kd or Ki
                has_kd = kd_str and kd_str.strip()
                has_ki = ki_str and ki_str.strip()
                if not has_kd and not has_ki:
                    continue

                # Get UniProt ID
                uniprot = ""
                if len(row) > uniprot_swiss_idx:
                    uniprot = row[uniprot_swiss_idx].strip()
                if not uniprot and len(row) > uniprot_trembl_idx:
                    uniprot = row[uniprot_trembl_idx].strip()
                if not uniprot:
                    continue

                # Get InChIKey — use pre-computed from BindingDB if available
                inchikey = ""
                if len(row) > inchikey_idx:
                    inchikey = row[inchikey_idx].strip()
                    # Validate InChIKey format (14 + 1 + 10 + 1 + 1 = 27 chars, uppercase)
                    if len(inchikey) != 27 or "-" not in inchikey:
                        inchikey = ""

                # If no pre-computed InChIKey, compute from SMILES
                if not inchikey:
                    smiles = row[smiles_idx] if len(row) > smiles_idx else ""
                    if smiles and smiles.strip():
                        mol = Chem.MolFromSmiles(smiles.strip())
                        if mol:
                            inchikey = MolToInchiKey(mol)

                if not inchikey:
                    continue

                # Parse values and convert to pKd
                pkd_values = []

                if has_kd:
                    try:
                        kd_nM = float(kd_str)
                        if kd_nM > 0:
                            # pKd = -log10(Kd in M) = -log10(Kd_nM * 1e-9)
                            pkd = -np.log10(kd_nM * 1e-9)
                            if 0 <= pkd <= 14:
                                pkd_values.append(pkd)
                                kept_kd += 1
                    except ValueError:
                        pass

                if has_ki:
                    try:
                        ki_nM = float(ki_str)
                        if ki_nM > 0:
                            pkd = -np.log10(ki_nM * 1e-9)
                            if 0 <= pkd <= 14:
                                pkd_values.append(pkd)
                                kept_ki += 1
                    except ValueError:
                        pass

                if pkd_values:
                    key = (uniprot, inchikey)
                    index[key].extend(pkd_values)
                    if len(index[key]) == len(pkd_values):  # first time seeing this key
                        kept_total += 1

    print(f"\n  Scanned {row_count:,} rows total")
    print(f"  Kd measurements kept: {kept_kd:,}")
    print(f"  Ki measurements kept: {kept_ki:,}")
    print(f"  Unique (uniprot, inchikey) pairs: {kept_total:,}")

    # Aggregate: median pKd per key
    agg_index = {}
    for key, vals in index.items():
        agg_index[key] = float(np.median(vals))

    print(f"  Aggregated index: {len(agg_index):,} entries")

    # Cache for future use
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(agg_index, f)
    print(f"  Cached → {CACHE_PATH}")

    return agg_index


def match_and_fill(tz_df: pd.DataFrame, pkd_index: dict) -> tuple[pd.DataFrame, int]:
    """Match TZ rows missing pKd against BindingDB pKd index.

    Tier 1: exact (uniprot_id, ligand_inchikey) match.
    """
    filled = 0
    for idx, row in tz_df.iterrows():
        if pd.notna(row.get("pkd_value")):
            continue

        up = row.get("uniprot_id")
        ik = row.get("ligand_inchikey")
        if not isinstance(up, str) or not isinstance(ik, str):
            continue

        key = (up.strip(), ik.strip())
        if key not in pkd_index:
            continue

        pkd_val = pkd_index[key]
        if 0 <= pkd_val <= 14:
            tz_df.at[idx, "pkd_value"] = pkd_val
            tz_df.at[idx, "pkd_value_source"] = "bindingdb_direct"
            filled += 1

    return tz_df, filled


def main():
    print("█" * 70)
    print("█  BINDINGDB DIRECT pKd SUPPLEMENT")
    print("█  Extracting Kd/Ki from BindingDB raw TSV (no IC50)")
    print("█" * 70)

    # Extract BindingDB pKd index
    print("\n" + "=" * 60)
    print("STEP 1: Extract Kd/Ki from BindingDB TSV")
    print("=" * 60)
    pkd_index = extract_bindingdb_pkd()

    # Load trenzition
    print("\n" + "=" * 60)
    print("STEP 2: Match with Trenzition")
    print("=" * 60)
    tz = pd.read_parquet(TZ_INPUT)
    tz_original = tz.copy()

    # Track source if not already present
    if "pkd_value_source" not in tz.columns:
        tz["pkd_value_source"] = None
        tz.loc[tz["pkd_value"].notna(), "pkd_value_source"] = "rts_original"

    pKd_before = tz["pkd_value"].notna().sum()
    print(f"  pKd before: {pKd_before:,}")

    tz, filled = match_and_fill(tz, pkd_index)
    print(f"  pKd filled: {filled:,}")

    pKd_after = tz["pkd_value"].notna().sum()
    print(f"  pKd after:  {pKd_after:,} (+{pKd_after - pKd_before:,})")

    # Stats
    perfect_before = (tz_original["kcat_per_s"].notna() & tz_original["km_M"].notna() & tz_original["pkd_value"].notna()).sum()
    perfect_after = (tz["kcat_per_s"].notna() & tz["km_M"].notna() & tz["pkd_value"].notna()).sum()

    print(f"\n  Perfect (all 3) before: {perfect_before:,} ({100*perfect_before/len(tz):.1f}%)")
    print(f"  Perfect (all 3) after:  {perfect_after:,} ({100*perfect_after/len(tz):.1f}%)")

    # Source breakdown
    print(f"\n  pKd source breakdown:")
    if "pkd_value_source" in tz.columns:
        for src, cnt in tz["pkd_value_source"].value_counts().items():
            print(f"    {src:<30s}: {cnt:>8,}")

    # Save
    print(f"\n  Saving...")
    tz.to_parquet(TZ_OUTPUT, index=False)
    print(f"    → {TZ_OUTPUT}")

    print("\n" + "█" * 70)
    print("  ✅ BindingDB direct pKd supplement complete!")
    print("█" * 70)

    return tz


if __name__ == "__main__":
    main()
