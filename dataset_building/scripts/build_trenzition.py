#!/usr/bin/env python3
"""
构建 Trenzition 数据集：CataPro 动力学数据 + 结合亲和力补充

以 CataPro（BRENDA + SABIO-RK）的 kcat/Km/kcatKm 数据为骨架，
通过蛋白-配体双键匹配补充 BindingDB/Ki/Kd 结合亲和力数据。

匹配策略（三级）：
  L1: (protein_seq_hash, ligand_inchikey) — 精确蛋白+配体匹配
  L2: protein_seq_hash — 同一蛋白，不同配体
  L3: uniprot_id — UniProt ID 匹配（序列哈希不一致时）

质量优先级：Kd (w=1.0) > Ki (w=0.7) > kinetics (w=0.5)

输出：
  - release/trenzition_full.parquet      — 完整数据集（含所有匹配和未匹配行）
  - release/trenzition_matched.parquet   — 仅成功匹配 pKd 的行
  - release/trenzition_stats.json        — 统计报告

用法：
  source /home/domi/BINN/.venv/bin/activate
  python build_trenzition.py
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

# Suppress RDKit warnings for cleaner output
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path("/home/domi/AI4enz/dataset_building")
CATAPRO_DIR = BASE_DIR / "CataPro" / "datasets"
RELEASE_DIR = BASE_DIR / "release"
PROCESSED_DIR = BASE_DIR / "processed"

CATAPRO_FILES = {
    "kcat": CATAPRO_DIR / "kcat-data_0.4simi-10fold.csv",
    "km": CATAPRO_DIR / "Km-data_0.4simi-10fold.csv",
    "kcatkm": CATAPRO_DIR / "kcat-over-Km-data_0.4simi-10fold.csv",
}

# Use the most complete existing dataset for pKd lookup
PKD_SOURCE = RELEASE_DIR / "recommended_training_set_enriched.parquet"
INCHIKEY_SMILES_MAP = PROCESSED_DIR / "inchikey_smiles_map.pkl"

OUTPUT_FULL = RELEASE_DIR / "trenzition_full.parquet"
OUTPUT_MATCHED = RELEASE_DIR / "trenzition_matched.parquet"
OUTPUT_STATS = RELEASE_DIR / "trenzition_stats.json"


# ============================================================
# Hash & Identifier utilities
# ============================================================
def compute_protein_hash(seq: str) -> str:
    """SHA256[:16] of uppercased sequence — matches existing project format."""
    return hashlib.sha256(str(seq).upper().encode()).hexdigest()[:16]


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
# Data Loading
# ============================================================

def load_catapro_data() -> pd.DataFrame:
    """Load all 3 CataPro CSVs and merge into a unified DataFrame.

    Each CataPro row is an enzyme-substrate pair with a kinetic measurement.
    The same pair may appear across multiple files (kcat-only, Km-only, kcat/Km).

    Returns a DataFrame with all rows and computed standard identifiers.
    """
    print("=" * 70)
    print("STEP 1: Loading CataPro raw data")
    print("=" * 70)

    dfs = []
    for dtype, fpath in CATAPRO_FILES.items():
        print(f"  Loading {fpath.name}...")
        df = pd.read_csv(fpath)
        df["data_type"] = dtype  # track origin
        dfs.append(df)
        print(f"    → {len(df):,} rows, {len(df.columns)} columns")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total combined: {len(combined):,} rows")

    # Compute standard identifiers
    print("\n  Computing protein_seq_hash (SHA256[:16])...")
    combined["protein_seq_hash"] = combined["Sequence"].apply(compute_protein_hash)

    print("  Computing ligand_inchikey from SMILES...")
    combined["ligand_inchikey"] = combined["Smiles"].apply(smiles_to_inchikey)
    n_fail = combined["ligand_inchikey"].isna().sum()
    if n_fail > 0:
        print(f"    ⚠ {n_fail:,} SMILES failed InChIKey conversion ({100*n_fail/len(combined):.2f}%)")

    print("  Computing canonical SMILES...")
    combined["canonical_smiles"] = combined["Smiles"].apply(canonicalize_smiles)
    n_canon_fail = combined["canonical_smiles"].isna().sum()
    if n_canon_fail > 0:
        print(f"    ⚠ {n_canon_fail:,} canonicalization failures")

    # Rename columns for clarity
    combined.rename(columns={
        "EC": "ec_number",
        "EnzymeType": "enzyme_type",
        "Organism": "organism_name",
        "Sequence": "sequence",
        "Substrate": "substrate_name",
        "Smiles": "smiles",
        "UniProtID": "uniprot_id",
        "fold": "cv_fold",
    }, inplace=True)

    # Standardize kinetic columns
    if "kcat(s^-1)" in combined.columns:
        combined.rename(columns={"kcat(s^-1)": "kcat_per_s"}, inplace=True)
    if "Km(M)" in combined.columns:
        combined.rename(columns={"Km(M)": "km_M"}, inplace=True)

    print(f"\n  Final: {len(combined):,} rows")
    print(f"    Unique proteins (seq_hash): {combined['protein_seq_hash'].nunique():,}")
    print(f"    Unique ligands (inchikey):  {combined['ligand_inchikey'].nunique():,}")
    print(f"    Unique UniProt IDs:         {combined['uniprot_id'].nunique():,}")

    return combined


def load_pkd_data() -> pd.DataFrame:
    """Load pKd data from existing project dataset.

    Filters to rows that actually have pKd values and organizes
    them for efficient lookups.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Loading existing pKd data")
    print("=" * 70)

    df = pd.read_parquet(PKD_SOURCE)
    print(f"  Source: {PKD_SOURCE.name}")
    print(f"  Total rows: {len(df):,}")

    # Only keep rows with actual pKd values
    has_pkd = df["pkd_aligned"].notna()
    print(f"  With pKd_aligned: {has_pkd.sum():,} ({100*has_pkd.sum()/len(df):.1f}%)")

    df_pkd = df[has_pkd].copy()

    # Map measurement types to quality tiers
    quality_tier = {
        "Kd": 1,       # highest — direct binding
        "Ki": 2,       # medium — inhibition constant ≈ binding
        "kinetics": 3,  # lower — derived from kinetics
    }

    df_pkd["quality_tier"] = df_pkd["measurement_type"].map(quality_tier).fillna(4).astype(int)
    df_pkd["quality_weight"] = df_pkd["quality_weight"].fillna(0.0)

    print(f"\n  pKd measurement types:")
    for mt, cnt in df_pkd["measurement_type"].value_counts().items():
        tier = quality_tier.get(mt, 4)
        print(f"    {mt}: {cnt:,} (tier={tier}, mean_w={df_pkd[df_pkd['measurement_type']==mt]['quality_weight'].mean():.2f})")

    print(f"\n  pKd statistics:")
    print(f"    mean={df_pkd['pkd_aligned'].mean():.3f}")
    print(f"    median={df_pkd['pkd_aligned'].median():.3f}")
    print(f"    std={df_pkd['pkd_aligned'].std():.3f}")
    print(f"    min={df_pkd['pkd_aligned'].min():.3f}")
    print(f"    max={df_pkd['pkd_aligned'].max():.3f}")
    print(f"    valid range [0,14]: {((df_pkd['pkd_aligned'] >= 0) & (df_pkd['pkd_aligned'] <= 14)).sum():,}")

    return df_pkd


def build_smiles_to_inchikey_lookup() -> dict[str, set[str]]:
    """Build canonical SMILES → InChIKey(s) reverse mapping.

    Returns:
        dict mapping canonical SMILES to set of InChIKeys
    """
    print("\n" + "=" * 70)
    print("STEP 3: Building SMILES → InChIKey reverse lookup")
    print("=" * 70)

    with open(INCHIKEY_SMILES_MAP, "rb") as f:
        ik2smiles = pickle.load(f)
    print(f"  Loaded {len(ik2smiles):,} InChIKey → SMILES mappings")

    smiles2ik = defaultdict(set)
    failures = 0
    for ik, smi in ik2smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canon = Chem.MolToSmiles(mol, canonical=True)
            smiles2ik[canon].add(ik)
        else:
            failures += 1

    print(f"  Reverse mapping: {len(smiles2ik):,} canonical SMILES → InChIKeys")
    print(f"  RDKit failures: {failures}")
    print(f"  Avg InChIKeys per SMILES: {sum(len(v) for v in smiles2ik.values()) / max(1, len(smiles2ik)):.2f}")

    return dict(smiles2ik)


# ============================================================
# Matching Engine
# ============================================================

class PKdMatcher:
    """Multi-level pKd matcher for CataPro rows.

    Match hierarchy (best → fallback):
      L1_exact:        (protein_hash, ligand_inchikey) identical pair
      L1_smiles_bridge: same pair found via canonical SMILES bridge
      L1_ligand:       same ligand (inchikey) binding to any protein
      L2_protein:      same protein binding any ligand
      L3_uniprot:      same UniProt ID (hash mismatch fallback)

    Confidence:
      high:      L1_exact, L1_smiles_bridge — specific to this pair
      medium:    L1_ligand — same substrate, different protein context
      low:       L2_protein, L3_uniprot — protein-level reference only
    """

    def __init__(self, df_pkd: pd.DataFrame, smiles2ik: dict[str, set[str]]):
        self.df_pkd = df_pkd
        self.smiles2ik = smiles2ik
        self._build_indexes()

    def _build_indexes(self):
        """Pre-build lookup indexes from pKd DataFrame."""
        df = self.df_pkd

        # L1: (protein_seq_hash, ligand_inchikey) → list of row index labels
        print("\n  Building L1 index (protein + ligand)...")
        self.l1_index = defaultdict(list)
        for idx, row in df.iterrows():
            key = (row["protein_seq_hash"], row["ligand_inchikey"])
            self.l1_index[key].append(idx)
        print(f"    {len(self.l1_index):,} unique (hash, inchikey) pairs")

        # L1_ligand: ligand_inchikey → list of row index labels
        print("  Building Ligand index (ligand only)...")
        self.ligand_index = defaultdict(list)
        for idx, row in df.iterrows():
            self.ligand_index[row["ligand_inchikey"]].append(idx)
        n_lig = sum(1 for k in self.ligand_index if isinstance(k, str))
        print(f"    {n_lig:,} unique ligands with pKd data")

        # L2: protein_seq_hash → list of row index labels
        print("  Building L2 index (protein only)...")
        self.l2_index = defaultdict(list)
        for idx, row in df.iterrows():
            self.l2_index[row["protein_seq_hash"]].append(idx)
        n_l2 = sum(1 for k in self.l2_index if isinstance(k, str))
        print(f"    {n_l2:,} unique protein hashes with pKd")

        # L3: uniprot_id → list of row index labels
        print("  Building L3 index (UniProt only)...")
        self.l3_index = defaultdict(list)
        for idx, row in df.iterrows():
            uid = row.get("uniprot_id")
            if isinstance(uid, str) and uid:
                self.l3_index[uid].append(idx)
        print(f"    {len(self.l3_index):,} unique UniProt IDs with pKd")

    def _best_from_indices(self, indices: list) -> dict | None:
        """Select the best pKd value from a list of row index labels.

        Priority: Kd (tier=1) > Ki (tier=2), then highest quality_weight,
        then median pKd of same-quality measurements.
        """
        if not indices:
            return None

        candidates = self.df_pkd.loc[indices]
        if isinstance(candidates, pd.Series):
            candidates = candidates.to_frame().T

        candidates = candidates.sort_values(
            ["quality_tier", "quality_weight"],
            ascending=[True, False]
        )

        best_tier = int(candidates.iloc[0]["quality_tier"])
        best_weight = float(candidates.iloc[0]["quality_weight"])

        same_quality = candidates[
            (candidates["quality_tier"] == best_tier) &
            (candidates["quality_weight"] == best_weight)
        ]

        pkd_values = same_quality["pkd_aligned"].dropna()
        if len(pkd_values) == 0:
            return None

        return {
            "pkd_value": float(pkd_values.median()),
            "pkd_mean": float(pkd_values.mean()),
            "pkd_std": float(pkd_values.std()) if len(pkd_values) > 1 else 0.0,
            "pkd_min": float(pkd_values.min()),
            "pkd_max": float(pkd_values.max()),
            "n_pkd_measurements": int(len(pkd_values)),
            "measurement_type": str(candidates.iloc[0]["measurement_type"]),
            "quality_tier": best_tier,
            "quality_weight": best_weight,
            "source_db_pkd": str(candidates.iloc[0].get("source_db", "unknown")),
            "match_level": None,     # set by caller
            "pkd_confidence": None,  # set by caller
        }

    def match_one(self, row: pd.Series) -> dict | None:
        """Find best pKd for a single CataPro row using 5-level hierarchy."""
        seq_hash = row.get("protein_seq_hash")
        inchikey = row.get("ligand_inchikey")
        canon_smi = row.get("canonical_smiles")
        uniprot = row.get("uniprot_id")

        # ---- L1: Exact (protein + ligand) match ----
        if seq_hash and inchikey:
            key = (seq_hash, inchikey)
            if key in self.l1_index:
                result = self._best_from_indices(self.l1_index[key])
                if result:
                    result["match_level"] = "L1_exact"
                    result["pkd_confidence"] = "high"
                    return result

            # L1_smiles_bridge: same pair via canonical SMILES bridge
            if canon_smi and canon_smi in self.smiles2ik:
                for mapped_ik in self.smiles2ik[canon_smi]:
                    key_alt = (seq_hash, mapped_ik)
                    if key_alt in self.l1_index:
                        result = self._best_from_indices(self.l1_index[key_alt])
                        if result:
                            result["match_level"] = "L1_smiles_bridge"
                            result["pkd_confidence"] = "high"
                            result["matched_inchikey"] = mapped_ik
                            return result

            # ---- L1_ligand: Same ligand, any protein ----
            if inchikey in self.ligand_index:
                result = self._best_from_indices(self.ligand_index[inchikey])
                if result:
                    result["match_level"] = "L1_ligand"
                    result["pkd_confidence"] = "medium"
                    return result

            # L1_ligand via SMILES bridge
            if canon_smi and canon_smi in self.smiles2ik:
                for mapped_ik in self.smiles2ik[canon_smi]:
                    if mapped_ik in self.ligand_index:
                        result = self._best_from_indices(self.ligand_index[mapped_ik])
                        if result:
                            result["match_level"] = "L1_ligand_smiles"
                            result["pkd_confidence"] = "medium"
                            result["matched_inchikey"] = mapped_ik
                            return result

        # ---- L2: Protein-level match (same protein, different ligand) ----
        if seq_hash and seq_hash in self.l2_index:
            result = self._best_from_indices(self.l2_index[seq_hash])
            if result:
                result["match_level"] = "L2_protein"
                result["pkd_confidence"] = "low"
                return result

        # ---- L3: UniProt ID match (hash mismatch fallback) ----
        if isinstance(uniprot, str) and uniprot and uniprot in self.l3_index:
            result = self._best_from_indices(self.l3_index[uniprot])
            if result:
                result["match_level"] = "L3_uniprot"
                result["pkd_confidence"] = "low"
                return result

        return None


# ============================================================
# Main Pipeline
# ============================================================

def build_trenzition():
    """Main entry point: build the Trenzition dataset."""
    print("\n" + "█" * 70)
    print("█  TRENZITION DATASET BUILDER")
    print("█  CataPro kinetics + Binding affinity supplement")
    print("█" * 70)

    # ---- Load data ----
    df_catapro = load_catapro_data()
    df_pkd = load_pkd_data()
    smiles2ik = build_smiles_to_inchikey_lookup()

    # ---- Match ----
    print("\n" + "=" * 70)
    print("STEP 4: Matching CataPro rows → pKd data")
    print("=" * 70)

    matcher = PKdMatcher(df_pkd, smiles2ik)

    match_results = []
    match_counts = defaultdict(int)

    for idx, row in df_catapro.iterrows():
        result = matcher.match_one(row)
        match_results.append(result)
        level = result["match_level"] if result else "no_match"
        match_counts[level] += 1

        if (idx + 1) % 20000 == 0:
            print(f"  Processed {idx+1:,}/{len(df_catapro):,} rows...")

    total_matched = sum(v for k, v in match_counts.items() if k != "no_match")
    print(f"\n  Matching complete!")
    print(f"    {'Level':<25s} {'Count':>8s}  {'Rate':>7s}  Confidence")
    print(f"    {'-'*55}")
    for level in ["L1_exact", "L1_smiles_bridge", "L1_ligand", "L1_ligand_smiles",
                   "L2_protein", "L3_uniprot", "no_match"]:
        cnt = match_counts.get(level, 0)
        rate = f"{100*cnt/len(df_catapro):.1f}%"
        if level.startswith("L1_"):
            conf = "high" if "exact" in level or "smiles_bridge" in level else "medium"
        elif level == "no_match":
            conf = "—"
        else:
            conf = "low"
        print(f"    {level:<25s} {cnt:>8,}  {rate:>7s}  {conf}")
    print(f"    {'─'*55}")
    print(f"    {'Total matched':<25s} {total_matched:>8,}  {100*total_matched/len(df_catapro):>6.1f}%")

    # ---- Assemble output ----
    print("\n" + "=" * 70)
    print("STEP 5: Assembling Trenzition dataset")
    print("=" * 70)

    # Add pKd columns to CataPro DataFrame
    pKd_cols = [
        "pkd_value", "pkd_mean", "pkd_std", "pkd_min", "pkd_max",
        "n_pkd_measurements", "measurement_type_pkd", "quality_tier",
        "quality_weight_pkd", "match_level", "pkd_confidence",
        "source_db_pkd", "matched_inchikey",
    ]
    for col in pKd_cols:
        df_catapro[col] = None

    for i, result in enumerate(match_results):
        if result is None:
            continue
        for col in pKd_cols:
            if col == "measurement_type_pkd":
                df_catapro.at[i, col] = result.get("measurement_type")
            elif col == "quality_weight_pkd":
                df_catapro.at[i, col] = result.get("quality_weight")
            elif col == "source_db_pkd":
                df_catapro.at[i, col] = result.get("source_db_pkd")
            elif col in result:
                df_catapro.at[i, col] = result[col]

    # Convert cols to proper types
    for col in ["pkd_value", "pkd_mean", "pkd_std", "pkd_min", "pkd_max", "quality_weight_pkd"]:
        df_catapro[col] = pd.to_numeric(df_catapro[col], errors="coerce")
    df_catapro["n_pkd_measurements"] = pd.to_numeric(
        df_catapro["n_pkd_measurements"], errors="coerce"
    ).astype("Int64")
    df_catapro["quality_tier"] = pd.to_numeric(
        df_catapro["quality_tier"], errors="coerce"
    ).astype("Int64")

    # ---- Quality filters ----
    # Flag rows where pKd is out of valid range
    df_catapro["pkd_valid"] = df_catapro["pkd_value"].apply(
        lambda x: 0 <= x <= 14 if pd.notna(x) else None
    )

    # Create data_source label
    df_catapro["data_source"] = "catapro"
    df_catapro["has_pkd"] = df_catapro["pkd_value"].notna()

    # ---- Save ----
    print("\n  Saving full dataset...")
    df_catapro.to_parquet(OUTPUT_FULL, index=False)
    print(f"    → {OUTPUT_FULL}")
    print(f"    {len(df_catapro):,} rows × {len(df_catapro.columns)} columns")

    df_matched = df_catapro[df_catapro["has_pkd"]].copy()
    print(f"\n  Saving matched-only dataset...")
    df_matched.to_parquet(OUTPUT_MATCHED, index=False)
    print(f"    → {OUTPUT_MATCHED}")
    print(f"    {len(df_matched):,} rows (only rows with pKd data)")

    # ---- Statistics ----
    stats = compute_statistics(df_catapro, df_matched, match_counts)
    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=json_safe)
    print(f"\n  Statistics saved → {OUTPUT_STATS}")

    return df_catapro, df_matched, stats


def json_safe(obj):
    """JSON serializer for non-standard Python/numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_statistics(df_full, df_matched, match_counts):
    """Generate comprehensive statistics for the Trenzition dataset."""
    total = len(df_full)
    n_matched = len(df_matched)

    # Build match_levels from the dict
    match_levels = {}
    for level in ["L1_exact", "L1_smiles_bridge", "L1_ligand", "L1_ligand_smiles",
                   "L2_protein", "L3_uniprot", "no_match"]:
        cnt = int(match_counts.get(level, 0))
        match_levels[level] = {"count": cnt, "pct": round(100 * cnt / total, 2)}

    # Convert value_counts to plain-int-keyed dicts
    def safe_vc_dict(series):
        d = {}
        for k, v in series.value_counts().items():
            d[str(k)] = int(v)
        return d

    stats = {
        "dataset_name": "Trenzition",
        "description": "CataPro kinetic data supplemented with binding affinity (pKd/Ki/Kd)",
        "total_rows": int(total),
        "total_matched_with_pkd": int(n_matched),
        "match_rate": round(100 * n_matched / total, 2),

        "match_levels": match_levels,

        "data_types": {str(k): int(v) for k, v in df_full["data_type"].value_counts().items()},

        "kinetic_params": {
            "has_kcat": int(df_full["kcat_per_s"].notna().sum()),
            "has_km": int(df_full["km_M"].notna().sum()),
            "has_both_kcat_km": int((df_full["kcat_per_s"].notna() & df_full["km_M"].notna()).sum()),
        },

        "pkd_by_measurement_type": (
            safe_vc_dict(df_matched["measurement_type_pkd"])
            if n_matched > 0 else {}
        ),

        "pkd_by_match_level": (
            safe_vc_dict(df_matched["match_level"])
            if n_matched > 0 else {}
        ),

        "pkd_by_confidence": (
            safe_vc_dict(df_matched["pkd_confidence"])
            if n_matched > 0 else {}
        ),

        "pkd_statistics": {
            "mean": float(df_matched["pkd_value"].mean()) if n_matched > 0 else None,
            "median": float(df_matched["pkd_value"].median()) if n_matched > 0 else None,
            "std": float(df_matched["pkd_value"].std()) if n_matched > 0 else None,
            "min": float(df_matched["pkd_value"].min()) if n_matched > 0 else None,
            "max": float(df_matched["pkd_value"].max()) if n_matched > 0 else None,
            "in_valid_range_0_14": int(
                ((df_matched["pkd_value"] >= 0) & (df_matched["pkd_value"] <= 14)).sum()
            ) if n_matched > 0 else 0,
        },

        "uniqueness": {
            "unique_proteins": int(df_full["protein_seq_hash"].nunique()),
            "unique_ligands": int(df_full["ligand_inchikey"].nunique()),
            "unique_uniprot": int(df_full["uniprot_id"].nunique()),
            "unique_ec": int(df_full["ec_number"].nunique()),
            "unique_pairs_matched": int(
                df_matched.drop_duplicates(subset=["protein_seq_hash", "ligand_inchikey"]).shape[0]
            ) if n_matched > 0 else 0,
        },

        "pkd_confidence_distribution": (
            safe_vc_dict(df_matched["pkd_confidence"])
            if n_matched > 0 else {}
        ),

        "pkd_per_confidence_stats": {},
    }

    # Per-confidence-level pKd stats
    for conf in ["high", "medium", "low"]:
        sub = df_matched[df_matched["pkd_confidence"] == conf]
        if len(sub) > 0:
            stats["pkd_per_confidence_stats"][conf] = {
                "count": int(len(sub)),
                "mean": float(sub["pkd_value"].mean()),
                "median": float(sub["pkd_value"].median()),
                "std": float(sub["pkd_value"].std()),
            }

    # Per-data-type breakdown
    for dtype in ["kcat", "km", "kcatkm"]:
        subset = df_full[df_full["data_type"] == dtype]
        subset_m = df_matched[df_matched["data_type"] == dtype]
        stats[f"breakdown_{dtype}"] = {
            "total": int(len(subset)),
            "matched_with_pkd": int(len(subset_m)),
            "match_rate": round(100 * len(subset_m) / max(1, len(subset)), 2),
        }

    return stats


# ============================================================
# Print Final Report
# ============================================================

def print_report(stats: dict):
    """Print a human-readable summary."""
    print("\n" + "█" * 70)
    print("█  TRENZITION DATASET — FINAL REPORT")
    print("█" * 70)

    print(f"\n  📊 Overview:")
    print(f"     Total rows:              {stats['total_rows']:>10,}")
    print(f"     Matched with pKd:        {stats['total_matched_with_pkd']:>10,} ({stats['match_rate']}%)")
    print(f"     Unique proteins:         {stats['uniqueness']['unique_proteins']:>10,}")
    print(f"     Unique ligands:          {stats['uniqueness']['unique_ligands']:>10,}")
    print(f"     Unique EC numbers:       {stats['uniqueness']['unique_ec']:>10,}")

    print(f"\n  🔬 Kinetic coverage:")
    kp = stats["kinetic_params"]
    print(f"     With kcat:               {kp['has_kcat']:>10,}")
    print(f"     With Km:                 {kp['has_km']:>10,}")
    print(f"     With both kcat+Km:       {kp['has_both_kcat_km']:>10,}")

    print(f"\n  🎯 Match levels:")
    match_level_order = ["L1_exact", "L1_smiles_bridge", "L1_ligand", "L1_ligand_smiles",
                          "L2_protein", "L3_uniprot", "no_match"]
    conf_map = {"L1_exact": "high", "L1_smiles_bridge": "high",
                "L1_ligand": "medium", "L1_ligand_smiles": "medium",
                "L2_protein": "low", "L3_uniprot": "low", "no_match": "—"}
    for level in match_level_order:
        info = stats["match_levels"].get(level, {"count": 0, "pct": 0})
        bar = "█" * int(info["pct"] / 2)
        print(f"     {level:<22s}: {info['count']:>8,} ({info['pct']:5.1f}%)  [{conf_map[level]:>6s}] {bar}")

    print(f"\n  🏷️  pKd measurement types (matched):")
    for mt, cnt in stats.get("pkd_by_measurement_type", {}).items():
        print(f"     {mt:<20s}: {cnt:>8,}")

    print(f"\n  ✅ pKd confidence distribution:")
    for conf, cnt in stats.get("pkd_confidence_distribution", {}).items():
        pct = 100 * cnt / max(1, stats["total_matched_with_pkd"])
        print(f"     {conf:<10s}: {cnt:>8,} ({pct:.1f}%)")

    print(f"\n  📈 pKd statistics (overall matched):")
    ps = stats["pkd_statistics"]
    if ps["mean"]:
        print(f"     Mean:   {ps['mean']:.3f}")
        print(f"     Median: {ps['median']:.3f}")
        print(f"     Std:    {ps['std']:.3f}")
        print(f"     Range:  [{ps['min']:.3f}, {ps['max']:.3f}]")

    print(f"\n  📈 pKd by confidence level:")
    for conf, pstats in stats.get("pkd_per_confidence_stats", {}).items():
        print(f"     {conf:<10s}: n={pstats['count']:>6,}  mean={pstats['mean']:.3f}  median={pstats['median']:.3f}  std={pstats['std']:.3f}")

    print(f"\n  📂 Per data type:")
    for dtype in ["kcat", "km", "kcatkm"]:
        bd = stats[f"breakdown_{dtype}"]
        print(f"     {dtype:<8s}: {bd['total']:>8,} total, {bd['matched_with_pkd']:>8,} matched ({bd['match_rate']}%)")

    print("\n" + "█" * 70)
    print("  ✅ Trenzition dataset built successfully!")
    print(f"  📁 Full:     {OUTPUT_FULL}")
    print(f"  📁 Matched:  {OUTPUT_MATCHED}")
    print(f"  📁 Stats:    {OUTPUT_STATS}")
    print("█" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    df_full, df_matched, stats = build_trenzition()
    print_report(stats)
