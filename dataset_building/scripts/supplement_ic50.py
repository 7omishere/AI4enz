#!/usr/bin/env python3
"""
Trenzition IC50 Supplement: 从 BindingDB IC50-only 数据补充近似 pKd。

原理: IC50 ≈ Ki（当底物浓度远小于 Km 时，Cheng-Prusoff 方程简化）
置信度: IC50 比直接 Ki/Kd 低一级（最多 medium），标注 source=ic50_approx

用法:
  source /home/domi/BINN/.venv/bin/activate
  python supplement_ic50.py
"""

import pandas as pd
import numpy as np
import hashlib
import zipfile
import json
import sys
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

# ============================================================
BASE_DIR = Path("/home/domi/AI4enz/dataset_building")
RELEASE_DIR = BASE_DIR / "release"
TRENZITION_FULL = RELEASE_DIR / "trenzition_full.parquet"
BINDINGDB_ZIP = BASE_DIR / "external_data" / "BindingDB_All_202605_tsv.zip"
INCHIKEY_SMILES_MAP = BASE_DIR / "processed" / "inchikey_smiles_map.pkl"

OUTPUT_TRENZITION_V2 = RELEASE_DIR / "trenzition_full_v2.parquet"
OUTPUT_STATS_V2 = RELEASE_DIR / "trenzition_stats_v2.json"


def compute_protein_hash(seq: str) -> str:
    return hashlib.sha256(str(seq).upper().encode()).hexdigest()[:16]


def load_trenzition() -> pd.DataFrame:
    print("Loading Trenzition dataset...")
    df = pd.read_parquet(TRENZITION_FULL)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    return df


def extract_ic50_data() -> pd.DataFrame:
    """Extract IC50-only entries from raw BindingDB TSV.

    Returns DataFrame with columns:
      protein_seq_hash, ligand_inchikey, uniprot_id,
      ic50_nm, pic50 (approximate pKd), measurement_type='IC50_approx'
    """
    print("\nExtracting IC50-only data from BindingDB...")

    with zipfile.ZipFile(BINDINGDB_ZIP, 'r') as zf:
        tsv_name = zf.namelist()[0]
        with zf.open(tsv_name) as f:
            df = pd.read_csv(f, sep='\t', low_memory=False,
                             usecols=['IC50 (nM)', 'Ki (nM)', 'Kd (nM)',
                                      'UniProt (SwissProt) Primary ID of Target Chain 1',
                                      'BindingDB Target Chain Sequence 1',
                                      'Ligand SMILES',
                                      'Curation/DataSource'])

    # Convert to numeric
    for c in ['IC50 (nM)', 'Ki (nM)', 'Kd (nM)']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    has_ic50 = df['IC50 (nM)'].notna() & (df['IC50 (nM)'] > 0)
    has_kikd = ((df['Ki (nM)'].notna() & (df['Ki (nM)'] > 0)) |
                 (df['Kd (nM)'].notna() & (df['Kd (nM)'] > 0)))
    only_ic50 = has_ic50 & ~has_kikd

    df_ic50 = df[only_ic50].copy()
    print(f"  IC50-only rows: {len(df_ic50):,}")

    # Require UniProt + Sequence + SMILES
    complete = (df_ic50['UniProt (SwissProt) Primary ID of Target Chain 1'].notna() &
                df_ic50['BindingDB Target Chain Sequence 1'].notna() &
                df_ic50['Ligand SMILES'].notna())
    df_ic50 = df_ic50[complete].copy()
    print(f"  Complete (UniProt+Seq+SMILES): {len(df_ic50):,}")

    # Compute identifiers
    print("  Computing protein_seq_hash...")
    df_ic50['protein_seq_hash'] = df_ic50['BindingDB Target Chain Sequence 1'].apply(compute_protein_hash)

    print("  Computing ligand_inchikey...")
    def safe_inchikey(smi):
        if pd.isna(smi):
            return None
        mol = Chem.MolFromSmiles(str(smi).strip())
        return MolToInchiKey(mol) if mol else None

    df_ic50['ligand_inchikey'] = df_ic50['Ligand SMILES'].apply(safe_inchikey)
    n_fail = df_ic50['ligand_inchikey'].isna().sum()
    print(f"  InChIKey conversion failures: {n_fail:,}")
    df_ic50 = df_ic50[df_ic50['ligand_inchikey'].notna()]

    # Compute pIC50 = -log10(IC50 in M). IC50 is in nM, so:
    # pIC50 = -log10(IC50_nM * 1e-9) = 9 - log10(IC50_nM)
    df_ic50['ic50_nm'] = df_ic50['IC50 (nM)']
    df_ic50['pic50'] = 9.0 - np.log10(df_ic50['ic50_nm'].astype(float))

    # Filter valid range: pIC50 between 0 and 14
    valid_pic50 = (df_ic50['pic50'] >= 0) & (df_ic50['pic50'] <= 14)
    print(f"  pIC50 in [0,14]: {valid_pic50.sum():,} / {len(df_ic50):,}")
    df_ic50 = df_ic50[valid_pic50]

    # Rename for clarity
    df_ic50.rename(columns={
        'UniProt (SwissProt) Primary ID of Target Chain 1': 'uniprot_id',
        'Curation/DataSource': 'source',
    }, inplace=True)

    print(f"  Final IC50 entries: {len(df_ic50):,}")
    print(f"  pIC50: mean={df_ic50['pic50'].mean():.2f}, median={df_ic50['pic50'].median():.2f}, "
          f"range=[{df_ic50['pic50'].min():.2f}, {df_ic50['pic50'].max():.2f}]")

    return df_ic50


def supplement_trenzition(df_tren: pd.DataFrame, df_ic50: pd.DataFrame) -> pd.DataFrame:
    """Match IC50 data to Trenzition rows and supplement pKd.

    Match hierarchy (IC50 has lower confidence than direct Ki/Kd):
      L1_ic50_exact:   (protein_hash, ligand_inchikey) match → medium confidence
      L1_ic50_ligand:  same ligand → medium confidence
      L1_ic50_smiles:  ligand match via canonical SMILES → low confidence

    Only supplements rows that currently have NO pKd data.
    """
    print("\n" + "=" * 60)
    print("Supplementing Trenzition with IC50 data")
    print("=" * 60)

    # Build IC50 indexes
    print("Building IC50 indexes...")
    l1_ic50 = defaultdict(list)   # (hash, inchikey) -> indices
    lig_ic50 = defaultdict(list)  # inchikey -> indices

    for idx, row in df_ic50.iterrows():
        key = (row['protein_seq_hash'], row['ligand_inchikey'])
        l1_ic50[key].append(idx)
        lig_ic50[row['ligand_inchikey']].append(idx)

    print(f"  L1 (protein+ligand): {len(l1_ic50):,} unique pairs")
    print(f"  Ligand-only: {len(lig_ic50):,} unique ligands")

    # Pre-build SMILES → InChIKeys reverse lookup (for SMILES bridge)
    print("Building SMILES→InChIKey reverse lookup for bridge matching...")
    import pickle
    with open(INCHIKEY_SMILES_MAP, 'rb') as f:
        ik2smiles = pickle.load(f)
    smiles2ik = defaultdict(list)
    for ik, smi in ik2smiles.items():
        smiles2ik[smi].append(ik)
    del ik2smiles  # free memory
    # Create canonical → InChIKeys mapping for fast lookup
    canon2ik = defaultdict(list)
    n_ik_total = 0
    for smi, iks in smiles2ik.items():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canon = Chem.MolToSmiles(mol, canonical=True)
            canon2ik[canon].extend(iks)
            n_ik_total += 1
        else:
            # Keep original SMILES as fallback
            canon2ik[smi].extend(iks)
    print(f"  Canonical SMILES → InChIKeys: {len(canon2ik):,} entries")

    # Aggregate IC50 values per key
    def agg_ic50(indices):
        subset = df_ic50.loc[indices]
        vals = subset['pic50']
        return {
            'ic50_pkd': float(vals.median()),
            'ic50_mean': float(vals.mean()),
            'ic50_std': float(vals.std()) if len(vals) > 1 else 0.0,
            'ic50_n': int(len(vals)),
            'ic50_source': str(subset['source'].mode().iloc[0]) if len(subset) > 0 else 'unknown',
        }

    # Only supplement rows without existing pKd
    rows_without_pkd = df_tren['pkd_value'].isna()
    n_candidates = rows_without_pkd.sum()
    print(f"\nTrenzition rows without pKd: {n_candidates:,}")

    # Pre-compute canonical SMILES for Trenzition rows (faster lookup)
    print("Pre-computing canonical SMILES for Trenzition...")
    tren_canon = {}
    for idx in df_tren[rows_without_pkd].index:
        canon = df_tren.at[idx, 'canonical_smiles']
        if isinstance(canon, str):
            tren_canon[idx] = canon

    # Match counters
    n_l1_ic50 = n_lig_ic50 = n_smiles_ic50 = 0
    new_ic50_data = {}

    for i, idx in enumerate(df_tren[rows_without_pkd].index):
        row = df_tren.loc[idx]
        seq_hash = row['protein_seq_hash']
        inchikey = row['ligand_inchikey']
        canon_smi = tren_canon.get(idx)

        # L1: exact (protein + ligand) IC50 match
        if seq_hash and inchikey:
            key = (seq_hash, inchikey)
            if key in l1_ic50:
                result = agg_ic50(l1_ic50[key])
                result['match_level'] = 'L1_ic50_exact'
                result['pkd_confidence'] = 'medium'
                result['measurement_type'] = 'IC50_approx'
                result['quality_tier'] = 3
                result['quality_weight'] = 0.5
                new_ic50_data[idx] = result
                n_l1_ic50 += 1
                continue

            # L1_ic50_ligand: same ligand (InChIKey match), any protein
            if inchikey in lig_ic50:
                result = agg_ic50(lig_ic50[inchikey])
                result['match_level'] = 'L1_ic50_ligand'
                result['pkd_confidence'] = 'medium'
                result['measurement_type'] = 'IC50_approx'
                result['quality_tier'] = 3
                result['quality_weight'] = 0.35
                new_ic50_data[idx] = result
                n_lig_ic50 += 1
                continue

            # L1_ic50_smiles: ligand match via canonical SMILES bridge
            if canon_smi and canon_smi in canon2ik:
                found = False
                for mapped_ik in canon2ik[canon_smi]:
                    if mapped_ik in lig_ic50:
                        result = agg_ic50(lig_ic50[mapped_ik])
                        result['match_level'] = 'L1_ic50_smiles'
                        result['pkd_confidence'] = 'low'
                        result['measurement_type'] = 'IC50_approx'
                        result['quality_tier'] = 3
                        result['quality_weight'] = 0.2
                        new_ic50_data[idx] = result
                        n_smiles_ic50 += 1
                        found = True
                        break
                if found:
                    continue

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{n_candidates:,}... "
                  f"(L1={n_l1_ic50}, Lig={n_lig_ic50}, SMILES={n_smiles_ic50})")

    total_new = n_l1_ic50 + n_lig_ic50 + n_smiles_ic50
    print(f"\n  IC50 supplement results:")
    print(f"    L1_ic50_exact (protein+ligand):  {n_l1_ic50:>8,}")
    print(f"    L1_ic50_ligand (same ligand):    {n_lig_ic50:>8,}")
    print(f"    L1_ic50_smiles (SMILES bridge):  {n_smiles_ic50:>8,}")
    print(f"    Total new IC50-supplemented:     {total_new:>8,}")

    # Apply supplements to dataframe
    for idx, data in new_ic50_data.items():
        df_tren.at[idx, 'pkd_value'] = data['ic50_pkd']
        df_tren.at[idx, 'pkd_mean'] = data['ic50_mean']
        df_tren.at[idx, 'pkd_std'] = data['ic50_std']
        df_tren.at[idx, 'n_pkd_measurements'] = data['ic50_n']
        df_tren.at[idx, 'measurement_type_pkd'] = data['measurement_type']
        df_tren.at[idx, 'quality_tier'] = data['quality_tier']
        df_tren.at[idx, 'quality_weight_pkd'] = data['quality_weight']
        df_tren.at[idx, 'match_level'] = data['match_level']
        df_tren.at[idx, 'pkd_confidence'] = data['pkd_confidence']
        df_tren.at[idx, 'source_db_pkd'] = f"bindingdb_{data['measurement_type']}"
        df_tren.at[idx, 'has_pkd'] = True
        df_tren.at[idx, 'pkd_valid'] = True

    return df_tren, total_new


def compute_v2_stats(df: pd.DataFrame) -> dict:
    """Compute updated statistics."""
    total = len(df)
    n_with_pkd = df['has_pkd'].sum()

    def safe_vc(series):
        return {str(k): int(v) for k, v in series.value_counts().items()}

    stats = {
        "dataset_name": "Trenzition v2 (with IC50 supplement)",
        "total_rows": int(total),
        "total_with_pkd": int(n_with_pkd),
        "pkd_coverage": round(100 * n_with_pkd / total, 1),

        "pkd_by_source": safe_vc(df['measurement_type_pkd']),
        "pkd_by_confidence": safe_vc(df['pkd_confidence']),
        "pkd_by_match_level": safe_vc(df['match_level']),

        "pkd_global_stats": {
            "mean": float(df.loc[df['has_pkd'], 'pkd_value'].mean()),
            "median": float(df.loc[df['has_pkd'], 'pkd_value'].median()),
            "std": float(df.loc[df['has_pkd'], 'pkd_value'].std()),
        },

        "pkd_by_source_stats": {},
    }

    # Per-source statistics
    for src in df['measurement_type_pkd'].dropna().unique():
        sub = df[df['measurement_type_pkd'] == src]
        stats["pkd_by_source_stats"][str(src)] = {
            "count": int(len(sub)),
            "mean": float(sub['pkd_value'].mean()),
            "median": float(sub['pkd_value'].median()),
            "std": float(sub['pkd_value'].std()),
        }

    return stats


def main():
    print("█" * 60)
    print("█  TRENZITION IC50 SUPPLEMENT")
    print("█  Adding approximate pKd from BindingDB IC50 data")
    print("█" * 60)

    # Load
    df_tren = load_trenzition()
    df_ic50 = extract_ic50_data()

    # Supplement
    df_v2, n_new = supplement_trenzition(df_tren, df_ic50)

    # Stats
    stats_v2 = compute_v2_stats(df_v2)

    # Save
    print("\n" + "=" * 60)
    print("Saving...")
    df_v2.to_parquet(OUTPUT_TRENZITION_V2, index=False)
    print(f"  → {OUTPUT_TRENZITION_V2}")
    print(f"  {len(df_v2):,} rows × {len(df_v2.columns)} columns")

    with open(OUTPUT_STATS_V2, 'w') as f:
        json.dump(stats_v2, f, indent=2, ensure_ascii=False, default=str)
    print(f"  → {OUTPUT_STATS_V2}")

    # Summary
    print("\n" + "█" * 60)
    print("█  IC50 SUPPLEMENT — COMPLETE")
    print("█" * 60)
    orig_pkd = df_v2['has_pkd'].sum() - n_new
    print(f"\n  Before supplement:      {orig_pkd:,} / {len(df_v2):,} ({100*orig_pkd/len(df_v2):.1f}%)")
    print(f"  New IC50 entries:       {n_new:,}")
    print(f"  After supplement:       {df_v2['has_pkd'].sum():,} / {len(df_v2):,} ({100*df_v2['has_pkd'].sum()/len(df_v2):.1f}%)")

    pS = stats_v2['pkd_by_source_stats']
    for src, s in pS.items():
        print(f"\n  {src}:")
        print(f"    n={s['count']:,}, mean={s['mean']:.3f}, median={s['median']:.3f}, std={s['std']:.3f}")

    return df_v2, stats_v2


if __name__ == "__main__":
    df_v2, stats_v2 = main()
