"""
Apply IC50→Ki correction model to unified_metadata.parquet.

Reads the trained correction model from ic50_ki_correction.json,
applies pKi = a * pIC50 + b to all IC50 records,
merges ChEMBL Kd supplements, and updates the parquet file.

Usage:
    python datepre/apply_ic50_correction.py
    python datepre/apply_ic50_correction.py --dry-run    # Preview only
"""

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent / "dataset_building"
OXI_DIR = PROJECT_DIR / "processed" / "oxidoreductase"
UNIFIED_META = OXI_DIR / "unified_metadata.parquet"
CORRECTION_PATH = OXI_DIR / "ic50_ki_correction.json"
KD_SUPP_PATH = OXI_DIR / "chembl_kd_supplement.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Apply IC50→Ki correction to unified_metadata"
    )
    parser.add_argument("--unified-meta", default=str(UNIFIED_META))
    parser.add_argument("--correction", default=str(CORRECTION_PATH))
    parser.add_argument("--kd-supplement", default=str(KD_SUPP_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    meta_path = Path(args.unified_meta)
    correction_path = Path(args.correction)
    kd_path = Path(args.kd_supplement)

    # Backup
    if not args.dry_run:
        bak = meta_path.with_suffix(".parquet.pre_correction")
        shutil.copy(meta_path, bak)
        log.info(f"Backup: {bak}")

    # Load model
    if correction_path.exists():
        correction = json.loads(correction_path.read_text())
        log.info(f"Correction model: {correction.get('model_type')}")
        log.info(f"  {correction.get('formula', 'Ki=IC50/2')}")
    else:
        log.warning(f"Correction model not found: {correction_path}")
        log.info("Using theoretical correction: Ki = IC50/2")
        correction = {'model_type': 'theoretical', 'a': 1.0, 'b': 0.301}

    a, b = correction['a'], correction['b']

    # Load data
    df = pd.read_parquet(meta_path)
    log.info(f"Loaded: {len(df):,} rows")

    # Stats before
    ic50_mask = df['measurement_type'] == 'IC50'
    n_ic50_before = ic50_mask.sum()
    log.info(f"\nBefore correction:")
    log.info(f"  IC50 records:  {n_ic50_before:,}")
    log.info(f"  Ki records:    {(df['measurement_type'] == 'Ki').sum():,}")
    log.info(f"  Kd records:    {(df['measurement_type'] == 'Kd').sum():,}")
    log.info(f"  pkd_aligned mean (IC50): {df.loc[ic50_mask, 'pkd_aligned'].mean():.3f}")

    # Apply correction
    if not args.dry_run:
        # Add/update columns
        if 'pkd_corrected' not in df.columns:
            df['pkd_corrected'] = df['pkd_aligned'].copy()

        if 'correction_source' not in df.columns:
            df['correction_source'] = 'none'

        # Correct IC50 records
        df.loc[ic50_mask, 'pkd_corrected'] = (
            a * df.loc[ic50_mask, 'pkd_aligned'] + b
        )
        df.loc[ic50_mask, 'pkd_aligned'] = df.loc[ic50_mask, 'pkd_corrected']
        df.loc[ic50_mask, 'correction_source'] = 'chembl_ic50_ki_model'

        # Ki/Kd records: pkd_corrected = pkd_aligned (unchanged)
        non_ic50 = ~ic50_mask
        df.loc[non_ic50, 'pkd_corrected'] = df.loc[non_ic50, 'pkd_aligned']

        log.info(f"\nAfter correction (IC50):")
        log.info(f"  pkd_aligned mean:     {df.loc[ic50_mask, 'pkd_aligned'].mean():.3f}")
        log.info(f"  Shift:                {df.loc[ic50_mask, 'pkd_aligned'].mean() - df.loc[ic50_mask, 'pkd_raw'].mean():.3f}")
        log.info(f"  correction_source:    {df['correction_source'].value_counts().to_dict()}")

    # ── Merge Kd supplements ──────────────────────────
    if kd_path.exists():
        kd_df = pd.read_parquet(kd_path)
        log.info(f"\nChEMBL Kd supplements: {len(kd_df)} records")

        if len(kd_df) > 0 and not args.dry_run:
            # Align with existing protein seq_hashes
            up_to_sh = df.groupby('uniprot_id')['protein_seq_hash'].first().to_dict()

            kd_supplements = []
            for _, row in kd_df.iterrows():
                uid = row['uniprot_id']
                sh = up_to_sh.get(uid)
                if not sh:
                    continue

                kd_supplements.append({
                    'protein_seq_hash': sh,
                    'uniprot_id': uid,
                    'ligand_inchikey': row['ligand_inchikey'],
                    'measurement_type': 'Kd',
                    'pkd_raw': row['pkd_raw'],
                    'pkd_aligned': row['pkd_raw'],
                    'pkd_corrected': row['pkd_raw'],
                    'correction_source': 'chembl_kd_direct',
                    'source_db': 'ChEMBL',
                    'quality_weight': 1.0,
                    'w_multiplier': 1.0,
                    'is_censored': False,
                    'n_measurements': row.get('n_measurements', 1),
                    'pkd_std': row.get('pkd_std', 0.0),
                    'has_structure': df[df['protein_seq_hash'] == sh]['has_structure'].iloc[0]
                        if sh in df['protein_seq_hash'].values else False,
                    'has_binding_site': False,
                    'has_domain_annotation': False,
                    'n_domains': 0,
                    'cofactor_domain_types': '',
                    'domains_json': '[]',
                    'has_kcat': False,
                    'kcat_source': 'none',
                    'kcat_median_s': np.nan,
                    'log_kcat_median': np.nan,
                    'kcat_outlier': False,
                })

            if kd_supplements:
                supp_df = pd.DataFrame(kd_supplements)

                # Fill missing columns from existing df schema
                for col in df.columns:
                    if col not in supp_df.columns:
                        if df[col].dtype == 'float64':
                            supp_df[col] = np.nan
                        elif df[col].dtype == 'bool':
                            supp_df[col] = False
                        elif df[col].dtype == 'int64':
                            supp_df[col] = 0
                        else:
                            supp_df[col] = ''

                supp_df = supp_df[df.columns]
                df = pd.concat([df, supp_df], ignore_index=True)
                log.info(f"  Merged: {len(supp_df)} new Kd records")

    # ── Save ──────────────────────────────────────────
    if not args.dry_run:
        df.to_parquet(meta_path, index=False)
        log.info(f"\nSaved: {meta_path} ({len(df):,} rows × {len(df.columns)} cols)")

        # Final stats
        log.info(f"\n{'='*50}")
        log.info(f"Final dataset:")
        log.info(f"  Total records:        {len(df):,}")
        log.info(f"  IC50:                 {(df['measurement_type']=='IC50').sum():,}")
        log.info(f"  Ki:                   {(df['measurement_type']=='Ki').sum():,}")
        log.info(f"  Kd:                   {(df['measurement_type']=='Kd').sum():,}")
        log.info(f"  Corrected IC50:       {(df['correction_source']=='chembl_ic50_ki_model').sum():,}")
        log.info(f"  Direct Kd (ChEMBL):   {(df['correction_source']=='chembl_kd_direct').sum():,}")
        log.info(f"  Proteins:             {df['protein_seq_hash'].nunique()}")
        log.info(f"  Ligands:              {df['ligand_inchikey'].nunique()}")
    else:
        log.info("\nDry run complete. No changes written.")


if __name__ == "__main__":
    main()
