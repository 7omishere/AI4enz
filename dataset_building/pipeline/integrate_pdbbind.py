"""
integrate_pdbbind.py
====================
Integrate newly parsed PDBbind records into the unified dataset.

Steps:
  1. Load PDBbind records from pickle
  2. Write protein features to proteins.h5 (append, skip existing seq_hashes)
  3. Build ligand graphs for new ligands
  4. Convert to unified metadata Parquet format
  5. Deduplicate against existing dataset (by seq_hash + inchikey)
  6. Merge with existing recommended_training_set_enriched.parquet
  7. Also merge kcat data from existing sources for matching proteins

Usage:
  cd dataset_building/pipeline
  python integrate_pdbbind.py --pdbbind ../processed/pdbbind_records.pkl \
      --existing-metadata ../release/recommended_training_set_enriched.parquet \
      --proteins-h5 ../processed/proteins.h5 \
      --ligand-dir ../processed/ligands \
      --out-dir ../processed \
      --workers 8
"""

import io
import os
import pickle
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Optional
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def _safe_int64_array(arr):
    """Convert to safe int64 array, ensure within valid range."""
    arr = np.asarray(arr, dtype=np.int64)
    arr = arr[(arr >= -2**62) & (arr < 2**62-1)]
    return arr


def write_protein_to_h5(h5_path: str, record: dict) -> bool:
    """
    Write a single protein's features to proteins.h5.
    Skip if seq_hash already exists.
    Returns True if written, False if skipped.
    """
    seq_hash = record['protein_seq_hash']
    with h5py.File(h5_path, 'a') as h5:
        if seq_hash in h5:
            return False

        grp = h5.create_group(seq_hash)
        grp.create_dataset('sequence', data=record['sequence'].encode('utf-8'))

        if record.get('binding_site_mask'):
            mask = _safe_int64_array(record['binding_site_mask'])
            grp.create_dataset('binding_site_mask', data=mask)
        else:
            grp.create_dataset('binding_site_mask', data=np.array([], dtype=np.int64))

        cm = record.get('contact_map')
        cn = record.get('contact_number')
        pi = record.get('protrusion_index')

        if cm is not None and cn is not None and pi is not None:
            L = len(cn)
            if L <= 500:
                grp.create_dataset('contact_map', data=cm)
                grp.attrs['contact_map_sparse'] = False
            else:
                # Store as sparse CSR
                from scipy.sparse import csr_matrix
                cm_sparse = csr_matrix(cm)
                buf = io.BytesIO()
                cm_sparse.tofile(buf)
                grp.create_dataset('contact_map_sparse_bytes', data=np.frombuffer(buf.getvalue(), dtype=np.uint8))
                grp.attrs['contact_map_sparse'] = True

            grp.create_dataset('contact_number', data=cn.astype(np.float32))
            grp.create_dataset('protrusion_index', data=pi.astype(np.float32))
        else:
            grp.create_dataset('binding_site_mask', data=np.array([], dtype=np.int64))
            grp.attrs['no_structure'] = True

    return True


def build_ligand_graphs(records: list, out_dir: str, n_workers: int = 4,
                        overwrite: bool = False):
    """Build ligand molecular graphs for new ligands only."""
    # Import here to avoid circular dependency
    import importlib
    build = importlib.import_module('04_build_ligand_graphs').build_ligand_graphs
    build(records=records, out_dir=out_dir, n_workers=n_workers, overwrite=overwrite)


def convert_to_dataframe(records: list) -> pd.DataFrame:
    """Convert PDBbind records to a unified metadata DataFrame."""
    rows = []
    for r in records:
        rows.append({
            'sample_id': f"pdbbind_{r['pdb_id']}",
            'protein_seq_hash': r['protein_seq_hash'],
            'ligand_inchikey': r['ligand_inchikey'],
            'uniprot_id': None,
            'pdb_id': r['pdb_id'],
            'source_db': 'PDBbind',
            'ec_numbers': None,
            'cofactors': None,
            'protein_name': None,
            'reviewed': False,
            'pkd_aligned': r['pkd_aligned'],
            'pkd_raw': r['pkd_raw'],
            'measurement_type': r['measurement_type'],
            'quality_weight': r['quality_weight'],
            'w_multiplier': 1.0,
            'is_censored': r['is_censored'],
            'n_measurements': 1,
            'pkd_std': 0.0,
            'has_structure': r['has_structure'],
            'has_binding_site': r['has_binding_site'],
            'has_domain_annotation': False,
            'n_domains': 0,
            'cofactor_domain_types': None,
            'domains_json': None,
            # kcat fields — none from PDBbind directly
            'bdb_n_km': 0, 'bdb_n_kcat': 0, 'bdb_n_kcatkm': 0,
            'bdb_km_median_uM': None, 'bdb_kcat_median_s': None,
            'bdb_kcatkm_median_M1s1': None,
            'n_km_sabio': 0, 'n_kcat_sabio': 0, 'n_kcatkm_sabio': 0,
            'km_median_uM_sabio': None, 'kcat_median_s_sabio': None,
            'kcatkm_median_M1s1_sabio': None,
            'up_has_kinetics': False, 'up_n_kcat': 0,
            'has_kcat': False, 'kcat_source': None,
            'kcat_median_s': None, 'log_kcat_median': None,
            'kcat_outlier': False,
            'split': None,  # To be assigned
            'structure_source': 'PDBbind_v2020R1',
            'pkd_corrected': None, 'correction_source': None,
            'temperature': None, 'ph_val': None,
            'organism_name': None, 'substrate_name': None,
            'enzymetype': None,
            'is_mutant': False, 'mutation_info': None,
            'split_v2': None,
        })
    return pd.DataFrame(rows)


def integrate(pdbbind_path: str,
              existing_parquet: str,
              proteins_h5: str,
              ligand_dir: str,
              out_dir: str,
              n_workers: int = 4,
              dry_run: bool = False):
    """
    Main integration function.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load ---
    log.info("Loading PDBbind records...")
    with open(pdbbind_path, 'rb') as f:
        pdbbind_records = pickle.load(f)
    log.info(f"Loaded {len(pdbbind_records):,} PDBbind records")

    log.info("Loading existing dataset...")
    existing = pd.read_parquet(existing_parquet)
    log.info(f"Loaded {len(existing):,} existing records")

    # --- 2. Deduplicate against existing ---
    existing_keys = set(zip(existing['protein_seq_hash'], existing['ligand_inchikey']))
    new_records = []
    dup_count = 0
    for r in pdbbind_records:
        key = (r['protein_seq_hash'], r['ligand_inchikey'])
        if key in existing_keys:
            dup_count += 1
        else:
            new_records.append(r)

    log.info(f"Deduplication: {len(new_records):,} new, {dup_count:,} duplicates removed")

    if dry_run:
        log.info("DRY RUN — exiting without writing.")
        return

    # --- 3. Write protein features to HDF5 ---
    log.info("Writing protein features to proteins.h5...")
    written = 0
    for r in tqdm(new_records, desc="Proteins to HDF5"):
        if write_protein_to_h5(proteins_h5, r):
            written += 1
    log.info(f"Proteins written: {written:,} new entries")

    # --- 4. Build ligand graphs ---
    log.info("Building ligand graphs (new ligands only)...")
    build_ligand_graphs(new_records, ligand_dir, n_workers=n_workers, overwrite=False)

    # --- 5. Convert to DataFrame ---
    new_df = convert_to_dataframe(new_records)

    # --- 6. Merge with existing ---
    merged = pd.concat([existing, new_df], ignore_index=True)
    log.info(f"Merged dataset: {len(merged):,} total records")

    # --- 7. Assign split (PDBbind goes to train, unless core set) ---
    # All PDBbind data goes to 'train' split (test is from PDBbind core set 2016)
    merged.loc[merged['split'].isna(), 'split'] = 'train'
    merged.loc[merged['split_v2'].isna(), 'split_v2'] = 'train'

    # --- 8. Save ---
    out_parquet = out_dir / 'recommended_training_set_with_pdbbind.parquet'
    merged.to_parquet(out_parquet, index=False)
    log.info(f"Saved → {out_parquet}")

    # Statistics
    n_pdbbind_final = (merged['source_db'] == 'PDBbind').sum()
    n_pdbbind_with_structure = ((merged['source_db'] == 'PDBbind') & merged['has_structure']).sum()
    n_pdbbind_with_binding_site = ((merged['source_db'] == 'PDBbind') & merged['has_binding_site']).sum()
    log.info(f"\nFinal PDBbind records: {n_pdbbind_final:,}")
    log.info(f"  With structure: {n_pdbbind_with_structure:,}")
    log.info(f"  With binding site: {n_pdbbind_with_binding_site:,}")

    source_counts = merged['source_db'].value_counts()
    log.info(f"\nSource distribution:")
    for src, count in source_counts.items():
        log.info(f"  {src}: {count:,}")

    return merged


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate PDBbind into unified dataset')
    parser.add_argument('--pdbbind', required=True, help='PDBbind records pickle')
    parser.add_argument('--existing-metadata', required=True, help='Existing unified parquet')
    parser.add_argument('--proteins-h5', required=True, help='proteins.h5 path')
    parser.add_argument('--ligand-dir', required=True, help='Ligand graphs directory')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true', help='Only preview, no writing')
    args = parser.parse_args()

    integrate(
        pdbbind_path=args.pdbbind,
        existing_parquet=args.existing_metadata,
        proteins_h5=args.proteins_h5,
        ligand_dir=args.ligand_dir,
        out_dir=args.out_dir,
        n_workers=args.workers,
        dry_run=args.dry_run,
    )
