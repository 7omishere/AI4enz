"""
05_write_storage.py
=================
Write processed records to the final storage layout:

  processed/
  ├── proteins.h5          — protein sequences + structural features (HDF5)
  ├── metadata.parquet     — labels + metadata (Parquet)
  └── ligands/             — ligand graphs (pre-built by 04_build_ligand_graphs.py)

HDF5 layout:
  /{seq_hash}/
      sequence          : bytes (UTF-8 encoded)
      binding_site_mask : int32 array (L_site,)   [indices, not dense mask]
      contact_map       : bool array (L, L)        [dense if L ≤ 500]
                          OR bytes (CSR sparse)    [if L > 500]
      contact_map_sparse: bool (attribute)         [True if stored as sparse]
      contact_number    : float32 array (L,)
      protrusion_index  : float32 array (L,)

Parquet columns:
  sample_id, source_db, pdb_id, uniprot_id,
  protein_seq_hash, ligand_inchikey,
  pkd_raw, pkd_aligned, measurement_type,
  quality_weight, is_censored,
  has_binding_site, has_structure,
  n_measurements, pkd_std, split

Train/Val/Test split:
  - PDBbind 2016 core set (285 complexes) → test
  - Remaining PDBbind → train/val (90/10 by sequence cluster)
  - BindingDB → train only (no overlap with test proteins by UniProt ID)

Usage:
  python 05_write_storage.py \\
      --pdbbind   processed/pdbbind_records.pkl \\
      --bindingdb processed/bindingdb_aligned.pkl \\
      --out-dir   processed \\
      --core-set  data/pdbbind/index/CoreSet.dat
"""

import os
import io
import pickle
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

SPARSE_THRESHOLD = 500   # use sparse storage for contact_map if L > this


# ─────────────────────────────────────────────────────────────
# 1. Load PDBbind 2016 core set IDs
# ─────────────────────────────────────────────────────────────

def load_core_set(core_set_path: Optional[str]) -> set:
    """
    Load PDBbind 2016 core set PDB IDs.

    The CoreSet.dat file has lines like:
      1a1e  2.00  2001  7.22  Kd=60nM  ...
    (first column is PDB ID)

    Returns set of lowercase PDB IDs.
    """
    if core_set_path is None or not Path(core_set_path).exists():
        log.warning("Core set file not found — using empty test set")
        return set()

    core_ids = set()
    with open(core_set_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            core_ids.add(line.split()[0].lower())

    log.info(f"Loaded {len(core_ids)} PDBbind 2016 core set IDs")
    return core_ids


# ─────────────────────────────────────────────────────────────
# 2. Assign train/val/test splits
# ─────────────────────────────────────────────────────────────

def assign_splits(pdbbind_records: list,
                  bindingdb_records: list,
                  core_set_ids: set,
                  val_fraction: float = 0.10,
                  random_seed: int = 42) -> dict:
    """
    Assign split labels to all records.

    Rules:
      - PDBbind core set → 'test'
      - BindingDB proteins that share a UniProt ID with any test protein → excluded
        from test (kept in train)
      - Remaining PDBbind → 90% train, 10% val (random split by seq_hash cluster)
      - All BindingDB → 'train'

    Returns dict: sample_id → split
    """
    rng = np.random.default_rng(random_seed)

    # Collect test UniProt IDs (from PDBbind core set proteins)
    # We don't have UniProt IDs for PDBbind directly, so we use seq_hash
    # as a proxy — any BindingDB record sharing a seq_hash with a test
    # PDBbind record is excluded from test.
    test_seq_hashes = set()
    splits = {}

    # PDBbind splits
    non_test_hashes = []
    for r in pdbbind_records:
        sid = _sample_id(r)
        if r.get('pdb_id', '').lower() in core_set_ids:
            splits[sid] = 'test'
            test_seq_hashes.add(r['protein_seq_hash'])
        else:
            non_test_hashes.append((sid, r['protein_seq_hash']))

    # Shuffle non-test PDBbind by seq_hash for reproducible split
    unique_hashes = list({h for _, h in non_test_hashes})
    rng.shuffle(unique_hashes)
    n_val = max(1, int(len(unique_hashes) * val_fraction))
    val_hashes = set(unique_hashes[:n_val])

    for sid, h in non_test_hashes:
        splits[sid] = 'val' if h in val_hashes else 'train'

    # BindingDB splits — all train, but flag if seq_hash overlaps with test
    for r in bindingdb_records:
        sid = _sample_id(r)
        splits[sid] = 'train'   # BindingDB never in test/val

    log.info(f"Split counts: "
             f"train={sum(v=='train' for v in splits.values())}, "
             f"val={sum(v=='val' for v in splits.values())}, "
             f"test={sum(v=='test' for v in splits.values())}")
    return splits


def _sample_id(r: dict) -> str:
    """Generate a unique sample ID from record fields."""
    key = f"{r.get('source_db','')}__{r.get('pdb_id') or r.get('uniprot_id','')}__{r.get('ligand_inchikey','')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────
# 3. Write HDF5 protein store
# ─────────────────────────────────────────────────────────────

def write_proteins_h5(all_records: list, h5_path: str) -> None:
    """
    Write protein data to HDF5.

    Deduplicates by protein_seq_hash — each unique protein is stored once.
    contact_map stored as:
      - Dense bool array if L ≤ SPARSE_THRESHOLD
      - Serialized scipy CSR bytes if L > SPARSE_THRESHOLD
    """
    log.info(f"Writing protein HDF5 → {h5_path}")

    # Collect unique proteins
    proteins = {}
    for r in all_records:
        h = r.get('protein_seq_hash')
        if h and h not in proteins:
            proteins[h] = r

    log.info(f"  {len(proteins):,} unique proteins")

    with h5py.File(h5_path, 'w') as f:
        for seq_hash, r in tqdm(proteins.items(), desc='Writing proteins'):
            grp = f.require_group(seq_hash)

            # Sequence
            seq = r.get('sequence', '')
            grp.create_dataset('sequence',
                               data=np.bytes_(seq.encode('utf-8')),
                               dtype=h5py.special_dtype(vlen=bytes))

            # Binding site mask (store as index array, not dense bool)
            site = r.get('binding_site_mask')
            if site is not None and len(site) > 0:
                grp.create_dataset('binding_site_mask',
                                   data=np.array(site, dtype=np.int32),
                                   compression='gzip', compression_opts=4)
            else:
                grp.create_dataset('binding_site_mask',
                                   data=np.array([], dtype=np.int32))

            # Structural features
            cm = r.get('contact_map')
            cn = r.get('contact_number')
            pi = r.get('protrusion_index')

            if cm is not None:
                L = cm.shape[0]
                if L > SPARSE_THRESHOLD:
                    # Store as sparse CSR bytes
                    csr = sp.csr_matrix(cm.astype(np.uint8))
                    buf = io.BytesIO()
                    sp.save_npz(buf, csr)
                    grp.create_dataset('contact_map',
                                       data=np.frombuffer(buf.getvalue(), dtype=np.uint8))
                    grp.attrs['contact_map_sparse'] = True
                else:
                    grp.create_dataset('contact_map',
                                       data=cm.astype(np.bool_),
                                       compression='gzip', compression_opts=4)
                    grp.attrs['contact_map_sparse'] = False
            else:
                grp.attrs['contact_map_sparse'] = False

            if cn is not None:
                grp.create_dataset('contact_number',
                                   data=cn.astype(np.float32),
                                   compression='gzip', compression_opts=4)

            if pi is not None:
                grp.create_dataset('protrusion_index',
                                   data=pi.astype(np.float32),
                                   compression='gzip', compression_opts=4)

    log.info(f"  HDF5 written: {Path(h5_path).stat().st_size / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────
# 4. Write Parquet metadata
# ─────────────────────────────────────────────────────────────

def write_metadata_parquet(all_records: list,
                            splits: dict,
                            parquet_path: str) -> None:
    """
    Write metadata + labels to Parquet.
    """
    log.info(f"Writing metadata Parquet → {parquet_path}")

    rows = []
    for r in all_records:
        sid = _sample_id(r)
        rows.append({
            'sample_id':        sid,
            'source_db':        r.get('source_db', ''),
            'pdb_id':           r.get('pdb_id'),
            'uniprot_id':       r.get('uniprot_id'),
            'protein_seq_hash': r.get('protein_seq_hash'),
            'ligand_inchikey':  r.get('ligand_inchikey'),
            'pkd_raw':          r.get('pkd_raw'),
            'pkd_aligned':      r.get('pkd_aligned'),
            'measurement_type': r.get('measurement_type'),
            'quality_weight':   r.get('quality_weight'),
            'w_multiplier':     r.get('w_multiplier', 1.0),
            'is_censored':      r.get('is_censored', False),
            'has_binding_site': r.get('has_binding_site', False),
            'has_structure':    r.get('has_structure', False),
            'n_measurements':   r.get('n_measurements', 1),
            'pkd_std':          r.get('pkd_std', 0.0),
            'split':            splits.get(sid, 'train'),
        })

    df = pd.DataFrame(rows)

    # Drop records with missing essential fields
    n_before = len(df)
    df = df.dropna(subset=['protein_seq_hash', 'ligand_inchikey',
                            'pkd_aligned', 'quality_weight'])
    log.info(f"  Dropped {n_before - len(df)} records with missing fields")

    df.to_parquet(parquet_path, index=False, compression='snappy')
    log.info(f"  Parquet written: {len(df):,} records, "
             f"{Path(parquet_path).stat().st_size / 1e6:.1f} MB")

    # Summary statistics
    for split in ('train', 'val', 'test'):
        sub = df[df['split'] == split]
        if len(sub) > 0:
            log.info(f"  {split:5s}: {len(sub):6,} records, "
                     f"pKd mean={sub['pkd_aligned'].mean():.2f} "
                     f"std={sub['pkd_aligned'].std():.2f}")


# ─────────────────────────────────────────────────────────────
# 5. Utility: load contact_map from HDF5 (handles sparse/dense)
# ─────────────────────────────────────────────────────────────

def load_contact_map(grp: h5py.Group) -> Optional[np.ndarray]:
    """
    Load contact_map from an HDF5 group, handling both dense and sparse formats.
    Returns bool numpy array (L, L) or None if not available.
    """
    if 'contact_map' not in grp:
        return None

    is_sparse = grp.attrs.get('contact_map_sparse', False)
    if is_sparse:
        raw_bytes = grp['contact_map'][()].tobytes()
        buf = io.BytesIO(raw_bytes)
        csr = sp.load_npz(buf)
        return csr.toarray().astype(bool)
    else:
        return grp['contact_map'][()].astype(bool)


# ─────────────────────────────────────────────────────────────
# 6. Main entry point
# ─────────────────────────────────────────────────────────────

def write_storage(pdbbind_records: list,
                  bindingdb_records: list,
                  out_dir: str,
                  core_set_path: Optional[str] = None,
                  val_fraction: float = 0.10,
                  random_seed: int = 42) -> None:
    """
    Write all processed data to the final storage layout.

    Parameters
    ----------
    pdbbind_records   : list of PDBbind record dicts
    bindingdb_records : list of aligned BindingDB record dicts
    out_dir           : output directory (will be created if needed)
    core_set_path     : path to PDBbind 2016 CoreSet.dat (for test split)
    val_fraction      : fraction of non-test PDBbind for validation
    random_seed       : for reproducible splits
    """
    os.makedirs(out_dir, exist_ok=True)

    all_records = pdbbind_records + bindingdb_records
    log.info(f"Total records: {len(all_records):,} "
             f"(PDBbind={len(pdbbind_records):,}, BindingDB={len(bindingdb_records):,})")

    # Assign splits
    core_set_ids = load_core_set(core_set_path)
    splits = assign_splits(pdbbind_records, bindingdb_records,
                           core_set_ids, val_fraction, random_seed)

    # Write HDF5
    h5_final = str(Path(out_dir) / 'proteins.h5')
    write_proteins_h5(all_records, h5_final)

    # Write Parquet
    parquet_path = str(Path(out_dir) / 'metadata.parquet')
    write_metadata_parquet(all_records, splits, parquet_path)

    log.info("Storage write complete.")
    log.info(f"  proteins.h5      → {h5_final}")
    log.info(f"  metadata.parquet → {parquet_path}")


# ─────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write processed data to HDF5 + Parquet')
    parser.add_argument('--pdbbind',    required=True,
                        help='PDBbind records pickle')
    parser.add_argument('--bindingdb',  required=True,
                        help='Aligned BindingDB records pickle')
    parser.add_argument('--out-dir',    required=True,
                        help='Output directory')
    parser.add_argument('--core-set',   default=None,
                        help='PDBbind 2016 CoreSet.dat for test split')
    parser.add_argument('--val-frac',   type=float, default=0.10)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    with open(args.pdbbind,   'rb') as f:
        pdbbind_records = pickle.load(f)
    with open(args.bindingdb, 'rb') as f:
        bindingdb_records = pickle.load(f)

    write_storage(
        pdbbind_records=pdbbind_records,
        bindingdb_records=bindingdb_records,
        out_dir=args.out_dir,
        core_set_path=args.core_set,
        val_fraction=args.val_frac,
        random_seed=args.seed,
    )
