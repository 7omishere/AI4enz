"""
02_parse_bindingdb.py  (v2 — fixed)
==================================
Parse raw BindingDB_All.tsv (or .tsv.gz) into structured records.

Key fixes vs v1:
  - Censored affinity values ('>'/''<') flagged with is_censored=True;
    quality_weight halved for censored entries.
  - Sequence length capped at 1020 AA (ESM-2 650M max).
  - Binding site mask indices validated against truncated sequence length.

Steps:
  1. Read TSV in chunks (file is ~2.8 GB compressed)
  2. Filter: valid SMILES, valid sequence (50–1020 AA), at least one affinity
  3. Select measurement with priority Kd > Ki > IC50; detect censoring
  4. Convert to pKd = -log10(value_nM * 1e-9)
  5. Fetch UniProt binding site annotations via REST API
  6. Deduplicate by (canonical_smiles, sequence_hash)

Usage:
  python 02_parse_bindingdb.py \\
      --tsv   data/BindingDB_All.tsv \\
      --out   processed/bindingdb_records.pkl \\
      --workers 8

The TSV may be gzipped (.tsv.gz) or uncompressed (.tsv).
"""

import os
import re
import time
import hashlib
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

COLS = {
    'smiles':      'Ligand SMILES',
    'sequence':    'BindingDB Target Chain Sequence 1',
    'uniprot':     'UniProt (SwissProt) Primary ID of Target Chain 1',
    'kd':          'Kd (nM)',
    'ki':          'Ki (nM)',
    'ic50':        'IC50 (nM)',
    'target_name': 'Target Name',
}

# Base quality weights (before censoring penalty)
QUALITY_WEIGHTS = {'Kd': 1.0, 'Ki': 0.85, 'IC50': 0.40}

# ESM-2 650M hard limit
MAX_SEQ_LEN = 1020

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


# ─────────────────────────────────────────────────────────────
# 1. pKd conversion
# ─────────────────────────────────────────────────────────────

def nM_to_pkd(value_nM: float) -> Optional[float]:
    """Convert affinity in nM to pKd. Returns None if out of range [2, 15]."""
    if value_nM <= 0:
        return None
    pkd = -np.log10(value_nM * 1e-9)
    return pkd if 2.0 <= pkd <= 15.0 else None


def parse_affinity_value(s) -> tuple:
    """
    Parse affinity string from BindingDB.
    Handles: '1.5', '>1000', '<0.1', '1.5e3', '~100', '1,500'

    Returns (value_nM: float | None, is_censored: bool).
    is_censored=True when the original string contained '>' or '<'.
    """
    if pd.isna(s) or str(s).strip() == '':
        return None, False

    s_orig = str(s).strip()
    is_censored = ('>' in s_orig or '<' in s_orig)

    s_clean = s_orig.replace(',', '')
    s_clean = re.sub(r'^[><=~\s]+', '', s_clean)
    try:
        return float(s_clean), is_censored
    except ValueError:
        return None, False


def select_best_affinity(row: pd.Series) -> tuple:
    """
    Select best affinity value with priority Kd > Ki > IC50.

    Returns (value_nM, measurement_type, base_quality_weight, is_censored)
    or (None, None, None, False) if no valid value found.
    """
    for col, mtype in [('kd', 'Kd'), ('ki', 'Ki'), ('ic50', 'IC50')]:
        raw = row.get(col, '')  # col already renamed from COLS mapping
        val, censored = parse_affinity_value(raw)
        if val is not None:
            return val, mtype, QUALITY_WEIGHTS[mtype], censored
    return None, None, None, False


# ─────────────────────────────────────────────────────────────
# 2. SMILES validation and canonicalization
# ─────────────────────────────────────────────────────────────

def canonicalize_smiles(smiles) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    if pd.isna(smiles) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """Return InChIKey for ligand deduplication."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.inchi.MolToInchiKey(mol)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# 3. Sequence validation
# ─────────────────────────────────────────────────────────────

def validate_sequence(seq) -> Optional[str]:
    """
    Validate and clean protein sequence.
    - Removes non-standard amino acid characters.
    - Enforces length in [50, MAX_SEQ_LEN].
    Returns cleaned sequence or None if invalid.
    """
    if pd.isna(seq) or not seq:
        return None
    seq = str(seq).strip().upper()
    seq = ''.join(c for c in seq if c in VALID_AA)
    if len(seq) < 50:
        return None
    # Truncate to ESM-2 limit (with a warning logged at build_records level)
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
    return seq


# ─────────────────────────────────────────────────────────────
# 4. UniProt binding site annotation
# ─────────────────────────────────────────────────────────────

_uniprot_cache: dict = {}


def fetch_uniprot_binding_sites(uniprot_id: str,
                                 max_retries: int = 3,
                                 timeout: int = 10) -> Optional[list]:
    """
    Fetch active site / binding site residue positions from UniProt REST API.

    Returns list of 0-based residue indices (relative to UniProt canonical
    sequence, which matches BindingDB's 'Target Sequence' field).
    Returns None if unavailable.

    Note: indices are validated against the truncated sequence length
    in build_records() to handle the 1020 AA cap.
    """
    if not uniprot_id or pd.isna(uniprot_id):
        return None

    uniprot_id = str(uniprot_id).strip()
    if uniprot_id in _uniprot_cache:
        return _uniprot_cache[uniprot_id]

    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.json'

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                _uniprot_cache[uniprot_id] = None
                return None
            if resp.status_code != 200:
                time.sleep(2 ** attempt)
                continue

            data = resp.json()
            features = data.get('features', [])

            site_positions = []
            for feat in features:
                if feat.get('type') in ('Active site', 'Binding site',
                                         'active site', 'binding site'):
                    loc   = feat.get('location', {})
                    start = loc.get('start', {}).get('value')
                    end   = loc.get('end',   {}).get('value')
                    if start is not None:
                        # UniProt is 1-based → convert to 0-based
                        if end is not None and end > start:
                            site_positions.extend(range(start - 1, end))
                        else:
                            site_positions.append(start - 1)

            result = sorted(set(site_positions)) if site_positions else None
            _uniprot_cache[uniprot_id] = result
            return result

        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)

    _uniprot_cache[uniprot_id] = None
    return None


# ─────────────────────────────────────────────────────────────
# 5. Parse TSV in chunks
# ─────────────────────────────────────────────────────────────

def parse_bindingdb_tsv(tsv_path: str,
                         chunk_size: int = 100_000) -> pd.DataFrame:
    """
    Read BindingDB TSV (possibly gzipped) in chunks.
    Returns filtered DataFrame with standardized columns.
    """
    log.info(f"Reading {tsv_path} ...")

    needed_cols = list(COLS.values())
    chunks = []
    total_rows = 0
    kept_rows  = 0

    reader = pd.read_csv(
        tsv_path,
        sep='\t',
        usecols=lambda c: c in needed_cols,
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines='skip',
        encoding='utf-8',
        encoding_errors='replace',
    )

    for chunk in tqdm(reader, desc='Reading BindingDB chunks'):
        total_rows += len(chunk)

        rename = {v: k for k, v in COLS.items() if v in chunk.columns}
        chunk = chunk.rename(columns=rename)

        chunk = chunk.dropna(subset=['smiles', 'sequence'])

        affinity_cols = [c for c in ('kd', 'ki', 'ic50') if c in chunk.columns]
        if affinity_cols:
            has_affinity = chunk[affinity_cols].notna().any(axis=1)
            chunk = chunk[has_affinity]

        if len(chunk) > 0:
            chunks.append(chunk)
            kept_rows += len(chunk)

    log.info(f"Read {total_rows:,} rows, kept {kept_rows:,} after basic filter")

    if not chunks:
        raise ValueError("No valid rows found in BindingDB TSV")

    return pd.concat(chunks, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# 6. Build records from filtered DataFrame
# ─────────────────────────────────────────────────────────────

def build_records(df: pd.DataFrame,
                  fetch_binding_sites: bool = True,
                  max_uniprot_workers: int = 8) -> list:
    """
    Convert filtered DataFrame rows into record dicts.

    Quality weight formula:
      quality_weight = base_weight * (0.5 if is_censored else 1.0) * 0.85
      (the 0.85 factor reflects BindingDB's lower data quality vs PDBbind)
    """
    records = []
    failed  = 0
    truncated = 0

    # Pre-fetch unique UniProt IDs in parallel
    if fetch_binding_sites and 'uniprot' in df.columns:
        unique_uniprots = df['uniprot'].dropna().unique().tolist()
        log.info(f"Fetching UniProt annotations for {len(unique_uniprots):,} proteins ...")

        def _fetch(uid):
            return uid, fetch_uniprot_binding_sites(uid)

        with ThreadPool(max_uniprot_workers) as pool:
            for uid, sites in tqdm(
                    pool.imap_unordered(_fetch, unique_uniprots),
                    total=len(unique_uniprots),
                    desc='UniProt API'):
                _uniprot_cache[uid] = sites

    log.info("Building records ...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Building records'):

        # Validate SMILES
        smiles = canonicalize_smiles(row.get('smiles', ''))
        if smiles is None:
            failed += 1
            continue

        # Validate + truncate sequence
        raw_seq = row.get('sequence', '')
        seq = validate_sequence(raw_seq)
        if seq is None:
            failed += 1
            continue
        if len(str(raw_seq).strip()) > MAX_SEQ_LEN:
            truncated += 1

        # Select affinity
        val_nM, mtype, base_qw, is_censored = select_best_affinity(row)
        if val_nM is None:
            failed += 1
            continue

        pkd = nM_to_pkd(val_nM)
        if pkd is None:
            failed += 1
            continue

        # Quality weight: base × censoring penalty × BindingDB penalty
        quality_weight = base_qw * (0.5 if is_censored else 1.0) * 0.85

        # Hashes
        seq_hash = hashlib.sha256(seq.encode()).hexdigest()[:16]
        inchikey  = smiles_to_inchikey(smiles)

        # Binding site from UniProt cache
        uniprot_id = row.get('uniprot', None)
        site = None
        if fetch_binding_sites and uniprot_id and not pd.isna(uniprot_id):
            raw_site = _uniprot_cache.get(str(uniprot_id).strip(), None)
            if raw_site is not None:
                # Filter out indices beyond truncated sequence length
                site = [i for i in raw_site if i < len(seq)]
                if not site:
                    site = None

        records.append({
            # Identity
            'pdb_id':             None,
            'uniprot_id':         str(uniprot_id).strip() if uniprot_id and not pd.isna(uniprot_id) else None,
            'source_db':          'BindingDB',
            'protein_seq_hash':   seq_hash,
            'ligand_inchikey':    inchikey,
            # Labels (pkd_aligned filled later by alignment step)
            'pkd_raw':            pkd,
            'pkd_aligned':        pkd,   # placeholder; overwritten after GMM alignment
            'measurement_type':   mtype,
            'quality_weight':     quality_weight,
            'is_censored':        is_censored,
            'resolution':         np.nan,
            # Protein data
            'sequence':           seq,
            'binding_site_mask':  site,
            'has_binding_site':   site is not None,
            'has_structure':      False,
            # Structural features: not available for BindingDB
            'contact_map':        None,
            'contact_number':     None,
            'protrusion_index':   None,
            # Ligand
            'ligand_smiles':      smiles,
        })

    log.info(f"Built {len(records):,} records, {failed:,} failed, "
             f"{truncated:,} sequences truncated to {MAX_SEQ_LEN} AA")
    return records


# ─────────────────────────────────────────────────────────────
# 7. Deduplication
# ─────────────────────────────────────────────────────────────

def deduplicate(records: list) -> list:
    """
    For identical (protein_seq_hash, ligand_inchikey) pairs,
    keep the quality-weighted median pKd and record measurement spread.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in records:
        key = (r['protein_seq_hash'], r['ligand_inchikey'])
        groups[key].append(r)

    deduped = []
    for key, group in groups.items():
        if len(group) == 1:
            r = group[0].copy()
            r['n_measurements'] = 1
            r['pkd_std']        = 0.0
            deduped.append(r)
        else:
            pkd_vals = np.array([g['pkd_raw']       for g in group])
            weights  = np.array([g['quality_weight'] for g in group])

            # Quality-weighted median
            sorted_idx = np.argsort(pkd_vals)
            pkd_s = pkd_vals[sorted_idx]
            w_s   = weights[sorted_idx]
            cumw  = np.cumsum(w_s)
            med_i = np.searchsorted(cumw, cumw[-1] / 2)

            best = group[0].copy()
            best['pkd_raw']        = float(pkd_s[med_i])
            best['pkd_aligned']    = float(pkd_s[med_i])
            best['n_measurements'] = len(group)
            best['pkd_std']        = float(np.std(pkd_vals))
            # Prefer record with binding site annotation
            for g in group:
                if g['has_binding_site']:
                    best['binding_site_mask'] = g['binding_site_mask']
                    best['has_binding_site']  = True
                    break
            deduped.append(best)

    log.info(f"After deduplication: {len(deduped):,} unique complexes "
             f"(from {len(records):,})")
    return deduped


# ─────────────────────────────────────────────────────────────
# 8. Main entry point
# ─────────────────────────────────────────────────────────────

def parse_bindingdb(tsv_path: str,
                    out_path: str,
                    fetch_binding_sites: bool = True,
                    uniprot_workers: int = 8,
                    max_rows: Optional[int] = None) -> list:
    """
    Full BindingDB parsing pipeline.

    Parameters
    ----------
    tsv_path           : path to BindingDB_All.tsv.gz
    out_path           : output pickle file path
    fetch_binding_sites: whether to query UniProt API for binding sites
    uniprot_workers    : parallel threads for UniProt API calls
    max_rows           : limit rows for debugging (None = all)
    """
    df = parse_bindingdb_tsv(tsv_path)
    if max_rows:
        df = df.head(max_rows)

    records = build_records(df,
                            fetch_binding_sites=fetch_binding_sites,
                            max_uniprot_workers=uniprot_workers)

    records = deduplicate(records)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(records, f)
    log.info(f"Saved {len(records):,} records to {out_path}")

    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse BindingDB TSV')
    parser.add_argument('--tsv',        required=True)
    parser.add_argument('--out',        default='processed/bindingdb_records.pkl')
    parser.add_argument('--no-uniprot', action='store_true',
                        help='Skip UniProt binding site fetching')
    parser.add_argument('--workers',    type=int, default=8)
    parser.add_argument('--max-rows',   type=int, default=None)
    args = parser.parse_args()

    parse_bindingdb(
        tsv_path=args.tsv,
        out_path=args.out,
        fetch_binding_sites=not args.no_uniprot,
        uniprot_workers=args.workers,
        max_rows=args.max_rows,
    )
