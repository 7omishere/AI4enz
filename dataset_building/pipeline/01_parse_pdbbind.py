"""
01_parse_pdbbind.py  (v2 — fixed)
================================
Parse raw PDBbind v2020 download into structured records.

Key fixes vs v1:
  - Multi-chain proteins: select the chain that contains the binding site
    (or the longest chain if no binding site is available).
    ESM-2 has no ':' token, so we NEVER join chains with ':'.
  - Binding site mask indices are now 0-based relative to the SELECTED chain,
    not the concatenated multi-chain sequence.
  - Sequence length capped at 1020 AA (ESM-2 650M max = 1022 tokens incl. special).
  - Censored affinity values ('>'/''<') flagged with is_censored=True.

Outputs per complex:
  - pdb_id, pkd_raw, measurement_type, quality_weight, is_censored, resolution
  - protein_sequence   (str, single chain, ≤ 1020 AA)
  - selected_chain_id  (str)
  - binding_site_mask  (list[int], 0-based within selected chain)
  - contact_map        (np.ndarray bool, L×L)
  - contact_number     (np.ndarray float32, L)
  - protrusion_index   (np.ndarray float32, L)
  - ligand_smiles      (str, canonical)
  - protein_seq_hash   (str, SHA-256[:16])
  - ligand_inchikey    (str)

Usage:
  python 01_parse_pdbbind.py \\
      --index  data/index/INDEX_general_PL.2020R1.lst \\
      --struct data/P-L \\
      --out    processed/pdbbind_records.pkl \\
      --workers 8

The --struct directory may be flat ({pdb_id}/) or nested by year
(e.g. P-L/2001-2010/{pdb_id}/). The script searches subdirectories automatically.
"""

import os
import re
import hashlib
import pickle
import argparse
import logging
import warnings
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# BioPython
from Bio.PDB import PDBParser, PPBuilder, NeighborSearch, Selection
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# RDKit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings('ignore', category=PDBConstructionWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ESM-2 650M hard limit (1022 tokens = 1020 AA + <cls> + <eos>)
MAX_SEQ_LEN = 1020


# ─────────────────────────────────────────────────────────────
# 1. Parse index file → label DataFrame
# ─────────────────────────────────────────────────────────────

def parse_index(index_path: str) -> pd.DataFrame:
    """
    Parse INDEX_general_PL.2020R1.lst (PDBbind v2020).

    Format (space-separated, lines starting with # are comments):
      PDB_ID  resolution  year  affinity  //  reference  ligand_name

    Affinity column examples: Kd=49uM, Ki=0.068nM, Kd<10uM, IC50~10nM
    Units: pM, nM, uM, mM.
    """
    records = []
    aff_pat = re.compile(
        r'^(Kd|Ki|IC50)\s*([<>=~]*)\s*([\d.]+)\s*(pM|nM|uM|mM)?'
    )
    UNIT_TO_NM = {'pM': 0.001, 'nM': 1.0, 'uM': 1000.0, 'mM': 1e6}

    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            pdb_id      = parts[0].lower()
            resolution  = parts[1]
            year        = int(parts[2])
            affinity_str = parts[3]

            m = aff_pat.match(affinity_str)
            if not m:
                continue

            mtype_str   = m.group(1)
            operator    = m.group(2) or ''
            value       = float(m.group(3))
            unit        = m.group(4) or 'uM'

            is_censored = ('<' in operator or '>' in operator)
            value_nM    = value * UNIT_TO_NM.get(unit, 1000.0)
            pkd_raw     = -np.log10(value_nM * 1e-9)

            if mtype_str == 'Kd':
                mtype = 'Kd'
            elif mtype_str == 'Ki':
                mtype = 'Ki'
            else:
                mtype = 'IC50'

            base_weight    = 1.0 if mtype in ('Kd', 'Ki') else 0.4
            quality_weight = base_weight * (0.5 if is_censored else 1.0)

            try:
                res_float = float(resolution)
            except ValueError:
                res_float = np.nan

            records.append({
                'pdb_id':           pdb_id,
                'pkd_raw':          pkd_raw,
                'measurement_type': mtype,
                'quality_weight':   quality_weight,
                'is_censored':      is_censored,
                'resolution':       res_float,
                'year':             year,
            })

    df = pd.DataFrame(records)
    df = df[(df['pkd_raw'] >= 2) & (df['pkd_raw'] <= 12)].reset_index(drop=True)
    log.info(f"Index parsed: {len(df)} valid complexes "
             f"({df['is_censored'].sum()} censored)")
    return df


# ─────────────────────────────────────────────────────────────
# 2. Extract protein sequence — single chain selection
# ─────────────────────────────────────────────────────────────

def extract_chains(pdb_path: str) -> Optional[dict]:
    """
    Parse PDB and return a dict of {chain_id: sequence_str} for all
    polypeptide chains. Chains shorter than 20 AA are excluded.
    Returns None if parsing fails.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('prot', pdb_path)
    except Exception:
        return None

    ppb = PPBuilder()
    chains = {}
    for model in structure:
        for chain in model:
            peptides = ppb.build_peptides(chain)
            seq = ''.join(str(pp.get_sequence()) for pp in peptides)
            if len(seq) >= 20:
                chains[chain.get_id()] = seq
        break  # first model only

    return chains if chains else None


def select_chain(chains: dict,
                 pocket_chain_ids: Optional[set] = None) -> Tuple[str, str]:
    """
    Select a single chain to represent the protein.

    Priority:
      1. Chain that contains the most pocket residues (if pocket_chain_ids given)
      2. Longest chain (fallback)

    Returns (chain_id, sequence).
    """
    if pocket_chain_ids:
        # Count how many pocket residues each chain has
        overlap = {cid: 0 for cid in chains}
        for cid in pocket_chain_ids:
            if cid in chains:
                overlap[cid] += 1
        best = max(overlap, key=lambda c: (overlap[c], len(chains[c])))
        return best, chains[best]

    # Fallback: longest chain
    best = max(chains, key=lambda c: len(chains[c]))
    return best, chains[best]


# ─────────────────────────────────────────────────────────────
# 3. Extract binding site residue indices (per-chain, 0-based)
# ─────────────────────────────────────────────────────────────

def extract_binding_site_from_pocket(pocket_pdb: str,
                                     protein_pdb: str,
                                     selected_chain_id: str
                                     ) -> Tuple[Optional[list], Optional[set]]:
    """
    Use the pre-computed pocket PDB to identify binding site residue indices.

    Returns:
      (site_indices, pocket_chain_ids)
      site_indices    : list of 0-based indices within selected_chain_id
      pocket_chain_ids: set of chain IDs that appear in the pocket
    """
    parser = PDBParser(QUIET=True)

    # Parse pocket residues → (chain_id, res_seq_num)
    try:
        pocket_struct = parser.get_structure('pocket', pocket_pdb)
    except Exception:
        return None, None

    pocket_residues = {}   # chain_id → set of res_seq_nums
    pocket_chain_ids = set()
    for model in pocket_struct:
        for chain in model:
            cid = chain.get_id()
            for res in chain:
                if res.get_id()[0] == ' ':  # ATOM records only
                    pocket_chain_ids.add(cid)
                    pocket_residues.setdefault(cid, set()).add(res.get_id()[1])
        break

    if not pocket_residues:
        return None, None

    # Parse full protein → build per-chain (resnum → 0-based index) map
    try:
        prot_struct = parser.get_structure('prot', protein_pdb)
    except Exception:
        return None, pocket_chain_ids

    ppb = PPBuilder()
    chain_res_to_idx = {}   # chain_id → {res_seq_num: 0-based idx within chain}
    for model in prot_struct:
        for chain in model:
            cid = chain.get_id()
            peptides = ppb.build_peptides(chain)
            idx = 0
            chain_res_to_idx[cid] = {}
            for pp in peptides:
                for res in pp:
                    chain_res_to_idx[cid][res.get_id()[1]] = idx
                    idx += 1
        break

    # Map pocket residues in selected chain → 0-based indices
    site_indices = []
    if selected_chain_id in pocket_residues and selected_chain_id in chain_res_to_idx:
        for resnum in pocket_residues[selected_chain_id]:
            if resnum in chain_res_to_idx[selected_chain_id]:
                site_indices.append(chain_res_to_idx[selected_chain_id][resnum])

    return (sorted(set(site_indices)) if site_indices else None), pocket_chain_ids


def extract_binding_site_from_contact(protein_pdb: str,
                                      ligand_sdf: str,
                                      selected_chain_id: str,
                                      cutoff: float = 6.0
                                      ) -> Optional[list]:
    """
    Fallback: compute binding site by finding residues in selected_chain_id
    within `cutoff` Å of any ligand heavy atom.
    Returns list of 0-based indices within selected_chain_id.
    """
    parser = PDBParser(QUIET=True)

    # Load ligand atoms
    try:
        mol = Chem.MolFromMolFile(ligand_sdf, removeHs=True)
        if mol is None:
            return None
        conf = mol.GetConformer()
        lig_coords = np.array([list(conf.GetAtomPosition(i))
                               for i in range(mol.GetNumAtoms())])
    except Exception:
        return None

    # Load protein atoms — only selected chain
    try:
        prot_struct = parser.get_structure('prot', protein_pdb)
    except Exception:
        return None

    ppb = PPBuilder()
    res_coords = []   # list of (0-based idx within chain, Cα coord)
    for model in prot_struct:
        for chain in model:
            if chain.get_id() != selected_chain_id:
                continue
            peptides = ppb.build_peptides(chain)
            idx = 0
            for pp in peptides:
                for res in pp:
                    try:
                        ca = res['CA'].get_vector().get_array()
                        res_coords.append((idx, ca))
                    except KeyError:
                        atoms = list(res.get_atoms())
                        if atoms:
                            res_coords.append((idx, np.mean(
                                [a.get_vector().get_array() for a in atoms], axis=0)))
                    idx += 1
        break

    if not res_coords:
        return None

    site_indices = []
    for res_idx, ca in res_coords:
        dists = np.linalg.norm(lig_coords - ca, axis=1)
        if dists.min() <= cutoff:
            site_indices.append(res_idx)

    return sorted(set(site_indices)) if site_indices else None


# ─────────────────────────────────────────────────────────────
# 4. Compute structural features from selected chain only
# ─────────────────────────────────────────────────────────────

def compute_structural_features(protein_pdb: str,
                                 selected_chain_id: str,
                                 contact_cutoff: float = 8.0
                                 ) -> Optional[dict]:
    """
    Compute per-residue structural features from Cα coordinates
    of the selected chain only.

    contact_map[i,j]     = 1 if Cα(i)–Cα(j) ≤ contact_cutoff (excl. self)
    contact_number[i]    = number of residues j≠i within contact_cutoff
    protrusion_index[i]  = contact_number[i] / max(contact_number)

    Returns dict with keys: contact_map, contact_number, protrusion_index
    Returns None on parse failure or < 5 residues.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('prot', protein_pdb)
    except Exception:
        return None

    ppb = PPBuilder()
    ca_coords = []
    for model in structure:
        for chain in model:
            if chain.get_id() != selected_chain_id:
                continue
            peptides = ppb.build_peptides(chain)
            for pp in peptides:
                for res in pp:
                    try:
                        ca = res['CA'].get_vector().get_array()
                        ca_coords.append(ca)
                    except KeyError:
                        atoms = list(res.get_atoms())
                        if atoms:
                            ca_coords.append(np.mean(
                                [a.get_vector().get_array() for a in atoms], axis=0))
                        else:
                            ca_coords.append(np.zeros(3))
        break

    if len(ca_coords) < 5:
        return None

    coords = np.array(ca_coords, dtype=np.float32)   # (L, 3)
    L = len(coords)

    # Pairwise Cα distances
    diff = coords[:, None, :] - coords[None, :, :]    # (L, L, 3)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))   # (L, L)

    contact_map    = (dist_matrix <= contact_cutoff) & (dist_matrix > 0)
    contact_number = contact_map.sum(axis=1).astype(np.float32)
    max_cn         = contact_number.max()
    protrusion_index = (contact_number / max_cn
                        if max_cn > 0
                        else np.zeros(L, dtype=np.float32))

    return {
        'contact_map':      contact_map,        # bool (L, L)
        'contact_number':   contact_number,     # float32 (L,)
        'protrusion_index': protrusion_index,   # float32 (L,)
    }


# ─────────────────────────────────────────────────────────────
# 5. Extract ligand SMILES from SDF
# ─────────────────────────────────────────────────────────────

def extract_ligand_smiles(ligand_sdf: str) -> Optional[str]:
    """Read ligand SDF → canonical SMILES. Returns None if parsing fails."""
    try:
        mol = Chem.MolFromMolFile(ligand_sdf, removeHs=True, sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# 6. Per-complex worker (called in parallel)
# ─────────────────────────────────────────────────────────────

def process_complex(args) -> Optional[dict]:
    """
    Process one PDBbind complex. Returns a record dict or None on failure.

    Chain selection logic:
      1. Parse pocket PDB → identify which chains appear in the pocket
      2. select_chain() picks the chain with the most pocket residues
      3. If no pocket PDB, select longest chain
      4. Binding site mask indices are 0-based within the selected chain
      5. Structural features computed from selected chain only
    """
    pdb_id, struct_dir, row = args
    struct_dir = Path(struct_dir)

    # Find the PDB directory — supports both flat and nested layouts
    pdb_dir = struct_dir / pdb_id
    if not pdb_dir.is_dir():
        for subdir in sorted(struct_dir.iterdir()):
            if subdir.is_dir():
                candidate = subdir / pdb_id
                if candidate.is_dir():
                    pdb_dir = candidate
                    break

    protein_pdb = pdb_dir / f'{pdb_id}_protein.pdb'
    ligand_sdf  = pdb_dir / f'{pdb_id}_ligand.sdf'
    pocket_pdb  = pdb_dir / f'{pdb_id}_pocket.pdb'

    if not protein_pdb.exists():
        return None

    # --- Step A: parse all chains ---
    chains = extract_chains(str(protein_pdb))
    if not chains:
        return None

    # --- Step B: get pocket chain info (needed for chain selection) ---
    pocket_chain_ids = None
    site_indices     = None

    if pocket_pdb.exists():
        # Temporarily parse pocket to get chain IDs before chain selection
        parser = PDBParser(QUIET=True)
        try:
            pocket_struct = parser.get_structure('pocket', str(pocket_pdb))
            pocket_chain_ids = set()
            for model in pocket_struct:
                for chain in model:
                    for res in chain:
                        if res.get_id()[0] == ' ':
                            pocket_chain_ids.add(chain.get_id())
                break
        except Exception:
            pocket_chain_ids = None

    # --- Step C: select chain ---
    selected_chain_id, sequence = select_chain(chains, pocket_chain_ids)

    # Enforce ESM-2 length limit
    if len(sequence) > MAX_SEQ_LEN:
        sequence = sequence[:MAX_SEQ_LEN]

    if len(sequence) < 30:
        return None

    # --- Step D: extract binding site (per-chain indices) ---
    if pocket_pdb.exists():
        site_indices, _ = extract_binding_site_from_pocket(
            str(pocket_pdb), str(protein_pdb), selected_chain_id)
    elif ligand_sdf.exists():
        site_indices = extract_binding_site_from_contact(
            str(protein_pdb), str(ligand_sdf), selected_chain_id)

    # Filter out-of-range indices (can happen after sequence truncation)
    if site_indices:
        site_indices = [i for i in site_indices if i < len(sequence)]
        if not site_indices:
            site_indices = None

    # --- Step E: structural features (selected chain only) ---
    struct_feats = compute_structural_features(
        str(protein_pdb), selected_chain_id)

    # Truncate structural features to match sequence length
    if struct_feats and len(struct_feats['contact_number']) > len(sequence):
        L = len(sequence)
        struct_feats['contact_map']      = struct_feats['contact_map'][:L, :L]
        struct_feats['contact_number']   = struct_feats['contact_number'][:L]
        struct_feats['protrusion_index'] = struct_feats['protrusion_index'][:L]

    # --- Step F: ligand SMILES ---
    smiles = extract_ligand_smiles(str(ligand_sdf)) if ligand_sdf.exists() else None

    # --- Step G: hashes ---
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    inchikey = None
    if smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchikey = Chem.inchi.MolToInchiKey(mol)
        except Exception:
            pass

    return {
        # Identity
        'pdb_id':             pdb_id,
        'source_db':          'PDBbind',
        'protein_seq_hash':   seq_hash,
        'ligand_inchikey':    inchikey,
        'selected_chain_id':  selected_chain_id,
        # Labels
        'pkd_raw':            row['pkd_raw'],
        'pkd_aligned':        row['pkd_raw'],   # PDBbind is the reference
        'measurement_type':   row['measurement_type'],
        'quality_weight':     row['quality_weight'],
        'is_censored':        row['is_censored'],
        'resolution':         row['resolution'],
        # Protein data
        'sequence':           sequence,
        'binding_site_mask':  site_indices,     # list[int] 0-based in selected chain
        'has_binding_site':   site_indices is not None,
        'has_structure':      struct_feats is not None,
        # Structural features (None if unavailable)
        'contact_map':        struct_feats['contact_map']      if struct_feats else None,
        'contact_number':     struct_feats['contact_number']   if struct_feats else None,
        'protrusion_index':   struct_feats['protrusion_index'] if struct_feats else None,
        # Ligand
        'ligand_smiles':      smiles,
    }


# ─────────────────────────────────────────────────────────────
# 7. Main entry point
# ─────────────────────────────────────────────────────────────

def parse_pdbbind(index_path: str,
                  struct_dir: str,
                  out_path: str,
                  n_workers: int = 4,
                  max_complexes: Optional[int] = None) -> list:
    """
    Full PDBbind parsing pipeline.

    Parameters
    ----------
    index_path    : path to INDEX_general_PL_data.2020
    struct_dir    : path to general-set or refined-set directory
    out_path      : output pickle file path
    n_workers     : number of parallel workers
    max_complexes : limit for debugging (None = process all)

    Returns
    -------
    List of record dicts
    """
    df = parse_index(index_path)
    if max_complexes:
        df = df.head(max_complexes)

    tasks = [(row['pdb_id'], struct_dir, row)
             for _, row in df.iterrows()]

    records = []
    failed  = 0
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_complex, tasks),
                           total=len(tasks),
                           desc='Parsing PDBbind complexes'):
            if result is not None:
                records.append(result)
            else:
                failed += 1

    log.info(f"Parsed {len(records)} complexes, {failed} failed")
    log.info(f"  With binding site : {sum(r['has_binding_site'] for r in records)}")
    log.info(f"  With structure    : {sum(r['has_structure']    for r in records)}")
    log.info(f"  With ligand SMILES: {sum(r['ligand_smiles'] is not None for r in records)}")
    log.info(f"  Censored labels   : {sum(r['is_censored']    for r in records)}")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(records, f)
    log.info(f"Saved → {out_path}")
    return records


# ─────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse PDBbind v2020')
    parser.add_argument('--index',    required=True,
                        help='INDEX_general_PL_data.2020')
    parser.add_argument('--struct',   required=True,
                        help='Path to PDBbind structure directory (flat or nested by year)')
    parser.add_argument('--out',      required=True,
                        help='Output pickle path')
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--max',      type=int, default=None,
                        help='Max complexes (for debugging)')
    args = parser.parse_args()

    parse_pdbbind(args.index, args.struct, args.out,
                  n_workers=args.workers, max_complexes=args.max)
