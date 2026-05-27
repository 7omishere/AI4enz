"""
04_build_ligand_graphs.py
========================
Convert canonical SMILES strings to PyTorch Geometric Data objects.

Node features (dim=74 per atom):
  - Atom type one-hot (44 types: common elements + UNK)
  - Degree one-hot (0–10)
  - Formal charge one-hot (-3, -2, -1, 0, 1, 2, 3)
  - Hybridization one-hot (SP, SP2, SP3, SP3D, SP3D2, OTHER)
  - Aromaticity (bool, 1 dim)
  - H count one-hot (0–4)
  - Ring membership (bool, 1 dim)
  - Chirality one-hot (CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW, OTHER)

Edge features (dim=12 per bond, bidirectional):
  - Bond type one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC)
  - Conjugation (bool, 1 dim)
  - Ring membership (bool, 1 dim)
  - Stereo one-hot (STEREONONE, STEREOZ, STEREOE, STEREOANY)

Output:
  - One .pt file per unique ligand, keyed by InChIKey
  - Stored in {out_dir}/{inchikey}.pt

Usage:
  python 04_build_ligand_graphs.py \\
      --records  processed/pdbbind_records.pkl processed/bindingdb_aligned.pkl \\
      --out-dir  processed/ligands \\
      --workers  4
"""

import os
import pickle
import argparse
import logging
from pathlib import Path
from multiprocessing import Pool
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Feature vocabularies
# ─────────────────────────────────────────────────────────────

ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'UNK',
]  # 44 types

DEGREES       = list(range(11))          # 0–10
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.OTHER,
]
H_COUNTS      = list(range(5))           # 0–4
CHIRALITIES   = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOANY,
]

# Verify total feature dimensions
_ATOM_DIM = (len(ATOM_TYPES) +       # 44
             len(DEGREES) +           # 11
             len(FORMAL_CHARGES) +    # 7
             len(HYBRIDIZATIONS) +    # 6
             1 +                      # aromaticity
             len(H_COUNTS) +          # 5
             1 +                      # ring
             len(CHIRALITIES))        # 4
# Total: 44+11+7+6+1+5+1+4 = 79
# Note: PLAN says 74; we use 79 (more complete). Update PLAN comment accordingly.

_BOND_DIM = (len(BOND_TYPES) +       # 4
             1 +                      # conjugation
             1 +                      # ring
             len(BOND_STEREOS))       # 4
# Total: 4+1+1+4 = 10
# Note: PLAN says 12; we use 10. Consistent within the codebase.

ATOM_FEATURE_DIM = _ATOM_DIM   # 79
BOND_FEATURE_DIM = _BOND_DIM   # 10


# ─────────────────────────────────────────────────────────────
# Feature encoding helpers
# ─────────────────────────────────────────────────────────────

def one_hot(value, vocab: list) -> list:
    """One-hot encode value against vocab. Unknown values → last slot."""
    if value not in vocab:
        value = vocab[-1]
    return [int(value == v) for v in vocab]


def atom_features(atom: rdchem.Atom) -> torch.Tensor:
    """
    Encode a single RDKit atom into a feature vector of dim=ATOM_FEATURE_DIM.
    """
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES:
        symbol = 'UNK'

    feats = (
        one_hot(symbol,                    ATOM_TYPES)       +  # 44
        one_hot(atom.GetDegree(),          DEGREES)          +  # 11
        one_hot(atom.GetFormalCharge(),    FORMAL_CHARGES)   +  # 7
        one_hot(atom.GetHybridization(),   HYBRIDIZATIONS)   +  # 6
        [int(atom.GetIsAromatic())]                          +  # 1
        one_hot(atom.GetTotalNumHs(),      H_COUNTS)         +  # 5
        [int(atom.IsInRing())]                               +  # 1
        one_hot(atom.GetChiralTag(),       CHIRALITIES)         # 4
    )
    return torch.tensor(feats, dtype=torch.float32)


def bond_features(bond: rdchem.Bond) -> torch.Tensor:
    """
    Encode a single RDKit bond into a feature vector of dim=BOND_FEATURE_DIM.
    """
    feats = (
        one_hot(bond.GetBondType(),   BOND_TYPES)    +  # 4
        [int(bond.GetIsConjugated())]                +  # 1
        [int(bond.IsInRing())]                       +  # 1
        one_hot(bond.GetStereo(),     BOND_STEREOS)     # 4
    )
    return torch.tensor(feats, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# SMILES → PyG Data
# ─────────────────────────────────────────────────────────────

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert a canonical SMILES string to a PyG Data object.

    Returns None if the molecule cannot be parsed or has < 2 atoms.

    Graph conventions:
      - Bidirectional edges (both directions for each bond)
      - No self-loops
      - x:         atom features (N_atoms, ATOM_FEATURE_DIM)
      - edge_index: (2, 2*N_bonds)
      - edge_attr:  bond features (2*N_bonds, BOND_FEATURE_DIM)
      - smiles:     canonical SMILES string (stored for reference)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)   # explicit H → implicit (cleaner graph)
    except Exception:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return None

    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])  # (N, 79)

    # Edge index and features (bidirectional)
    edge_indices = []
    edge_attrs   = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Forward edge
        edge_indices.append([i, j])
        edge_attrs.append(bf)
        # Reverse edge
        edge_indices.append([j, i])
        edge_attrs.append(bf)

    if not edge_indices:
        # Molecule with no bonds (e.g. single atom after H removal)
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # (2, E)
    edge_attr  = torch.stack(edge_attrs)                                          # (E, 10)

    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        smiles     = smiles,
        num_nodes  = n_atoms,
    )


# ─────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────

def _worker(args) -> tuple:
    """Worker function for multiprocessing: (inchikey, smiles) → (inchikey, Data|None)."""
    inchikey, smiles = args
    graph = smiles_to_graph(smiles)
    return inchikey, graph


def build_ligand_graphs(records: list,
                         out_dir: str,
                         n_workers: int = 4,
                         overwrite: bool = False) -> dict:
    """
    Build and save PyG ligand graphs for all unique ligands in records.

    Parameters
    ----------
    records   : list of record dicts (from PDBbind or BindingDB parsers)
    out_dir   : directory to save .pt files
    n_workers : number of parallel workers
    overwrite : if False, skip ligands whose .pt file already exists

    Returns
    -------
    Dict mapping inchikey → success (True/False)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect unique (inchikey, smiles) pairs
    seen = {}
    for r in records:
        ik = r.get('ligand_inchikey')
        sm = r.get('ligand_smiles')
        if ik and sm and ik not in seen:
            seen[ik] = sm

    log.info(f"Found {len(seen):,} unique ligands")

    # Filter already-processed
    tasks = []
    for ik, sm in seen.items():
        out_path = Path(out_dir) / f'{ik}.pt'
        if not overwrite and out_path.exists():
            continue
        tasks.append((ik, sm))

    log.info(f"Processing {len(tasks):,} ligands "
             f"({len(seen) - len(tasks):,} already cached)")

    results = {}
    failed  = 0

    with Pool(n_workers) as pool:
        for inchikey, graph in tqdm(
                pool.imap_unordered(_worker, tasks),
                total=len(tasks),
                desc='Building ligand graphs'):
            if graph is not None:
                out_path = Path(out_dir) / f'{inchikey}.pt'
                torch.save(graph, str(out_path))
                results[inchikey] = True
            else:
                results[inchikey] = False
                failed += 1

    log.info(f"Saved {len(results) - failed:,} graphs, {failed:,} failed")
    log.info(f"Atom feature dim: {ATOM_FEATURE_DIM}, Bond feature dim: {BOND_FEATURE_DIM}")
    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build ligand molecular graphs')
    parser.add_argument('--records',   nargs='+', required=True,
                        help='One or more record pickle files')
    parser.add_argument('--out-dir',   required=True,
                        help='Output directory for .pt files')
    parser.add_argument('--workers',   type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    all_records = []
    for path in args.records:
        with open(path, 'rb') as f:
            all_records.extend(pickle.load(f))
    log.info(f"Loaded {len(all_records):,} total records from {len(args.records)} files")

    build_ligand_graphs(all_records, args.out_dir,
                        n_workers=args.workers,
                        overwrite=args.overwrite)
