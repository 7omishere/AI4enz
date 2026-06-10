#!/usr/bin/env python3
"""
补齐 metadata 中缺失的配体 GNN 图 (.pt 文件)。

用法：
  source /home/domi/BINN/.venv/bin/activate
  python scripts/fill_missing_ligands.py
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
RELEASE_DIR = BASE_DIR / "release"
LIGAND_DIR = PROCESSED_DIR / "ligands"

# ─────────────────────────────────────────────────────────────
# Feature vocabularies (from 04_build_ligand_graphs.py)
# ─────────────────────────────────────────────────────────────

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "UNK",
]

DEGREES = list(range(11))
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.OTHER,
]
H_COUNTS = list(range(5))
CHIRALITIES = [
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

def one_hot(value, vocab: list) -> list:
    if value not in vocab:
        value = vocab[-1]
    return [int(value == v) for v in vocab]


def atom_features(atom: rdchem.Atom) -> torch.Tensor:
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES:
        symbol = "UNK"
    feats = (
        one_hot(symbol, ATOM_TYPES)
        + one_hot(atom.GetDegree(), DEGREES)
        + one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
        + one_hot(atom.GetTotalNumHs(), H_COUNTS)
        + [int(atom.IsInRing())]
        + one_hot(atom.GetChiralTag(), CHIRALITIES)
    )
    return torch.tensor(feats, dtype=torch.float32)


def bond_features(bond: rdchem.Bond) -> torch.Tensor:
    feats = (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
        + one_hot(bond.GetStereo(), BOND_STEREOS)
    )
    return torch.tensor(feats, dtype=torch.float32)


def smiles_to_graph(smiles: str) -> Optional[Data]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices.append([i, j])
        edge_attrs.append(bf)
        edge_indices.append([j, i])
        edge_attrs.append(bf)

    if not edge_indices:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        num_nodes=n_atoms,
    )


def main():
    os.makedirs(LIGAND_DIR, exist_ok=True)

    # 1. 加载 metadata → 获取所有需要的 ligand_inchikey
    meta = pd.read_parquet(PROCESSED_DIR / "metadata.parquet")
    all_ligands = set(meta["ligand_inchikey"].dropna().unique())
    print(f"Metadata unique ligands: {len(all_ligands)}")

    # 2. 检查磁盘上已有的 .pt 文件
    existing = set(f.stem for f in LIGAND_DIR.iterdir() if f.suffix == ".pt")
    print(f"Existing .pt files: {len(existing)}")

    missing = all_ligands - existing
    print(f"Missing: {len(missing)}")

    if not missing:
        print("All ligands already encoded!")
        return

    # 3. 从 V5 获取 SMILES
    v5 = pd.read_parquet(RELEASE_DIR / "trenzition_full_v5.parquet")
    inchikey_to_smiles = dict(zip(v5["ligand_inchikey"], v5["canonical_smiles"]))
    print(f"V5 inchikey→SMILES mappings: {len(inchikey_to_smiles)}")

    # 4. 编码缺失配体
    failed = []
    done = 0
    for ik in tqdm(sorted(missing), desc="Encoding ligands"):
        sm = inchikey_to_smiles.get(ik)
        if not sm:
            failed.append((ik, "no SMILES in V5"))
            continue

        graph = smiles_to_graph(sm)
        if graph is None:
            failed.append((ik, "graph build failed"))
            continue

        torch.save(graph, str(LIGAND_DIR / f"{ik}.pt"))
        done += 1

    print(f"\nDone: {done} encoded, {len(failed)} failed")
    if failed:
        print("Failed ligands:")
        for ik, reason in failed[:10]:
            print(f"  {ik}: {reason}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
