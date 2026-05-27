"""
酶挖掘推理脚本：给定目标底物 SMILES → 从蛋白库中排序推荐最适配的酶。

用法：
  source /home/domi/BINN/.venv/bin/activate
  cd /home/domi/BINN/AI4enz/dataset_building

  # 单底物查询
  python inference_enzyme_mining.py --smiles "O=C1CCCCC1" --top-k 10

  # 从文件批量查询
  python inference_enzyme_mining.py --smiles-file substrates.txt --top-k 5

  # 指定辅因子过滤
  python inference_enzyme_mining.py --smiles "c1ccccc1O" --cofactor HEME --top-k 10

  # 输出为 JSON
  python inference_enzyme_mining.py --smiles "O=C1CCCCC1" --top-k 5 --output results.json

依赖：需要训练好的模型检查点（checkpoints/best.ckpt）。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data as PyGData

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "datepre"))
from ranking_model import MarcusPINN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Paths
PROJECT_DIR = SCRIPT_DIR
PROCESSED_DIR = PROJECT_DIR / "processed"
PROTEIN_H5 = PROCESSED_DIR / "proteins.h5"
META_PATH = PROCESSED_DIR / "oxidoreductase" / "unified_metadata.parquet"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# Atom/bond feature vocabularies — must match pipeline/04_build_ligand_graphs.py exactly
from rdkit import Chem as _Chem

ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'UNK',
]
DEGREES = list(range(11))
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
HYBRIDIZATIONS = [
    _Chem.rdchem.HybridizationType.SP, _Chem.rdchem.HybridizationType.SP2,
    _Chem.rdchem.HybridizationType.SP3, _Chem.rdchem.HybridizationType.SP3D,
    _Chem.rdchem.HybridizationType.SP3D2, _Chem.rdchem.HybridizationType.OTHER,
]
H_COUNTS = list(range(5))
CHIRALITIES = [
    _Chem.rdchem.ChiralType.CHI_UNSPECIFIED, _Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    _Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, _Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    _Chem.rdchem.BondType.SINGLE, _Chem.rdchem.BondType.DOUBLE,
    _Chem.rdchem.BondType.TRIPLE, _Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    _Chem.rdchem.BondStereo.STEREONONE, _Chem.rdchem.BondStereo.STEREOZ,
    _Chem.rdchem.BondStereo.STEREOE, _Chem.rdchem.BondStereo.STEREOANY,
]
ATOM_DIM = 79
EDGE_DIM = 10


class ProteinLibrary:
    """预加载所有蛋白特征到内存的离线库。

    对 541 个蛋白预计算 ESM-2 嵌入 + 口袋特征 + 辅因子类型，
    推理时直接遍历（541 蛋白 < 1 秒）。
    """

    def __init__(self, proteins_h5_path: str, metadata_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        log.info("Loading metadata...")
        meta = pd.read_parquet(metadata_path)
        self.meta = meta

        # 每个 seq_hash → uniprot_id, protein_name, cofactors
        self.protein_info = {}
        for phash, grp in meta.groupby("protein_seq_hash"):
            row = grp.iloc[0]
            self.protein_info[phash] = {
                "uniprot_id": row.get("uniprot_id", "unknown"),
                "protein_name": row.get("protein_name", row.get("entry_name", "unknown")),
                "cofactor": str(row.get("cofactors", "") or ""),
            }

        log.info("Loading protein features from HDF5...")
        self.h5 = h5py.File(proteins_h5_path, "r")

        # 预加载所有氧化还原酶蛋白特征
        self.protein_features = {}  # seq_hash → feature dict (on CPU)
        ox_hashes = set(meta["protein_seq_hash"].unique())
        n_loaded = 0
        for phash in sorted(ox_hashes):
            if phash not in self.h5:
                continue
            group = self.h5[phash]
            feats = {}

            # ESM-2 embedding
            if "esm2_embed" in group:
                feats["esm2_embed"] = torch.from_numpy(group["esm2_embed"][:]).float()

            # Structure features
            if "contact_number" in group:
                feats["cn_mean"] = float(group["contact_number"][:].mean())
                feats["pi_mean"] = float(group["protrusion_index"][:].mean())
                feats["has_structure"] = True
            else:
                feats["cn_mean"] = 0.0
                feats["pi_mean"] = 0.0
                feats["has_structure"] = False

            # Pocket features
            if "pocket_ca_distances" in group:
                feats["pocket_cn"] = torch.from_numpy(group["pocket_contact_number"][:]).float()
                feats["pocket_pi"] = torch.from_numpy(group["pocket_protrusion_index"][:]).float()
                feats["pocket_dist"] = torch.from_numpy(group["pocket_ca_distances"][:]).float()
                feats["pocket_mask"] = torch.ones(len(feats["pocket_cn"]), dtype=torch.bool)
            else:
                feats["pocket_cn"] = None
                feats["pocket_pi"] = None
                feats["pocket_dist"] = None
                feats["pocket_mask"] = None

            # Domain masks
            if "domain_masks" in group:
                feats["domain_masks"] = torch.from_numpy(group["domain_masks"][:]).float()
            else:
                feats["domain_masks"] = None

            self.protein_features[phash] = feats
            n_loaded += 1

        log.info(f"Loaded {n_loaded} proteins into library")
        self.h5.close()

    def get_info(self, seq_hash: str) -> dict:
        return self.protein_info.get(seq_hash, {})

    def __len__(self):
        return len(self.protein_features)


def smiles_to_graph(smiles: str) -> PyGData:
    """将 SMILES 转换为 PyG 分子图（79 维原子特征 + 10 维键特征）。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atom_feats = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol() if atom.GetSymbol() in ATOM_TYPES else "UNK"
        feat = (
            _one_hot(symbol, ATOM_TYPES) +
            _one_hot(atom.GetDegree(), DEGREES) +
            _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES) +
            _one_hot(atom.GetHybridization(), HYBRIDIZATIONS) +
            [float(atom.GetIsAromatic())] +
            _one_hot(atom.GetTotalNumHs(), H_COUNTS) +
            [float(atom.IsInRing())] +
            _one_hot(atom.GetChiralTag(), CHIRALITIES)
        )
        atom_feats.append(feat)
    x = torch.tensor(atom_feats, dtype=torch.float32)

    edge_index = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = (
            _one_hot(bond.GetBondType(), BOND_TYPES) +
            [float(bond.GetIsConjugated())] +
            [float(bond.IsInRing())] +
            _one_hot(bond.GetStereo(), BOND_STEREOS)
        )
        edge_index.extend([[i, j], [j, i]])
        edge_attrs.extend([bf, bf])

    if len(edge_index) == 0:
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, EDGE_DIM)
    else:
        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        ea = torch.tensor(edge_attrs, dtype=torch.float32)

    return PyGData(x=x, edge_index=ei, edge_attr=ea)


def _one_hot(value, allowed) -> list[float]:
    return [1.0 if value == a else 0.0 for a in allowed]


class EnzymeMiningInference:
    """酶挖掘推理器：给定底物→从蛋白库排序推荐酶。"""

    def __init__(self, checkpoint_path: str, library: ProteinLibrary, device: str = "cpu"):
        self.device = torch.device(device)
        self.library = library

        # 加载模型
        log.info(f"Loading model from {checkpoint_path}...")
        self.model = MarcusPINN(hidden_dim=256)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        log.info(f"Model loaded (step {checkpoint.get('global_step', 'N/A')})")

    @torch.no_grad()
    def rank_enzymes(
        self,
        smiles: str,
        top_k: int = 10,
        cofactor_filter: Optional[str] = None,
    ) -> list[dict]:
        """给定底物 SMILES，返回 Top-K 推荐酶，按预测 pKd 降序排列。"""
        from torch_geometric.data import Batch as PyGBatch

        # Build ligand graph
        ligand_graph = smiles_to_graph(smiles)

        # Collect all proteins (apply cofactor filter)
        items = []
        for seq_hash, feats in self.library.protein_features.items():
            info = self.library.get_info(seq_hash)
            if cofactor_filter and cofactor_filter.upper() not in info.get("cofactor", "").upper():
                continue
            items.append((seq_hash, feats, info))

        # Batch inference: process all proteins in one forward pass
        n = len(items)
        results = []
        batch_size = 128  # avoid OOM

        for start in range(0, n, batch_size):
            batch_items = items[start:start + batch_size]
            B = len(batch_items)

            # Batch protein features
            esm2_list = []
            struct_feat_list = []
            has_structure_list = []
            cofactor_list = []
            pocket_cn_batch = []
            pocket_pi_batch = []
            pocket_dist_batch = []
            pocket_mask_batch = []
            max_K = 0
            has_any_pocket = False

            for _, feats, info in batch_items:
                esm2_list.append(feats["esm2_embed"])
                struct_feat_list.append(torch.tensor(
                    [feats["cn_mean"], feats["pi_mean"], 1.0 if feats["has_structure"] else 0.0],
                    dtype=torch.float32,
                ))
                has_structure_list.append(feats["has_structure"])
                cofactor_list.append(info.get("cofactor", ""))

                if feats["pocket_cn"] is not None:
                    has_any_pocket = True
                    max_K = max(max_K, feats["pocket_cn"].size(0))

            if has_any_pocket:
                for _, feats, _ in batch_items:
                    if feats["pocket_cn"] is not None:
                        K = feats["pocket_cn"].size(0)
                        pc = torch.zeros(max_K); pc[:K] = feats["pocket_cn"]
                        pp = torch.zeros(max_K); pp[:K] = feats["pocket_pi"]
                        pd = torch.zeros(max_K, max_K); pd[:K, :K] = feats["pocket_dist"]
                        pm = torch.zeros(max_K, dtype=torch.bool); pm[:K] = True
                    else:
                        pc = torch.zeros(max_K)
                        pp = torch.zeros(max_K)
                        pd = torch.zeros(max_K, max_K)
                        pm = torch.zeros(max_K, dtype=torch.bool)
                    pocket_cn_batch.append(pc)
                    pocket_pi_batch.append(pp)
                    pocket_dist_batch.append(pd)
                    pocket_mask_batch.append(pm)

            # Stack tensors
            esm2_batch = torch.stack(esm2_list).to(self.device)
            struct_feat_batch = torch.stack(struct_feat_list).to(self.device)
            has_structure_batch = torch.tensor(has_structure_list, dtype=torch.bool, device=self.device)

            # Replicate ligand graph for batch
            ligand_batch = PyGBatch.from_data_list([ligand_graph] * B)

            # Pocket features
            pc_b, pp_b, pd_b, pm_b = None, None, None, None
            if has_any_pocket:
                pc_b = torch.stack(pocket_cn_batch).to(self.device)
                pp_b = torch.stack(pocket_pi_batch).to(self.device)
                pd_b = torch.stack(pocket_dist_batch).to(self.device)
                pm_b = torch.stack(pocket_mask_batch).to(self.device)

            outputs = self.model(
                ligand_batch,
                esm2_batch,
                cofactor_list,
                struct_feat=struct_feat_batch,
                has_structure=has_structure_batch,
                pocket_cn=pc_b,
                pocket_pi=pp_b,
                pocket_dist=pd_b,
                pocket_mask=pm_b,
            )

            pkd_values = outputs["pkd"].cpu().tolist()
            for i, (seq_hash, _, info) in enumerate(batch_items):
                results.append({
                    "seq_hash": seq_hash,
                    "uniprot_id": info.get("uniprot_id", "unknown"),
                    "protein_name": info.get("protein_name", "unknown"),
                    "cofactor": info.get("cofactor", ""),
                    "pred_pkd": round(pkd_values[i], 2),
                })

        # Sort by predicted pKd descending
        results.sort(key=lambda x: x["pred_pkd"], reverse=True)
        for i, r in enumerate(results[:top_k]):
            r["rank"] = i + 1

        return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Enzyme mining inference")
    parser.add_argument("--smiles", default=None, help="Target substrate SMILES")
    parser.add_argument("--smiles-file", default=None, help="File with one SMILES per line")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top recommendations")
    parser.add_argument("--cofactor", default=None, help="Filter by cofactor type (e.g. HEME, FAD)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "best.ckpt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.smiles and not args.smiles_file:
        parser.error("Must provide --smiles or --smiles-file")

    device = args.device

    # Build protein library
    library = ProteinLibrary(
        proteins_h5_path=str(PROTEIN_H5),
        metadata_path=str(META_PATH),
        device=device,
    )

    # Create inference engine
    engine = EnzymeMiningInference(
        checkpoint_path=args.checkpoint,
        library=library,
        device=device,
    )

    # Query substrates
    queries = []
    if args.smiles:
        queries.append(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file) as f:
            queries.extend(line.strip() for line in f if line.strip())

    all_results = []
    for smiles in queries:
        print(f"\n{'='*60}")
        print(f"  Query: {smiles}")
        print(f"{'='*60}")

        try:
            results = engine.rank_enzymes(
                smiles=smiles,
                top_k=args.top_k,
                cofactor_filter=args.cofactor,
            )
        except ValueError as e:
            log.error(f"  Error: {e}")
            continue

        print(f"\n  {'Rank':<6} {'UniProt':<12} {'Pred pKd':<10} {'Seq Hash':<18} Cofactor")
        print(f"  {'-'*70}")
        for r in results:
            print(f"  {r['rank']:<6} {r['uniprot_id']:<12} {r['pred_pkd']:<10.2f} {r['seq_hash'][:16]:<18} {r['cofactor']}")

        all_results.append({"smiles": smiles, "results": results})

    # Save output
    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        log.info(f"Results saved to {args.output}")

    log.info("Done.")


if __name__ == "__main__":
    main()
