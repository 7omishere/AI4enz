#!/usr/bin/env python3
"""
Trenzition 三头推理脚本：给定底物 + 酶序列 → 预测 Kd/Ki + kcat + Km

三头输出：
  - pKd (Kd 分支): -log10(Kd)，底物结合亲和力，越高越强
  - pKi (Ki 分支): -log10(Ki)，抑制剂结合近似常数
  - log₁₀(kcat):   催化速率常数，越高催化越快
  - ΔG‡:           活化吉布斯自由能 (kJ/mol)，越低越容易催化
  - log₁₀(Km):     米氏常数，越高结合越弱

用法：
  # 单样本预测
  python predict.py --smiles "CCO" --sequence "MKTVW..."

  # 批量预测 (CSV: smiles, sequence, cofactor)
  python predict.py --csv candidates.csv --output results.csv

  # 使用预计算的 ESM-2 嵌入 (跳过 ESM-2 加载，大幅加速)
  python predict.py --csv candidates_with_esm2.h5 --esm2-h5 embeddings.h5
"""

import argparse, sys, os, logging
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("predict")

# ── 路径 ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ranking_model import Trenzition

BEST_CKPT = SCRIPT_DIR / "checkpoints" / "best.ckpt"

# ── 归一化参数 (与 ThreeHeadLoss / NORM_PARAMS 保持一致) ──
PKD_MIN, PKD_MAX = 0.0, 12.0
KCAT_MIN, KCAT_MAX = -7.0, 8.0
KM_MIN, KM_MAX = -13.0, 3.0

# ── SMILES → 分子图 (与 dataset_building/pipeline 一致) ──

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

ATOM_TYPES = [
    'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B',
    'V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu',
    'Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','UNK',
]
DEGREES = list(range(11))
FORMAL_CHARGES = [-3,-2,-1,0,1,2,3]
HYBRIDIZATIONS = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
                  rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
                  rdchem.HybridizationType.SP3D2, rdchem.HybridizationType.OTHER]
H_COUNTS = list(range(5))
CHIRALITIES = [rdchem.ChiralType.CHI_UNSPECIFIED, rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
               rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, rdchem.ChiralType.CHI_OTHER]
BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
              rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
BOND_STEREOS = [rdchem.BondStereo.STEREONONE, rdchem.BondStereo.STEREOZ,
                rdchem.BondStereo.STEREOE, rdchem.BondStereo.STEREOANY]


def one_hot(value, vocab):
    if value not in vocab:
        value = vocab[-1]
    return [int(value == v) for v in vocab]


def atom_features(atom):
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES:
        symbol = 'UNK'
    return torch.tensor(
        one_hot(symbol,                    ATOM_TYPES)       +
        one_hot(atom.GetDegree(),          DEGREES)          +
        one_hot(atom.GetFormalCharge(),    FORMAL_CHARGES)   +
        one_hot(atom.GetHybridization(),   HYBRIDIZATIONS)   +
        [int(atom.GetIsAromatic())]                          +
        one_hot(atom.GetTotalNumHs(),      H_COUNTS)         +
        [int(atom.IsInRing())]                               +
        one_hot(atom.GetChiralTag(),       CHIRALITIES),
        dtype=torch.float32,
    )


def bond_features(bond):
    return torch.tensor(
        one_hot(bond.GetBondType(),   BOND_TYPES)    +
        [int(bond.GetIsConjugated())]                +
        [int(bond.IsInRing())]                       +
        one_hot(bond.GetStereo(),     BOND_STEREOS),
        dtype=torch.float32,
    )


def smiles_to_graph(smiles: str):
    """SMILES → PyG Data。与训练时预处理完全一致。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() < 2:
        return None

    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
    edge_idx, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_idx.extend([[i,j], [j,i]])
        edge_attr.extend([bf, bf])

    if not edge_idx:
        return None

    return Data(
        x=torch.stack([atom_features(a) for a in mol.GetAtoms()]),
        edge_index=torch.tensor(edge_idx, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_attr),
        smiles=smiles,
    )


# ── 蛋白序列 → ESM-2 嵌入 ──

def compute_esm2_embedding(sequence: str, model, tokenizer, device="cpu"):
    """单条序列 → ESM-2 mean-pooled embedding (1280-dim)"""
    import esm
    tokens = tokenizer(sequence, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        result = model(**tokens, output_hidden_states=True)
    # 取最后一层，mean pool (排除特殊 token)
    hidden = result["hidden_states"][-1][0, 1:-1, :]  # (L, 1280)
    return hidden.mean(dim=0).cpu()


def sequence_to_aa_props(seq: str) -> torch.Tensor:
    """轻量级 AA 物化性质编码 (6-dim)，不需要 ESM-2"""
    AA_PROPERTIES = {
        "A":[1.8,31.0,0.0,0.0,1.0,1.0], "R":[-4.5,124.0,1.0,1.0,1.0,6.13],
        "N":[-3.5,56.0,0.0,1.0,1.0,2.95], "D":[-3.5,54.0,-1.0,1.0,1.0,2.78],
        "C":[2.5,55.0,0.0,0.0,0.0,1.0], "Q":[-3.5,85.0,0.0,1.0,1.0,3.0],
        "E":[-3.5,83.0,-1.0,1.0,1.0,3.0], "G":[-0.4,3.0,0.0,0.0,2.0,0.0],
        "H":[-3.2,96.0,0.1,1.0,1.0,2.98], "I":[4.5,111.0,0.0,0.0,1.0,4.0],
        "L":[3.8,111.0,0.0,0.0,1.0,4.0], "K":[-3.9,119.0,1.0,1.0,1.0,5.0],
        "M":[1.9,105.0,0.0,0.0,1.0,3.8], "F":[2.8,132.0,0.0,0.0,0.0,5.89],
        "P":[-1.6,32.0,0.0,0.0,2.0,2.5], "S":[-0.8,32.0,0.0,1.0,1.0,1.5],
        "T":[-0.7,61.0,0.0,1.0,1.0,2.6], "W":[-0.9,170.0,0.0,1.0,0.0,8.08],
        "Y":[-1.3,136.0,0.0,1.0,0.0,6.47], "V":[4.2,84.0,0.0,0.0,1.0,3.0],
    }
    feats = [AA_PROPERTIES.get(aa, [0.0]*6) for aa in seq[:1020].upper()]
    return torch.tensor(feats, dtype=torch.float32).mean(dim=0) if feats else torch.zeros(6)


# ── 主推理 ──

def predict_single(model, smiles, seq_embed, cofactor_str="", device="cpu",
                   measurement_type=0, temperature_K=298.15):
    """单样本推理 (三头模型, EyringKcatHead 需要温度)"""
    g = smiles_to_graph(smiles)
    if g is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")

    model.eval()
    with torch.no_grad():
        out = model(
            g.to(device),
            seq_embed.unsqueeze(0).to(device),
            [cofactor_str],
            measurement_types=torch.tensor([measurement_type], device=device),
            temperature_K=torch.tensor([temperature_K], device=device),
        )

    # 反归一化
    pkd_kd = out["kd_pred"].item() * (PKD_MAX - PKD_MIN) + PKD_MIN
    pkd_ki = out["ki_pred"].item() * (PKD_MAX - PKD_MIN) + PKD_MIN
    kcat = out["kcat_pred"].item() * (KCAT_MAX - KCAT_MIN) + KCAT_MIN
    dG = out["dG_eyring"].item()
    km = out["log_km_pred"].item() * (KM_MAX - KM_MIN) + KM_MIN

    return {
        "pKd_Kd": pkd_kd,
        "pKd_Ki": pkd_ki,
        "log10_kcat": kcat,
        "dG_eyring_kJmol": dG,
        "log10_Km": km,
    }


def main():
    parser = argparse.ArgumentParser(description="Trenzition 三头酶-底物预测")
    parser.add_argument("--smiles", help="底物 SMILES")
    parser.add_argument("--sequence", help="蛋白氨基酸序列 (单字母)")
    parser.add_argument("--cofactor", default="", help="辅因子，如 'NAD|FAD'")
    parser.add_argument("--csv", help="批量预测 CSV (columns: smiles, sequence, cofactor)")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-esm2", action="store_true",
                        help="使用 ESM-2 编码 (需安装 fair-esm，约 2.5 GB 显存)")
    parser.add_argument("--esm2-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--measurement-type", type=int, default=0,
                        help="测量类型: 0=Kd (默认), 1=Ki, 2=IC50, 3=none")
    parser.add_argument("--temperature", type=float, default=298.15,
                        help="反应温度 (K), 默认 298.15")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载模型 (三头)
    log.info(f"Loading Trenzition (three-head) from {BEST_CKPT}...")
    model = Trenzition(hidden_dim=256, gnn_layers=3,
                       three_head=True, kcat_ode_steps=10).to(device)
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    log.info(f"Loaded (best val loss: {ckpt.get('best_val_loss', 'N/A'):.6f})")

    # ESM-2 (可选)
    esm2_model = None
    tokenizer = None
    use_esm2 = args.use_esm2
    if use_esm2:
        try:
            import esm
            log.info(f"Loading ESM-2 ({args.esm2_model})...")
            esm2_model, alphabet = esm.pretrained.load_model_and_alphabet(args.esm2_model)
            esm2_model = esm2_model.to(device).eval()
            tokenizer = alphabet.get_batch_converter()
            log.info("ESM-2 loaded")
        except ImportError:
            log.warning("fair-esm not installed, falling back to AA properties")
            use_esm2 = False

    # ── 单样本 ──
    if args.smiles and args.sequence:
        if use_esm2 and esm2_model:
            embed = compute_esm2_embedding(args.sequence, esm2_model, tokenizer, device)
        else:
            embed = sequence_to_aa_props(args.sequence)
            if not use_esm2:
                log.info("使用 AA 物化性质编码 (6-dim)，如需更高精度请加 --use-esm2")

        result = predict_single(model, args.smiles, embed, args.cofactor, device,
                                measurement_type=args.measurement_type,
                                temperature_K=args.temperature)
        print(f"\n底物: {args.smiles}")
        print(f"蛋白: {args.sequence[:50]}{'...' if len(args.sequence) > 50 else ''}")
        print(f"辅因子: {args.cofactor or '(无)'}")
        print(f"温度: {args.temperature:.1f}K")
        print(f"测量类型: {['Kd','Ki','IC50','none'][args.measurement_type]}")
        print(f"{'='*55}")
        print(f"pKd (Kd分支):     {result['pKd_Kd']:.2f}  [-log10(Kd), 越高越强]")
        print(f"pKd (Ki分支):     {result['pKd_Ki']:.2f}  [-log10(Ki), 抑制常数]")
        print(f"kcat (催化速率):   {result['log10_kcat']:.2f}  [log10(s⁻¹), 越高越快]")
        print(f"ΔG‡ (活化能):     {result['dG_eyring_kJmol']:.1f}  kJ/mol")
        print(f"log₁₀(Km):        {result['log10_Km']:.2f}  [log10(M), 越高结合越弱]")
        return

    # ── 批量 ──
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        required = {"smiles", "sequence"}
        missing = required - set(df.columns)
        if missing:
            log.error(f"CSV 缺少列: {missing}")
            sys.exit(1)
        if "cofactor" not in df.columns:
            df["cofactor"] = ""

        results = []
        for i, row in df.iterrows():
            try:
                if use_esm2 and esm2_model:
                    embed = compute_esm2_embedding(row["sequence"], esm2_model, tokenizer, device)
                else:
                    embed = sequence_to_aa_props(row["sequence"])
                r = predict_single(model, row["smiles"], embed, str(row.get("cofactor", "")), device)
                results.append({**r, "smiles": row["smiles"]})
                if (i+1) % 100 == 0:
                    log.info(f"  {i+1}/{len(df)} done")
            except Exception as e:
                log.warning(f"  [{i}] 跳过: {e}")
                results.append({"pKd_Kd": np.nan, "pKd_Ki": np.nan,
                                "log10_kcat": np.nan, "dG_eyring_kJmol": np.nan,
                                "log10_Km": np.nan, "smiles": row["smiles"]})

        out = pd.DataFrame(results)
        out.to_csv(args.output, index=False)
        log.info(f"结果保存至 {args.output} ({len(out)} 条)")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
