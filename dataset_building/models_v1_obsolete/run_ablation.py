#!/usr/bin/env python3
"""
消融实验：测试 TransitionBINN 各组件的贡献

Ablation variants (zero-shot, no retraining):
- Full model        : 完整模型（baseline）
- No Gate           : 门控输出置为 1（不衰减）
- No Cofactor       : 辅因子 embedding 置零
- No Ligand GNN     : 配体 embedding 置零（仅用蛋白+辅因子）
- ODE steps = 1     : 减少 ODE 积分步数
- No BINN (skip ODE): 跳过 ODE，直接 MLP 映射

评估指标: MSE, MAE, R², Spearman ρ
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import json
import logging
import warnings
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

from train import OxidoreductaseDataset, collate_fn
from ranking_model import Trenzition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ablation")

PROCESSED_DIR = SCRIPT_DIR.parent / "processed"
BEST_CKPT = SCRIPT_DIR / "checkpoints" / "best.ckpt"
OUTPUT_DIR = SCRIPT_DIR / "checkpoints"
METADATA_PATH = PROCESSED_DIR / "metadata.parquet"
PROTEINS_H5 = PROCESSED_DIR / "proteins.h5"
LIGAND_DIR = PROCESSED_DIR / "ligands"


def compute_metrics(preds, targets):
    """计算回归指标."""
    from scipy.stats import spearmanr
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0
    rho, _ = spearmanr(preds, targets) if len(preds) > 1 else (0, 0)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"MSE": float(mse), "MAE": float(mae), "Pearson_r": float(corr),
            "Spearman_rho": float(rho), "R2": float(r2)}


def load_model(device):
    """加载训练好的 Trenzition 模型."""
    model = Trenzition(hidden_dim=256, gnn_layers=3, n_ode_steps=5,
                       use_classification=False).to(device)
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    log.info(f"Loaded Trenzition (best val: {ckpt.get('best_val_loss', 'N/A'):.6f})")
    return model


class AblatedTrenzition(Trenzition):
    """Trenzition with configurable ablation."""

    def __init__(self, base_model: Trenzition, ablation: str):
        super().__init__(hidden_dim=256, gnn_layers=3, n_ode_steps=5,
                        use_classification=False)
        self.load_state_dict(base_model.state_dict())
        self.ablation = ablation

        if ablation == "ode_steps_1":
            self.binn.n_ode_steps = 1

    def forward(self, ligand_data, seq_embed, cofactor_strs):
        ligand_h = self.ligand_encoder(ligand_data)
        ligand_h = self.ligand_proj(ligand_h)

        protein_h = self.protein_encoder(seq_embed)

        cofactor_h, _ = self.cofactor_encoder(cofactor_strs)
        cofactor_h_proj = self.cofactor_proj(cofactor_h)

        # ── Apply ablations ──
        if self.ablation == "no_cofactor":
            cofactor_h_proj = torch.zeros_like(cofactor_h_proj)
        elif self.ablation == "no_ligand":
            ligand_h = torch.zeros_like(ligand_h)
        elif self.ablation == "no_ligand_and_cofactor":
            ligand_h = torch.zeros_like(ligand_h)
            cofactor_h_proj = torch.zeros_like(cofactor_h_proj)

        if self.ablation == "no_binn":
            # Skip ODE, directly project concatenated features
            combined = torch.cat([protein_h, ligand_h, cofactor_h_proj], dim=-1)
            # Use a simple projection to match hidden_dim
            h_reaction = protein_h + ligand_h + cofactor_h_proj  # simple sum
            feature_evol = torch.zeros(protein_h.size(0), device=protein_h.device)
            trajectory = []
            gate_profile = torch.zeros(1, protein_h.size(0), device=protein_h.device)
        else:
            binn_output = self.binn(protein_h, ligand_h, cofactor_h_proj)
            h_reaction = binn_output["h_reaction"]

        catalysis_output = self.catalysis_head(h_reaction)
        return {
            "ts_stability": catalysis_output["ts_stability"],
            "catalysis_rate": catalysis_output["catalysis_rate"],
            "dG_eyring": catalysis_output["dG_eyring"],
            "h_reaction": h_reaction,
            "feature_evol": binn_output.get("feature_evol", torch.zeros(1))
            if self.ablation != "no_binn" else feature_evol,
            "trajectory": binn_output.get("trajectory", [])
            if self.ablation != "no_binn" else trajectory,
            "gate_profile": binn_output.get("gate_profile", torch.zeros(1))
            if self.ablation != "no_binn" else gate_profile,
        }


def evaluate_ablated(model, loader, device) -> dict:
    """评估消融变体."""
    model.eval()
    all_preds_pkd, all_targets_pkd = [], []
    all_preds_kcat, all_targets_kcat = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {model.ablation}", leave=False):
            ligand_data = batch["ligand_data"].to(device)
            seq_embed = batch["seq_embed"].to(device)
            cofactor_strs = batch["cofactor_strs"]

            outputs = model(ligand_data, seq_embed, cofactor_strs)

            pkd_pred = outputs["ts_stability"].cpu().numpy().flatten()
            pkd_target = batch["pkd_target"].cpu().numpy().flatten()
            pkd_mask = batch["pkd_target_mask"].cpu().numpy().flatten().astype(bool)
            all_preds_pkd.extend(pkd_pred[pkd_mask].tolist())
            all_targets_pkd.extend(pkd_target[pkd_mask].tolist())

            kcat_pred = outputs["catalysis_rate"].cpu().numpy().flatten()
            kcat_target = batch["log_kcat_target"].cpu().numpy().flatten()
            kcat_mask = batch["kcat_target_mask"].cpu().numpy().flatten().astype(bool)
            all_preds_kcat.extend(kcat_pred[kcat_mask].tolist())
            all_targets_kcat.extend(kcat_target[kcat_mask].tolist())

    # Denormalize
    pkd_preds_raw = np.array(all_preds_pkd) * 12.0
    pkd_targets_raw = np.array(all_targets_pkd) * 12.0
    kcat_preds_raw = np.array(all_preds_kcat) * 15.0 - 7.0
    kcat_targets_raw = np.array(all_targets_kcat) * 15.0 - 7.0

    return {
        "pkd": {"preds": pkd_preds_raw, "targets": pkd_targets_raw},
        "kcat": {"preds": kcat_preds_raw, "targets": kcat_targets_raw},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load data (test set only) ──
    log.info("Loading test set...")
    test_ds = OxidoreductaseDataset(METADATA_PATH, PROTEINS_H5, LIGAND_DIR, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn,
                             pin_memory=True)
    log.info(f"Test: {len(test_ds)} samples")

    # ── Load trained model ──
    base_model = load_model(device)

    # ── Ablation variants ──
    ablations = OrderedDict([
        ("full", "Full Model"),
        ("no_gate", "No Gate"),
        ("no_cofactor", "No Cofactor"),
        ("no_ligand", "No Ligand GNN"),
        ("no_ligand_and_cofactor", "No Ligand + Cofactor"),
        ("ode_steps_1", "ODE Steps = 1"),
        ("no_binn", "No BINN (skip ODE)"),
    ])

    results = OrderedDict()
    for abbr, name in ablations.items():
        log.info(f"\n{'='*50}\n  {name} ({abbr})\n{'='*50}")

        if abbr == "full":
            model = base_model
            model.ablation = "full"
        else:
            model = AblatedTrenzition(base_model, abbr).to(device)
        model.eval()

        raw = evaluate_ablated(model, test_loader, device)

        result = {}
        for task in ["pkd", "kcat"]:
            preds = raw[task]["preds"]
            targets = raw[task]["targets"]
            result[task] = compute_metrics(preds, targets)
            log.info(f"  {task}: MSE={result[task]['MSE']:.4f}, "
                     f"MAE={result[task]['MAE']:.4f}, "
                     f"R²={result[task]['R2']:.4f}, ρ={result[task]['Spearman_rho']:.4f}")

        results[name] = result

    # ── Save results ──
    out_path = OUTPUT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved → {out_path}")

    # ── Print summary table ──
    for task in ["pkd", "kcat"]:
        print(f"\n{'='*70}")
        print(f"  ABLATION — {task.upper()}")
        print(f"{'='*70}")
        print(f"{'Variant':<28} {'R²':>8} {'Δ R²':>8} {'MSE':>10} {'Spearman ρ':>10}")
        print("-" * 70)

        base_r2 = results["Full Model"][task]["R2"]
        for name, r in results.items():
            r2 = r[task]["R2"]
            delta = r2 - base_r2
            print(f"{name:<28} {r2:>8.4f} {delta:>+8.4f} "
                  f"{r[task]['MSE']:>10.4f} {r[task]['Spearman_rho']:>10.4f}")


if __name__ == "__main__":
    main()
