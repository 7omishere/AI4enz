#!/usr/bin/env python3
""""
评估 Trenzition 三头模型：PCC / SCC / R² on test set
用法: python3 evaluate.py [--checkpoint checkpoints/best.ckpt]

依赖: proteins_token.h5 (token 级 ESM-2 嵌入, 100% 覆盖率)
"""
import argparse, sys, logging, json, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranking_model import Trenzition
from train import OxidoreductaseDataset, collate_fn, NORM_PARAMS, PROJECT_DIR, PROCESSED_DIR, LIGAND_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate")

def pearson(x, y):
    """PCC"""
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum()
    sx = ((x - mx) ** 2).sum().sqrt()
    sy = ((y - my) ** 2).sum().sqrt()
    return cov / (sx * sy + 1e-8)

def spearman(x, y):
    """SCC — rank-based"""
    xr = x.argsort().argsort().float()
    yr = y.argsort().argsort().float()
    return pearson(xr, yr)

def r2_score(x, y):
    """R²"""
    ss_res = ((x - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-8)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = {k: [] for k in ['kd', 'ki', 'kcat', 'km', 'dG']}
    targets = {k: [] for k in ['kd', 'ki', 'kcat', 'km']}

    for batch in tqdm(loader, desc="Evaluating"):
        bg = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}
        bg["ligand_data"] = bg["ligand_data"].to(device)

        out = model(
            ligand_data=bg["ligand_data"],
            protein_tokens=bg["protein_tokens"],
            protein_mask=bg["protein_mask"],
            cofactor_strs=bg["cofactor_strs"],
            measurement_types=bg.get("measurement_type"),
            temperature_K=bg.get("temperature_K"),
        )

        # pKd
        mask_kd = bg["pkd_target_mask"]
        if mask_kd.any():
            kd = _denorm(_pred_mean(out["kd_pred"][mask_kd]), 0, 12)
            tk = _denorm(bg["pkd_target"][mask_kd], 0, 12)
            preds["kd"].append(kd.cpu()); targets["kd"].append(tk.cpu())

        # Ki
        mask_ki = mask_kd & (bg.get("measurement_type") == 1)
        if mask_ki.any():
            ki = _denorm(_pred_mean(out["ki_pred"][mask_ki]), 0, 12)
            tki = _denorm(bg["pkd_target"][mask_ki], 0, 12)
            preds["ki"].append(ki.cpu()); targets["ki"].append(tki.cpu())

        # kcat
        mask_c = bg["kcat_target_mask"]
        if mask_c.any():
            kc = _denorm(_pred_mean(out["kcat_pred"][mask_c]), -7, 8)
            tc = _denorm(bg["log_kcat_target"][mask_c], -7, 8)
            preds["kcat"].append(kc.cpu()); targets["kcat"].append(tc.cpu())

        # Km
        mask_m = bg["km_target_mask"]
        if mask_m.any():
            km = _denorm(_pred_mean(out["log_km_pred"][mask_m]), -13, 3)
            tm = bg["log_km_target"][mask_m].cpu()
            preds["km"].append(km.cpu()); targets["km"].append(tm.cpu())

        # dG
        if "dG_eyring" in out:
            preds["dG"].append(out["dG_eyring"].cpu())

    return {k: torch.cat(v) for k, v in preds.items()}, \
           {k: torch.cat(v) for k, v in targets.items()}

def _pred_mean(v):
    """异方差模式下取 mean（第一列），普通模式直接返回"""
    return v[..., 0] if v.dim() > 1 and v.size(-1) > 1 else v

def _denorm(v, mn, mx):
    return v * (mx - mn) + mn

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(PROJECT_DIR / "checkpoints/best.ckpt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--heteroscedastic", action="store_true", help="启用异方差 NLL 损失")
    # 数据路径
    p.add_argument("--metadata", default=str(PROCESSED_DIR / "metadata.parquet"),
                   help="元数据 parquet 路径")
    p.add_argument("--proteins-token-h5", default=str(PROCESSED_DIR / "proteins_token.h5"),
                   help="蛋白 token 级 ESM-2 嵌入 H5 路径")
    p.add_argument("--ligand-dir", default=str(LIGAND_DIR),
                   help="配体图目录")
    args = p.parse_args()

    device = torch.device(args.device)

    # ── 数据集 (token-only) ──
    ds = OxidoreductaseDataset(
        unified_metadata_path=args.metadata,
        proteins_token_h5_path=args.proteins_token_h5,
        ligand_dir=args.ligand_dir,
        split="test",
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn,
                        num_workers=4, pin_memory=device.type == "cuda")

    # ── 模型 ──
    model = Trenzition(
        hidden_dim=256, gnn_layers=3, three_head=True, kcat_ode_steps=10,
        use_cross_attn=False,
        heteroscedastic=args.heteroscedastic,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── 评估 ──
    preds, targets = evaluate(model, loader, device)

    print("\n========== Test Set Metrics ==========")
    for key, label in [("kd", "pKd (Kd)"), ("ki", "pKi (Ki)"),
                        ("kcat", "log\u2081\u2080(kcat)"), ("km", "log\u2081\u2080(Km)")]:
        if key in preds and key in targets and len(preds[key]) > 5:
            pcc = pearson(preds[key], targets[key]).item()
            scc = spearman(preds[key], targets[key]).item()
            r2 = r2_score(preds[key], targets[key]).item()
            n = len(preds[key])
            print(f"  {label:>15s}:  PCC={pcc:.4f}  SCC={scc:.4f}  R\u00b2={r2:.4f}  (n={n})")
        else:
            print(f"  {label:>15s}:  (insufficient data)")

    # dG stats
    if "dG" in preds and len(preds["dG"]) > 0:
        dG = preds["dG"]
        print(f"  {'ΔG‡ (kJ/mol)':>15s}:  mean={dG.mean():.1f}  std={dG.std():.1f}  "
              f"min={dG.min():.1f}  max={dG.max():.1f}  (n={len(dG)})")

    print("=====================================\n")

if __name__ == "__main__":
    main()
