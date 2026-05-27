#!/usr/bin/env python3
"""
评估模型在验证集和测试集上的表现
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# 导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from datepre.ranking_model import MarcusPINN
from dataset_building.train import OxidoreductaseDataset, collate_fn
from torch.utils.data import DataLoader

PROCESSED_DIR = Path(__file__).parent.parent / "dataset_building/processed"
OXIDOREDUCTASE_DIR = PROCESSED_DIR / "oxidoreductase"
LIGAND_DIR = PROCESSED_DIR / "ligands"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

@torch.no_grad()
def evaluate_model():
    """评估模型性能"""
    print("=" * 80)
    print("AI4enz 模型评估")
    print("=" * 80)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备：{device}")
    
    # 加载最佳模型
    checkpoint_path = CHECKPOINT_DIR / "best.ckpt"
    print(f"加载模型：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = MarcusPINN(hidden_dim=256, gnn_layers=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载数据集
    print("\n加载数据集...")
    unified_metadata = OXIDOREDUCTASE_DIR / "unified_metadata.parquet"
    proteins_h5 = PROCESSED_DIR / "proteins.h5"
    
    val_dataset = OxidoreductaseDataset(
        unified_metadata, proteins_h5, LIGAND_DIR,
        split="val", use_esm2=True,
    )
    test_dataset = OxidoreductaseDataset(
        unified_metadata, proteins_h5, LIGAND_DIR,
        split="test", use_esm2=True,
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    
    print(f"验证集：{len(val_dataset)} 样本，{len(val_loader)} 批次")
    print(f"测试集：{len(test_dataset)} 样本，{len(test_loader)} 批次")
    
    # 评估函数
    def evaluate_loader(loader, desc):
        all_pkd_true = []
        all_pkd_pred = []
        all_kcat_true = []
        all_kcat_pred = []
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(loader, desc=desc):
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(device)
                elif hasattr(v, 'to') and hasattr(v, 'edge_index'):
                    batch_gpu[k] = v.to(device)
                else:
                    batch_gpu[k] = v
            
            outputs = model(
                batch_gpu["ligand_data"],
                batch_gpu["seq_embed"],
                batch_gpu["cofactor_strs"],
                batch_gpu["struct_feat"],
                batch_gpu["has_structure"],
                domain_masks=batch_gpu.get("domain_masks"),
                domain_padding_mask=batch_gpu.get("domain_padding_mask"),
                pocket_cn=batch_gpu.get("pocket_cn"),
                pocket_pi=batch_gpu.get("pocket_pi"),
                pocket_dist=batch_gpu.get("pocket_dist"),
                pocket_mask=batch_gpu.get("pocket_mask"),
            )
            
            # 计算损失
            _, losses = model.compute_loss(
                outputs,
                {
                    "pkd_target": batch_gpu["pkd_target"],
                    "pkd_target_mask": batch_gpu["pkd_target_mask"],
                    "log_kcat_target": batch_gpu["log_kcat_target"],
                    "kcat_target_mask": batch_gpu["kcat_target_mask"],
                    "kcat_weights": batch_gpu.get("kcat_weights"),
                    "quality_weight": batch_gpu["quality_weight"],
                },
            )
            
            total_loss += losses.get("total", 0).item()
            n_batches += 1
            
            # 收集预测值
            pkd_mask = batch_gpu["pkd_target_mask"].cpu().bool()
            all_pkd_true.extend(batch_gpu["pkd_target"][pkd_mask].cpu().numpy())
            all_pkd_pred.extend(outputs["pkd"][pkd_mask].cpu().numpy())
            
            kcat_mask = batch_gpu.get("kcat_target_mask", torch.zeros_like(pkd_mask)).cpu().bool()
            if kcat_mask.any():
                all_kcat_true.extend(batch_gpu["log_kcat_target"][kcat_mask].cpu().numpy())
                all_kcat_pred.extend(outputs["log_kcat"][kcat_mask].cpu().numpy())
        
        # 计算指标
        metrics = {
            "loss": total_loss / max(n_batches, 1),
        }
        
        if len(all_pkd_true) > 0:
            metrics["pkd_mae"] = mean_absolute_error(all_pkd_true, all_pkd_pred)
            metrics["pkd_rmse"] = np.sqrt(mean_squared_error(all_pkd_true, all_pkd_pred))
            metrics["pkd_r2"] = r2_score(all_pkd_true, all_pkd_pred)
        
        if len(all_kcat_true) > 0:
            metrics["kcat_mae"] = mean_absolute_error(all_kcat_true, all_kcat_pred)
            metrics["kcat_rmse"] = np.sqrt(mean_squared_error(all_kcat_true, all_kcat_pred))
            metrics["kcat_r2"] = r2_score(all_kcat_true, all_kcat_pred)
        
        return metrics
    
    # 评估验证集
    print("\n" + "=" * 80)
    val_metrics = evaluate_loader(val_loader, "验证集")
    print("\n验证集指标:")
    print("-" * 80)
    print(f"   总损失：{val_metrics['loss']:.4f}")
    if "pkd_mae" in val_metrics:
        print(f"   pKd MAE:  {val_metrics['pkd_mae']:.4f}")
        print(f"   pKd RMSE: {val_metrics['pkd_rmse']:.4f}")
        print(f"   pKd R²:   {val_metrics['pkd_r2']:.4f}")
    if "kcat_mae" in val_metrics:
        print(f"   kcat MAE:  {val_metrics['kcat_mae']:.4f}")
        print(f"   kcat RMSE: {val_metrics['kcat_rmse']:.4f}")
        print(f"   kcat R²:   {val_metrics['kcat_r2']:.4f}")
    
    # 评估测试集
    print("\n" + "=" * 80)
    test_metrics = evaluate_loader(test_loader, "测试集")
    print("\n测试集指标:")
    print("-" * 80)
    print(f"   总损失：{test_metrics['loss']:.4f}")
    if "pkd_mae" in test_metrics:
        print(f"   pKd MAE:  {test_metrics['pkd_mae']:.4f}")
        print(f"   pKd RMSE: {test_metrics['pkd_rmse']:.4f}")
        print(f"   pKd R²:   {test_metrics['pkd_r2']:.4f}")
    if "kcat_mae" in test_metrics:
        print(f"   kcat MAE:  {test_metrics['kcat_mae']:.4f}")
        print(f"   kcat RMSE: {test_metrics['kcat_rmse']:.4f}")
        print(f"   kcat R²:   {test_metrics['kcat_r2']:.4f}")
    
    # 合理性判断
    print("\n" + "=" * 80)
    print("指标合理性评估:")
    print("-" * 80)
    
    # pKd 指标评估
    if "pkd_mae" in val_metrics:
        pkd_mae = val_metrics['pkd_mae']
        if pkd_mae < 0.5:
            print(f"   ✓ pKd MAE ({pkd_mae:.3f}): 优秀")
        elif pkd_mae < 1.5:
            print(f"   ✓ pKd MAE ({pkd_mae:.3f}): 合理")
        else:
            print(f"   ⚠ pKd MAE ({pkd_mae:.3f}): 需改进")
        
        pkd_r2 = val_metrics['pkd_r2']
        if pkd_r2 > 0.6:
            print(f"   ✓ pKd R² ({pkd_r2:.3f}): 优秀")
        elif pkd_r2 > 0.3:
            print(f"   ✓ pKd R² ({pkd_r2:.3f}): 合理")
        else:
            print(f"   ⚠ pKd R² ({pkd_r2:.3f}): 需改进")
    
    # kcat 指标评估
    if "kcat_mae" in val_metrics:
        kcat_mae = val_metrics['kcat_mae']
        if kcat_mae < 0.3:
            print(f"   ✓ kcat MAE ({kcat_mae:.3f}): 优秀")
        elif kcat_mae < 1.0:
            print(f"   ✓ kcat MAE ({kcat_mae:.3f}): 合理")
        else:
            print(f"   ⚠ kcat MAE ({kcat_mae:.3f}): 需改进")
        
        kcat_r2 = val_metrics['kcat_r2']
        if kcat_r2 > 0.5:
            print(f"   ✓ kcat R² ({kcat_r2:.3f}): 优秀")
        elif kcat_r2 > 0.2:
            print(f"   ✓ kcat R² ({kcat_r2:.3f}): 合理")
        else:
            print(f"   ⚠ kcat R² ({kcat_r2:.3f}): 需改进")
    
    # 过拟合检查
    print("\n过拟合检查:")
    print("-" * 80)
    # 这里需要训练集的损失来比较，暂时跳过
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    
    # 保存评估结果到文件
    results = {
        'val': val_metrics,
        'test': test_metrics,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    result_path = CHECKPOINT_DIR / "evaluation_results.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n评估结果已保存到：{result_path}")
    
    return val_metrics, test_metrics

if __name__ == "__main__":
    evaluate_model()
