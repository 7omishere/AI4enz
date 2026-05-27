#!/usr/bin/env python3
"""
分析训练完成后的验证集和测试集评估指标
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

def load_training_log():
    """加载训练日志"""
    log_file = CHECKPOINT_DIR / "training_log.json"
    if log_file.exists():
        with open(log_file, 'r') as f:
            return json.load(f)
    return None

def analyze_metrics():
    """分析评估指标"""
    print("=" * 80)
    print("AI4enz 训练评估指标分析")
    print("=" * 80)
    
    # 1. 检查 checkpoint 文件
    print("\n1. 检查点文件:")
    checkpoint_files = list(CHECKPOINT_DIR.glob("*.ckpt"))
    for ckpt in sorted(checkpoint_files):
        size_mb = ckpt.stat().st_size / 1024 / 1024
        print(f"   - {ckpt.name}: {size_mb:.2f} MB")
    
    # 2. 加载最佳模型的训练历史
    print("\n2. 训练历史分析:")
    print("-" * 80)
    
    # 尝试从日志或 checkpoint 中恢复训练历史
    # 这里需要根据实际保存的格式来调整
    
    # 3. 合理的评估指标范围
    print("\n3. 合理的评估指标范围参考:")
    print("-" * 80)
    print("   pKd 任务 (结合亲和力预测):")
    print("     - MAE: 0.5-1.5 (合理), <0.5 (优秀), >2.0 (需改进)")
    print("     - RMSE: 0.7-2.0 (合理), <0.7 (优秀), >2.5 (需改进)")
    print("     - R²: 0.3-0.6 (合理), >0.6 (优秀), <0.2 (需改进)")
    print("")
    print("   kcat 任务 (催化效率预测):")
    print("     - MAE: 0.3-1.0 (合理), <0.3 (优秀), >1.5 (需改进)")
    print("     - RMSE: 0.5-1.5 (合理), <0.5 (优秀), >2.0 (需改进)")
    print("     - R²: 0.2-0.5 (合理), >0.5 (优秀), <0.1 (需改进)")
    print("")
    print("   损失值参考:")
    print("     - L_pkd: 0.01-0.1 (合理), <0.01 (优秀), >0.2 (需改进)")
    print("     - L_kcat: 0.01-0.1 (合理), <0.01 (优秀), >0.2 (需改进)")
    print("")
    
    # 4. 过拟合检测
    print("\n4. 过拟合检测标准:")
    print("-" * 80)
    print("   - train_loss / val_loss < 1.2: 正常")
    print("   - 1.2 <= train_loss / val_loss < 2.0: 轻微过拟合")
    print("   - train_loss / val_loss >= 2.0: 严重过拟合")
    print("")
    
    # 5. 数据集统计
    print("\n5. 数据集统计:")
    print("-" * 80)
    unified_metadata = Path(__file__).parent.parent / "dataset_building/processed/oxidoreductase/unified_metadata.parquet"
    if unified_metadata.exists():
        df = pd.read_parquet(unified_metadata)
        print(f"   总记录数：{len(df):,}")
        print(f"   唯一蛋白数：{df['uniprot_id'].nunique():,}")
        print(f"   唯一配体数：{df['ligand_inchikey'].nunique():,}")
        
        if 'split' in df.columns:
            for split in ['train', 'val', 'test']:
                split_df = df[df['split'] == split]
                print(f"   {split}: {len(split_df):,} 样本 ({len(split_df)/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    analyze_metrics()
