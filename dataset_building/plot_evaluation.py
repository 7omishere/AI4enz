#!/usr/bin/env python3
"""
绘制训练和评估结果的可视化图表
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 150

def plot_model_comparison():
    """绘制验证集和测试集的指标对比"""
    
    # 从评估结果创建数据
    metrics = {
        'Metric': ['Total Loss', 'pKd MAE', 'pKd RMSE', 'pKd R²', 
                   'kcat MAE', 'kcat RMSE', 'kcat R²'],
        'Validation': [0.0584, 0.827, 1.056, 0.183, 0.714, 0.930, 0.420],
        'Test': [0.0695, 0.668, 0.886, -0.104, 0.667, 0.907, 0.183]
    }
    
    df = pd.DataFrame(metrics)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图 1：损失对比
    ax1 = axes[0, 0]
    losses = df[df['Metric'].isin(['Total Loss'])]
    x = np.arange(len(losses))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, losses['Validation'].values, width, 
                    label='Validation', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, losses['Test'].values, width, 
                    label='Test', color='coral', alpha=0.8)
    
    ax1.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(losses['Metric'].values, rotation=0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 图 2：pKd 指标对比
    ax2 = axes[0, 1]
    pkd_metrics = df[df['Metric'].str.contains('pKd')]
    x = np.arange(len(pkd_metrics))
    
    bars1 = ax2.bar(x - width/2, pkd_metrics['Validation'].values, width, 
                    label='Validation', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, pkd_metrics['Test'].values, width, 
                    label='Test', color='coral', alpha=0.8)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('pKd Prediction Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pkd_metrics['Metric'].values, rotation=15)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 图 3：kcat 指标对比
    ax3 = axes[1, 0]
    kcat_metrics = df[df['Metric'].str.contains('kcat')]
    x = np.arange(len(kcat_metrics))
    
    bars1 = ax3.bar(x - width/2, kcat_metrics['Validation'].values, width, 
                    label='Validation', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, kcat_metrics['Test'].values, width, 
                    label='Test', color='coral', alpha=0.8)
    
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('kcat Prediction Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(kcat_metrics['Metric'].values, rotation=15)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 图 4：R²对比（重点关注）
    ax4 = axes[1, 1]
    r2_metrics = df[df['Metric'].str.contains('R²')]
    x = np.arange(len(r2_metrics))
    
    colors = []
    for val in r2_metrics['Test'].values:
        if val < 0:
            colors.append('red')
        elif val < 0.3:
            colors.append('orange')
        else:
            colors.append('green')
    
    bars = ax4.bar(x, r2_metrics['Test'].values, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, 
                label='Critical threshold (R²=0)')
    ax4.axhline(y=0.3, color='orange', linestyle='--', linewidth=1.5, 
                label='Acceptable threshold (R²=0.3)')
    ax4.axhline(y=0.5, color='green', linestyle='--', linewidth=1.5, 
                label='Good threshold (R²=0.5)')
    
    ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax4.set_title('R² Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(r2_metrics['Metric'].values, rotation=15)
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(-0.5, 0.6)
    
    # 添加数值标签
    for bar, color in zip(bars, colors):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va=va, fontsize=11, 
                fontweight='bold', color='black')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = CHECKPOINT_DIR / "model_evaluation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"评估指标对比图已保存到：{save_path}")
    
    # 打印关键发现
    print("\n" + "=" * 80)
    print("关键发现:")
    print("=" * 80)
    print(f"✓ 验证集 pKd R² = {df[df['Metric']=='pKd R²']['Validation'].values[0]:.3f}")
    print(f"✗ 测试集 pKd R² = {df[df['Metric']=='pKd R²']['Test'].values[0]:.3f} (负值！)")
    print(f"✓ 验证集 kcat R² = {df[df['Metric']=='kcat R²']['Validation'].values[0]:.3f}")
    print(f"⚠️ 测试集 kcat R² = {df[df['Metric']=='kcat R²']['Test'].values[0]:.3f} (下降明显)")
    print("\n建议:")
    print("1. 检查训练集/测试集数据分布是否一致")
    print("2. 分析测试集中表现最差的样本")
    print("3. 考虑增加训练数据多样性或使用正则化")
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    plot_model_comparison()
