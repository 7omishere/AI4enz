#!/usr/bin/env python3
"""
使用已保存的基线比较结果生成对比图
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison():
    # 加载结果
    result_path = Path("baseline_comparison_real") / "baseline_comparison.json"
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    model_names = list(results.keys())
    colors = []
    types = []
    for name in model_names:
        model_type = results[name]['model_type']
        types.append(model_type)
        if model_type == 'Baseline':
            colors.append('gray')
        elif model_type == 'MLP':
            colors.append('orange')
        elif model_type == 'GNN':
            colors.append('green')
        elif model_type == 'Hybrid':
            colors.append('blue')
        else:
            colors.append('purple')
    
    # 绘制 pKd MAE
    fig, ax = plt.subplots(figsize=(12, 6))
    values = [results[name]['metrics'].get('pkd_mae', 0) for name in model_names]
    bars = ax.bar(model_names, values, color=colors, edgecolor='black')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('pKd MAE', fontsize=12, fontweight='bold')
    ax.set_title('pKd MAE Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('baseline_comparison_real/pkd_mae_comparison.png', dpi=300, bbox_inches='tight')
    print("pKd MAE对比图已保存")
    
    # 绘制 pKd R²
    fig, ax = plt.subplots(figsize=(12, 6))
    values = [results[name]['metrics'].get('pkd_r2', 0) for name in model_names]
    bars = ax.bar(model_names, values, color=colors, edgecolor='black')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('pKd R²', fontsize=12, fontweight='bold')
    ax.set_title('pKd R² Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('baseline_comparison_real/pkd_r2_comparison.png', dpi=300, bbox_inches='tight')
    print("pKd R²对比图已保存")
    
    # 绘制 kcat R²
    fig, ax = plt.subplots(figsize=(12, 6))
    values = [results[name]['metrics'].get('kcat_r2', 0) for name in model_names]
    bars = ax.bar(model_names, values, color=colors, edgecolor='black')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('kcat R²', fontsize=12, fontweight='bold')
    ax.set_title('kcat R² Comparison', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('baseline_comparison_real/kcat_r2_comparison.png', dpi=300, bbox_inches='tight')
    print("kcat R²对比图已保存")
    
    # 打印对比表
    print("\n" + "=" * 100)
    print("模型对比表 (使用真实评估指标)")
    print("=" * 100)
    print(f"{'Model':<20} {'Type':<10} {'pKd MAE':<10} {'pKd R²':<10} {'kcat MAE':<10} {'kcat R²':<10}")
    print("-" * 100)
    for name in model_names:
        m = results[name]['metrics']
        t = results[name]['model_type']
        print(f"{name:<20} {t:<10} {m.get('pkd_mae', 'N/A'):<10.4f} {m.get('pkd_r2', 'N/A'):<10.4f} {m.get('kcat_mae', 'N/A'):<10.4f} {m.get('kcat_r2', 'N/A'):<10.4f}")
    print("=" * 100)

if __name__ == "__main__":
    plot_comparison()