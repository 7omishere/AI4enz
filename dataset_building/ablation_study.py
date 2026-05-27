#!/usr/bin/env python3
"""
消融实验框架：评估模型各组件的重要性
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

class AblationStudy:
    """消融实验管理器"""
    
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir or "ablation_results")
        self.results_dir.mkdir(exist_ok=True)
        self.experiments = {}
    
    def add_experiment(self, name, config, metrics):
        """添加消融实验结果"""
        self.experiments[name] = {
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_results(self, filename="ablation_results.json"):
        """保存消融实验结果"""
        with open(self.results_dir / filename, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def load_results(self, filename="ablation_results.json"):
        """加载消融实验结果"""
        if (self.results_dir / filename).exists():
            with open(self.results_dir / filename, 'r') as f:
                self.experiments = json.load(f)
    
    def compare_metrics(self, metrics_to_compare=None):
        """比较不同实验的metric"""
        if not self.experiments:
            print("没有实验数据")
            return
        
        metrics_to_compare = metrics_to_compare or ['total_loss', 'pkd_mae', 'pkd_rmse', 'pkd_r2', 'kcat_mae', 'kcat_r2']
        
        # 创建对比表格
        data = []
        for exp_name, exp_data in self.experiments.items():
            row = {'Experiment': exp_name}
            for metric in metrics_to_compare:
                row[metric] = exp_data['metrics'].get(metric, 'N/A')
            data.append(row)
        
        df = pd.DataFrame(data)
        print("\n" + "=" * 100)
        print("消融实验对比表")
        print("=" * 100)
        print(df.to_string(index=False))
        print("\n" + "=" * 100)
        
        return df
    
    def plot_comparison(self, metric='total_loss'):
        """绘制不同实验的metric对比图"""
        if not self.experiments:
            print("没有实验数据")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        experiment_names = list(self.experiments.keys())
        values = [self.experiments[name]['metrics'].get(metric, np.nan) for name in experiment_names]
        
        bars = ax.bar(experiment_names, values, color='skyblue', edgecolor='black')
        
        ax.set_xlabel('Ablation Experiments', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = self.results_dir / f'{metric}_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到：{save_path}")
        
        plt.show()

def generate_ablation_configs(base_config):
    """生成消融实验配置"""
    ablation_configs = []
    
    # 完整模型（基准）
    ablation_configs.append({
        'name': 'Full_Model',
        'description': '完整模型（所有组件都启用）',
        'config': base_config.copy()
    })
    
    # 移除蛋白编码器
    config = base_config.copy()
    config['use_protein_encoder'] = False
    ablation_configs.append({
        'name': 'No_Protein_Encoder',
        'description': '移除蛋白序列编码器（ESM-2）',
        'config': config
    })
    
    # 移除配体编码器
    config = base_config.copy()
    config['use_ligand_encoder'] = False
    ablation_configs.append({
        'name': 'No_Ligand_Encoder',
        'description': '移除配体图编码器（GNN）',
        'config': config
    })
    
    # 移除辅因子编码器
    config = base_config.copy()
    config['use_cofactor_encoder'] = False
    ablation_configs.append({
        'name': 'No_Cofactor_Encoder',
        'description': '移除辅因子编码器',
        'config': config
    })
    
    # 移除交互模块
    config = base_config.copy()
    config['use_interaction_module'] = False
    ablation_configs.append({
        'name': 'No_Interaction_Module',
        'description': '移除蛋白-配体交互模块',
        'config': config
    })
    
    # 使用简化模型结构
    config = base_config.copy()
    config['hidden_dim'] = 128
    config['gnn_layers'] = 2
    ablation_configs.append({
        'name': 'Smaller_Model',
        'description': '使用更小的模型（hidden_dim=128, gnn_layers=2）',
        'config': config
    })
    
    # 移除kcat任务
    config = base_config.copy()
    config['use_kcat_loss'] = False
    ablation_configs.append({
        'name': 'No_Kcat_Loss',
        'description': '移除kcat预测任务（只训练pKd）',
        'config': config
    })
    
    # 使用不同的学习率
    config = base_config.copy()
    config['lr'] = 5e-5
    ablation_configs.append({
        'name': 'Lower_LR',
        'description': '使用更低的学习率（5e-5）',
        'config': config
    })
    
    return ablation_configs

def run_ablation_experiment(config):
    """运行单个消融实验（示例函数）"""
    print(f"\n正在运行消融实验: {config['name']}")
    print(f"描述: {config['description']}")
    
    # 这里应该是实际的训练代码
    # 为了演示，我们模拟一些结果
    
    # 模拟指标（基于完整模型的性能变化）
    base_metrics = {
        'total_loss': 0.050,
        'pkd_mae': 0.82,
        'pkd_rmse': 1.05,
        'pkd_r2': 0.25,
        'kcat_mae': 0.72,
        'kcat_rmse': 0.93,
        'kcat_r2': 0.45,
        'best_epoch': 34
    }
    
    # 根据消融组件调整指标
    metrics = base_metrics.copy()
    name = config['name']
    
    if 'No_Protein_Encoder' in name:
        metrics['pkd_mae'] *= 1.3
        metrics['pkd_r2'] *= 0.6
        metrics['kcat_mae'] *= 1.2
        metrics['kcat_r2'] *= 0.7
    elif 'No_Ligand_Encoder' in name:
        metrics['pkd_mae'] *= 1.4
        metrics['pkd_r2'] *= 0.5
        metrics['kcat_mae'] *= 1.3
        metrics['kcat_r2'] *= 0.6
    elif 'No_Cofactor_Encoder' in name:
        metrics['pkd_mae'] *= 1.1
        metrics['pkd_r2'] *= 0.85
        metrics['kcat_mae'] *= 1.15
        metrics['kcat_r2'] *= 0.8
    elif 'No_Interaction_Module' in name:
        metrics['pkd_mae'] *= 1.2
        metrics['pkd_r2'] *= 0.7
        metrics['kcat_mae'] *= 1.15
        metrics['kcat_r2'] *= 0.75
    elif 'Smaller_Model' in name:
        metrics['pkd_mae'] *= 1.1
        metrics['pkd_r2'] *= 0.9
        metrics['kcat_mae'] *= 1.08
        metrics['kcat_r2'] *= 0.88
    elif 'No_Kcat_Loss' in name:
        metrics['pkd_mae'] *= 0.95
        metrics['pkd_r2'] *= 1.05
        metrics['kcat_mae'] = float('inf')
        metrics['kcat_r2'] = float('nan')
    elif 'Lower_LR' in name:
        metrics['pkd_mae'] *= 1.02
        metrics['pkd_r2'] *= 0.98
    
    # 添加一些噪声
    for key in metrics:
        if isinstance(metrics[key], (int, float)) and np.isfinite(metrics[key]):
            metrics[key] *= (1 + np.random.uniform(-0.02, 0.02))
    
    print(f"实验完成，验证损失: {metrics['total_loss']:.4f}")
    
    return metrics

def main():
    """运行完整的消融实验流程"""
    print("=" * 100)
    print("消融实验框架")
    print("=" * 100)
    
    # 基准配置
    base_config = {
        'hidden_dim': 256,
        'gnn_layers': 3,
        'lr': 1e-4,
        'batch_size': 128,
        'epochs': 64,
        'use_protein_encoder': True,
        'use_ligand_encoder': True,
        'use_cofactor_encoder': True,
        'use_interaction_module': True,
        'use_kcat_loss': True,
        'weight_decay': 1e-5
    }
    
    # 生成消融实验配置
    ablation_configs = generate_ablation_configs(base_config)
    print(f"\n生成了 {len(ablation_configs)} 个消融实验配置")
    
    # 创建消融实验管理器
    study = AblationStudy()
    
    # 运行所有消融实验
    for config in ablation_configs:
        metrics = run_ablation_experiment(config)
        study.add_experiment(config['name'], config['config'], metrics)
    
    # 保存结果
    study.save_results()
    print("\n消融实验结果已保存")
    
    # 比较metrics
    df = study.compare_metrics()
    
    # 绘制对比图
    study.plot_comparison('pkd_mae')
    study.plot_comparison('pkd_r2')
    study.plot_comparison('kcat_r2')
    
    # 生成分析报告
    generate_analysis_report(df)

def generate_analysis_report(df):
    """生成消融实验分析报告"""
    print("\n" + "=" * 100)
    print("消融实验分析报告")
    print("=" * 100)
    
    # 找出对pkd任务最重要的组件
    pkd_mae_increases = {}
    base_pkd_mae = df[df['Experiment'] == 'Full_Model']['pkd_mae'].values[0]
    
    for _, row in df.iterrows():
        if row['Experiment'] != 'Full_Model' and pd.notna(row['pkd_mae']):
            increase = (row['pkd_mae'] - base_pkd_mae) / base_pkd_mae * 100
            pkd_mae_increases[row['Experiment']] = increase
    
    print("\n📊 pKd任务组件重要性（MAE上升百分比）:")
    for exp, increase in sorted(pkd_mae_increases.items(), key=lambda x: x[1], reverse=True):
        print(f"  {exp}: +{increase:.1f}%")
    
    # 找出对kcat任务最重要的组件
    kcat_r2_decreases = {}
    base_kcat_r2 = df[df['Experiment'] == 'Full_Model']['kcat_r2'].values[0]
    
    for _, row in df.iterrows():
        if row['Experiment'] != 'Full_Model' and pd.notna(row['kcat_r2']):
            decrease = (base_kcat_r2 - row['kcat_r2']) / base_kcat_r2 * 100
            kcat_r2_decreases[row['Experiment']] = decrease
    
    print("\n📊 kcat任务组件重要性（R²下降百分比）:")
    for exp, decrease in sorted(kcat_r2_decreases.items(), key=lambda x: x[1], reverse=True):
        print(f"  {exp}: -{decrease:.1f}%")
    
    # 总结最重要的组件
    print("\n🎯 关键发现:")
    print("-" * 50)
    
    most_important_pkd = max(pkd_mae_increases, key=pkd_mae_increases.get)
    print(f"• 对pKd任务最重要的组件: {most_important_pkd}")
    
    most_important_kcat = max(kcat_r2_decreases, key=kcat_r2_decreases.get)
    print(f"• 对kcat任务最重要的组件: {most_important_kcat}")
    
    least_important = min(pkd_mae_increases, key=pkd_mae_increases.get)
    print(f"• 相对不重要的组件: {least_important}")
    
    print("\n💡 建议:")
    print("-" * 50)
    print("1. 优先优化最重要的组件")
    print("2. 可以考虑移除或简化相对不重要的组件")
    print("3. 基于消融结果调整模型架构")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()