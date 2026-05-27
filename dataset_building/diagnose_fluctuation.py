#!/usr/bin/env python3
"""
诊断验证损失波动的原因
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

def analyze_validation_fluctuation():
    """分析验证损失波动的原因"""
    
    print("=" * 80)
    print("验证损失波动诊断分析")
    print("=" * 80)
    
    # 加载训练历史
    checkpoint = torch.load(CHECKPOINT_DIR / "best.ckpt", map_location='cpu', weights_only=False)
    val_losses = checkpoint['val_losses']
    train_losses = checkpoint['train_losses']
    epochs = range(1, len(val_losses) + 1)
    
    # 计算统计指标
    val_std = np.std(val_losses)
    val_mean = np.mean(val_losses)
    val_cv = val_std / val_mean * 100  # 变异系数
    
    train_std = np.std(train_losses)
    train_mean = np.mean(train_losses)
    train_cv = train_std / train_mean * 100
    
    print(f"\n📊 损失统计分析:")
    print(f"验证损失: 均值={val_mean:.4f}, 标准差={val_std:.4f}, 变异系数={val_cv:.2f}%")
    print(f"训练损失: 均值={train_mean:.4f}, 标准差={train_std:.4f}, 变异系数={train_cv:.2f}%")
    
    # 检测异常波动点
    print(f"\n⚠️ 异常波动检测:")
    threshold = val_mean + 2 * val_std
    high_points = [(i+1, val_losses[i]) for i in range(len(val_losses)) if val_losses[i] > threshold]
    print(f"超过2σ阈值 ({threshold:.4f}) 的波动点:")
    for epoch, loss in high_points[:5]:
        print(f"  Epoch {epoch}: {loss:.4f}")
    
    # 计算连续波动
    val_diff = np.diff(val_losses)
    large_fluctuations = [(i+2, val_diff[i]) for i in range(len(val_diff)) if abs(val_diff[i]) > 0.01]
    print(f"\n🔄 大幅波动 (变化 > 0.01):")
    for epoch, diff in large_fluctuations[:5]:
        direction = "上升" if diff > 0 else "下降"
        print(f"  Epoch {epoch-1}→{epoch}: {direction} {abs(diff):.4f}")
    
    # 可能原因分析
    print("\n" + "=" * 80)
    print("🎯 可能原因分析:")
    print("=" * 80)
    
    # 原因1: 验证集数据分布
    if val_cv > 10:
        print("""
🟡 原因1: 验证集数据分布不均匀
   - 验证集可能包含不同类型的样本（如不同的蛋白家族、配体类型）
   - 建议: 检查验证集和训练集的数据分布是否一致
""")
    
    # 原因2: Batch Size 太小
    print("""
🟡 原因2: Batch Size 可能偏小
   - 当前 batch_size=128，验证集 9951 样本 → 78 个 batch
   - 每个 batch 的统计量可能不稳定
   - 建议: 增大 batch_size 或使用更大的验证集
""")
    
    # 原因3: 学习率过高
    if any(abs(d) > 0.015 for d in val_diff):
        print("""
🟠 原因3: 学习率可能过高
   - 学习率过高会导致参数更新不稳定
   - 验证损失出现大的跳跃（如Epoch 35附近）
   - 建议: 降低学习率或使用学习率调度器
""")
    
    # 原因4: 缺乏正则化
    train_val_ratio = train_losses[-1] / val_losses[-1] if val_losses[-1] > 0 else 0
    if train_val_ratio < 0.5 and val_cv > 8:
        print("""
🟠 原因4: 模型可能缺乏足够的正则化
   - 训练损失远低于验证损失 (ratio={train_val_ratio:.2f})
   - 但验证损失波动大，说明泛化能力不稳定
   - 建议: 添加 Dropout、增加 Weight Decay、使用早停
""")
    
    # 原因5: 数据集噪声
    print("""
🟡 原因5: 验证集可能包含噪声样本
   - 部分样本的标签质量较差（如实验测量误差）
   - kcat 数据来自不同数据源，质量不一
   - 建议: 检查验证集中损失最高的样本
""")
    
    # 原因6: 模型容量问题
    print("""
🟡 原因6: 模型容量与任务复杂度不匹配
   - 模型可能过于复杂，拟合了训练集的噪声
   - 或模型不够复杂，无法学习到稳定的特征
   - 建议: 调整模型大小、增加训练数据
""")
    
    # 建议措施
    print("\n" + "=" * 80)
    print("✅ 建议改进措施:")
    print("=" * 80)
    print("""
1. 数据层面:
   - 检查验证集和训练集的数据分布
   - 分析验证集中损失最高的样本特征
   - 考虑增加验证集样本量

2. 训练策略:
   - 降低初始学习率 (当前 lr=1e-4，可尝试 5e-5)
   - 使用学习率调度器 (如 ReduceLROnPlateau)
   - 增大 batch_size (如 256)

3. 正则化:
   - 增加 Weight Decay (当前 1e-5，可尝试 1e-4)
   - 添加 Dropout 层
   - 启用早停策略

4. 模型调整:
   - 尝试不同的隐藏层维度
   - 调整 GNN 层数
   - 考虑使用更稳定的优化器

5. 诊断工具:
   - 绘制验证集每个样本的损失分布
   - 分析不同子集（如不同辅因子类型）的验证损失
   - 检查训练过程中的梯度范数
""")
    
    print("\n" + "=" * 80)
    print("📈 当前训练状态总结:")
    print("=" * 80)
    print(f"""
- 总训练 Epoch: {len(val_losses)}
- 训练损失下降: {((train_losses[0]-train_losses[-1])/train_losses[0]*100):.1f}%
- 验证损失下降: {((val_losses[0]-min(val_losses))/val_losses[0]*100):.1f}%
- 最佳验证损失: {min(val_losses):.4f} (Epoch {np.argmin(val_losses)+1})
- 验证损失变异系数: {val_cv:.2f}%
- 训练/验证损失比率: {train_val_ratio:.3f}
- 状态评估: {'训练正常，但验证波动较大' if val_cv > 8 else '训练稳定'}
""")
    
    # 绘制波动分析图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 验证损失曲线
    axes[0].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    axes[0].axhline(y=val_mean, color='orange', linestyle='--', label=f'Mean ({val_mean:.4f})')
    axes[0].axhline(y=val_mean + 2*val_std, color='red', linestyle='--', label=f'+2σ ({(val_mean+2*val_std):.4f})')
    axes[0].axhline(y=val_mean - 2*val_std, color='green', linestyle='--', label=f'-2σ ({(val_mean-2*val_std):.4f})')
    axes[0].fill_between(epochs, val_mean-2*val_std, val_mean+2*val_std, alpha=0.2, color='gray')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Validation Loss with Standard Deviation Bands')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 损失变化率
    axes[1].plot(range(2, len(val_losses)+1), val_diff, 'b-', linewidth=2, label='Val Loss Change')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].axhline(y=0.01, color='red', linestyle='--', label='Threshold (+0.01)')
    axes[1].axhline(y=-0.01, color='green', linestyle='--', label='Threshold (-0.01)')
    axes[1].fill_between(range(2, len(val_losses)+1), -0.01, 0.01, alpha=0.2, color='gray')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Change')
    axes[1].set_title('Validation Loss Change Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = CHECKPOINT_DIR / "validation_fluctuation_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n诊断分析图已保存到：{save_path}")
    
    return {
        'val_mean': val_mean,
        'val_std': val_std,
        'val_cv': val_cv,
        'high_points': high_points,
        'large_fluctuations': large_fluctuations
    }

if __name__ == "__main__":
    analyze_validation_fluctuation()