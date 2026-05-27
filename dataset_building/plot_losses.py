#!/usr/bin/env python3
"""
绘制训练过程的损失函数曲线
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

def load_training_history():
    """从 checkpoint 加载训练历史"""
    # 尝试从 best.ckpt 或 last.ckpt 加载
    checkpoint_files = ['best.ckpt', 'last.ckpt']
    
    for ckpt_name in checkpoint_files:
        ckpt_path = CHECKPOINT_DIR / ckpt_name
        if ckpt_path.exists():
            print(f"加载 checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            # 检查是否有训练历史
            if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']
                print(f"找到训练历史：{len(train_losses)} 个 epoch")
                return train_losses, val_losses
    
    print("未找到训练历史，尝试从日志文件加载...")
    return None, None

def plot_losses(train_losses, val_losses):
    """绘制损失曲线"""
    if train_losses is None or val_losses is None:
        print("无法加载训练历史")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # 图 1：总损失
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    
    # 标记最佳验证损失点
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', linewidth=1.5, 
                    label=f'Best Epoch {best_epoch}')
    axes[0].scatter([best_epoch], [best_val_loss], c='green', s=100, zorder=5)
    
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=1)
    
    # 图 2：训练/验证损失比率（过拟合检测）
    ratio = [t/v if v > 0 else 0 for t, v in zip(train_losses, val_losses)]
    axes[1].plot(epochs, ratio, 'm-', linewidth=2)
    axes[1].axhline(y=1.0, color='g', linestyle='--', linewidth=1.5, label='Ideal (1.0)')
    axes[1].axhline(y=1.2, color='orange', linestyle='--', linewidth=1.5, 
                    label='Overfitting threshold (1.2)')
    axes[1].fill_between(epochs, 0, 1.2, alpha=0.2, color='green', label='Normal region')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Train/Val Loss Ratio', fontsize=12, fontweight='bold')
    axes[1].set_title('Overfitting Detection', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=1)
    axes[1].set_ylim(bottom=0)
    
    # 图 3：损失变化率
    if len(train_losses) > 1:
        train_diff = [train_losses[i] - train_losses[i-1] for i in range(1, len(train_losses))]
        val_diff = [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]
        diff_epochs = range(2, len(train_losses) + 1)
        
        axes[2].plot(diff_epochs, train_diff, 'b-', linewidth=2, 
                     label='Train Loss Change', alpha=0.7)
        axes[2].plot(diff_epochs, val_diff, 'r-', linewidth=2, 
                     label='Val Loss Change', alpha=0.7)
        axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[2].fill_between(diff_epochs, 0, train_diff, alpha=0.3, color='blue')
        axes[2].fill_between(diff_epochs, 0, val_diff, alpha=0.3, color='red')
        axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Loss Change', fontsize=12, fontweight='bold')
        axes[2].set_title('Loss Convergence Rate', fontsize=14, fontweight='bold')
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(left=2)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = CHECKPOINT_DIR / "training_losses.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n损失曲线已保存到：{save_path}")
    
    # 显示统计信息
    print("\n" + "=" * 80)
    print("训练统计信息:")
    print("=" * 80)
    print(f"总训练 epoch 数：{len(train_losses)}")
    print(f"最佳验证损失：{best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"初始训练损失：{train_losses[0]:.4f}")
    print(f"最终训练损失：{train_losses[-1]:.4f}")
    print(f"初始验证损失：{val_losses[0]:.4f}")
    print(f"最终验证损失：{val_losses[-1]:.4f}")
    
    # 过拟合分析
    final_ratio = train_losses[-1] / val_losses[-1] if val_losses[-1] > 0 else 0
    print(f"\n最终训练/验证损失比率：{final_ratio:.3f}")
    if final_ratio < 1.2:
        print("✓ 模型处于正常训练状态（无明显过拟合）")
    elif final_ratio < 2.0:
        print("⚠️ 模型有轻微过拟合迹象")
    else:
        print("⚠️ 模型存在严重过拟合")
    
    print("=" * 80)
    
    plt.show()

if __name__ == "__main__":
    train_losses, val_losses = load_training_history()
    plot_losses(train_losses, val_losses)
