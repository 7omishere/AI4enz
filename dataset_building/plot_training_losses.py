#!/usr/bin/env python3
"""
绘制完整的训练损失曲线
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 150

def plot_training_losses():
    """绘制训练和验证损失曲线"""
    
    # 加载 checkpoint
    checkpoint = torch.load(CHECKPOINT_DIR / "best.ckpt", map_location='cpu', weights_only=False)
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图 1：训练和验证损失曲线
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', linewidth=2.5, label='train loss', marker='o', markersize=5, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2.5, label='val loss', marker='s', markersize=5, alpha=0.8)    
    
    # 标记最佳验证损失点
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
                label=f'Best Epoch ({best_epoch})')
    ax1.scatter([best_epoch], [best_val_loss], c='green', s=120, zorder=5, 
                edgecolor='white', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=14, fontweight='bold')
    ax1.set_title('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    ax1.tick_params(axis='both', labelsize=12)
    
    # 图 2：训练/验证损失比率（过拟合检测）
    ax2 = axes[1]
    ratio = [t/v for t, v in zip(train_losses, val_losses)]
    ax2.plot(epochs, ratio, 'm-', linewidth=2.5, marker='^', markersize=5, alpha=0.8)
    
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Ideal State Ratio (1.0)')
    ax2.axhline(y=1.2, color='orange', linestyle='--', linewidth=2, label='Overfitting Threshold Ratio (1.2)')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Severe Overfitting Threshold Ratio (2.0)')
    
    ax2.fill_between(epochs, 0, 1.2, alpha=0.15, color='green', label='Normal Region')
    ax2.fill_between(epochs, 1.2, 2.0, alpha=0.15, color='orange', label='Minor Overfitting Region')
    ax2.fill_between(epochs, 2.0, max(ratio)*1.1, alpha=0.15, color='red', label='Severe Overfitting Region')
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Training/Validation Loss Ratio', fontsize=14, fontweight='bold')
    ax2.set_title('Overfitting Detection', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    ax2.set_ylim(bottom=0, top=min(max(ratio)*1.1, 3.0))
    ax2.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = CHECKPOINT_DIR / "training_losses_detailed.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training losses curves saved to: {save_path}")
    
    # 显示统计信息
    print("\n" + "=" * 80)
    print("Training Loss Statistics Analysis")
    print("=" * 80)
    print(f"Total Training Epochs: {len(train_losses)}")
    print(f"Initial Training Loss: {train_losses[0]:.4f}")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Training Loss Decrease: {(train_losses[0]-train_losses[-1])/train_losses[0]*100:.1f}%")
    print(f"\nInitial Validation Loss: {val_losses[0]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Validation Loss Decrease: {(val_losses[0]-best_val_loss)/val_losses[0]*100:.1f}%")
    
    # 过拟合分析
    final_ratio = train_losses[-1] / val_losses[-1]
    print(f"\nFinal Training/Validation Loss Ratio: {final_ratio:.3f}")
    if final_ratio < 1.2:
        print("✓ Model is in normal training state (no significant overfitting)")
    elif final_ratio < 2.0:
        print("⚠️ Minor overfitting signs")
    else:
        print("⚠️ Severe overfitting")
    
    # 收敛分析
    loss_diff = train_losses[-5:]
    avg_change = np.mean(np.diff(loss_diff))
    print(f"\nLast 5 Epoch Training Loss Average Change: {avg_change:.6f}")
    if abs(avg_change) < 0.0005:
        print("✓ Training loss is stable enough")
    else:
        print("⚠️ Training loss is still decreasing, can continue training further")
    
    print("=" * 80)

if __name__ == "__main__":
    plot_training_losses()
