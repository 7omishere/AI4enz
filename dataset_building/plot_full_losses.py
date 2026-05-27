#!/usr/bin/env python3
"""
从多个checkpoint中提取完整的训练历史并绘制
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 150

def extract_full_training_history():
    """从多个checkpoint提取完整的训练历史"""
    
    print("=" * 80)
    print("从多个checkpoint提取完整训练历史")
    print("=" * 80)
    
    # 收集所有checkpoint
    checkpoint_files = []
    
    # 添加 last.ckpt
    last_ckpt = CHECKPOINT_DIR / "last.ckpt"
    if last_ckpt.exists():
        checkpoint_files.append(('last', last_ckpt))
    
    # 添加定期保存的checkpoint
    for f in sorted(CHECKPOINT_DIR.glob("epoch_*.ckpt")):
        epoch_num = int(f.stem.split('_')[1])
        checkpoint_files.append((epoch_num, f))
    
    # 按epoch排序
    checkpoint_files.sort(key=lambda x: x[0] if isinstance(x[0], int) else 999)
    
    print(f"\nFound {len(checkpoint_files)} checkpoints:")
    for name, path in checkpoint_files:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', 'N/A')
        train_len = len(ckpt.get('train_losses', []))
        val_len = len(ckpt.get('val_losses', []))
        print(f"  {name}: epoch={epoch}, train_losses={train_len}, val_losses={val_len}")
    
    # 尝试从定期保存的checkpoint中提取完整历史
    full_train_losses = []
    full_val_losses = []
    
    # 方法1：从epoch_60.ckpt获取（如果有完整历史）
    epoch_60_ckpt = CHECKPOINT_DIR / "epoch_0060.ckpt"
    if epoch_60_ckpt.exists():
        ckpt = torch.load(epoch_60_ckpt, map_location='cpu', weights_only=False)
        if 'train_losses' in ckpt and len(ckpt['train_losses']) > 42:
            print(f"\nExtract full training history from epoch_0060.ckpt")
            full_train_losses = ckpt['train_losses']
            full_val_losses = ckpt['val_losses']
    
    # 方法2：从best.ckpt获取前42个，然后从last.ckpt获取剩余（如果有）
    if len(full_train_losses) < 64:
        best_ckpt = torch.load(CHECKPOINT_DIR / "best.ckpt", map_location='cpu', weights_only=False)
        last_ckpt = torch.load(CHECKPOINT_DIR / "last.ckpt", map_location='cpu', weights_only=False)
        
        if 'train_losses' in best_ckpt and 'train_losses' in last_ckpt:
            print(f"\n合并 best.ckpt 和 last.ckpt 的历史")
            full_train_losses = best_ckpt['train_losses'] + last_ckpt['train_losses'][len(best_ckpt['train_losses']):]
            full_val_losses = best_ckpt['val_losses'] + last_ckpt['val_losses'][len(best_ckpt['val_losses']):]
    
    # 方法3：如果以上都失败，只绘制前42个
    if len(full_train_losses) < 42:
        best_ckpt = torch.load(CHECKPOINT_DIR / "best.ckpt", map_location='cpu', weights_only=False)
        full_train_losses = best_ckpt['train_losses']
        full_val_losses = best_ckpt['val_losses']
        print(f"\n只能使用 best.ckpt 的历史（前42个epoch）")
    
    print(f"\n最终提取到 {len(full_train_losses)} 个epoch的训练历史")
    
    return full_train_losses, full_val_losses

def plot_full_training_history(train_losses, val_losses):
    """绘制完整的训练历史"""
    
    epochs = range(1, len(train_losses) + 1)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 图 1：训练和验证损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_losses, 'b-', linewidth=2.5, label='Train Loss', marker='o', markersize=3, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2.5, label='Val Loss', marker='s', markersize=3, alpha=0.8)
    
    # 标记最佳验证损失点
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
                label=f'Best Epoch ({best_epoch})')
    ax1.scatter([best_epoch], [best_val_loss], c='green', s=120, zorder=5, 
                edgecolor='white', linewidth=2)
    
    # 标记训练结束点
    ax1.axvline(x=len(train_losses), color='orange', linestyle='--', linewidth=2, 
                label=f'Final Epoch ({len(train_losses)})')
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title(f'Training & Validation Loss ({len(train_losses)} Epochs)', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    ax1.tick_params(axis='both', labelsize=12)
    
    # 图 2：训练/验证损失比率
    ax2 = fig.add_subplot(gs[0, 1])
    ratio = [t/v for t, v in zip(train_losses, val_losses)]
    ax2.plot(epochs, ratio, 'm-', linewidth=2.5, marker='^', markersize=3, alpha=0.8)
    
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Ideal (1.0)')
    ax2.axhline(y=1.2, color='orange', linestyle='--', linewidth=2, label='Overfitting (1.2)')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Severe (2.0)')
    
    ax2.fill_between(epochs, 0, 1.2, alpha=0.15, color='green')
    ax2.fill_between(epochs, 1.2, 2.0, alpha=0.15, color='orange')
    ax2.fill_between(epochs, 2.0, max(ratio)*1.1, alpha=0.15, color='red')
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Train/Val Loss Ratio', fontsize=14, fontweight='bold')
    ax2.set_title('Overfitting Detection', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    ax2.set_ylim(bottom=0, top=min(max(ratio)*1.1, 2.5))
    ax2.tick_params(axis='both', labelsize=12)
    
    # 图 3：损失变化率
    ax3 = fig.add_subplot(gs[1, 0])
    if len(train_losses) > 1:
        train_diff = np.diff(train_losses)
        val_diff = np.diff(val_losses)
        diff_epochs = range(2, len(train_losses) + 1)
        
        ax3.plot(diff_epochs, train_diff, 'b-', linewidth=2, label='Train Loss Change', alpha=0.7)
        ax3.plot(diff_epochs, val_diff, 'r-', linewidth=2, label='Val Loss Change', alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax3.fill_between(diff_epochs, 0, train_diff, alpha=0.2, color='blue')
        ax3.fill_between(diff_epochs, 0, val_diff, alpha=0.2, color='red')
        
        ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Loss Change', fontsize=14, fontweight='bold')
        ax3.set_title('Loss Convergence Rate', fontsize=16, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=2)
        ax3.tick_params(axis='both', labelsize=12)
    
    # 图 4：训练进度总结
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    train_drop = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    val_drop = (val_losses[0] - best_val_loss) / val_losses[0] * 100
    final_ratio = train_losses[-1] / val_losses[-1]
    
    recent_train_change = np.mean(np.diff(train_losses[-10:])) if len(train_losses) >= 10 else 0
    recent_val_change = np.mean(np.diff(val_losses[-10:])) if len(val_losses) >= 10 else 0
    
    summary_text = f"""
Training Summary ({len(train_losses)} Epochs)
{'='*50}

Loss Metrics:
  Initial Train Loss: {train_losses[0]:.4f}
  Final Train Loss:   {train_losses[-1]:.4f}
  Train Loss Drop:    {train_drop:.1f}%

  Initial Val Loss:   {val_losses[0]:.4f}
  Best Val Loss:      {best_val_loss:.4f} (Epoch {best_epoch})
  Final Val Loss:     {val_losses[-1]:.4f}
  Val Loss Drop:      {val_drop:.1f}%

Overfitting Analysis:
  Train/Val Ratio:    {final_ratio:.3f}
  Status:             {'No Overfitting' if final_ratio < 1.2 else 'Mild Overfitting' if final_ratio < 2.0 else 'Severe Overfitting'}

Convergence Status:
  Recent Train Change: {recent_train_change:.6f}
  Recent Val Change:   {recent_val_change:.6f}
  Status:              {'Converged' if abs(recent_train_change) < 0.0005 and abs(recent_val_change) < 0.0005 else 'Still Learning'}

Best Model: Epoch {best_epoch}
Final Model: Epoch {len(train_losses)}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    save_path = CHECKPOINT_DIR / "training_losses_full.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n完整损失曲线已保存到：{save_path}")
    
    print("\n" + "=" * 80)
    print("训练损失统计分析")
    print("=" * 80)
    print(f"总训练 Epoch 数：{len(train_losses)}")
    print(f"初始训练损失：{train_losses[0]:.4f}")
    print(f"最终训练损失：{train_losses[-1]:.4f}")
    print(f"训练损失下降幅度：{train_drop:.1f}%")
    print(f"\n初始验证损失：{val_losses[0]:.4f}")
    print(f"最终验证损失：{val_losses[-1]:.4f}")
    print(f"最佳验证损失：{best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"验证损失下降幅度：{val_drop:.1f}%")
    
    print(f"\n最终训练/验证损失比率：{final_ratio:.3f}")
    if final_ratio < 1.2:
        print("Model status: No overfitting")
    elif final_ratio < 2.0:
        print("Model status: Mild overfitting")
    else:
        print("Model status: Severe overfitting")
    
    print(f"\n最近 10 个 Epoch 训练损失平均变化：{recent_train_change:.6f}")
    print(f"最近 10 个 Epoch 验证损失平均变化：{recent_val_change:.6f}")
    if abs(recent_train_change) < 0.0005 and abs(recent_val_change) < 0.0005:
        print("Convergence status: Converged")
    else:
        print("Convergence status: Still learning")
    
    print("=" * 80)

if __name__ == "__main__":
    train_losses, val_losses = extract_full_training_history()
    plot_full_training_history(train_losses, val_losses)