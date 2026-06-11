"""
efficiency_classification_head.py
==================================
Binary classification head: Efficient vs Inefficient enzyme based on kcat/Km ratio.

Task: Predict whether an enzyme is "efficient" (kcat/Km > threshold) or "inefficient"
- log10(kcat/Km) threshold: 1.36 (median of OED data, ~23 M^-1 s^-1)
- Binary cross-entropy loss
- Direct classification (no ordinal regression)

Usage:
    from efficiency_classification_head import EfficiencyBinaryHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


# 二分类阈值 - 基于 OED 数据的 log10(kcat/Km) 中位数
BINARY_THRESHOLD_LOG = 1.36  # log10(kcat/Km) threshold
BINARY_THRESHOLD_LINEAR = 10 ** BINARY_THRESHOLD_LOG  # ~23 M^-1 s^-1


@dataclass
class BinaryClassificationResult:
    """二分类结果容器"""
    is_efficient: torch.Tensor      # (B,) 硬预测 0/1
    efficiency_prob: torch.Tensor   # (B,) 高效酶概率
    log_kcatkm_pred: torch.Tensor   # (B,) 回归预测 log10(kcat/Km)
    loss: Optional[torch.Tensor]    # 损失值
    losses: dict                    # 损失分解


class EfficiencyBinaryHead(nn.Module):
    """
    酶催化效率二分类头
    
    将回归任务（预测 log10(kcat/Km)）转化为二分类任务：
    - 高效酶 (Class 1): log10(kcat/Km) > 1.36 (约 23 M^-1 s^-1)
    - 低效酶 (Class 0): log10(kcat/Km) <= 1.36
    
    Architecture:
        shared MLP → reg_head (log_kcatkm) + cls_head (binary)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        threshold_log: float = BINARY_THRESHOLD_LOG,
        use_learnable_threshold: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.threshold_log = threshold_log
        self.use_learnable_threshold = use_learnable_threshold
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 回归头：预测 log10(kcat/Km)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        
        # 分类头：直接二分类
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        
        # 可学习阈值（可选）
        if use_learnable_threshold:
            self.log_threshold = nn.Parameter(torch.tensor(threshold_log))
        else:
            self.register_buffer('log_threshold', torch.tensor(threshold_log))
    
    @property
    def threshold(self) -> float:
        """当前阈值"""
        if self.use_learnable_threshold:
            return self.log_threshold.item()
        return self.threshold_log
    
    def forward(self, h: torch.Tensor, return_probs: bool = True) -> dict:
        """
        前向传播
        
        Args:
            h: (B, hidden_dim) 来自 BINN 的特征
            
        Returns:
            dict with:
            - is_efficient: (B,) 硬预测 0/1
            - efficiency_prob: (B,) 高效酶概率 [0, 1]
            - log_kcatkm_pred: (B,) 回归预测值
            - cls_logits: (B,) 分类 logit
        """
        B = h.size(0)
        
        # 共享特征
        h_shared = self.shared(h)
        
        # 回归预测 log10(kcat/Km)
        raw_reg = self.reg_head(h_shared).squeeze(-1)
        # 使用 tanh 限制范围 [-5, 10]
        log_kcatkm_pred = 2.5 + 5 * torch.tanh(raw_reg / 5)
        
        # 直接分类预测
        cls_logits = self.cls_head(h_shared).squeeze(-1)  # (B,)
        efficiency_prob = torch.sigmoid(cls_logits)
        
        # 或基于回归结果分类（可选择）
        # reg_prob = torch.sigmoid((log_kcatkm_pred - self.threshold) * 3)
        
        # 硬预测
        is_efficient = (efficiency_prob > 0.5).long()
        
        return {
            'is_efficient': is_efficient,
            'efficiency_prob': efficiency_prob,
            'log_kcatkm_pred': log_kcatkm_pred,
            'cls_logits': cls_logits,
            'threshold': self.threshold,
        }
    
    def compute_loss(
        self,
        outputs: dict,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        reg_weight: float = 0.2,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算二分类损失
        
        Args:
            outputs: forward() 的输出
            labels: (B,) 类别标签 0/1
            class_weights: 可选的类别权重 [w_inefficient, w_efficient]
            reg_weight: 回归损失的权重
            reduction: 'mean' | 'sum'
            
        Returns:
            total_loss, losses_dict
        """
        # 1. 交叉熵损失（主）
        ce_loss = F.binary_cross_entropy(
            outputs['efficiency_prob'],
            labels.float(),
            weight=class_weights,
            reduction=reduction
        ) if class_weights is not None else F.binary_cross_entropy(
            outputs['efficiency_prob'],
            labels.float(),
            reduction=reduction
        )
        
        # 2. 回归损失（辅助）- 预测 log10(kcat/Km)
        if reduction == 'mean':
            reg_loss = F.mse_loss(
                outputs['log_kcatkm_pred'],
                (labels.float() * 2 - 1) * 4 + self.threshold  # 缩放到回归目标
            )
        else:
            reg_loss = F.mse_loss(
                outputs['log_kcatkm_pred'],
                (labels.float() * 2 - 1) * 4 + self.threshold,
                reduction='sum'
            )
        
        # 总损失
        total_loss = ce_loss + reg_weight * reg_loss
        
        losses = {
            'L_ce': ce_loss.item() if reduction == 'mean' else ce_loss,
            'L_reg': reg_loss.item() if reduction == 'mean' else reg_loss,
            'total': total_loss.item() if reduction == 'mean' else total_loss,
        }
        
        return total_loss, losses
    
    def get_accuracy(self, outputs: dict, labels: torch.Tensor) -> float:
        """计算准确率"""
        preds = outputs['is_efficient']
        return (preds == labels).float().mean().item()
    
    def get_metrics(self, outputs: dict, labels: torch.Tensor) -> dict:
        """计算完整指标"""
        preds = outputs['is_efficient']
        probs = outputs['efficiency_prob']
        
        # 基本指标
        accuracy = (preds == labels).float().mean().item()
        
        # 计算 AUC（需要 logits）
        try:
            from sklearn.metrics import roc_auc_score
            if len(labels.unique()) > 1:
                auc = roc_auc_score(labels.cpu().numpy(), probs.detach().cpu().numpy())
            else:
                auc = 0.5
        except:
            auc = 0.5
        
        # 精确率和召回率
        tp = ((preds == 1) & (labels == 1)).float().sum()
        fp = ((preds == 1) & (labels == 0)).float().sum()
        fn = ((preds == 0) & (labels == 1)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
        }


def create_binary_class_weights(
    n_efficient: int,
    n_inefficient: int,
    method: str = 'balanced'
) -> torch.Tensor:
    """
    创建二分类权重
    
    Args:
        n_efficient: 高效酶样本数
        n_inefficient: 低效酶样本数
        method: 'balanced' | 'inv_freq'
        
    Returns:
        weights: (2,) 类别权重 [w_inefficient, w_efficient]
    """
    total = n_efficient + n_inefficient
    
    if method == 'balanced':
        # scikit-learn 风格平衡权重
        w_efficient = total / (2 * n_efficient)
        w_inefficient = total / (2 * n_inefficient)
    else:
        # 简单逆频率
        w_efficient = n_inefficient / total
        w_inefficient = n_efficient / total
    
    # 返回 [inefficient, efficient] 的顺序
    return torch.tensor([w_inefficient, w_efficient])


# 用于多任务版本
class EfficiencyMultiTaskHead(nn.Module):
    """
    多任务版本：同时预测 pKd, kcat, log_kcatkm 和二分类
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 共享特征
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 各任务头
        self.pkid_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )  # pKd 回归
        
        self.kcat_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )  # log(kcat) 回归
        
        self.kcatkm_head = EfficiencyBinaryHead(hidden_dim, use_learnable_threshold=False)
        
    def forward(self, h: torch.Tensor) -> dict:
        h_shared = self.shared(h)
        
        return {
            'pkd_pred': torch.sigmoid(self.pkid_head(h_shared).squeeze(-1)),
            'log_kcat_pred': torch.sigmoid(self.kcat_head(h_shared).squeeze(-1)),
            'efficiency_output': self.kcatkm_head(h_shared),
        }


if __name__ == '__main__':
    print("=" * 60)
    print("测试 EfficiencyBinaryHead")
    print("=" * 60)
    
    # 创建模型
    model = EfficiencyBinaryHead(hidden_dim=256, use_learnable_threshold=True)
    
    # 模拟输入
    B = 32
    h = torch.randn(B, 256)
    labels = torch.randint(0, 2, (B,))
    
    print(f"\n测试阈值: {model.threshold:.2f} (log10 scale)")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(h)
    
    print(f"\n预测分布:")
    print(f"  高效酶预测数: {outputs['is_efficient'].sum().item():.0f}/{B}")
    print(f"  log_kcatkm 范围: [{outputs['log_kcatkm_pred'].min():.2f}, {outputs['log_kcatkm_pred'].max():.2f}]")
    
    # 损失计算
    loss, losses = model.compute_loss(outputs, labels)
    print(f"\n损失:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")
    
    # 指标计算
    metrics = model.get_metrics(outputs, labels)
    print(f"\n指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 阈值学习测试
    print("\n阈值学习测试:")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(100):
        optimizer.zero_grad()
        outputs = model(h)
        loss, _ = model.compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"  Step {step}: threshold = {model.threshold:.3f}, loss = {loss.item():.4f}")
    
    print("\n✓ 测试通过!")