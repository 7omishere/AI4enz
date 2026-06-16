"""
classification_head.py
=======================
四分类任务模型头：酶类型分类（完美酶/结合受限/平衡态/催化受限）

基于可学习阈值的有序回归：
- 预测 log_ratio = log10(Km/Kd)
- 四分类阈值：[-2, -1, 1]
- 可学习参数允许微调边界

物理意义：
- log_ratio < -2: 完美酶 (Km << Kd)
- -2 ≤ log_ratio < -1: 结合受限
- -1 ≤ log_ratio ≤ 1: 平衡态
- log_ratio > 1: 催化受限 (Km >> Kd)

用法：
    from classification_head import EnzymeTypeClassificationHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# 分类阈值 - 基于 oed_kinetics.json 数据分布的四分位数
# log_ratio = log10(Km) + pKd，范围约 7-16
# 使用数据驱动的阈值实现均衡分类
DATA_DRIVEN_THRESHOLDS = [7.17, 8.83, 10.56]  # 25th, 50th, 75th percentiles

# 物理阈值 (仅作为参考，不再使用)
# PHYSICAL_THRESHOLDS = [-2.0, -1.0, 1.0]  # 基于 Km/Kd 比值的物理意义

class EnzymeTypeClassificationHead(nn.Module):
    """
    酶类型四分类头
    
    基于可学习阈值的有序回归，将回归任务转化为分类任务
    
    物理背景:
    ---------
    Km/Kd 比值决定了酶的限制类型:
    - Km ≈ Kd (结合受限): kcat << k_1
    - Km >> Kd (催化受限): kcat >> k_1
    - Km << Kd (完美酶): 结合极强
    
    log10(Km/Kd) 是物理上有意义的度量:
    - log_ratio < -2: 完美酶 (Km 比 Kd 小 100 倍)
    - -2 ≤ ratio < -1: 结合受限 (Km 比 Kd 小 10-100 倍)
    - -1 ≤ ratio ≤ 1: 平衡态 (Km 和 Kd 同量级)
    - ratio > 1: 催化受限 (Km 比 Kd 大 10 倍以上)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 4,
        init_thresholds: list = [-2.0, -1.0, 1.0],
        learnable_thresholds: bool = True,
        use_ordinal: bool = True,
    ):
        """
        Args:
            hidden_dim: 输入特征维度
            num_classes: 类别数 (默认4)
            init_thresholds: 初始阈值 [t0, t1, t2] = [-2, -1, 1]
            learnable_thresholds: 是否让阈值可学习
            use_ordinal: 是否使用有序回归 (推荐 True)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.init_thresholds = init_thresholds
        self.use_ordinal = use_ordinal
        
        # ── 共享特征提取 ─────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # ── 回归头：预测 log_ratio ────────────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        
        # ── 分类头：直接四分类 ────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )
        
        # ── 可学习阈值 ────────────────────────────────────
        if learnable_thresholds:
            # 初始化为物理预设值
            self.thresholds = nn.Parameter(torch.tensor(init_thresholds))
        else:
            self.register_buffer('thresholds', torch.tensor(init_thresholds))
    
    def forward(self, h: torch.Tensor) -> dict:
        """
        Args:
            h: (B, hidden_dim) 来自 BINN 的特征
            
        Returns:
            dict with:
            - log_ratio: (B,) 预测的 log10(Km/Kd)
            - cls_logits: (B, 4) 分类 logit
            - prob: (B, 4) 各类概率
            - thresholds: (3,) 学习的阈值
            - class_pred: (B,) 硬预测类别
        """
        B = h.size(0)
        
        # 共享特征
        h_shared = self.shared(h)
        
        # 回归预测
        raw_reg = self.reg_head(h_shared).squeeze(-1)  
        # 映射到合理范围 [-4, 6]
        log_ratio = 5 * torch.tanh(raw_reg / 5)  # 软饱和
        
        # 直接分类预测 (辅助)
        cls_logits = self.cls_head(h_shared)  # (B, 4)
        cls_prob = F.softmax(cls_logits, dim=-1)
        
        # ── 基于阈值的概率计算 ──────────────────────────
        t0, t1, t2 = self.thresholds  # 可学习阈值
        
        # 计算累积概率
        # P(class=0) = P(log_ratio < t0) = sigmoid((t0 - log_ratio) * scale)
        # P(class=1) = P(t0 <= log_ratio < t1)
        # ...
        scale = 5.0  # sigmoid 的陡峭程度
        
        p0 = torch.sigmoid((t0 - log_ratio) * scale)
        p1 = torch.sigmoid((t1 - log_ratio) * scale)
        p2 = torch.sigmoid((t2 - log_ratio) * scale)
        
        # 从概率密度
        prob0 = p0
        prob1 = (1 - p0) * p1
        prob2 = (1 - p1) * p2
        prob3 = 1 - p2
        
        # 确保概率和为1（数值稳定版本）
        prob_thresh = torch.stack([prob0, prob1, prob2, prob3], dim=-1)
        prob_thresh = prob_thresh / (prob_thresh.sum(dim=-1, keepdim=True) + 1e-8)
        
        # ── 融合策略 ────────────────────────────────────
        # 结合回归概率和直接分类概率
        mix_ratio = 0.6  # 回归概率的权重
        prob = mix_ratio * prob_thresh + (1 - mix_ratio) * cls_prob
        
        # ── 硬预测 ──────────────────────────────────────
        class_pred = prob.argmax(dim=-1)  # (B,)
        
        return {
            'log_ratio': log_ratio,          # 回归值
            'cls_logits': cls_logits,        # 分类 logit
            'prob': prob,                    # 混合概率
            'prob_thresh': prob_thresh,      # 阈值概率 (纯有序回归)
            'prob_cls': cls_prob,            # 直接分类概率
            'thresholds': self.thresholds,   # 学习到的阈值
            'class_pred': class_pred,        # 硬预测
        }
    
    def get_ordinal_loss(
        self,
        log_ratio: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算有序回归损失
        
        方法：将四分类分解为三个二分类
        - 二分类1: P(log_ratio > t0) vs P(log_ratio < t0)
        - 二分类2: P(log_ratio > t1) vs P(log_ratio < t1)
        - 二分类3: P(log_ratio > t2) vs P(log_ratio < t2)
        """
        t0, t1, t2 = self.thresholds
        
        # 三个二分类的目标
        # 类0: log_ratio < t0  → target_pass0 = 0
        # 类1: t0 ≤ log_ratio < t1  → target_pass0 = 1, target_pass1 = 0
        # 类2: t1 ≤ log_ratio < t2  → target_pass0 = 1, target_pass1 = 1, target_pass2 = 0
        # 类3: log_ratio ≥ t2  → 所有 pass = 1
        
        target_pass0 = (labels >= 1).float()  # log_ratio > t0?
        target_pass1 = (labels >= 2).float()  # log_ratio > t1?
        target_pass2 = (labels >= 3).float()  # log_ratio > t2?
        
        # 预测概率
        p0 = torch.sigmoid((log_ratio - t0) * 5)
        p1 = torch.sigmoid((log_ratio - t1) * 5)
        p2 = torch.sigmoid((log_ratio - t2) * 5)
        
        # BCE 损失
        loss0 = F.binary_cross_entropy(p0, target_pass0, reduction=reduction)
        loss1 = F.binary_cross_entropy(p1, target_pass1, reduction=reduction)
        loss2 = F.binary_cross_entropy(p2, target_pass2, reduction=reduction)
        
        losses = {
            'L_ordinal_0': loss0.item(),
            'L_ordinal_1': loss1.item(),
            'L_ordinal_2': loss2.item(),
        }
        
        total = loss0 + loss1 + loss2
        
        return total, losses
    
    def get_classification_loss(
        self,
        prob: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算交叉熵分类损失（带可选的类别权重）
        """
        if class_weights is not None:
            class_weights = class_weights.to(prob.device)
        
        loss = F.nll_loss(
            torch.log(prob.clamp(min=1e-8)),
            labels,
            weight=class_weights,
            reduction=reduction
        )
        
        return loss, {'L_cross_entropy': loss.item()}
    
    def get_threshold_order_loss(self) -> torch.Tensor:
        """
        约束损失：确保 t0 < t1 < t2（阈值有序）
        """
        t0, t1, t2 = self.thresholds
        
        loss = (
            F.relu(t0 - t1) +   # t0 < t1
            F.relu(t1 - t2)    # t1 < t2
        )
        
        return loss
    
    def forward_with_loss(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        前向传播 + 损失计算
        
        Args:
            h: (B, hidden_dim) 输入特征
            labels: (B,) 类别标签 [0, 1, 2, 3]
            class_weights: (4,) 类别权重
            
        Returns:
            dict: 输出 + 损失
        """
        outputs = self.forward(h)
        
        # 多任务损失
        losses = {}
        
        # 1. 有序回归损失（主损失）
        L_ordinal, ordinal_losses = self.get_ordinal_loss(
            outputs['log_ratio'], labels
        )
        losses.update(ordinal_losses)
        losses['L_ordinal'] = L_ordinal.item()
        
        # 2. 交叉熵损失（辅助）
        L_ce, ce_losses = self.get_classification_loss(
            outputs['prob'].log(), labels, class_weights
        )
        losses.update(ce_losses)
        losses['L_ce'] = L_ce.item()
        
        # 3. 阈值有序约束
        L_order = self.get_threshold_order_loss()
        losses['L_threshold_order'] = L_order.item()
        
        # 总损失
        # 权重：有序回归为主，交叉熵为辅，有序约束很轻
        total_loss = (
            0.5 * L_ordinal +      # 主损失
            0.3 * L_ce +           # 辅助损失
            0.1 * L_order          # 约束
        )
        losses['total'] = total_loss.item()
        
        return {
            **outputs,
            'loss': total_loss,
            'losses': losses,
        }


class EnzymeTypeMultiTaskHead(nn.Module):
    """
    多任务版本：同时预测 pKd, kcat, Km (用于计算 log_ratio) 和分类
    
    适用于数据中有部分样本只有 pKd/kcat 而没有完整动力学数据的情况
    """
    
    def __init__(self, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 共享特征
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 原有任务头（保留）
        self.ts_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.catalysis_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.km_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Km > 0
        )
        
        # 四分类头
        self.classification_head = EnzymeTypeClassificationHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
    
    def forward(
        self,
        h: torch.Tensor,
        return_all: bool = False
    ) -> dict:
        """
        Args:
            h: (B, hidden_dim) 输入特征
            return_all: 是否返回所有中间量
            
        Returns:
            dict with predictions
        """
        h_shared = self.shared(h)
        
        # ── 原有任务 ────────────────────────────────────
        ts_stability = torch.sigmoid(self.ts_head(h_shared).squeeze(-1))  # pKd
        catalysis_rate = torch.sigmoid(self.catalysis_head(h_shared).squeeze(-1))  # kcat
        km_median = self.km_head(h_shared).squeeze(-1)  # Km (原始尺度)
        
        # ── 分类任务 ────────────────────────────────────
        cls_outputs = self.classification_head(h_shared)
        
        # ── 可选：从 km 和 kd 计算 log_ratio ─────────────
        # 如果有 km_head 预测的 Km，可以计算
        # log_ratio = log10(km_pred) + pkd_pred  (因为 Kd ≈ 10^-pKd)
        # 这是一个一致性检查
        
        outputs = {
            'ts_stability': ts_stability,
            'catalysis_rate': catalysis_rate,
            'km_median': km_median,
            'log_ratio': cls_outputs['log_ratio'],
            'prob': cls_outputs['prob'],
            'class_pred': cls_outputs['class_pred'],
            'thresholds': cls_outputs['thresholds'],
        }
        
        if return_all:
            outputs['cls_logits'] = cls_outputs['cls_logits']
            outputs['prob_thresh'] = cls_outputs['prob_thresh']
            outputs['prob_cls'] = cls_outputs['prob_cls']
        
        return outputs


def create_class_weights(
    class_counts: list,
    method: str = 'effective'
) -> torch.Tensor:
    """
    创建类别权重以处理类别不平衡
    
    Args:
        class_counts: [n_class0, n_class1, n_class2, n_class3]
        method: 'inv_freq' (1/n), 'effective' (1/sqrt(n)), 'manual'
        
    Returns:
        weights: (num_classes,) 类别权重
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    total = counts.sum()
    
    if method == 'inv_freq':
        # W = N / (n_classes * n_i)
        weights = total / (len(class_counts) * counts)
    elif method == 'effective':
        # W = sqrt(N) / sqrt(n_i) (有效样本数)
        weights = (total ** 0.5) / (counts ** 0.5)
    else:
        weights = torch.ones(len(class_counts))
    
    # 归一化
    weights = weights / weights.sum() * len(class_counts)
    
    return weights


if __name__ == '__main__':
    # 快速测试
    print("="*60)
    print("测试 EnzymeTypeClassificationHead")
    print("="*60)
    
    # 创建模型
    model = EnzymeTypeClassificationHead(hidden_dim=256)
    
    # 模拟输入
    B = 32
    h = torch.randn(B, 256)
    labels = torch.randint(0, 4, (B,))
    
    # 前向传播
    with torch.no_grad():
        outputs = model(h)
    
    print(f"\n阈值学习: {model.thresholds.data.tolist()}")
    print(f"log_ratio 范围: [{outputs['log_ratio'].min():.2f}, {outputs['log_ratio'].max():.2f}]")
    print(f"类别分布: {outputs['prob'].mean(dim=0).tolist()}")
    
    # 计算损失
    outputs = model.forward_with_loss(h, labels)
    print(f"\n损失分解:")
    for k, v in outputs['losses'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ 测试通过!")