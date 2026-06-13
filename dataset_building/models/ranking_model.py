# -*- coding: utf-8 -*-
"""
ranking_model.py
================
酶挖掘排序模型：基于蛋白序列嵌入 + 配体分子图 + 辅因子类型预测底物结合亲和力 (pKd)。

架构（精简版 v2）：
  LigandEncoder (GATv2×3) + ProteinEncoder (ESM-2 序列直通) + CofactorEncoder
  → LatentPathwayBINN (Neural ODE 多步特征变换 + 门控)
  → TrenzitionCatalysisHead (ts_stability + catalysis_rate)  [回归模式]
      或
  → EnzymeTypeClassificationHead (四分类) [分类模式]

精简设计说明：
  - ProteinEncoder 移除结构/口袋/域特征路径（Loss 不约束，纯浪费 GPU 算力）
  - ESM-2 已捕获进化信息和隐式结构倾向，足够支撑催化预测
  - 基于过渡态理论，不依赖显式结构约束

使用模式切换：
  - use_classification=False (默认): 回归模式，预测 pKd 和 kcat
  - use_classification=True: 分类模式，同时预测 pKd/kcat/log_ratio 和四分类

训练目标：
  L_total = L_ts + L_catalysis + 0.01*L_eyring
  - L_ts: 结合亲和力（SmoothL1，归一化目标[0,1]）
  - L_catalysis: 催化效率（SmoothL1，归一化目标[0,1]）
  - L_eyring: Eyring 自洽约束（MSE，权重 0.01）

用法：
  from ranking_model import Trenzition
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool

# 导入分类头
from classification_head import EnzymeTypeClassificationHead, create_class_weights

# ─────────────────────────────────────────────────────────────
# 辅因子域类型顺序（必须与 extract_domains.py:COFACTOR_INDEX 一致！）
# domain_masks 在 proteins.h5 中按此顺序存储 (dim 0..14)
# ─────────────────────────────────────────────────────────────
COFACTOR_DOMAIN_TYPES: list[str] = [
    "NAD", "NADP", "FAD", "FMN", "HEME",
    "FES", "CU", "MPT", "COQ", "PQQ",
    "TPP", "PLP", "COA", "B12", "THF",
]
N_COFACTOR_DOMAIN_TYPES: int = len(COFACTOR_DOMAIN_TYPES)


# ─────────────────────────────────────────────────────────────
# 物理常数
# ─────────────────────────────────────────────────────────────

R_kcal = 1.987e-3          # kcal/(mol·K)
R_kJ   = 8.314e-3          # kJ/(mol·K)
T_ref  = 298.15             # K (25°C)
k_B    = 1.380649e-23       # J/K
h      = 6.62607015e-34     # J·s
kBT_h  = k_B * T_ref / h    # ≈ 6.21e12 s⁻¹  (过渡态理论指前因子)
RT_kcal = R_kcal * T_ref    # ≈ 0.592 kcal/mol

# ΔG° = RT ln(10) · pKd
# pKd = -log10(Kd), Kd in M
DELTA_G_FACTOR = R_kJ * T_ref * math.log(10)  # ≈ 5.71 kJ/mol per pKd unit


# ─────────────────────────────────────────────────────────────
# 辅因子 λ 先验（文献值）
# ─────────────────────────────────────────────────────────────

@dataclass
class CofactorPrior:
    """λ 参考值及其电子转移机制类型"""
    lambda_mean: float   # eV
    lambda_std: float    # eV
    mechanism: str       # 'et' | 'hydride' | 'pcet'
    delta: float = 0.0   # Marcus-Hammond 修正系数 (仅 hydride)
    lambda_p: float = 0.0  # 质子重组能 (仅 pcet)

# fmt: off
COFACTOR_PRIORS: dict[str, CofactorPrior] = {
    # ── 纯电子转移 (et) ──
    "HEME":  CofactorPrior(0.60, 0.15, "et"),
    "FES":   CofactorPrior(0.40, 0.20, "et"),    # Fe-S 簇
    "CU":    CofactorPrior(0.90, 0.30, "et"),    # 蓝铜/铜中心
    "COQ":   CofactorPrior(0.80, 0.15, "et"),    # Coenzyme Q
    "PQQ":   CofactorPrior(0.80, 0.15, "et"),    # Pyrroloquinoline quinone
    "MPT":   CofactorPrior(0.70, 0.20, "et"),    # Molybdopterin
    "FMN":   CofactorPrior(0.70, 0.15, "et"),    # 也可以做单电子转移
    "FAD":   CofactorPrior(0.70, 0.15, "et"),    # 同上
    "SULFUR": CofactorPrior(0.50, 0.20, "et"),   # 硫中心
    "NI":    CofactorPrior(0.60, 0.20, "et"),    # 镍中心
    "ZN":    CofactorPrior(0.70, 0.20, "et"),    # 锌中心（非氧化还原但常见）
    "MG":    CofactorPrior(0.30, 0.15, "et"),    # 镁（配位作用）
    "TPP":   CofactorPrior(0.50, 0.20, "et"),    # Thiamine pyrophosphate
    "PLP":   CofactorPrior(0.50, 0.20, "et"),    # Pyridoxal phosphate

    # ── 氢负离子转移 (hydride) ── 使用 Marcus-Hammond
    "NAD":   CofactorPrior(1.00, 0.20, "hydride", delta=0.15),
    "NADP":  CofactorPrior(1.00, 0.20, "hydride", delta=0.15),

    # ── 质子耦合电子转移 (pcet) ──
    # 氢原子转移中的 PCET
    "PCET_General": CofactorPrior(0.85, 0.20, "pcet", lambda_p=0.25),
}
# fmt: on


def get_prior_for_cofactors(cofactor_str: str) -> CofactorPrior:
    """
    对 'HEME|FAD|FMN' 这样的多辅因子字符串，
    以主要电子转移辅因子的机制为准（优先级：HEME > FES > CU > FMN > FAD > NAD > NADP > ...）
    """
    if not cofactor_str:
        return CofactorPrior(0.70, 0.25, "et")

    cofactors = [c.strip() for c in cofactor_str.split("|")]

    # 取文献 λ 最小的那个（通常决定电子转移瓶颈）
    best = None
    best_lambda = float("inf")
    for cf in cofactors:
        if cf in COFACTOR_PRIORS:
            p = COFACTOR_PRIORS[cf]
            if p.lambda_mean < best_lambda:
                best_lambda = p.lambda_mean
                best = p

    if best is None:
        return CofactorPrior(0.70, 0.25, "et")
    return best


# ─────────────────────────────────────────────────────────────
# 编码器
# ─────────────────────────────────────────────────────────────

class LigandEncoder(nn.Module):
    """配体分子图 GNN 编码器 (GATv2 + 全局注意力池化)"""

    def __init__(self,
                 atom_dim: int = 79,
                 edge_dim: int = 10,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.atom_proj = nn.Linear(atom_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                      edge_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        self.global_attn = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ),
            nn=nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data):
        x = self.atom_proj(data.x)
        e = self.edge_proj(data.edge_attr)

        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, data.edge_index, e)
            x_new = bn(x_new)
            x = x + x_new
            x = F.silu(x)

        return self.global_attn(x, data.batch)


class CofactorEncoder(nn.Module):
    """
    辅因子编码器：对多辅因子组合的离散编码（参考 Cui 2024 CASTLE 的 VQ-VAE 离散潜码思路）。

    每个独立辅因子类型有可学习的 embedding，组合时使用注意力加权求和。
    """

    def __init__(self,
                 cofactor_types: list[str],
                 embed_dim: int = 64):
        super().__init__()
        self.cofactor_types = cofactor_types
        self.num_types = len(cofactor_types)
        self.embed_dim = embed_dim

        # 可学习的辅因子类型 embedding
        self.type_embed = nn.Embedding(self.num_types + 1, embed_dim, padding_idx=0)

        # 类型索引映射
        self.type_to_idx = {t: i + 1 for i, t in enumerate(cofactor_types)}

        # 从 embedding 预测 λ 先验偏移
        self.embed_to_lambda_offset = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 1),
            nn.Tanh(),  # 输出在 [-1, 1] eV 范围内
        )

    def _parse_cofactors(self, cofactor_str: str) -> torch.Tensor:
        """将 'HEME|FAD|FMN' 转为 multi-hot 编码"""
        if not cofactor_str or isinstance(cofactor_str, float):
            return torch.zeros(1, self.num_types + 1)

        cofactors = [c.strip() for c in str(cofactor_str).split("|")]
        indices = torch.tensor([
            self.type_to_idx.get(cf, 0) for cf in cofactors
        ])
        multi_hot = torch.zeros(self.num_types + 1)
        for idx in indices:
            multi_hot[idx] = 1.0
        return multi_hot.unsqueeze(0)

    def forward(self, cofactor_strs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          cofactor_embed: (B, embed_dim)
          lambda_offset:  (B, 1)  — λ 相对于辅因子先验的偏移
        """
        device = self.type_embed.weight.device
        batch_multihot = torch.cat([
            self._parse_cofactors(cf).to(device) for cf in cofactor_strs
        ], dim=0)  # (B, num_types+1)

        # 加权混合多个辅因子
        weights = batch_multihot / (batch_multihot.sum(dim=-1, keepdim=True) + 1e-8)
        cofactor_embed = weights @ self.type_embed.weight  # (B, embed_dim)

        lambda_offset = self.embed_to_lambda_offset(cofactor_embed)  # (B, 1)
        return cofactor_embed, lambda_offset


class ProteinEncoder(nn.Module):
    """
    精简版蛋白质编码器：纯序列特征 → hidden_dim。

    双路径设计：
      - ESM-2 路径: 1280-dim → seq_proj → hidden_dim（主力，已捕获进化信息和隐式结构倾向）
      - AA属性路径: AA_PROP_DIM (6) → aa_proj → hidden_dim（备选，轻量级物理化学特征）

    设计说明：
      - 基于过渡态理论，不依赖显式结构约束
      - ESM-2 已经捕获进化信息和隐式结构倾向，足够支撑催化预测
      - 移除结构/口袋/域特征路径：Loss 不约束这些特征，纯浪费 GPU 算力
      - 精简后 ProteinEncoder 参数 ~397K，forward FLOPs 降至约 25%
    """

    def __init__(self,
                 seq_embed_dim: int = 1280,
                 hidden_dim: int = 256,
                 aa_prop_dim: int = 6,
                 ):
        super().__init__()

        # ESM-2 路径（主力编码器，与 ESM-2 使用一致的 GELU 激活）
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # AA 属性路径（备选，无 ESM-2 预计算嵌入时使用）
        self.aa_proj = nn.Linear(aa_prop_dim, hidden_dim)
        self.aa_norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_embed: (B, 1280) ESM-2 嵌入 或 (B, AA_PROP_DIM) AA 物化性质

        Returns:
            (B, hidden_dim) 蛋白质潜在表示
        """
        if seq_embed.size(-1) == self.aa_proj.in_features:
            # AA 属性路径（备选）
            seq_h = self.aa_proj(seq_embed)
            seq_h = self.aa_norm(seq_h)
            seq_h = F.gelu(seq_h)
        else:
            # ESM-2 路径（主力）
            seq_h = self.seq_proj(seq_embed)
        return seq_h


# ═══════════════════════════════════════════════════════════════════════════════
# LatentPathwayBINN: 潜在特征路径上的多步特征变换
# ═══════════════════════════════════════════════════════════════════════════════
# 设计原则：
#   - 不假设任何物理机制（Marcus/ET/Hydride/PCET）
#   - 使用 Neural ODE + 门控机制进行多步特征变换
#   - "推理深度"的概念：不同酶-底物对可能需要不同的处理步数
#   - 门控 profile 可作为可解释性工具（哪些步信息流量大）

class LatentGate(nn.Module):
    """
    信息流门控机制。
    
    核心思想：
    - 每个推理步决定"多少信息可以通过"
    - 门控值 ∈ [0, 1]：1=完全通过，0=完全阻挡
    - 门控由当前状态 + 酶催化上下文 + 底物信息共同决定
    
    设计动机：
    - 不同的酶-底物对可能需要不同的"推理深度"
    - 门控让模型学会自适应地控制每步的信息流
    - 训练后可以通过 gate_profile 观察："哪些步骤信息流量大"
    
    注意：这不是物理"能垒"——门控值不代表能量。
    它是潜在特征空间中信息流量的可学习控制机制。
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # h + catalyst + ligand context
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, 
                h: torch.Tensor,           # (B, D) 当前状态
                catalyst_h: torch.Tensor,  # (B, D) 酶催化上下文
                ligand_h: torch.Tensor,    # (B, D) 底物状态
               ) -> torch.Tensor:
        """
        Returns:
            gate_value: (B,) ∈ [0, 1]，门控值
        """
        context = torch.cat([h, catalyst_h, ligand_h], dim=-1)
        gate = self.gate_net(context).squeeze(-1)
        return gate


class LatentPathwayBINN(nn.Module):
    """
    潜在特征路径上的多步特征变换。

    核心思想：
    1. 酶-底物复合物的初始表示 h₀ 经过 N 步逐步变换
    2. 每一步 dh = f(h, 酶上下文, 底物) 由共享的动力学网络预测
    3. 门控机制控制每步信息流量：dh_actual = gate × dh
    4. 最终输出 h_reaction = output_proj(concat(h_final, h_initial))

    设计动机：
    - 共享的"动力学网络"学到对所有酶-底物对最优的通用特征变换路径
    - 门控让每个具体样本决定"需要多少步处理"
    - 训练后 gate_profile 揭示哪些酶-底物对需要更深的"推理"
    - 这不是物理反应坐标——ξ 只是步索引，不是空间坐标
    
    关键区别（与物理反应坐标不同）：
    - ξ 没有物理单位，不表示"反应进度"
    - 门控值不是"能垒"，而是信息流量
    - 步数 N 是超参数，不是物理决定的
    """

    def __init__(
        self, 
        hidden_dim: int = 256,
        n_ode_steps: int = 1,
        use_gate: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_steps = n_ode_steps
        self.use_gate = use_gate
        
        # Step 1: 初始状态构建（酶-底物复合物特征融合）
        self.initial_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # protein + ligand
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        
        # Step 2: 特征变换动力学函数
        #   learns: dh = f(h, catalyst_context, ligand)
        #   注：使用GeLU替代SiLU，与ESM-2保持一致，梯度更流畅
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # h + catalyst + ligand
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # 保证动力学稳定（有界输出）
        )
        
        # 步长
        self.dxi = 1.0 / n_ode_steps
        
        # Step 3: 门控机制
        if use_gate:
            self.gate = LatentGate(hidden_dim)
        
        # Step 4: 输出投影（最终状态 + 初始状态）
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, 
                protein_h: torch.Tensor,   # (B, D)
                ligand_h: torch.Tensor,    # (B, D)
                cofactor_h: torch.Tensor   # (B, D_cf)
               ) -> dict[str, torch.Tensor]:
        """
        Args:
            protein_h: 酶的隐藏状态
            ligand_h:  底物的隐藏状态
            cofactor_h: 辅因子的隐藏状态
        
        Returns:
            h_reaction: (B, D) 多步变换后的最终状态
            trajectory: list of (B, D) 每一步的状态
            gate_profile: (B, n_steps) 每步的门控值（信息流量）
            feature_evol: (B,) 特征变化幅度（诊断用）
        """
        B = protein_h.size(0)
        device = protein_h.device
        
        # 催化剂上下文 = 酶 + 辅因子
        catalyst_h = protein_h + cofactor_h
        
        # ── Step 1: 构建初始状态 ──────────────────
        es_complex = torch.cat([protein_h, ligand_h], dim=-1)
        h0 = self.initial_state_proj(es_complex)
        
        trajectory = [h0]
        gate_values = []
        
        # ── Step 2: 多步特征变换 ──────────────────
        h = h0
        for step in range(self.n_steps):
            dynamics_input = torch.cat([h, catalyst_h, ligand_h], dim=-1)
            dh = self.dynamics_net(dynamics_input)
            
            # 门控
            if self.use_gate:
                gate = self.gate(h, catalyst_h, ligand_h)
                gate_values.append(gate.detach())
                dh = dh * gate.unsqueeze(-1)
            
            # 二阶（中点）Euler 积分
            h_mid = h + 0.5 * self.dxi * dh
            dynamics_mid = torch.cat([h_mid, catalyst_h, ligand_h], dim=-1)
            dh_correction = self.dynamics_net(dynamics_mid)
            h = h + self.dxi * (dh + dh_correction) / 2
            
            trajectory.append(h.detach())
        
        # ── Step 3: 输出 ──────────────────────────────────
        # 特征变化幅度
        feature_evol = (h - h0).pow(2).mean(dim=-1)  # (B,)
        
        # 最终输出
        h_reaction = self.output_proj(torch.cat([h, h0], dim=-1))
        
        return {
            'h_reaction': h_reaction,
            'trajectory': trajectory,
            'gate_profile': torch.stack(gate_values) if gate_values else None,
            'feature_evol': feature_evol,
        }


class TrenzitionCatalysisHead(nn.Module):
    """
    Trenzition 催化预测头

    预测：
    1. ts_stability: 结合亲和力（与pKd对应）→ 输出 [0, 1]（归一化后）
    2. catalysis_rate: 催化效率（与log_kcat对应）→ 输出 [0, 1]（归一化后）
    3. dG_eyring: 活化吉布斯自由能 ΔG‡ [kJ/mol]

    Eyring 约束：
    - dG_eyring 与 catalysis_rate 通过 Eyring 方程关联
    - 训练时由 TrenzitionLoss.eyring_loss() 确保一致性
    - 这与之前在 Neural ODE 中的"barrier"不同——这是对催化效率
      的独立热力学预测，有明确的物理意义
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 结合亲和力头：预测归一化后的 pKd
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        
        # 催化效率头：预测归一化后的 log10(kcat)
        self.catalytic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # 活化能垒头：预测 ΔG‡ [kJ/mol]
        # 意义：模型从特征中推断反应的活化自由能
        # 范围约束在 [20, 200] kJ/mol（一般酶催化范围）
        self.dG_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h_reaction: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            h_reaction: (B, D) 多步变换后的最终状态

        Returns:
            ts_stability:   (B,) 预测的归一化pKd [0, 1]
            catalysis_rate: (B,) 预测的归一化log10(kcat) [0, 1]
            dG_eyring:      (B,) 预测的活化能 ΔG‡ [kJ/mol]
        """
        h = self.shared(h_reaction)

        ts_stability = torch.sigmoid(self.affinity_head(h).squeeze(-1))
        catalysis_rate = torch.sigmoid(self.catalytic_head(h).squeeze(-1))

        # ΔG‡ 预测：范围 [20, 200] kJ/mol
        raw_dG = self.dG_head(h).squeeze(-1)
        dG_eyring = 20.0 + 180.0 * torch.sigmoid(raw_dG / 30.0)
        
        return {
            'ts_stability': ts_stability,
            'catalysis_rate': catalysis_rate,
            'dG_eyring': dG_eyring,
        }


class TrenzitionLoss(nn.Module):
    """
    回归损失函数 + Eyring 物理约束

    损失项：
    - L_ts: 结合亲和力回归（pKd）
    - L_catalysis: 催化效率回归（log_kcat）
    - L_eyring: Eyring 方程自洽约束

    Eyring 约束设计：
    Eyring 方程：kcat = κ · (kBT/h) · exp(-ΔG‡ / RT)
    
    反解 ΔG‡：ΔG‡ = RT · ln(κ·kBT/h / kcat)
    
    L_eyring 约束"模型预测的 ΔG‡"与"从 kcat 反推的 ΔG‡"必须一致。
    这确保了两个预测头的热力学自洽性。
    
    注意：
    - 这不是施加外部标签，而是内部一致性约束
    - log10_prefactor 是可学习的（全局，所有样本共享）
      初始化为 log10(0.5 · kBT/h) ≈ 12.19
      学习后反映数据整体的有效透射系数 κ
    """

    def __init__(self, use_learnable_weights: bool = False):
        """
        Args:
            use_learnable_weights: 是否使用可学习权重（建议初期 False）
        """
        super().__init__()
        self.use_learnable_weights = use_learnable_weights
        if use_learnable_weights:
            self.register_parameter(
                'log_var_ts', nn.Parameter(torch.tensor(0.0))
            )
            self.register_parameter(
                'log_var_cat', nn.Parameter(torch.tensor(0.0))
            )
        
        # Eyring 指前因子（可学习）
        # log10(κ · kBT/h), 初始 κ=0.5
        kappa_init = 0.5
        log10_prefactor_init = math.log10(kappa_init * k_B * T_ref / h)
        self.register_parameter(
            'log10_prefactor', nn.Parameter(torch.tensor(log10_prefactor_init))
        )

    def compute_eyring_loss(self, outputs: dict) -> torch.Tensor:
        """
        Eyring 自洽约束：
        ΔG‡_pred ≈ RT·ln(10) · (log10_prefactor - log10(kcat_pred))

        其中 log10(kcat_pred) 从归一化的 catalysis_rate 反归一化得到。
        """
        # 反归一化 catalysis_rate → log10(kcat)
        # 归一化区间: kcat_min=-7, kcat_max=8（与 train.py NORM_PARAMS 保持一致）
        kcat_min = -7.0
        kcat_max = 8.0
        log_kcat_pred = outputs['catalysis_rate'] * (kcat_max - kcat_min) + kcat_min
        
        # 从 Eyring 方程计算期望的 ΔG‡ (kJ/mol)
        dG_from_kcat = (
            R_kJ * T_ref * math.log(10) *
            (self.log10_prefactor - log_kcat_pred)
        )
        
        # 约束：预测的 dG 应与 Eyring 期望值一致
        dG_pred = outputs['dG_eyring']
        loss = F.mse_loss(dG_pred, dG_from_kcat)
        return loss

    def forward(self,
                outputs: dict,
                batch: dict,
                eyring_weight: float = 0.01,
                ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: 模型输出 dict
            batch: 数据 batch
            eyring_weight: Eyring 约束权重（建议 0.01-0.05）
        """
        device = outputs['ts_stability'].device
        losses = {}

        # ── 1. 结合亲和力损失 L_ts ──────────────────────
        pkd_mask = batch.get('pkd_target_mask', torch.ones_like(outputs['ts_stability'], dtype=torch.bool))

        if pkd_mask.any():
            l_ts = F.smooth_l1_loss(
                outputs['ts_stability'][pkd_mask],
                batch['pkd_target'][pkd_mask]
            )
        else:
            l_ts = torch.tensor(0.0, device=device)
        losses['L_ts'] = l_ts

        # ── 2. 催化效率损失 L_catalysis ───────────────────
        kcat_mask = batch.get('kcat_target_mask', torch.zeros_like(outputs['catalysis_rate'], dtype=torch.bool))

        if kcat_mask.any():
            l_cat = F.smooth_l1_loss(
                outputs['catalysis_rate'][kcat_mask],
                batch['log_kcat_target'][kcat_mask]
            )
        else:
            l_cat = torch.tensor(0.0, device=device)
        losses['L_catalysis'] = l_cat

        # ── 3. Eyring 自洽约束 ──────────────────────────
        if 'dG_eyring' in outputs and outputs['dG_eyring'] is not None:
            l_eyring = self.compute_eyring_loss(outputs) * eyring_weight
        else:
            l_eyring = torch.tensor(0.0, device=device)
        losses['L_eyring'] = l_eyring

        # ── 4. 总损失计算 ────────────────────────────────
        if self.use_learnable_weights and hasattr(self, 'log_var_ts'):
            w_ts = torch.exp(-self.log_var_ts)
            w_cat = torch.exp(-self.log_var_cat)
            total = (
                w_ts * losses['L_ts'] + self.log_var_ts +
                w_cat * losses['L_catalysis'] + self.log_var_cat +
                losses['L_eyring']
            )
            losses['weights'] = {
                'w_ts': w_ts.item(),
                'w_cat': w_cat.item(),
            }
        else:
            total = losses['L_ts'] + losses['L_catalysis'] + losses['L_eyring']
            losses['weights'] = {'w_ts': 1.0, 'w_cat': 1.0}

        losses['total'] = total
        losses['log10_prefactor'] = self.log10_prefactor.item()
        return total, losses


# ═══════════════════════════════════════════════════════════════════════════════
# 完整模型：Trenzition
# ═══════════════════════════════════════════════════════════════════════════════

class Trenzition(nn.Module):
    """
    酶-底物催化效率预测模型（精简版）

    架构：
      LigandEncoder (GATv2×3) + ProteinEncoder (ESM-2 序列直通) + CofactorEncoder
      → LatentPathwayBINN (Neural ODE 多步特征变换 + 门控)
      → CatalysisHead (binding_affinity + catalytic_rate)

    设计说明：
      - 不假设任何预定的物理机制（电子转移/Marcus/过渡态）
      - Neural ODE 的"反应坐标"实为潜在特征空间中的多步变换路径
      - 门控机制是信息流控制，不是物理"能垒"
      - 训练后的 gate_profile 可解释为"该样本需要多少步推理深度"
      - 精简版移除结构/域/口袋特征路径，ProteinEncoder 参数减少 ~26%，FLOPs 降至 ~25%

    使用模式：
      - use_classification=False: 回归模式，预测 pKd 和 log_kcat
      - use_classification=True: 分类模式，酶类型分类
    """

    COFACTOR_TYPES = sorted(COFACTOR_PRIORS.keys())

    def __init__(
        self,
        hidden_dim: int = 256,
        cofactor_embed_dim: int = 64,
        n_heads: int = 4,
        gnn_layers: int = 3,
        n_ode_steps: int = 1,
        use_gate: bool = True,
        use_learnable_weights: bool = False,
        use_classification: bool = False,  # ✅ 分类模式开关
        num_classes: int = 4,              # ✅ 分类类别数
        classification_init_thresholds: list = [-2.0, -1.0, 1.0],  # ✅ 初始阈值
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cofactor_embed_dim = cofactor_embed_dim
        self.use_classification = use_classification  # ✅ 保存模式标志
        self.num_classes = num_classes
        self.classification_init_thresholds = classification_init_thresholds

        # ─────── 编码器 ───────
        self.ligand_encoder = LigandEncoder(
            atom_dim=79, edge_dim=10, hidden_dim=hidden_dim,
            num_layers=gnn_layers, heads=n_heads,
        )
        self.protein_encoder = ProteinEncoder(
            seq_embed_dim=1280, hidden_dim=hidden_dim,
        )
        self.cofactor_encoder = CofactorEncoder(
            cofactor_types=self.COFACTOR_TYPES, embed_dim=cofactor_embed_dim,
        )

        self.ligand_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cofactor_proj = nn.Linear(cofactor_embed_dim, hidden_dim)

        # ─────── BINN交互层 ───────
        self.binn = LatentPathwayBINN(
            hidden_dim=hidden_dim,
            n_ode_steps=n_ode_steps,
            use_gate=use_gate,
        )

        # ─────── 预测头（根据模式选择）───────
        if use_classification:
            # ✅ 分类模式：使用多任务分类头
            self.classification_head = EnzymeTypeClassificationHead(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                init_thresholds=classification_init_thresholds,
                learnable_thresholds=True,
                use_ordinal=True,
            )
        else:
            # 回归模式：使用原始催化头
            self.catalysis_head = TrenzitionCatalysisHead(hidden_dim=hidden_dim)

        # ─────── 损失函数（可配置）───────
        self.loss_fn = TrenzitionLoss(use_learnable_weights=use_learnable_weights)

        # ─────── 参数初始化 ────────────────────────────────────
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        """统一的参数初始化策略。

        原则：
        - 输出头最后一层 (out_dim=1, 接 sigmoid)：用小 gain 避免饱和
        - 中间层 (接 GELU/SiLU)：Xavier uniform（比 Kaiming 更适合非 ReLU 激活）
        - BINN 动力学层：保守 gain 防止 ODE 多步积分放大
        - LayerNorm：保持默认 (1.0, 0.0)
        """
        if isinstance(module, nn.Linear):
            is_output_head = (
                module.out_features == 1
                and module.in_features == 64
            )
            is_dynamics = (
                module.in_features == module.out_features == 256
            )
            if is_output_head:
                # sigmoid 前的最后映射 → 小初始化防饱和
                nn.init.xavier_normal_(module.weight, gain=0.3)
            elif is_dynamics:
                # BINN ODE 动力学 → 保守初始化防 ODE 积分爆炸
                nn.init.xavier_uniform_(module.weight, gain=0.67)
            else:
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            # LayerNorm 保持默认 (weight=1, bias=0)
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self,
                ligand_data,
                seq_embed: torch.Tensor,
                cofactor_strs: list[str],
                ) -> dict[str, torch.Tensor]:
        """
        Returns:
            ts_stability:   (B,) 预测的归一化pKd [0, 1]
            catalysis_rate: (B,) 预测的归一化log10(kcat) [0, 1]
            gate_profile:   (n_steps, B) 每步门控值
            feature_evol:   (B,) 特征变化幅度
            trajectory:     list of (B, D) 每一步的状态
            h_reaction:     (B, D) 最终状态
        """
        # ── 编码 ────────────────────────────────────────────
        ligand_h = self.ligand_encoder(ligand_data)
        ligand_h = self.ligand_proj(ligand_h)

        protein_h = self.protein_encoder(seq_embed)

        cofactor_h, _ = self.cofactor_encoder(cofactor_strs)
        cofactor_h_proj = self.cofactor_proj(cofactor_h)
        
        # ── BINN反应坐标演化 ────────────────────────────────
        binn_output = self.binn(protein_h, ligand_h, cofactor_h_proj)
        h_reaction = binn_output['h_reaction']
        
        # ── 预测输出（根据模式选择）───────────────────────
        if self.use_classification:
            # 分类模式输出
            cls_output = self.classification_head(h_reaction)
            
            return {
                # 分类结果
                'class_pred': cls_output['class_pred'],
                'prob': cls_output['prob'],
                'log_ratio': cls_output['log_ratio'],
                'cls_logits': cls_output['cls_logits'],
                'thresholds': cls_output['thresholds'],
                
                # 回归结果（多任务）
                'ts_stability': cls_output.get('ts_stability'),
                'catalysis_rate': cls_output.get('catalysis_rate'),
                
                # 特征路径信息
                'h_reaction': h_reaction,
                'feature_evol': binn_output['feature_evol'],
                'trajectory': binn_output['trajectory'],
                'gate_profile': binn_output['gate_profile'],
                
                # 编码器输出
                'protein_h': protein_h,
                'ligand_h': ligand_h,
                'cofactor_h': cofactor_h,
            }
        else:
            # 回归模式输出
            catalysis_output = self.catalysis_head(h_reaction)
            
            return {
                # 主要预测
                'ts_stability': catalysis_output['ts_stability'],
                'catalysis_rate': catalysis_output['catalysis_rate'],
                'dG_eyring': catalysis_output['dG_eyring'],
                
                # 特征路径信息
                'h_reaction': h_reaction,
                'feature_evol': binn_output['feature_evol'],
                'trajectory': binn_output['trajectory'],
                'gate_profile': binn_output['gate_profile'],
                
                # 编码器输出
                'protein_h': protein_h,
                'ligand_h': ligand_h,
                'cofactor_h': cofactor_h,
            }

    def compute_loss(self,
                     outputs: dict,
                     batch: dict,
                     eyring_weight: float = 0.01,
                     ) -> tuple[torch.Tensor, dict]:
        """
        损失函数计算
        
        回归模式: L_ts + L_catalysis + L_eyring
        分类模式: L_ordinal + L_ce + L_order
        """
        if self.use_classification:
            return self._compute_classification_loss(outputs, batch)
        else:
            return self.loss_fn(outputs, batch, eyring_weight=eyring_weight)
    
    def _compute_classification_loss(self,
                                      outputs: dict,
                                      batch: dict,
                                      ) -> tuple[torch.Tensor, dict]:
        """
        分类模式的损失函数
        
        损失组成:
        - L_ordinal: 有序回归损失（三个二分类 BCE 之和）
        - L_ce: 多分类交叉熵损失
        - L_order: 阈值单调约束 (t0 < t1 < t2)
        """
        device = outputs['class_pred'].device
        losses = {}
        
        # 获取类别标签
        labels = batch.get('enzyme_type_class', None)
        if labels is None:
            raise ValueError("分类模式需要 batch['enzyme_type_class'] 标签")
        
        # 1. 有序回归损失（主损失）
        # 基于 threshold 的概率计算
        t0, t1, t2 = outputs['thresholds']
        log_ratio = outputs['log_ratio']
        scale = 5.0
        
        # 三个二分类目标
        target_pass0 = (labels >= 1).float()  # class > 0?
        target_pass1 = (labels >= 2).float()  # class > 1?
        target_pass2 = (labels >= 3).float()  # class > 2?
        
        # 预测概率
        p0 = torch.sigmoid((log_ratio - t0) * scale)
        p1 = torch.sigmoid((log_ratio - t1) * scale)
        p2 = torch.sigmoid((log_ratio - t2) * scale)
        
        # BCE 损失
        L_ordinal_0 = F.binary_cross_entropy(p0, target_pass0, reduction='mean')
        L_ordinal_1 = F.binary_cross_entropy(p1, target_pass1, reduction='mean')
        L_ordinal_2 = F.binary_cross_entropy(p2, target_pass2, reduction='mean')
        L_ordinal = L_ordinal_0 + L_ordinal_1 + L_ordinal_2
        losses['L_ordinal_0'] = L_ordinal_0.item()
        losses['L_ordinal_1'] = L_ordinal_1.item()
        losses['L_ordinal_2'] = L_ordinal_2.item()
        losses['L_ordinal'] = L_ordinal.item()
        
        # 2. 交叉熵损失（辅助）
        prob = outputs['prob']
        # 获取类别权重（如有）
        class_weights = batch.get('class_weights', None)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        
        L_ce = F.nll_loss(
            torch.log(prob.clamp(min=1e-8)),
            labels,
            weight=class_weights,
            reduction='mean'
        )
        losses['L_ce'] = L_ce.item()
        
        # 3. 阈值有序约束
        L_order = F.relu(t0 - t1) + F.relu(t1 - t2)
        losses['L_threshold_order'] = L_order.item()
        
        # 总损失
        total = (
            0.6 * L_ordinal +      # 主损失：有序回归
            0.3 * L_ce +           # 辅助：交叉熵
            0.1 * L_order          # 约束：阈值有序
        )
        losses['total'] = total.item()
        
        return total, losses

    def predict_catalytic_efficiency(self, outputs: dict) -> torch.Tensor:
        """
        辅助函数：计算 log10(kcat/KM) 催化效率指标
        
        简化估计：kcat/KM ≈ kcat * Kd^(-1) = kcat * 10^pKd
        """
        pkd = outputs['ts_stability']
        log_kcat = outputs['catalysis_rate']
        log_kcat_km = log_kcat + pkd
        return log_kcat_km


def create_trenzition_optimizer(model: Trenzition, lr: float = 1e-4, weight_decay: float = 1e-5,
                         use_learnable_weights: bool = False):
    """
    为Trenzition创建分组优化器

    不同组件使用不同学习率：
    - encoder: 正常学习率
    - binn: 稍慢（稳定性优先）
    - head: 稍快（快速学习预测映射）
    - loss weights: 仅在use_learnable_weights=True时训练（更慢）
    """
    encoder_params = []
    binn_params = []
    head_params = []
    loss_params = []

    for name, param in model.named_parameters():
        if 'loss_fn.log_var' in name and use_learnable_weights:
            loss_params.append(param)
        elif 'binn' in name:
            binn_params.append(param)
        elif 'catalysis_head' in name or 'classification_head' in name:  # ✅ 支持两种头
            head_params.append(param)
        else:
            encoder_params.append(param)

    optimizer_groups = [
        {'params': encoder_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': binn_params, 'lr': lr * 0.5, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr * 2, 'weight_decay': weight_decay},
    ]

    if loss_params:
        optimizer_groups.append({'params': loss_params, 'lr': lr * 0.1, 'weight_decay': 0})

    return torch.optim.AdamW(optimizer_groups)


# ═══════════════════════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  酶挖掘排序模型 — Trenzition")
    print("=" * 60)

    # 测试回归模式
    print("\n── 回归模式测试 ──")
    model_reg = Trenzition(hidden_dim=256, use_classification=False)
    total_params = sum(p.numel() for p in model_reg.parameters())
    print(f"  总参数量:    {total_params:,}")
    print(f"  模式:        回归 (pKd + kcat)")
    print(f"  BINN层:      {type(model_reg.binn).__name__}")
    print(f"  门控层:      {type(model_reg.binn.gate).__name__}")

    # 测试分类模式
    print("\n── 分类模式测试 ──")
    model_cls = Trenzition(
        hidden_dim=256,
        use_classification=True,
        num_classes=4,
        classification_init_thresholds=[-2.0, -1.0, 1.0]
    )
    total_params = sum(p.numel() for p in model_cls.parameters())
    print(f"  总参数量:    {total_params:,}")
    print(f"  模式:        分类 (四分类)")
    print(f"  预测头:      {type(model_cls.classification_head).__name__}")
    print(f"  初始阈值:    {model_cls.classification_head.thresholds.data.tolist()}")
    
    # 模拟前向传播
    print("\n── 前向传播测试 ──")
    B = 8
    from torch_geometric.data import Data
    
    # 模拟配体数据
    ligand_data = Data(
        x=torch.randn(B, 20, 79),
        edge_index=torch.randint(0, 20, (2, 60)),
        edge_attr=torch.randn(60, 10),
        batch=torch.randint(0, B, (20,))
    )
    
    # 模拟其他输入
    seq_embed = torch.randn(B, 1280)
    
    # 默认测试（回归模式）
    with torch.no_grad():
        outputs = model_reg(
            ligand_data=ligand_data,
            seq_embed=seq_embed,
            cofactor_strs=["NAD"] * B,
        )
    print(f"  回归模式输出键: {list(outputs.keys())}")
    
    # 分类模式测试
    with torch.no_grad():
        outputs_cls = model_cls(
            ligand_data=ligand_data,
            seq_embed=seq_embed,
            cofactor_strs=["NAD"] * B,
        )
    print(f"  分类模式输出键: {list(outputs_cls.keys())}")
    print(f"  类别预测:       {outputs_cls['class_pred'].tolist()}")
    print(f"  log_ratio范围:  [{outputs_cls['log_ratio'].min():.2f}, {outputs_cls['log_ratio'].max():.2f}]")

    # 测试损失计算（分类模式）
    print("\n── 分类损失测试 ──")
    labels = torch.randint(0, 4, (B,))
    batch = {
        'enzyme_type_class': labels,
        'class_weights': create_class_weights([100, 200, 500, 300]),
    }
    total_loss, losses = model_cls.compute_loss(outputs_cls, batch)
    print(f"  总损失:          {total_loss.item():.4f}")
    for k, v in losses.items():
        if k != 'total':
            print(f"    {k}: {v:.4f}")

    # 测试优化器创建
    print("\n── 优化器测试 ──")
    opt = create_trenzition_optimizer(model_cls, lr=1e-4)
    print(f"  参数组数:        {len(opt.param_groups)}")
    for i, g in enumerate(opt.param_groups):
        print(f"    组{i}: lr={g['lr']:.2e}, 参数数={len(g['params'])}")

    print("\n  ✓ 所有测试通过 — Trenzition Ready!")
