# -*- coding: utf-8 -*-
"""
ranking_model.py
================
酶挖掘排序模型：基于蛋白序列嵌入 + 配体分子图 + 辅因子类型预测底物结合亲和力 (pKd)
+ 催化效率 (kcat, Eyring 硬编码) + 米氏常数 (Km)。

架构（v3，三头 + Eyring 物理硬编码）：
  LigandEncoder (GATv2×3) + ProteinEncoder (ESM-2) + CofactorEncoder
  → [BindingDualHead, EyringKcatHead, KmHead]

预测头：
  1. BindingDualHead: BINN (1步) + Kd/Ki 双分支 → pKd
  2. EyringKcatHead:  BINN (多步ODE) + dG_predictor + Eyring 公式硬编码 → ΔG‡ → kcat
  3. KmHead:          简单 MLP → log₁₀(Km)

Eyring 硬编码:
  kcat = κ · (k_B·T/h) · exp(-ΔG‡ / RT)
  - κ (透射系数): 可学习, 初始 log₁₀(0.5)
  - T: 每样本独立温度 (K)
  - ΔG‡: 模型预测, 钳位 [5, 300] kJ/mol
  - 无需 L_eyring 软约束: 物理规律已在模型内部硬编码

训练目标：
  L_total = L_binding + L_kcat + L_km + 0.1*L_joint

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

class LatentPathwayBINN(nn.Module):
    """
    潜在特征路径上的多步特征变换 (Neural ODE)。

    架构:
      1. 酶-底物复合物的初始表示 h₀
      2. 多步中点 Euler 积分: dh/dt = f(h, catalyst, ligand)
      3. 输出投影: concat(h_final, h0) → h_reaction

    设计：
      - 纯回归，无 Gate
      - 用于 KcatHead 的 ODE 动力学
    """

    def __init__(self, hidden_dim: int = 256, n_ode_steps: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_steps = n_ode_steps

        # Step 1: 初始状态构建
        self.initial_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # protein + ligand
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        # Step 2: 动力学函数
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # h + catalyst + ligand
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.dxi = 1.0 / max(n_ode_steps, 1)

        # Step 3: 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self,
                protein_h: torch.Tensor,
                ligand_h: torch.Tensor,
                cofactor_h: torch.Tensor,
                ) -> dict:
        B = protein_h.size(0)

        # 催化剂上下文 = 酶 + 辅因子
        catalyst_h = protein_h + cofactor_h

        # 初始状态 (ES 复合物)
        es_complex = torch.cat([protein_h, ligand_h], dim=-1)
        h0 = self.initial_state_proj(es_complex)

        # 多步中点 Euler 积分
        h = h0
        for step in range(self.n_steps):
            dynamics_input = torch.cat([h, catalyst_h, ligand_h], dim=-1)
            dh = self.dynamics_net(dynamics_input)

            # 中点校正
            h_mid = h + 0.5 * self.dxi * dh
            dynamics_mid = torch.cat([h_mid, catalyst_h, ligand_h], dim=-1)
            dh_correction = self.dynamics_net(dynamics_mid)

            h = h + self.dxi * (dh + dh_correction) / 2

        # 特征变化幅度
        feature_evol = (h - h0).pow(2).mean(dim=-1)

        # 输出
        h_reaction = self.output_proj(torch.cat([h, h0], dim=-1))

        return {
            'h_reaction': h_reaction,
            'feature_evol': feature_evol,
        }


# 完整模型：Trenzition
# ═══════════════════════════════════════════════════════════════════════════════

class Trenzition(nn.Module):
    """
    三头独立预测模型（Trenzition v2）

    架构：
      LigandEncoder (GATv2×3) + ProteinEncoder (ESM-2) + CofactorEncoder
      → [BindingDualHead, EyringKcatHead(BINN+ODE+Eyring硬编码), KmHead]

    预测头：
      1. BindingDualHead: Kd 分支 + Ki 分支 → 结合亲和力 pKd
      2. EyringKcatHead: BINN + ODE + Eyring硬编码 → ΔG‡ → kcat
      3. KmHead: 简单 MLP → log₁₀(Km)
    """

    COFACTOR_TYPES = sorted(COFACTOR_PRIORS.keys())

    def __init__(
        self,
        hidden_dim: int = 256,
        cofactor_embed_dim: int = 64,
        n_heads: int = 4,
        gnn_layers: int = 3,
        n_ode_steps: int = 1,
        kcat_ode_steps: int = 10,
        three_head: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cofactor_embed_dim = cofactor_embed_dim
        self.three_head = three_head

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

        # ─────── 预测头 ───────
        if three_head:
            self.binding_head = BindingDualHead(hidden_dim=hidden_dim)
            self.kcat_head = EyringKcatHead(hidden_dim=hidden_dim, n_ode_steps=kcat_ode_steps)
            self.km_head = KmHead(hidden_dim=hidden_dim)
            self.loss_fn = ThreeHeadLoss()
        else:
            # 旧单头模式（向后兼容）
            self.kcat_head = EyringKcatHead(hidden_dim=hidden_dim, n_ode_steps=n_ode_steps)
            self.loss_fn = ThreeHeadLoss()

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
                measurement_types: torch.Tensor = None,
                temperature_K: torch.Tensor = None,
                ) -> dict[str, torch.Tensor]:
        """三头预测: binding(Kd/Ki) + kcat(Eyring硬编码) + Km"""
        # ── 编码 ──
        ligand_h = self.ligand_encoder(ligand_data)
        ligand_h = self.ligand_proj(ligand_h)
        protein_h = self.protein_encoder(seq_embed)
        cofactor_h, _ = self.cofactor_encoder(cofactor_strs)
        cofactor_h_proj = self.cofactor_proj(cofactor_h)

        # ── 三个独立预测头 ──
        binding_out = self.binding_head(
            protein_h, ligand_h, cofactor_h_proj, measurement_types
        )
        kcat_out = self.kcat_head(
            protein_h, ligand_h, cofactor_h_proj, temperature_K
        )
        km_out = self.km_head(protein_h, ligand_h, cofactor_h_proj)

        return {
            **binding_out, **kcat_out, **km_out,
            'protein_h': protein_h, 'ligand_h': ligand_h, 'cofactor_h': cofactor_h,
        }

    def compute_loss(self,
                     outputs: dict,
                     batch: dict,
                     joint_weight: float = 0.1,
                     ) -> tuple[torch.Tensor, dict]:
        """三头联合损失: L_binding + L_kcat + L_km + L_joint (Eyring 已硬编码在模型中)"""
        return self.loss_fn(outputs, batch, joint_weight=joint_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# 测量类型编码（BindingDualHead 用）
# ═══════════════════════════════════════════════════════════════════════════════

MTYPE_KD = 0   # Kd: 底物结合 (直接结合亲和力)
MTYPE_KI = 1   # Ki: 抑制剂结合 (抑制常数)
MTYPE_IC50 = 2 # IC50 近似
MTYPE_NONE = 3 # 无结合数据


class BindingDualHead(nn.Module):
    """
    双分支结合亲和力预测头 (BINN 1步 + Kd/Ki双分支)

    架构:
      1. 初始状态融合 (protein + ligand) → h₀
      2. 1步动力学变换: dh = f(h₀, catalyst, ligand) → h = h₀ + dh
      3. 输出投影: concat(h, h₀) → h_reaction
      4. 双分支:
        - Kd 分支: 预测底物结合 pKd
        - Ki 分支: 预测抑制剂结合 pKi
      5. 测量类型标志 [is_Kd, is_Ki] 作为输入特征，让模型知道当前预测类型

    设计说明:
      - BINN 结构保持 (LatentPathwayBINN 的简化版)
      - 无多步 ODE (动力学仅 1 步)
      - 两个分支共享底层表征，但输出头参数独立
      - 测量类型标志: one-hot [底物, 抑制剂] 拼接到 h_reaction
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # BINN-like 初始状态融合
        self.initial_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # protein + ligand
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        # 1步动力学 (无 ODE 迭代)
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # 输出投影 (shortcut: concat h_final + h0)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Kd 分支: 底物结合预测
        self.kd_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),  # +2 for type one-hot
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Ki 分支: 抑制剂结合预测
        self.ki_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self,
                protein_h: torch.Tensor,
                ligand_h: torch.Tensor,
                cofactor_h: torch.Tensor,
                measurement_types: torch.Tensor = None,
                ) -> dict[str, torch.Tensor]:
        """
        Args:
            protein_h: (B, D) 酶表征
            ligand_h: (B, D) 底物表征
            cofactor_h: (B, D) 辅因子表征
            measurement_types: (B,) 测量类型 {0=Kd, 1=Ki, 2=IC50, 3=none}

        Returns:
            kd_pred: (B,) 归一化 pKd (Kd 分支)
            ki_pred: (B,) 归一化 pKd (Ki 分支)
            binding_pred: (B,) 按测量类型选通的输出
        """
        B = protein_h.size(0)
        device = protein_h.device

        # 催化剂上下文
        catalyst_h = protein_h + cofactor_h

        # 初始状态 (ES 复合物)
        es_complex = torch.cat([protein_h, ligand_h], dim=-1)
        h0 = self.initial_state_proj(es_complex)

        # 1步动力学 (无 ODE 迭代)
        dynamics_input = torch.cat([h0, catalyst_h, ligand_h], dim=-1)
        dh = self.dynamics_net(dynamics_input)
        h = h0 + dh  # 一步变换

        # 输出投影
        h_reaction = self.output_proj(torch.cat([h, h0], dim=-1))

        # 测量类型 one-hot [is_Kd, is_Ki]
        type_onehot = torch.zeros(B, 2, device=device)
        if measurement_types is not None:
            type_onehot[:, 0] = (measurement_types == MTYPE_KD).float()
            type_onehot[:, 1] = (measurement_types == MTYPE_KI).float()

        h_with_type = torch.cat([h_reaction, type_onehot], dim=-1)

        # 双分支输出 (sigmoid → 归一化 [0, 1])
        kd_pred = torch.sigmoid(self.kd_head(h_with_type).squeeze(-1))
        ki_pred = torch.sigmoid(self.ki_head(h_with_type).squeeze(-1))

        # 按测量类型选择默认输出
        binding_pred = torch.where(
            measurement_types == MTYPE_KI if measurement_types is not None else torch.zeros(B, dtype=torch.bool, device=device),
            ki_pred,
            kd_pred,
        )

        return {
            'kd_pred': kd_pred,
            'ki_pred': ki_pred,
            'binding_pred': binding_pred,
        }


class EyringKcatHead(nn.Module):
    """
    kcat 预测头 (BINN + Eyring 物理硬编码)

    架构:
      1. 完整 LatentPathwayBINN (多步 ODE 积分) → 特征沿反应坐标演化
      2. 共享层 shared → h
      3. dG_predictor: h → ΔG‡ [kJ/mol] (小型 MLP: 256→64→1)
      4. Eyring 公式硬编码: ΔG‡ → kcat (不可学习的物理变换)

    Eyring 公式:
      kcat = κ · (k_B·T/h) · exp(-ΔG‡ / RT)
      log10(kcat) = log10(κ) + log10(k_B·T/h) - ΔG‡ / (R·T·ln10)

    设计:
      - ΔG‡ 钳位在 [5, 300] kJ/mol，避免训练初期发散
      - κ (透射系数): 可学习的全局标量，初始 log10(0.5)
      - T: 每样本独立温度 (K)，不是全局常数
      - 不再需要 L_eyring 软约束 —— Eyring 约束已硬编码，不可违背
    """

    # 物理常数
    R_kJ = 8.314e-3          # kJ/(mol·K)
    k_B = 1.380649e-23       # J/K
    h_planck = 6.62607015e-34  # J·s

    def __init__(self, hidden_dim: int = 256, n_ode_steps: int = 10):
        super().__init__()

        # BINN (完整多步 ODE)
        self.binn = LatentPathwayBINN(
            hidden_dim=hidden_dim,
            n_ode_steps=n_ode_steps,
        )

        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ΔG‡ 预测器 (小型 MLP)
        self.dG_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # 可学习的全局透射系数: κ = 10^{log_kappa}
        self.log_kappa = nn.Parameter(torch.tensor(math.log10(0.5)))

    def forward(self,
                protein_h: torch.Tensor,
                ligand_h: torch.Tensor,
                cofactor_h: torch.Tensor,
                temperature_K: torch.Tensor,  # (B,) 每样本独立温度
                ) -> dict[str, torch.Tensor]:
        """
        Args:
            temperature_K: (B,) 温度 (K)，范围 ~276-562

        Returns:
            kcat_pred: (B,) 归一化 log10(kcat) [0, 1]
            dG_eyring: (B,) 活化自由能 ΔG‡ [kJ/mol]
        """
        device = protein_h.device

        # 1) BINN 特征演化
        binn_out = self.binn(protein_h, ligand_h, cofactor_h)
        h = self.shared(binn_out['h_reaction'])

        # 2) 预测 ΔG‡
        dG = self.dG_predictor(h).squeeze(-1)          # (B,)
        dG_clamped = torch.clamp(dG, min=5.0, max=300.0)   # [5, 300] kJ/mol

        # 3) Eyring 公式硬编码 (批量计算)
        RT_ln10 = self.R_kJ * temperature_K * math.log(10)  # (B,)

        # log10(κ·k_B·T/h) = log_kappa + log10(k_B·T/h)
        log10_prefactor = self.log_kappa + torch.log10(
            self.k_B * temperature_K / self.h_planck
        )  # (B,)

        log_kcat = log10_prefactor - dG_clamped / RT_ln10   # (B,)

        # 4) 归一化到 [0,1] 适配标签
        kcat_normalized = ((log_kcat + 7.0) / 15.0).clamp(0.0, 1.0)

        return {
            'kcat_pred': kcat_normalized,
            'dG_eyring': dG_clamped,
            'kcat_binn_trajectory': binn_out.get('trajectory'),
            'kcat_feature_evol': binn_out.get('feature_evol'),
        }


class KmHead(nn.Module):
    """
    Km 预测头 (简单 MLP，归一化 [0,1])

    输出 sigmoid [0,1] 归一化 log₁₀(Km)，由 ThreeHeadLoss 统一反归一化到 [-13, 3]。
    与 Kd/Kcat 头一致的归一化逻辑。

    架构: 三层 MLP，输入为 (protein_h, ligand_h, cofactor_h) 拼接
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # protein + ligand + cofactor
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self,
                protein_h: torch.Tensor,
                ligand_h: torch.Tensor,
                cofactor_h: torch.Tensor,
                ) -> dict[str, torch.Tensor]:
        """
        Returns:
            log_km_pred: (B,) 预测的 log₁₀(Km) (归一化 [0,1])
        """
        x = torch.cat([protein_h, ligand_h, cofactor_h], dim=-1)
        raw = self.net(x).squeeze(-1)
        # sigmoid 直接输出 [0,1] 归一化值
        log_km_pred = torch.sigmoid(raw)
        return {'log_km_pred': log_km_pred}


class ThreeHeadLoss(nn.Module):
    """
    三头独立预测的联合损失函数

    损失组成:
    1. L_binding: 双分支结合亲和力回归 (Kd 分支 + Ki 分支，带测量类型 mask)
    2. L_kcat: 催化速率回归 (Eyring 硬编码在模型内，损失仅为回归项)
    3. L_km: Km 回归
    4. L_joint: kcat/Km 联合约束 (物理自洽)

    物理约束说明:
    - kcat: Eyring 公式已硬编码在 EyringKcatHead 中，不再需要 L_eyring 软约束
    - kcat/Km: log₁₀(kcat/Km) = log₁₀(kcat) - log₁₀(Km)
    - 扩散极限: kcat/Km ≤ 10⁹ M⁻¹s⁻¹
    """

    # 归一化参数 (与 train.py NORM_PARAMS 保持一致)
    PKD_MIN = 0.0
    PKD_MAX = 12.0
    KCAT_MIN = -7.0
    KCAT_MAX = 8.0
    KM_MIN = -13.0
    KM_MAX = 3.0

    def __init__(self):
        super().__init__()

    @staticmethod
    def denormalize(value: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        """反归一化 [0,1] → 原始值"""
        return value * (vmax - vmin) + vmin

    def forward(self,
                outputs: dict,
                batch: dict,
                binding_weight: float = 1.0,
                kcat_weight: float = 1.0,
                km_weight: float = 1.0,
                joint_weight: float = 0.1,
                ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: 模型输出
            batch: 数据 batch
            binding_weight: 结合损失权重
            kcat_weight: kcat 损失权重
            km_weight: Km 损失权重
            joint_weight: 联合约束权重
        """
        device = next(iter(outputs.values())).device if outputs else 'cpu'
        losses = {}

        # ── 1. 结合亲和力损失 L_binding ───────────────────
        pkd_target = batch.get('pkd_target')
        pkd_mask = batch.get('pkd_target_mask', torch.zeros(batch.get('pkd_target', torch.empty(0)).size(0), dtype=torch.bool, device=device))
        mtype = batch.get('measurement_type', None)

        l_binding = torch.tensor(0.0, device=device)
        if pkd_mask.any() and pkd_target is not None:
            has_kd = pkd_mask & (mtype == MTYPE_KD) if mtype is not None else pkd_mask
            has_ki = pkd_mask & (mtype == MTYPE_KI) if mtype is not None else torch.zeros_like(pkd_mask)

            # Kd 分支损失 (权重 1.0)
            if has_kd.any():
                l_kd = F.smooth_l1_loss(
                    outputs['kd_pred'][has_kd],
                    pkd_target[has_kd],
                )
            else:
                l_kd = torch.tensor(0.0, device=device)

            # Ki 分支损失 (权重 0.7，Ki 不是直接 Kd)
            if has_ki.any():
                l_ki = F.smooth_l1_loss(
                    outputs['ki_pred'][has_ki],
                    pkd_target[has_ki],
                )
            else:
                l_ki = torch.tensor(0.0, device=device)

            l_binding = l_kd + 0.7 * l_ki

        losses['L_binding'] = l_binding
        losses['L_binding_kd'] = l_kd if 'l_kd' in locals() else torch.tensor(0.0, device=device)
        losses['L_binding_ki'] = l_ki if 'l_ki' in locals() else torch.tensor(0.0, device=device)

        # ── 2. kcat 损失 L_kcat (Eyring 硬编码在模型头内) ───
        kcat_mask = batch.get('kcat_target_mask', torch.zeros_like(pkd_mask))

        if kcat_mask.any():
            l_kcat = F.smooth_l1_loss(
                outputs['kcat_pred'][kcat_mask],
                batch['log_kcat_target'][kcat_mask],
            )
        else:
            l_kcat = torch.tensor(0.0, device=device)
        losses['L_kcat'] = l_kcat

        # ── 3. Km 损失 L_km ───────────────────────────────
        km_mask = batch.get('km_target_mask', torch.zeros_like(pkd_mask))

        if km_mask.any():
            # log_km_pred 是归一化 [0,1]，target 也归一化到 [0,1]
            km_target_norm = (batch['log_km_target'][km_mask] - self.KM_MIN) / (self.KM_MAX - self.KM_MIN)
            l_km = F.smooth_l1_loss(
                outputs['log_km_pred'][km_mask],
                km_target_norm,
            )
        else:
            l_km = torch.tensor(0.0, device=device)
        losses['L_km'] = l_km

        # ── 4. 联合约束 L_joint (kcat/Km 物理自洽) ─────────
        both_mask = kcat_mask & km_mask
        l_joint = torch.tensor(0.0, device=device)

        if both_mask.any():
            # 从预测算 log10(kcat/Km)
            log_kcat_denorm = self.denormalize(
                outputs['kcat_pred'][both_mask],
                self.KCAT_MIN, self.KCAT_MAX
            )
            log_km_denorm = self.denormalize(
                outputs['log_km_pred'][both_mask],
                self.KM_MIN, self.KM_MAX
            )
            log_kcatKm_pred = log_kcat_denorm - log_km_denorm

            # 真实 log10(kcat/Km)
            log_kcat_true = batch['log_kcat_target_denorm'][both_mask]
            log_km_true = batch['log_km_target'][both_mask]
            log_kcatKm_true = log_kcat_true - log_km_true

            # kcat/Km 回归损失
            l_joint_reg = F.smooth_l1_loss(log_kcatKm_pred, log_kcatKm_true)

            # 扩散极限约束: kcat/Km ≤ 10⁹ (log10 ≤ 9)
            l_joint_limit = F.relu(log_kcatKm_pred - 9.0).mean()

            l_joint = (l_joint_reg + 0.01 * l_joint_limit) * joint_weight

        losses['L_joint'] = l_joint

        # ── 总损失 ────────────────────────────────────────
        total = (
            binding_weight * l_binding +
            kcat_weight * l_kcat +
            km_weight * l_km +
            l_joint
        )
        losses['total'] = total

        return total, losses


def create_threehead_optimizer(model: nn.Module,
                                lr: float = 1e-4,
                                weight_decay: float = 1e-5,
                                ) -> torch.optim.Optimizer:
    """
    三头模型的分组优化器

    分组策略:
      - 编码器 (shared):    lr × 1.0
      - BINN 层:            lr × 0.5
      - 预测头 (各分支):     lr × 2.0
    """
    encoder_params = []
    binn_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'ligand_encoder' in name or 'protein_encoder' in name or 'cofactor_encoder' in name:
            encoder_params.append(param)
        elif 'binn' in name or 'initial_state_proj' in name or 'dynamics_net' in name:
            binn_params.append(param)
        elif 'kd_head' in name or 'ki_head' in name or 'kcat_head' in name or 'dG_predictor' in name or 'log_kappa' in name or 'km_head' in name or 'shared' in name or 'output_proj' in name or 'net' in name:
            head_params.append(param)
        else:
            encoder_params.append(param)  # fallback

    return torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': binn_params, 'lr': lr * 0.5, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr * 2.0, 'weight_decay': weight_decay},
    ])
