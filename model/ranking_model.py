# -*- coding: utf-8 -*-
"""
ranking_model.py
================
酶挖掘排序模型：基于蛋白序列嵌入 + 配体分子图 + 辅因子类型预测底物结合亲和力 (pKd)
+ 催化效率 (kcat, Eyring 硬编码) + 米氏常数 (Km)。

架构（v5，交叉注意力 + 残基级查询注意力）：

  LigandEncoder (GATv2×3, per-atom) + ProteinEncoder (ESM-2, per-token) + CofactorEncoder
    → CrossAttentionFusion (全酶[蛋白+辅因子] ↔ 配体原子)
    → [ResidueQueryAttention × 3] (每个头独立关注不同残基)
    → [BindingDualHead, EyringKcatHead, KmHead]

v5 新特性：
  1. 残基级查询注意力: 各预测头用可学习查询向量从全长残基序列中提取
     任务相关信息，告别全局池化 (丢弃 ~75% 噪声)
  2. 全酶交叉注意力: 辅因子作为虚拟 token 拼到 Q 侧（蛋白侧），
     构成全酶 (holoenzyme) 与底物交互

v4 新特性：
  1. 交叉注意力融合: 蛋白 token 级别 × 配体原子级别的 cross-attention
  2. 异方差损失 (NLL): 预测 mean + variance，带置信度的回归

v3 继承特性:
  - Eyring 公式硬编码: 模型预测 ΔG‡，Eyring 公式不可学习变换为 kcat
  - 每样本独立温度: temperature_K 传入 EyringKcatHead
  - BindingDualHead: Kd/Ki 双分支，测量类型 one-hot

预测头：
  1. BindingDualHead: BINN (1步) + Kd/Ki 双分支 → pKd （可选异方差 + 残基注意）
  2. EyringKcatHead:  BINN (多步ODE) + dG_predictor + Eyring 硬编码 → ΔG‡ → kcat
  3. KmHead:          MLP → log₁₀(Km)

用法：
  # v3 向后兼容模式（无交叉注意力，SmoothL1 损失）
  model = Trenzition(use_cross_attn=False, heteroscedastic=False)

  # v4 完全体（交叉注意力 + 异方差 NLL）
  model = Trenzition(use_cross_attn=True, heteroscedastic=True)

  # v5 完全体（交叉注意力 + 残基级查询注意力 + 异方差）
  model = Trenzition(use_cross_attn=True, use_residue_attn=True, heteroscedastic=True)
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, AttentionalAggregation, global_mean_pool
from torch_geometric.utils import to_dense_batch

log = logging.getLogger(__name__)

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
    """
    配体编码器，GATv2 配体分子图编码。

    GATv2 (default):
      配体分子图 GNN 编码器 (GATv2 × 3)。


    """

    def __init__(self,
                 atom_dim: int = 79,
                 edge_dim: int = 10,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.1,
                 return_per_node: bool = False):
        super().__init__()
        self.return_per_node = return_per_node

        # GATv2 路径
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

        self.global_attn = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ),
            nn=nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data):
        # GATv2 路径
        x = self.atom_proj(data.x)
        e = self.edge_proj(data.edge_attr)

        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, data.edge_index, e)
            x_new = bn(x_new)
            x = x + x_new
            x = F.silu(x)

        if self.return_per_node:
            return x  # (total_atoms, hidden_dim)
        else:
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
    蛋白质编码器 — 仅 token 路径 (mean-pooled 路径已删除).

    架构:
      1. token_proj: (B, L, 1280) → (B, L, hidden_dim)
      2. AttentionPooling: 可学习的残基级注意力权重
         score = Linear(hidden_dim → 1), softmax, weighted sum → (B, hidden_dim)

    输出:
      - pooled_h: (B, hidden_dim) 注意力池化后的蛋白表示
      - token_h:  (B, L, hidden_dim) token 级表示（交叉注意力用）
    """

    def __init__(self,
                 seq_embed_dim: int = 1280,
                 hidden_dim: int = 256,
                 protein_dim: Optional[int] = None,
                 ):
        super().__init__()
        self.protein_dim = protein_dim or hidden_dim

        # Token 投影: (B, L, 1280) → (B, L, protein_dim)
        # 蛋白保留更高维避免信息压缩过度
        self.token_proj = nn.Sequential(
            nn.Linear(seq_embed_dim, self.protein_dim),
            nn.LayerNorm(self.protein_dim),
            nn.GELU(),
            nn.Linear(self.protein_dim, self.protein_dim),
        )

        # 可学习的注意力池化: 每个残基学习一个重要性权重
        self.attn_score = nn.Linear(self.protein_dim, 1, bias=False)

    def forward(self, token_embed: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embed: (B, L, 1280) 每残基 token 级 ESM-2 嵌入
            mask: (B, L) 布尔掩码, True=有效残基, False=填充
        Returns:
            pooled_h: (B, protein_dim) 注意力池化后的蛋白表示
            token_h:  (B, L, protein_dim) token 级表示
        """
        token_h = self.token_proj(token_embed)  # (B, L, protein_dim)

        # 注意力分数: (B, L, 1)
        scores = self.attn_score(token_h)
        # 掩码: 填充位置设为 -inf 使 softmax 后为 0
        scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        attn_weights = torch.softmax(scores, dim=1)  # (B, L, 1)

        # 加权求和
        pooled_h = (attn_weights * token_h).sum(dim=1)  # (B, protein_dim)

        return pooled_h, token_h


# ═══════════════════════════════════════════════════════════════════════════════
# TemperatureEncoder: 标量温度 → hidden_dim 嵌入
# ═══════════════════════════════════════════════════════════════════════════════


class TemperatureEncoder(nn.Module):
    """
    温度编码器: 将标量温度映射到 hidden_dim 特征向量.

    归一化: (T - 298.15) / 50  →  ~[-0.44, 2.88] 范围
    加入催化剂上下文作为残差调制:
      catalyst_context = cofactor_h_proj + temperature_h

    Eyring 公式仍接收原始 temperature_K 用于物理转换, 两者正交.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, hidden_dim),
        )

    def forward(self, temperature_K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temperature_K: (B,) 温度 (K)
        Returns:
            temp_h: (B, hidden_dim) 温度嵌入
        """
        t_norm = (temperature_K - 298.15) / 50.0  # (B,)
        return self.net(t_norm.unsqueeze(-1))  # (B, hidden_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# ResidueQueryAttention: 可学习查询 × 残基序列 交叉注意力（替代池化）
# ═══════════════════════════════════════════════════════════════════════════════


class ResidueQueryAttention(nn.Module):
    """
    可学习查询向量 × 蛋白残基序列的交叉注意力。

    架构:
      query (1, 1, D) — 可学习
        ×  protein_tokens (B, L, D) — 残基序列
        → MultiheadCrossAttention(query=查询, K/V=残基)
        → LayerNorm → (B, D)

    每个预测头拥有独立的查询向量，因此可以关注不同的残基集合：
      - BindingHead 关注结合位点（与底物有接触的残基）
      - KcatHead    关注催化位点（过渡态稳定残基 + 辅因子位置）
      - KmHead      关注底物结合 + 解离相关残基

    参考:
      - SetTransformer (Lee et al., 2019): 用可学习种子向量做交叉注意力
      - 本质上是可微的 top-K 选择：查询向量与哪个残基最相似，就提取哪个残基
    """

    def __init__(self, hidden_dim: int = 512, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                residue_tokens: torch.Tensor,
                mask: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            residue_tokens: (B, L, D) 残基级表示
            mask: (B, L) 布尔掩码，True=有效

        Returns:
            attended: (B, D) 从残基中提取的任务相关表示
            attn_w:   (B, 1, L) 每个残基的注意力权重（可解释性分析用）
        """
        B = residue_tokens.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, D)

        out, attn_w = self.cross_attn(
            query=q,
            key=residue_tokens,
            value=residue_tokens,
            key_padding_mask=~mask,
            need_weights=True,
        )
        # out: (B, 1, D), attn_w: (B, 1, L) (average_attn_weights=True, L_q=1)
        return self.norm(out).squeeze(1), attn_w  # (B, D), (B, 1, L)

class CrossAttentionFusion(nn.Module):
    """
    全酶（蛋白+辅因子）↔ 配体交叉注意力融合模块。

    架构:
      Q  = [protein_tokens (L, D); cofactor_token (1, D)]  — 全酶（holoenzyme）
      K/V = ligand_atoms (A, D)                             — 底物
        → MultiheadCrossAttention(Q=全酶残基+辅因子, K/V=配体原子)
        → LayerNorm + FFN
        → AttentionPooling（只池化蛋白残基，辅因子 token 提供上下文但不被池化）
        → (B, D)

    物理意义:
      - 辅因子是酶的一部分（全酶），与蛋白残基一起构成 "查询方"
      - 每个蛋白残基和辅因子 token 可以关注不同的配体原子
      - 辅因子 token 作为活性位点的"上下文"调制注意力模式：
        同一个蛋白绑了 NADH vs FAD，对同一底物的注意力分布应该不同
      - 池化时辅因子 token 只做调制、不计入最终表征（其影响已通过
        cross-attention 传递给蛋白残基）

    参考:
      - ERBA (2025) 中的 MRCA 模块 + 辅因子作为酶的一部分
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 protein_dim: Optional[int] = None,
                 ):
        super().__init__()
        protein_dim = protein_dim or hidden_dim

        # 非对称交叉注意力: Q(全酶) 在 protein_dim, K/V(配体) 在 hidden_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=protein_dim,
            kdim=hidden_dim,
            vdim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(protein_dim)

        self.ffn = nn.Sequential(
            nn.Linear(protein_dim, protein_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(protein_dim)

        # 可学习的残基级池化权重（只池化蛋白残基，不池化辅因子 token）
        self.attn_score = nn.Linear(protein_dim, 1, bias=False)

    def forward(
        self,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_atoms: torch.Tensor,
        ligand_mask: torch.Tensor,
        cofactor_token: torch.Tensor,
        return_tokens: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            protein_tokens: (B, L, D) 蛋白残基表示
            protein_mask: (B, L) 布尔掩码，True=有效残基，False=填充
            ligand_atoms: (B, A, D) 配体原子表示
            ligand_mask: (B, A) 布尔掩码，True=有效原子，False=填充
            cofactor_token: (B, 1, D) 辅因子表示，拼到 Q 侧作为全酶一部分
            return_tokens: 为 True 时额外返回 per-residue 表示

        Returns:
            fused: (B, D) 池化后的全酶表示
            token_h: (B, L, D) 或 None — 池化前的残基级表示
            attn_weights: (B, L+1, A) 注意力权重（最后一行为辅因子）
        """
        B, L, D = protein_tokens.shape
        device = protein_tokens.device

        # ── Q = [蛋白残基; 辅因子虚拟 token]  → 构成全酶 ──
        q = torch.cat([protein_tokens, cofactor_token], dim=1)  # (B, L+1, D)
        q_mask = torch.cat([
            protein_mask,
            torch.ones(B, 1, dtype=torch.bool, device=device),
        ], dim=1)  # (B, L+1)

        # MultiheadAttention key_padding_mask: True=忽略该位置
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=ligand_atoms,
            value=ligand_atoms,
            key_padding_mask=~ligand_mask,  # True=padding, 忽略
            need_weights=True,
            average_attn_weights=True,  # (B, L+1, A)
        )

        # 残差 + LayerNorm（仍在全酶序列上）
        h = self.norm1(q + attn_out)

        # FFN + 残差
        h = self.norm2(h + self.ffn(h))

        # 提取残基部分 [0:L] 用于池化和下游
        h_protein_only = h[:, :L, :]  # (B, L, D)

        # AttentionPooling：只池化蛋白残基部分 [0:L]
        # 辅因子 token 已通过 cross-attention 调制了蛋白残基的表示
        scores = self.attn_score(h_protein_only)  # (B, L, 1)
        scores = scores.masked_fill(~protein_mask.unsqueeze(-1), float('-inf'))
        attn_w = torch.softmax(scores, dim=1)  # (B, L, 1)
        fused = (attn_w * h_protein_only).sum(dim=1)  # (B, D)

        token_h = h_protein_only if return_tokens else None
        return fused, token_h, attn_weights



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

    def __init__(self, hidden_dim: int = 256, n_ode_steps: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_steps = n_ode_steps
        self.dropout = dropout

        # Step 1: 初始状态构建
        self.initial_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # protein + ligand
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Step 2: 动力学函数
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # h + catalyst + ligand
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.dxi = 1.0 / max(n_ode_steps, 1)

        # Step 3: 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self,
                protein_h: torch.Tensor,
                ligand_h: torch.Tensor,
                catalyst_context: torch.Tensor,
                ) -> dict:
        B = protein_h.size(0)

        # 催化剂上下文 = 酶 + 辅因子 + 温度
        catalyst_h = protein_h + catalyst_context

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
    Trenzition v4: 三头独立预测 + 交叉注意力融合 + 异方差损失

    v4 新特性:
      - Cross-attention: 蛋白残基 token ↔ 配体原子
      - Heteroscedastic loss: 预测 mean + variance, NLL 损失
      - 向后兼容: use_cross_attn=False, heteroscedastic=False = v3 行为

    预测头:
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
        dropout: float = 0.1,
        # ── v4 新参数 ──
        use_cross_attn: bool = False,
        cross_attn_heads: int = 4,
        heteroscedastic: bool = False,
        # ── v5 参数 ──
        use_residue_attn: bool = False,
        # ── 多任务平衡 ──
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cofactor_embed_dim = cofactor_embed_dim
        self.three_head = three_head
        self.use_cross_attn = use_cross_attn
        self.heteroscedastic = heteroscedastic
        self.use_residue_attn = use_residue_attn
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # ── 不确定性加权: 三任务可学习噪声参数 ──
        if use_uncertainty_weighting:
            self.task_log_sigma = nn.Parameter(torch.zeros(3))
        else:
            self.task_log_sigma = None

        # ─────── 编码器 ───────
        self.ligand_encoder = LigandEncoder(
            atom_dim=79, edge_dim=10, hidden_dim=hidden_dim,
            num_layers=gnn_layers, heads=n_heads, dropout=dropout,
            return_per_node=use_cross_attn,  # 交叉注意力需要原子级输出
        )
        self.protein_encoder = ProteinEncoder(
            seq_embed_dim=1280, hidden_dim=hidden_dim,
        )
        self.cofactor_encoder = CofactorEncoder(
            cofactor_types=self.COFACTOR_TYPES, embed_dim=cofactor_embed_dim,
        )

        self.cofactor_proj = nn.Linear(cofactor_embed_dim, hidden_dim)
        self.temperature_encoder = TemperatureEncoder(hidden_dim=hidden_dim)

        # ─── v4 交叉注意力融合 ───
        if use_cross_attn:
            # 配体原子级投影 (128→hidden_dim)
            self.ligand_atom_proj = nn.Linear(hidden_dim, hidden_dim)

            # 交叉注意力融合
            self.fusion = CrossAttentionFusion(
                hidden_dim=hidden_dim,
                n_heads=cross_attn_heads,
                dropout=dropout,
            )

        # ─────── 预测头 ───────
        if three_head:
            self.binding_head = BindingDualHead(
                hidden_dim=hidden_dim, dropout=dropout,
                heteroscedastic=heteroscedastic,
                use_residue_attn=use_residue_attn,
            )
            self.kcat_head = EyringKcatHead(
                hidden_dim=hidden_dim, n_ode_steps=kcat_ode_steps, dropout=dropout,
                heteroscedastic=heteroscedastic,
                use_residue_attn=use_residue_attn,
            )
            self.km_head = KmHead(
                hidden_dim=hidden_dim, dropout=dropout,
                heteroscedastic=heteroscedastic,
                use_residue_attn=use_residue_attn,
            )
            self.loss_fn = ThreeHeadLoss(heteroscedastic=heteroscedastic)
        else:
            # 旧单头模式（向后兼容）
            self.kcat_head = EyringKcatHead(
                hidden_dim=hidden_dim, n_ode_steps=n_ode_steps, dropout=dropout,
            )
            self.loss_fn = ThreeHeadLoss()

        # ─────── 参数初始化 ────────────────────────────────────
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        """统一的参数初始化策略。

        原则：
        - 输出头最后一层 (out_dim=1, 接 sigmoid)：用小 gain 避免饱和
        - v4 异方差输出头最后一层 (out_dim=2)：小 gain，因 log_var 需要稳定
        - 中间层 (接 GELU/SiLU)：Xavier uniform
        - BINN 动力学层：保守 gain 防止 ODE 多步积分放大
        - LayerNorm：保持默认 (1.0, 0.0)
        """
        if isinstance(module, nn.Linear):
            is_output_head = (
                module.out_features == 1
                and module.in_features == 64
            )
            is_hetero_head = (
                module.out_features == 2
                and module.in_features == 64
            )
            # BINN 动力学层检测: in ≥ 256 且 < 1280 且 out=256
            # 原 `in==out==256` 漏了 dynamics_net 中的 Linear(768→256)
            # 放宽检测: 捕获所有 BINN 相关的大维度层
            is_dynamics = (
                module.in_features >= 256
                and module.in_features < 1280
                and module.out_features == 256
            )
            if is_output_head or is_hetero_head:
                # sigmoid/log_var 前的最后映射 → 小初始化防饱和
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
                protein_tokens: torch.Tensor,
                protein_mask: torch.Tensor,
                cofactor_strs: list[str],
                measurement_types: torch.Tensor = None,
                temperature_K: torch.Tensor = None,
                ) -> dict[str, torch.Tensor]:
        """三头预测（token-only 模式, mean-pooled 路径已删除）

        Args:
            ligand_data: PyG batch，配体分子图
            protein_tokens: (B, L, 1280) token-level ESM-2 嵌入
            protein_mask: (B, L) 蛋白 token 有效掩码
            cofactor_strs: 辅因子字符串列表
            measurement_types: (B,) 测量类型编码
            temperature_K: (B,) 温度 (K)

        Returns:
            dict: 包含所有预测头输出
        """

        # ── 蛋白编码: (B, L, 1280) → attention pool → (B, hidden_dim) ──

        # 辅因子编码（cross-attention / 简单路径都用）
        cofactor_h, _ = self.cofactor_encoder(cofactor_strs)
        cofactor_h_proj = self.cofactor_proj(cofactor_h)

        protein_tokens_h = None  # 初始化，use_residue_attn=False 时保持 None

        if self.use_cross_attn:
            # 交叉注意力路径: 需要 token 级表示
            ligand_raw = self.ligand_encoder(ligand_data)
            ligand_atoms, ligand_mask = to_dense_batch(ligand_raw, ligand_data.batch)
            ligand_atoms = self.ligand_atom_proj(ligand_atoms)
            ligand_h = (ligand_atoms * ligand_mask.unsqueeze(-1).float()).sum(dim=1) \
                       / ligand_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            protein_pooled_h, protein_token_h = self.protein_encoder(protein_tokens, protein_mask)
            # LayerNorm 对齐蛋白和配体表示（稳定交叉注意力）
            protein_token_h = F.layer_norm(protein_token_h, [protein_token_h.size(-1)])
            # 辅因子投影到 hidden_dim 后作为虚拟 token 参与交叉注意力
            cofactor_token = cofactor_h_proj.unsqueeze(1)  # (B, 1, hidden_dim)

            fused_h, protein_tokens_h, attn_weights = self.fusion(
                protein_token_h, protein_mask, ligand_atoms, ligand_mask,
                cofactor_token=cofactor_token,
                return_tokens=self.use_residue_attn,  # 只在需要时返回残基级
            )
            protein_h = fused_h
        else:
            # 简单模式: GATv2 mean-pooled 配体 + 注意力池化蛋白
            ligand_h = self.ligand_encoder(ligand_data)  # (B, hidden_dim)
            if self.use_residue_attn:
                protein_h, protein_tokens_h = self.protein_encoder(
                    protein_tokens, protein_mask
                )
            else:
                protein_h, _ = self.protein_encoder(protein_tokens, protein_mask)
                protein_tokens_h = None

        # ── 温度编码 ──
        temp_h = self.temperature_encoder(temperature_K)  # (B, hidden_dim)

        # ── 催化剂上下文（含温度） ──
        catalyst_context = cofactor_h_proj + temp_h  # (B, hidden_dim)

        # ── 三个独立预测头 ──
        binding_out = self.binding_head(
            protein_h, ligand_h, catalyst_context, measurement_types,
            protein_tokens=protein_tokens_h, protein_mask=protein_mask,
        )
        kcat_out = self.kcat_head(
            protein_h, ligand_h, catalyst_context, temperature_K,
            protein_tokens=protein_tokens_h, protein_mask=protein_mask,
        )
        km_out = self.km_head(
            protein_h, ligand_h, catalyst_context,
            protein_tokens=protein_tokens_h, protein_mask=protein_mask,
        )

        return {
            **binding_out, **kcat_out, **km_out,
            'protein_h': protein_h, 'ligand_h': ligand_h, 'cofactor_h': cofactor_h,
            'temp_h': temp_h,
        }

    def compute_loss(self,
                     outputs: dict,
                     batch: dict,
                     binding_weight: float = 1.0,
                     joint_weight: float = 0.1,
                     kcat_weight: float = 1.0,
                     km_weight: float = 1.0,
                     dG_prior_weight: float = 0.05,
                     joint_km_weight: float = 0.0,
                     ) -> tuple[torch.Tensor, dict]:
        """三头联合损失: L_binding + L_kcat + L_km + L_joint + L_dG_prior

        当 use_uncertainty_weighting=True 时，各头 loss 除以对应 σ² + log(σ)，
        σ 自动学习以平衡不同数据量的任务。
        """
        total, losses = self.loss_fn(outputs, batch,
                                     binding_weight=binding_weight,
                                     joint_weight=joint_weight,
                                     kcat_weight=kcat_weight, km_weight=km_weight,
                                     joint_km_weight=joint_km_weight)

        # ── 不确定性加权: 自动平衡各任务噪声水平 ──
        if self.use_uncertainty_weighting and self.task_log_sigma is not None:
            # clamp log_sigma ∈ [-3, 3] → σ ∈ [0.05, 20]，防止极端值
            sigma = self.task_log_sigma.clamp(-3.0, 3.0).exp()  # (3,)

            # 原 loss 值（从 losses dict 里读，已在 loss_fn 中加权）
            raw_binding = losses.get('L_binding', torch.tensor(0.0, device=total.device))
            raw_kcat    = losses.get('L_kcat',    torch.tensor(0.0, device=total.device))
            raw_km      = losses.get('L_km',      torch.tensor(0.0, device=total.device))

            # 不确定性加权: L_i / (2·σ_i²) + log(σ_i)
            l_binding_w = raw_binding / (2.0 * sigma[0] ** 2) + torch.log(sigma[0])
            l_kcat_w    = raw_kcat    / (2.0 * sigma[1] ** 2) + torch.log(sigma[1])
            l_km_w      = raw_km      / (2.0 * sigma[2] ** 2) + torch.log(sigma[2])

            # 重新计算 total
            total = (binding_weight * l_binding_w
                     + kcat_weight * l_kcat_w
                     + km_weight * l_km_w
                     + losses.get('L_joint', torch.tensor(0.0, device=total.device)))

            # 记录 σ 值用于日志
            losses['task_sigma_binding'] = sigma[0].detach()
            losses['task_sigma_kcat'] = sigma[1].detach()
            losses['task_sigma_km'] = sigma[2].detach()

        # ── ΔG‡ 先验正则化（在 Trenzition 层处理而非丟进 ThreeHeadLoss）──
        l_dG_prior = torch.tensor(0.0, device=total.device if isinstance(total, torch.Tensor) else total)
        if dG_prior_weight > 0 and 'dG_eyring' in outputs:
            dG = outputs['dG_eyring']
            if dG.numel() > 0:
                dG_prior = 70.0
                dG_std = 20.0
                l_dG_prior = dG_prior_weight * ((dG - dG_prior) / dG_std).pow(2).mean()
        losses['L_dG_prior'] = l_dG_prior
        total = total + l_dG_prior

        return total, losses


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

    v4 新增:
      - heteroscedastic=True: 输出 (mean, log_var)，NLL 损失
      - 异方差模式下 Kd/Ki 各输出 2 维 [mean_raw, log_var_raw]

    v5 新增:
      - use_residue_attn=True: 用 ResidueQueryAttention 从残基序列提取
        表示，取代全局池化的 protein_h。每个头独立学习关注哪些残基。
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1,
                 heteroscedastic: bool = False,
                 use_residue_attn: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.use_residue_attn = use_residue_attn

        if use_residue_attn:
            self.residue_attn = ResidueQueryAttention(
                hidden_dim=hidden_dim, n_heads=4, dropout=dropout,
            )

        # BINN-like 初始状态融合
        self.initial_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # protein + ligand
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 1步动力学 (无 ODE 迭代)
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # 输出投影 (shortcut: concat h_final + h0)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Kd 分支
        out_dim = 2 if heteroscedastic else 1
        self.kd_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),  # +2 for type one-hot
            nn.SiLU(),
            nn.Linear(64, out_dim),
        )

        # Ki 分支
        self.ki_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self,
                protein_h: torch.Tensor,
                ligand_h: torch.Tensor,
                catalyst_context: torch.Tensor,
                measurement_types: torch.Tensor = None,
                protein_tokens: torch.Tensor = None,
                protein_mask: torch.Tensor = None,
                ) -> dict[str, torch.Tensor]:
        """
        Args:
            protein_h: (B, D) 酶表征（use_residue_attn=False 时使用）
            ligand_h: (B, D) 底物表征
            catalyst_context: (B, D) 辅因子 + 温度上下文
            measurement_types: (B,) 测量类型 {0=Kd, 1=Ki, 2=IC50, 3=none}
            protein_tokens: (B, L, D) 残基级表示（use_residue_attn=True 时使用）
            protein_mask: (B, L) 布尔掩码，True=有效

        Returns:
            kd_pred / ki_pred: (B,) 或 (B, 2) 取决于 heteroscedastic 标志
            binding_pred: (B,) 按测量类型选通的输出
        """
        B = protein_h.size(0)
        device = protein_h.device

        # ── 可选：用残基级注意力提取替代池化后的表示 ──
        if self.use_residue_attn and protein_tokens is not None and protein_mask is not None:
            protein_h, _ = self.residue_attn(protein_tokens, protein_mask)

        # 催化剂上下文 = 酶 + 辅因子 + 温度
        catalyst_h = protein_h + catalyst_context

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

        # 双分支输出
        kd_out = self.kd_head(h_with_type)  # (B, 1) or (B, 2)
        ki_out = self.ki_head(h_with_type)  # (B, 1) or (B, 2)

        if self.heteroscedastic:
            # 异方差: [mean_raw, log_var_raw]
            kd_mean = torch.sigmoid(kd_out[:, 0])
            kd_log_var = kd_out[:, 1]  # 无约束
            ki_mean = torch.sigmoid(ki_out[:, 0])
            ki_log_var = ki_out[:, 1]

            # 堆叠为 (B, 2) 方便损失函数读取
            kd_pred = torch.stack([kd_mean, kd_log_var], dim=-1)
            ki_pred = torch.stack([ki_mean, ki_log_var], dim=-1)

            # binding_pred 使用对应分支的 mean（用于评估）
            binding_pred = torch.where(
                measurement_types == MTYPE_KI if measurement_types is not None
                else torch.zeros(B, dtype=torch.bool, device=device),
                ki_mean,
                kd_mean,
            )
        else:
            # 标准: sigmoid → [0, 1]
            kd_pred = torch.sigmoid(kd_out.squeeze(-1))  # (B,)
            ki_pred = torch.sigmoid(ki_out.squeeze(-1))  # (B,)

            binding_pred = torch.where(
                measurement_types == MTYPE_KI if measurement_types is not None
                else torch.zeros(B, dtype=torch.bool, device=device),
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

    v4 新增:
      - heteroscedastic: dG_predictor 输出 (ΔG‡_mean, ΔG‡_log_var)
      - 方差通过 Eyring 公式解析传播到 kcat

    v5 新增:
      - use_residue_attn=True: 从残基序列提取催化相关表示

    方差传播:
      log10(kcat) = log10(κ) + log10(k_B·T/h) - ΔG‡ / (R·T·ln10)
      var(log10(kcat)) = var(ΔG‡) / (R·T·ln10)²
    """

    # 物理常数
    R_kJ = 8.314e-3          # kJ/(mol·K)
    k_B = 1.380649e-23       # J/K
    h_planck = 6.62607015e-34  # J·s

    def __init__(self, hidden_dim: int = 256, n_ode_steps: int = 10,
                 dropout: float = 0.1, heteroscedastic: bool = False,
                 use_residue_attn: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.use_residue_attn = use_residue_attn

        if use_residue_attn:
            self.residue_attn = ResidueQueryAttention(
                hidden_dim=hidden_dim, n_heads=4, dropout=dropout,
            )

        # BINN (完整多步 ODE)
        self.binn = LatentPathwayBINN(
            hidden_dim=hidden_dim,
            n_ode_steps=n_ode_steps,
            dropout=dropout,
        )

        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ΔG‡ 预测器
        if heteroscedastic:
            # 输出 [dG_raw, dG_log_var_raw]
            dG_out_dim = 2
        else:
            dG_out_dim = 1

        self.dG_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, dG_out_dim),
        )

        # 可学习的全局透射系数: κ = 10^{log_kappa}
        self.log_kappa = nn.Parameter(torch.tensor(math.log10(0.5)))

    def forward(self,
                protein_h: torch.Tensor,
                ligand_h: torch.Tensor,
                catalyst_context: torch.Tensor,
                temperature_K: torch.Tensor,  # (B,) 每样本独立温度, Eyring 公式用
                protein_tokens: torch.Tensor = None,
                protein_mask: torch.Tensor = None,
                ) -> dict[str, torch.Tensor]:
        """
        Args:
            catalyst_context: (B, D) 辅因子 + 温度上下文
            temperature_K: (B,) 温度 (K)，传入 Eyring 公式做物理转换
            protein_tokens: (B, L, D) 残基级表示（use_residue_attn=True 时使用）
            protein_mask: (B, L) 布尔掩码，True=有效

        Returns:
            kcat_pred: (B,) 或 (B, 2) 归一化 log10(kcat)
            dG_eyring: (B,) 活化自由能 ΔG‡ [kJ/mol]
        """
        device = protein_h.device

        # ── 可选：用残基级注意力提取替代池化后的表示 ──
        if self.use_residue_attn and protein_tokens is not None and protein_mask is not None:
            protein_h, _ = self.residue_attn(protein_tokens, protein_mask)

        # 1) BINN 特征演化 (催化剂上下文 = cofactor + temp)
        binn_out = self.binn(protein_h, ligand_h, catalyst_context)
        h = self.shared(binn_out['h_reaction'])

        # 2) 预测 ΔG‡
        dG_out = self.dG_predictor(h)  # (B, 1) or (B, 2)

        # ΔG‡ 映射: ELU + x/(x+1) bounded mapping
        # sigmoid 在 |x|>5 时梯度≈0，模型学不到极端 kcat
        # ELU: x>0 时梯度=1（线性），x<0 时梯度=exp(x)（平滑衰减）
        # x/(x+1): [0,∞) → [0,1) 梯度始终非零
        # 结果: 模型可以自由学习 [5, 300] 范围内的任意 dG 值
        def _dg_from_raw(dG_raw_in: torch.Tensor) -> torch.Tensor:
            offset = F.elu(dG_raw_in) + 1.0  # [0+, ∞)
            return 5.0 + 295.0 * (offset / (offset + 1.0))  # [5, 300]

        if self.heteroscedastic:
            # 异方差: dG_out = [dG_raw, dG_log_var_raw]
            dG_raw = dG_out[:, 0]
            dG_log_var_raw = dG_out[:, 1]
            dG = _dg_from_raw(dG_raw)

            # 方差传播: g(x) = 5 + 295 * f(ELU(x)+1), f(z)=z/(z+1)
            # 对 x>0: g'(x) = 295 / (x+2)²
            # 对 x<0: g'(x) = 295 * exp(x) / (exp(x)+2)²
            x_pos = dG_raw > 0
            grad_factor = torch.where(
                x_pos,
                295.0 / (dG_raw + 2.0).pow(2),
                295.0 * torch.exp(dG_raw) / (torch.exp(dG_raw) + 2.0).pow(2),
            )
            var_dG = grad_factor.pow(2) * torch.exp(dG_log_var_raw) + 1e-8
        else:
            dG_raw = dG_out.squeeze(-1)
            dG = _dg_from_raw(dG_raw)

        # 3) Eyring 公式硬编码 (批量计算)
        RT_ln10 = self.R_kJ * temperature_K * math.log(10)  # (B,)

        # log10(κ·k_B·T/h) = log_kappa + log10(k_B·T/h)
        log10_prefactor = self.log_kappa + torch.log10(
            self.k_B * temperature_K / self.h_planck
        )  # (B,)

        log_kcat = log10_prefactor - dG / RT_ln10  # (B,)

        if self.heteroscedastic:
            # 方差传播: var(log10(kcat)) = var(dG) / (R·T·ln10)²
            var_log_kcat = var_dG / (RT_ln10 ** 2)  # (B,)
            log_var_log_kcat = torch.log(var_log_kcat + 1e-8)  # (B,)

        # 4) 归一化到 [0,1] 适配标签
        # 不用 clamp —— 避免训练初期梯度消失（kcat 初始在 0 以下）
        # sigmoid(dG) 已保证 dG [5,300]，log_kcat 自然有界
        kcat_mean_norm = ((log_kcat + 7.0) / 15.0)

        if self.heteroscedastic:
            # 归一化方差传播: var_norm = var / 15²
            kcat_log_var_norm = log_var_log_kcat - 2 * math.log(15.0)
            kcat_pred = torch.stack([kcat_mean_norm, kcat_log_var_norm], dim=-1)
        else:
            kcat_pred = kcat_mean_norm

        return {
            'kcat_pred': kcat_pred,
            'dG_eyring': dG,
            'kcat_binn_trajectory': binn_out.get('trajectory'),
            'kcat_feature_evol': binn_out.get('feature_evol'),
        }


class KmHead(nn.Module):
    """
    Km 预测头 (简单 MLP，归一化 [0,1])

    v4 新增:
      - heteroscedastic: 输出 (mean, log_var)

    v5 新增:
      - use_residue_attn=True: 从残基序列提取底物结合相关表示
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1,
                 heteroscedastic: bool = False,
                 use_residue_attn: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.use_residue_attn = use_residue_attn

        if use_residue_attn:
            self.residue_attn = ResidueQueryAttention(
                hidden_dim=hidden_dim, n_heads=4, dropout=dropout,
            )

        # 中间层共享
        self.net_shared = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        out_dim = 2 if heteroscedastic else 1
        self.net_out = nn.Linear(hidden_dim // 2, out_dim)

    def forward(self,
                protein_h: torch.Tensor,
                ligand_h: torch.Tensor,
                catalyst_context: torch.Tensor,
                protein_tokens: torch.Tensor = None,
                protein_mask: torch.Tensor = None,
                ) -> dict[str, torch.Tensor]:
        """
        Returns:
            log_km_pred: (B,) 或 (B, 2) 预测的 log₁₀(Km)
        """
        # ── 可选：用残基级注意力提取替代池化后的表示 ──
        if self.use_residue_attn and protein_tokens is not None and protein_mask is not None:
            protein_h, _ = self.residue_attn(protein_tokens, protein_mask)

        x = torch.cat([protein_h, ligand_h, catalyst_context], dim=-1)
        h = self.net_shared(x)

        if self.heteroscedastic:
            raw = self.net_out(h)  # (B, 2)
            km_mean = torch.sigmoid(raw[:, 0])
            km_log_var = raw[:, 1]  # 无约束
            log_km_pred = torch.stack([km_mean, km_log_var], dim=-1)
        else:
            raw = self.net_out(h).squeeze(-1)
            log_km_pred = torch.sigmoid(raw)

        return {'log_km_pred': log_km_pred}


class ThreeHeadLoss(nn.Module):
    """
    三头独立预测的联合损失函数（v4: 支持异方差 NLL）

    损失组成:
    1. L_binding: 双分支结合亲和力回归（SmoothL1 或 NLL）
    2. L_kcat: 催化速率回归（SmoothL1 或 NLL）
    3. L_km: Km 回归（SmoothL1 或 NLL）
    4. L_joint: kcat/Km 联合约束（物理自洽，始终用 SmoothL1）

    v4 异方差模式:
      - heteroscedastic=True: 各头预测 (mean, log_var)，用 NLL 损失
      - heteroscedastic=False: 仅预测 mean，用 SmoothL1（v3 兼容）
    """

    # 归一化参数
    PKD_MIN = 0.0
    PKD_MAX = 12.0
    KCAT_MIN = -7.0
    KCAT_MAX = 8.0
    KM_MIN = -13.0
    KM_MAX = 3.0

    def __init__(self, heteroscedastic: bool = False):
        super().__init__()
        self.heteroscedastic = heteroscedastic

    @staticmethod
    def denormalize(value: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        """反归一化 [0,1] → 原始值"""
        return value * (vmax - vmin) + vmin

    @staticmethod
    def _smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """封装 SmoothL1 损失"""
        return F.smooth_l1_loss(pred, target)

    @staticmethod
    def _nll_loss(pred_mean: torch.Tensor, pred_log_var: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
        """
        异方差 NLL 损失（在归一化空间内计算）

        NLL = 0.5 * log(2π) + 0.5 * log_var + 0.5 * (y-μ)² / exp(log_var)
        常数项 0.5*log(2π) ≈ 0.9189 省略（不影响梯度）
        """
        variance = torch.exp(pred_log_var) + 1e-6
        return (0.5 * pred_log_var + 0.5 * (pred_mean - target) ** 2 / variance).mean()

    def _regression_loss(self, pred, target, mask):
        """根据模式选择 SmoothL1 或 NLL 损失"""
        if not mask.any():
            return torch.tensor(0.0, device=pred.device if isinstance(pred, torch.Tensor) else 'cpu')

        if self.heteroscedastic:
            # pred: (N, 2) = [mean, log_var]
            return self._nll_loss(pred[mask, 0], pred[mask, 1], target[mask])
        else:
            # pred: (N,)
            return self._smooth_l1(pred[mask], target[mask])

    def forward(self,
                outputs: dict,
                batch: dict,
                binding_weight: float = 1.0,
                kcat_weight: float = 1.0,
                km_weight: float = 1.0,
                joint_weight: float = 0.1,
                joint_km_weight: float = 0.1,
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
        pkd_mask = batch.get('pkd_target_mask',
                             torch.zeros(batch.get('pkd_target', torch.empty(0)).size(0),
                                         dtype=torch.bool, device=device))
        mtype = batch.get('measurement_type', None)

        l_binding = torch.tensor(0.0, device=device)
        if pkd_mask.any() and pkd_target is not None:
            has_kd = pkd_mask & (mtype == MTYPE_KD) if mtype is not None else pkd_mask
            has_ki = pkd_mask & (mtype == MTYPE_KI) if mtype is not None else torch.zeros_like(pkd_mask)

            # Kd 分支损失 (权重 1.0)
            if has_kd.any():
                l_kd = self._regression_loss(
                    outputs['kd_pred'], pkd_target, has_kd
                )
            else:
                l_kd = torch.tensor(0.0, device=device)

            # Ki 分支损失 (权重 0.3 — Ki pKd 中位数仅 1.68 vs Kd 的 5.09，信号弱)
            if has_ki.any():
                l_ki = 0.3 * self._regression_loss(
                    outputs['ki_pred'], pkd_target, has_ki
                )
            else:
                l_ki = torch.tensor(0.0, device=device)

            l_binding = l_kd + l_ki

        losses['L_binding'] = l_binding
        losses['L_binding_kd'] = l_kd if 'l_kd' in locals() else torch.tensor(0.0, device=device)
        losses['L_binding_ki'] = l_ki if 'l_ki' in locals() else torch.tensor(0.0, device=device)

        # ── 2. kcat 损失 L_kcat ───────────────────────────
        kcat_mask = batch.get('kcat_target_mask',
                              torch.zeros_like(pkd_mask) if 'pkd_target_mask' in batch
                              else torch.zeros(batch.get('log_kcat_target', torch.empty(0)).size(0),
                                               dtype=torch.bool, device=device))

        if kcat_mask.any():
            l_kcat = self._regression_loss(
                outputs['kcat_pred'], batch['log_kcat_target'], kcat_mask
            )
        else:
            l_kcat = torch.tensor(0.0, device=device)
        losses['L_kcat'] = l_kcat

        # ── 3. Km 损失 L_km ───────────────────────────────
        km_mask = batch.get('km_target_mask',
                            torch.zeros_like(kcat_mask) if 'kcat_target_mask' in batch
                            else torch.zeros(batch.get('log_km_target', torch.empty(0)).size(0),
                                             dtype=torch.bool, device=device))

        if km_mask.any():
            # log_km_pred 是归一化 [0,1]，target 也归一化到 [0,1]
            km_target_norm = (batch['log_km_target'] - self.KM_MIN) / (self.KM_MAX - self.KM_MIN)
            l_km = self._regression_loss(
                outputs['log_km_pred'], km_target_norm, km_mask
            )
        else:
            l_km = torch.tensor(0.0, device=device)
        losses['L_km'] = l_km

        # ── 4. 联合约束 L_joint (kcat/Km 物理自洽) ─────────
        both_mask = kcat_mask & km_mask
        l_joint = torch.tensor(0.0, device=device)

        if both_mask.any():
            # 从预测算 log10(kcat/Km) — 总是用均值（无方差）
            if self.heteroscedastic:
                kcat_mean = outputs['kcat_pred'][both_mask, 0]
                km_mean = outputs['log_km_pred'][both_mask, 0]
            else:
                kcat_mean = outputs['kcat_pred'][both_mask]
                km_mean = outputs['log_km_pred'][both_mask]

            log_kcat_denorm = self.denormalize(
                kcat_mean, self.KCAT_MIN, self.KCAT_MAX
            )
            log_km_denorm = self.denormalize(
                km_mean, self.KM_MIN, self.KM_MAX
            )
            log_kcatKm_pred = log_kcat_denorm - log_km_denorm

            # 真实 log10(kcat/Km)
            log_kcat_true = batch['log_kcat_target_denorm'][both_mask]
            log_km_true = batch['log_km_target'][both_mask]
            log_kcatKm_true = log_kcat_true - log_km_true

            # kcat/Km 回归损失（始终用 SmoothL1）
            l_joint_reg = F.smooth_l1_loss(log_kcatKm_pred, log_kcatKm_true)

            # 扩散极限约束: kcat/Km ≤ 10⁹ (log10 ≤ 9)
            # 物理意义：酶催化效率不可能无限高，扩散相遇速率上限 ~10⁹ M⁻¹s⁻¹
            # 约束项和回归项用相同权重（不再乘以 0.01），确保约束激活时信号充足
            l_joint_limit = F.relu(log_kcatKm_pred - 9.0).mean()

            # 扩散极限退化约束: log₁₀(Km) ≥ log₁₀(kcat) - 9
            # 防止模型预测的 Km 低到不合理（hack: 通过 joint_km_weight 可加重）
            l_joint_km = F.relu(log_kcatKm_pred - 9.0).mean() * joint_km_weight

            l_joint = (l_joint_reg + l_joint_limit) * joint_weight + l_joint_km

        losses['L_joint'] = l_joint
        losses['L_joint_reg'] = l_joint_reg if 'l_joint_reg' in dir() else torch.tensor(0.0, device=device)
        losses['L_joint_limit'] = l_joint_limit if 'l_joint_limit' in dir() else torch.tensor(0.0, device=device)
        losses['L_joint_km'] = l_joint_km if 'l_joint_km' in dir() else torch.tensor(0.0, device=device)

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
        if ('ligand_encoder' in name or 'protein_encoder' in name
            or 'cofactor_encoder' in name or 'temperature_encoder' in name):
            encoder_params.append(param)
        elif 'kcat_head.binn' in name:
            binn_params.append(param)
        elif ('kd_head' in name or 'ki_head' in name or 'kcat_head' in name
              or 'dG_predictor' in name or 'log_kappa' in name or 'km_head' in name
              or 'shared' in name or 'output_proj' in name or 'net' in name
              or 'initial_state_proj' in name or 'dynamics_net' in name
              or 'fusion' in name or 'ligand_atom_proj' in name or 'token_proj' in name):
            head_params.append(param)
        else:
            encoder_params.append(param)  # fallback

    return torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': binn_params, 'lr': lr * 0.5, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr * 2.0, 'weight_decay': weight_decay},
    ])
