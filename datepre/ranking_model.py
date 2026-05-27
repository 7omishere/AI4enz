"""
ranking_model.py
================
酶挖掘排序模型：基于蛋白序列嵌入 + 配体分子图 + 辅因子类型预测底物结合亲和力 (pKd)。

架构：
  LigandEncoder (GATv2×3) + ProteinEncoder (ESM-2 1280-dim) + CofactorEncoder
  → InteractionModule (交叉注意力)
  → MultiTaskHead (pKd + λ_offset + log_kcat)

训练目标：
  L_total = L_pkd + L_kcat
  - pKd 损失：按热力学分层权重 (Kd=1.0, Ki=0.7, IC50=0.15)
  - kcat 损失：蛋白级辅助回归，按数据源加权
  - Marcus 物理约束已移除（经验证不适用于当前数据集）
  - λ 预测头保留但不参与损失

用法：
  from ranking_model import MarcusPINN
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


class PocketEncoder(nn.Module):
    """口袋几何编码器：从口袋残基的局部结构特征学习几何表示。

    输入（变长 K，已 padding 到 max_K）：
      - pocket_cn:         (B, K) 口袋残基接触数
      - pocket_pi:         (B, K) 口袋残基突起指数
      - pocket_dist:       (B, K, K) 口袋残基间近似 Cα 距离 (Å)
      - pocket_mask:       (B, K) True=有效残基

    设计：
      - 旋转/平移不变：使用残基间距离矩阵而非绝对坐标
      - 变长处理：padding + mask
      - 距离信息通过 Gaussian 核注入加权消息传递
    """

    def __init__(self, pocket_dim: int = 64, d_model: int = 32):
        super().__init__()
        self.pocket_dim = pocket_dim
        self.d_model = d_model

        # 每个残基的特征：接触数 + 突起指数 → d_model
        self.residue_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # 距离→注意力权重的长度尺度（可学习）
        self.dist_sigma = nn.Parameter(torch.tensor(5.0))

        # 距离加权的消息传递
        self.msg_proj = nn.Linear(d_model, d_model)
        self.msg_norm = nn.LayerNorm(d_model)

        # 池化 → pocket_dim
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model * 2, pocket_dim),
            nn.LayerNorm(pocket_dim),
            nn.SiLU(),
        )

        # Fallback: 无口袋坐标时的可学习偏置
        self.no_pocket_bias = nn.Parameter(torch.zeros(pocket_dim))

    def forward(
        self,
        pocket_cn: torch.Tensor | None,        # (B, K)
        pocket_pi: torch.Tensor | None,        # (B, K)
        pocket_dist: torch.Tensor | None,      # (B, K, K)
        pocket_mask: torch.Tensor | None,      # (B, K)
    ) -> torch.Tensor:
        """Returns (B, pocket_dim)"""
        if pocket_cn is None or pocket_mask is None or not pocket_mask.any():
            B = pocket_cn.size(0) if pocket_cn is not None else 1
            return self.no_pocket_bias.unsqueeze(0).expand(B, -1)

        # Per-residue features: (B, K, 2) → (B, K, d_model)
        cn = pocket_cn.unsqueeze(-1)
        pi = pocket_pi.unsqueeze(-1)
        res_feat = torch.cat([cn, pi], dim=-1)
        x = self.residue_proj(res_feat)

        # Distance-weighted message passing
        if pocket_dist is not None and pocket_dist.size(-1) > 1:
            # Gaussian kernel attention: w_ij = exp(-d_ij^2 / (2 * sigma^2))
            attn = torch.exp(-(pocket_dist ** 2) / (2 * self.dist_sigma ** 2 + 1e-6))
            # Mask invalid positions
            mask_2d = pocket_mask.unsqueeze(-1) & pocket_mask.unsqueeze(-2)
            attn = attn * mask_2d.float()
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            # Weighted message: x'_i = sum_j w_ij * proj(x_j)
            msg = self.msg_proj(x)
            x_weighted = torch.bmm(attn, msg)
            x = self.msg_norm(x + x_weighted)
            x = F.silu(x)

        # Pool: mean + max over valid residues
        mask_expanded = pocket_mask.unsqueeze(-1).float()
        x_masked = x * mask_expanded
        mean_pool = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        max_pool = x_masked.max(dim=1).values

        return self.pool_proj(torch.cat([mean_pool, max_pool], dim=-1))


class ProteinEncoder(nn.Module):
    """
    蛋白质编码器：序列特征 + 结构特征 + 口袋几何 + 辅因子结合域。

    双路径设计：
      - ESM-2 路径: 1280-dim → seq_proj → hidden_dim
      - AA属性路径: AA_PROP_DIM (6) → aa_proj → hidden_dim
    """

    def __init__(self,
                 seq_embed_dim: int = 1280,
                 struct_feat_dim: int = 3,
                 hidden_dim: int = 256,
                 domain_types: list[str] | None = None,
                 domain_embed_dim: int = 32,
                 aa_prop_dim: int = 6,
                 pocket_dim: int = 64,
                 ):
        super().__init__()
        if domain_types is None:
            domain_types = COFACTOR_DOMAIN_TYPES
        self.n_domain_types = len(domain_types)

        # ESM-2 路径
        self.seq_proj = nn.Linear(seq_embed_dim, hidden_dim)
        # AA属性路径
        self.aa_proj = nn.Linear(aa_prop_dim, hidden_dim)
        self.aa_norm = nn.LayerNorm(hidden_dim)

        # 口袋几何编码器
        self.pocket_encoder = PocketEncoder(pocket_dim=pocket_dim)

        # 结构特征投影（输入维度增加 pocket_dim）
        self.struct_proj = nn.Linear(struct_feat_dim + pocket_dim, hidden_dim // 4)
        self.has_structure_bias = nn.Parameter(torch.zeros(hidden_dim // 4))

        # 域特征
        self.domain_embed_dim = domain_embed_dim
        self.domain_type_embed = nn.Embedding(self.n_domain_types, domain_embed_dim)
        self.domain_proj = nn.Linear(domain_embed_dim, hidden_dim // 8)

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4 + hidden_dim // 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def _aggregate_domain_features(
        self,
        domain_masks: torch.Tensor,
        domain_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        domain_masks: (B, n_domain_types, L_padded) — 各通道对应 COFACTOR_DOMAIN_TYPES
        domain_padding_mask: (B, L_padded) — True = 有效位置

        按辅因子类型计算域覆盖率，加权聚合各类型的可学习 embedding。
        """
        B, n_types, L = domain_masks.shape

        if domain_padding_mask is not None:
            valid_lengths = domain_padding_mask.float().sum(dim=-1).unsqueeze(-1)
            coverage = domain_masks.sum(dim=-1) / valid_lengths.clamp(min=1)
        else:
            coverage = domain_masks.mean(dim=-1)

        type_embeds = self.domain_type_embed.weight  # (n_types, domain_embed_dim)
        aggregated = coverage.unsqueeze(-1) * type_embeds.unsqueeze(0)
        return aggregated.sum(dim=1)  # (B, domain_embed_dim)

    def forward(self,
                seq_embed: torch.Tensor,           # (B, 1280) || (B, AA_PROP_DIM)
                struct_feat: Optional[torch.Tensor] = None,  # (B, 3)
                has_structure: Optional[torch.Tensor] = None,  # (B,)
                domain_masks: Optional[torch.Tensor] = None,   # (B, n_domain_types, L)
                domain_padding_mask: Optional[torch.Tensor] = None,  # (B, L)
                pocket_cn: Optional[torch.Tensor] = None,      # (B, K)
                pocket_pi: Optional[torch.Tensor] = None,      # (B, K)
                pocket_dist: Optional[torch.Tensor] = None,    # (B, K, K)
                pocket_mask: Optional[torch.Tensor] = None,    # (B, K)
                ) -> torch.Tensor:
        # 双路径序列编码
        if seq_embed.size(-1) == self.aa_proj.in_features:
            seq_h = self.aa_proj(seq_embed)
            seq_h = self.aa_norm(seq_h)
            seq_h = F.silu(seq_h)
        else:
            seq_h = self.seq_proj(seq_embed)

        # 口袋几何编码
        pocket_embed = self.pocket_encoder(pocket_cn, pocket_pi, pocket_dist, pocket_mask)

        # 结构特征 + 口袋特征合并
        if struct_feat is not None and has_structure is not None:
            struct_input = torch.cat([struct_feat, pocket_embed], dim=-1)
            struct_h = self.struct_proj(struct_input)
            mask = has_structure.float().unsqueeze(-1)
            struct_h = struct_h * mask + self.has_structure_bias * (1 - mask)
        else:
            struct_h = self.has_structure_bias.unsqueeze(0).expand(seq_h.size(0), -1)

        if domain_masks is not None:
            domain_h = self.domain_proj(
                self._aggregate_domain_features(domain_masks, domain_padding_mask)
            )
        else:
            domain_h = torch.zeros(seq_h.size(0), self.domain_proj.out_features,
                                   device=seq_h.device)

        x = torch.cat([seq_h, struct_h, domain_h], dim=-1)
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────
# 交互模块
# ─────────────────────────────────────────────────────────────

class InteractionModule(nn.Module):
    """蛋白-配体-辅因子交互模块，使用交叉注意力"""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.cross_attn_pl = nn.MultiheadAttention(hidden_dim, num_heads,
                                                    batch_first=True)
        self.cross_attn_pc = nn.MultiheadAttention(hidden_dim, num_heads,
                                                    batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self,
                protein_h: torch.Tensor,   # (B, D)
                ligand_h: torch.Tensor,    # (B, D)
                cofactor_h: torch.Tensor,  # (B, D_cofactor)
                ) -> torch.Tensor:
        # 投影配体和辅因子到相同维度
        D = protein_h.size(-1)
        if ligand_h.size(-1) != D:
            ligand_h = F.linear(ligand_h, torch.eye(D, ligand_h.size(-1), device=ligand_h.device))
        if cofactor_h.size(-1) != D:
            cofactor_h = F.linear(cofactor_h, torch.eye(D, cofactor_h.size(-1), device=cofactor_h.device))

        # 交叉注意力
        attn_pl, _ = self.cross_attn_pl(
            protein_h.unsqueeze(1), ligand_h.unsqueeze(1), ligand_h.unsqueeze(1)
        )
        attn_pc, _ = self.cross_attn_pc(
            protein_h.unsqueeze(1), cofactor_h.unsqueeze(1), cofactor_h.unsqueeze(1)
        )

        combined = torch.cat([
            self.norm1(protein_h + attn_pl.squeeze(1)),
            self.norm2(protein_h + attn_pc.squeeze(1)),
            protein_h,
        ], dim=-1)

        return self.output_proj(combined)


# ─────────────────────────────────────────────────────────────
# 预测头
# ─────────────────────────────────────────────────────────────

class MultiTaskHead(nn.Module):
    """双头预测器：pKd, λ（从蛋白-配体交互特征中预测）。

    kcat 不再从此头预测——由独立的 KcatPredictor 从蛋白序列+辅因子特征直接预测。
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        shared = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        ]

        self.shared = nn.Sequential(*shared)

        # pKd 预测头：输出在 [2, 15] 范围
        self.pkd_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # λ 预测头：输出 λ 偏移 (eV)
        self.lambda_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        pkd = self.pkd_head(h).squeeze(-1)           # (B,)
        pkd = 2.0 + 13.0 * torch.sigmoid(pkd)        # constrain to [2, 15]
        lambda_offset = self.lambda_head(h).squeeze(-1)  # (B,)  Δλ in eV
        return pkd, lambda_offset


# ─────────────────────────────────────────────────────────────
# kcat 预测器（蛋白级，独立路径）
# ─────────────────────────────────────────────────────────────

class KcatPredictor(nn.Module):
    """蛋白级 kcat 预测器：从蛋白序列 + 辅因子特征直接预测催化速率。

    设计原则：
      - kcat 是蛋白级属性（同一蛋白所有底物共享），不需要配体信息
      - 与 pKd 的交互路径分离，避免底物级和蛋白级表征冲突（消除负迁移）
      - 基线验证：纯 ESM-2 → MLP 的 kcat R²=0.721，本模块复用此结论

    输入：
      - seq_embed: (B, 1280) ESM-2 或 (B, 6) AA 物化性质
      - cofactor_embed: (B, cofactor_dim) 辅因子类型嵌入
    """

    def __init__(self, seq_embed_dim: int = 1280, cofactor_dim: int = 64,
                 hidden_dim: int = 256):
        super().__init__()
        input_dim = seq_embed_dim + cofactor_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq_embed: torch.Tensor, cofactor_embed: torch.Tensor) -> torch.Tensor:
        """Returns log10(kcat) (B,)"""
        x = torch.cat([seq_embed, cofactor_embed], dim=-1)
        return self.mlp(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# 物理损失函数（保留但不参与训练，Marcus 方程经验证不适用于当前数据集）
# ─────────────────────────────────────────────────────────────

class MarcusPhysicsLoss(nn.Module):
    """
    三种电子转移机制的 PDE 约束。

    形式：
      1. 纯 ET:     ΔG‡ = (λ + ΔG°)² / (4λ)
      2. Hydride:   ΔG‡ = (λ + ΔG°)² / (4λ) + δ·|ΔG°|      (Marcus-Hammond)
      3. PCET:      ΔG‡ = [(λ_e + λ_p) + ΔG°]² / [4(λ_e + λ_p)] + ΔG°_pcet

    损失基于过渡态理论检验：
      kcat_theory = kBT/h · exp(-ΔG‡ / (RT))
      但为了数值稳定，使用 log10(kcat) 比较。
    """

    def __init__(self, temperature: float = T_ref):
        super().__init__()
        self.RT = R_kcal * temperature      # kcal/mol
        self.RT_kJ = R_kJ * temperature     # kJ/mol
        self.kBT_h = k_B * temperature / h  # s⁻¹
        self.ev_to_kcal = 23.0605           # 1 eV = 23.0605 kcal/mol

    def _delta_g_from_pkd(self, pkd: torch.Tensor) -> torch.Tensor:
        """ΔG° (kcal/mol) = -RT ln(Kd). pKd = -log10(Kd)."""
        return DELTA_G_FACTOR * pkd           # kJ/mol, positive = unfavorable

    def _marcus_barrier(self,
                        delta_g: torch.Tensor,  # kJ/mol
                        lambda_: torch.Tensor,   # eV
                        ) -> torch.Tensor:
        """纯 Marcus 活化能 ΔG‡ (kcal/mol)"""
        lambda_kj = lambda_ * self.ev_to_kcal * 4.184  # eV → kcal → kJ
        numerator = (lambda_kj + delta_g) ** 2
        denominator = 4 * lambda_kj + 1e-8
        return numerator / denominator

    def _theoretical_log_kcat(self, barrier_kj: torch.Tensor) -> torch.Tensor:
        """ΔG‡ (kJ/mol) → log10(kcat) via transition state theory"""
        # kcat = kBT/h * exp(-ΔG‡ / RT) in s⁻¹
        log_kcat_natural = math.log(self.kBT_h) - barrier_kj / (self.RT_kJ + 1e-8)
        return log_kcat_natural / math.log(10)   # natural log → log10

    def forward(self,
                pkd: torch.Tensor,
                lambda_: torch.Tensor,
                log_kcat_pred: torch.Tensor,
                cofactor_strs: list[str],
                ) -> dict[str, torch.Tensor]:
        """
        Returns dict of losses keyed by mechanism type.
        """
        batch_size = pkd.size(0)

        # 分组：按电子转移机制
        et_mask = torch.zeros(batch_size, dtype=torch.bool)
        hydride_mask = torch.zeros(batch_size, dtype=torch.bool)
        pcet_mask = torch.zeros(batch_size, dtype=torch.bool)
        hydride_deltas = torch.zeros(batch_size)
        pcet_lambda_ps = torch.zeros(batch_size)

        for i, cf_str in enumerate(cofactor_strs):
            prior = get_prior_for_cofactors(cf_str)
            if prior.mechanism == "hydride":
                hydride_mask[i] = True
                hydride_deltas[i] = prior.delta
            elif prior.mechanism == "pcet":
                pcet_mask[i] = True
                pcet_lambda_ps[i] = prior.lambda_p
            else:
                et_mask[i] = True

        delta_g = self._delta_g_from_pkd(pkd)  # kJ/mol

        losses = {}
        total_physics_loss = torch.tensor(0.0, device=pkd.device)

        # ── 纯 ET ──
        if et_mask.any():
            lambda_et = lambda_[et_mask]
            dg_et = delta_g[et_mask]
            barrier = self._marcus_barrier(dg_et, lambda_et)
            log_kcat_theory = self._theoretical_log_kcat(barrier)
            log_kcat_pred_sub = log_kcat_pred[et_mask]
            loss = F.smooth_l1_loss(log_kcat_pred_sub, log_kcat_theory)
            losses["L_marcus_et"] = loss
            total_physics_loss = total_physics_loss + loss

        # ── Marcus-Hammond (hydride) ──
        if hydride_mask.any():
            lambda_h = lambda_[hydride_mask]
            dg_h = delta_g[hydride_mask]
            delta_h = hydride_deltas[hydride_mask].to(pkd.device)

            # Marcus 基础项
            barrier_base = self._marcus_barrier(dg_h, lambda_h)
            # Hammond 修正项: δ · |ΔG°|
            hammond_correction = delta_h * torch.abs(dg_h)
            barrier_h = barrier_base + hammond_correction

            log_kcat_theory = self._theoretical_log_kcat(barrier_h)
            log_kcat_pred_sub = log_kcat_pred[hydride_mask]
            loss = F.smooth_l1_loss(log_kcat_pred_sub, log_kcat_theory)
            losses["L_marcus_hydride"] = loss
            total_physics_loss = total_physics_loss + loss

        # ── PCET ──
        if pcet_mask.any():
            lambda_pcet = lambda_[pcet_mask]
            dg_pc = delta_g[pcet_mask]
            lambda_p = pcet_lambda_ps[pcet_mask].to(pkd.device)

            # PCET: 有效重组能 = λ_e + λ_p
            lambda_eff = lambda_pcet + lambda_p * 23.0605 * 4.184  # eV→kJ
            dg_eff = dg_pc  # 简化

            barrier = (lambda_eff + dg_eff) ** 2 / (4 * lambda_eff + 1e-8)
            log_kcat_theory = self._theoretical_log_kcat(barrier)
            log_kcat_pred_sub = log_kcat_pred[pcet_mask]
            loss = F.smooth_l1_loss(log_kcat_pred_sub, log_kcat_theory)
            losses["L_marcus_pcet"] = loss
            total_physics_loss = total_physics_loss + loss

        losses["L_physics_total"] = total_physics_loss
        return losses


# ─────────────────────────────────────────────────────────────
# OT 正则化（Zhu 2025）
# ─────────────────────────────────────────────────────────────

class OTRegularizer(nn.Module):
    """
    对学习到的 λ 分布施加基于辅因子类型的软约束。

    迭代 Sinkhorn 算法计算可微的 Wasserstein 距离，
    将每个 batch 中各类辅因子的 λ 经验分布拉到参考分布附近。
    """

    def __init__(self,
                 cofactor_priors: dict[str, CofactorPrior],
                 n_bins: int = 50,
                 lambda_range: tuple[float, float] = (0.1, 2.5),
                 sinkhorn_reg: float = 0.05,
                 sinkhorn_iters: int = 20,
                 ):
        super().__init__()
        self.priors = cofactor_priors
        self.n_bins = n_bins
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iters = sinkhorn_iters

        # 构建离散分 bin
        self.register_buffer(
            "bin_edges",
            torch.linspace(lambda_range[0], lambda_range[1], n_bins + 1),
        )
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        # 为每种辅因子构建参考分布 (Gaussian 近似)
        self.cofactor_to_idx = {}
        ref_dists = []
        for i, (cf_type, prior) in enumerate(sorted(cofactor_priors.items())):
            self.cofactor_to_idx[cf_type] = i
            # Gaussian 离散化到 bins
            sigma = prior.lambda_std
            mu = prior.lambda_mean
            # 计算每个 bin 的密度
            density = torch.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)
            density = density / (density.sum() + 1e-8)
            ref_dists.append(density)

        self.register_buffer("ref_distributions",
                             torch.stack(ref_dists))  # (n_cofactors, n_bins)

    def _empirical_distribution(self, lambda_values: torch.Tensor) -> torch.Tensor:
        """Soft bin assignment: 对 λ 值分配到离散 bin 中"""
        lambda_clamped = lambda_values.clamp(
            self.bin_edges[0].item(), self.bin_edges[-1].item()
        )
        idx = torch.searchsorted(self.bin_edges[1:-1], lambda_clamped)
        dist = torch.zeros(lambda_values.size(0), self.n_bins,
                           device=lambda_values.device)
        dist.scatter_(1, idx.unsqueeze(-1), 1.0)
        return dist.mean(dim=0)  # empirical mean distribution per batch

    def _sinkhorn_distance(self, mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """可微 Sinkhorn 距离。mu, nu 均为归一化直方图 (n_bins,)"""
        n = mu.size(0)

        # 代价矩阵：bin 中心之间的欧氏距离
        bins = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        C = torch.abs(bins.unsqueeze(0) - bins.unsqueeze(1))  # (n_bins, n_bins)

        # 核矩阵
        K = torch.exp(-C / self.sinkhorn_reg)

        # Sinkhorn 迭代
        v = torch.ones_like(nu)
        for _ in range(self.sinkhorn_iters):
            u = mu / (K @ v + 1e-8)
            v = nu / (K.T @ u + 1e-8)

        # 最优传输计划
        P = torch.diag(u) @ K @ torch.diag(v)

        # Wasserstein 距离
        return (C * P).sum()

    def forward(self,
                lambda_values: torch.Tensor,
                cofactor_strs: list[str],
                ) -> torch.Tensor:
        """
        对每个辅因子类别计算 Wasserstein 距离并求和。

        Args:
          lambda_values: (B,) 预测的 λ 值 (eV)
          cofactor_strs: list[str] 辅因子字符串列表
        """
        total_wass = torch.tensor(0.0, device=lambda_values.device)
        n_groups = 0

        # 按主辅因子类型分组
        groups: dict[str, list[int]] = {}
        for i, cf_str in enumerate(cofactor_strs):
            prior = get_prior_for_cofactors(cf_str)
            # 找到匹配的参考分布键
            cofactors = [c.strip() for c in str(cf_str).split("|")]
            key = None
            for cf in cofactors:
                if cf in self.cofactor_to_idx:
                    key = cf
                    break
            if key is not None:
                groups.setdefault(key, []).append(i)

        for cf_type, indices in groups.items():
            if len(indices) < 3:  # 至少 3 个样本才有意义
                continue
            idx = self.cofactor_to_idx[cf_type]
            ref_dist = self.ref_distributions[idx]

            lambda_sub = lambda_values[torch.tensor(indices, device=lambda_values.device)]
            emp_dist = self._empirical_distribution(lambda_sub)

            wass = self._sinkhorn_distance(emp_dist, ref_dist)
            total_wass = total_wass + wass
            n_groups += 1

        return total_wass / max(n_groups, 1)


# ─────────────────────────────────────────────────────────────
# 完整模型
# ─────────────────────────────────────────────────────────────

class MarcusPINN(nn.Module):
    """
    酶挖掘排序模型：预测酶-底物催化效率 (kcat/KM) 用于底物偏好排序。

    架构：
      - pKd 路径（底物级）：配体 GNN + 蛋白结构/口袋 + 辅因子 → InteractionModule → pKd
      - kcat 路径（蛋白级）：ESM-2 + 辅因子嵌入 → KcatPredictor MLP → log_kcat
      - 两条路径分离，输出端汇合：score = pKd + log_kcat = log10(kcat/KM)

    输入：蛋白序列嵌入 + 配体分子图 + 辅因子类型
    输出：pKd (约束[2,15]) + λ (重组能偏移) + log10(kcat)
    损失：L_total = L_score(kcat/KM) + L_pkd_fallback(无kcat标签的样本)
    """

    COFACTOR_TYPES = sorted(COFACTOR_PRIORS.keys())

    def __init__(self,
                 hidden_dim: int = 256,
                 cofactor_embed_dim: int = 64,
                 n_heads: int = 4,
                 gnn_layers: int = 3,
                 ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cofactor_embed_dim = cofactor_embed_dim

        # 编码器
        self.ligand_encoder = LigandEncoder(
            atom_dim=79, edge_dim=10, hidden_dim=hidden_dim,
            num_layers=gnn_layers, heads=n_heads,
        )
        self.protein_encoder = ProteinEncoder(
            seq_embed_dim=1280, struct_feat_dim=3, hidden_dim=hidden_dim,
        )
        self.cofactor_encoder = CofactorEncoder(
            cofactor_types=self.COFACTOR_TYPES, embed_dim=cofactor_embed_dim,
        )

        # 投影对齐
        self.ligand_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cofactor_proj = nn.Linear(cofactor_embed_dim, hidden_dim)

        # 交互模块
        self.interaction = InteractionModule(
            hidden_dim=hidden_dim, num_heads=n_heads,
        )

        # 预测头
        self.head = MultiTaskHead(hidden_dim=hidden_dim)

        # kcat 预测器（蛋白级，独立路径 — 不与 pKd 共享表征）
        self.kcat_predictor = KcatPredictor(
            seq_embed_dim=1280, cofactor_dim=cofactor_embed_dim,
            hidden_dim=hidden_dim,
        )

        # 物理损失和OT正则化已移除（Marcus方程经验证不适用于当前数据集）
        # λ 预测头保留，但不再参与损失计算

    def forward(self,
                ligand_data,                       # PyG Batch
                seq_embed: torch.Tensor,           # (B, 1280)
                cofactor_strs: list[str],
                struct_feat: Optional[torch.Tensor] = None,   # (B, 3)
                has_structure: Optional[torch.Tensor] = None,  # (B,)
                domain_masks: Optional[torch.Tensor] = None,   # (B, 15, L_padded)
                domain_padding_mask: Optional[torch.Tensor] = None,  # (B, L_padded)
                pocket_cn: Optional[torch.Tensor] = None,      # (B, K)
                pocket_pi: Optional[torch.Tensor] = None,      # (B, K)
                pocket_dist: Optional[torch.Tensor] = None,    # (B, K, K)
                pocket_mask: Optional[torch.Tensor] = None,    # (B, K)
                ) -> dict[str, torch.Tensor]:
        """
        Returns:
          pkd:           predicted pKd (B,) — 底物级结合亲和力 (InteractionModule → head)
          lambda_total:  predicted reorganization energy (eV) (B,)
          log_kcat:      predicted log10(kcat) (B,) — 蛋白级催化速率 (KcatPredictor)
          cofactor_embed: cofactor embedding (B, cofactor_embed_dim)
        """
        # 编码
        ligand_h = self.ligand_encoder(ligand_data)
        ligand_h = self.ligand_proj(ligand_h)

        protein_h = self.protein_encoder(
            seq_embed, struct_feat, has_structure,
            domain_masks=domain_masks,
            domain_padding_mask=domain_padding_mask,
            pocket_cn=pocket_cn,
            pocket_pi=pocket_pi,
            pocket_dist=pocket_dist,
            pocket_mask=pocket_mask,
        )

        cofactor_h, lambda_offset = self.cofactor_encoder(cofactor_strs)
        cofactor_h_proj = self.cofactor_proj(cofactor_h)

        # 交互 → pKd + λ (底物级路径)
        combined = self.interaction(protein_h, ligand_h, cofactor_h_proj)
        pkd, lambda_offset_head = self.head(combined)

        # kcat 预测 (蛋白级路径 — 独立，不经过交互模块)
        log_kcat = self.kcat_predictor(seq_embed, cofactor_h)

        # λ_total = λ_prior + λ_offset (cofactor) + λ_offset (head)
        lambda_prior_batch = torch.tensor([
            get_prior_for_cofactors(cf).lambda_mean for cf in cofactor_strs
        ], device=pkd.device)
        lambda_total = lambda_prior_batch + lambda_offset.squeeze(-1) + lambda_offset_head
        # soft clamp（可导）：λ 被柔和地约束在合理范围 [0.05, 3.0] eV
        lambda_total = 0.05 + F.softplus(lambda_total - 0.05)
        lambda_total = 3.0 - F.softplus(3.0 - lambda_total)

        return {
            "pkd": pkd,
            "lambda": lambda_total,
            "log_kcat": log_kcat,
            "lambda_prior": lambda_prior_batch,
            "lambda_offset": lambda_offset,
            "lambda_offset_head": lambda_offset_head,
            "cofactor_embed": cofactor_h,
        }

    def compute_loss(self,
                     outputs: dict,
                     batch: dict,
                     ) -> tuple[torch.Tensor, dict]:
        """
        统一催化效率损失函数：直接优化 kcat/KM（酶学适配性标准指标）。

        设计原则：
          - 主损失 L_score：直接优化 (pKd + log_kcat) = log10(kcat/KM)
            模型自由分配 pKd 和 kcat 的预测误差来最小化组合误差
          - 回退 L_pkd_fallback：对无 kcat 标签的样本 (18%)，退化为纯 pKd 损失
          - 监控指标：L_pkd_monitor 和 L_kcat_monitor 仅用于日志，不参与梯度

        Args:
          outputs: forward() 的输出
          batch:   {
            pkd_target, log_kcat_target,
            pkd_target_mask, kcat_target_mask,
            quality_weight (含 thermo_weight),
          }

        Returns:
          total_loss, loss_components
        """
        pkd_pred = outputs["pkd"]
        log_kcat_pred = outputs["log_kcat"]

        losses = {}
        quality_weight = batch.get("quality_weight", torch.ones_like(pkd_pred))
        pkd_mask = batch.get("pkd_target_mask", torch.ones_like(pkd_pred, dtype=torch.bool))
        kcat_mask = batch.get("kcat_target_mask", torch.zeros_like(log_kcat_pred, dtype=torch.bool))

        # 样本分组
        both_mask = pkd_mask & kcat_mask              # 有 pKd + kcat：用 kcat/KM
        pkd_only_mask = pkd_mask & ~kcat_mask          # 仅有 pKd：回退到 pKd 损失

        # ── 主损失：kcat/KM = pKd + log10(kcat) ──
        if both_mask.any():
            score_pred = pkd_pred[both_mask] + log_kcat_pred[both_mask]
            score_true = batch["pkd_target"][both_mask] + batch["log_kcat_target"][both_mask]
            diff = score_pred - score_true
            losses["L_score"] = (quality_weight[both_mask] * F.smooth_l1_loss(
                diff, torch.zeros_like(diff), reduction='none'
            )).mean()
        else:
            losses["L_score"] = torch.tensor(0.0, device=pkd_pred.device)

        # ── 回退损失：纯 pKd (无 kcat 标签的样本) ──
        if pkd_only_mask.any():
            diff = pkd_pred[pkd_only_mask] - batch["pkd_target"][pkd_only_mask]
            losses["L_pkd_fallback"] = (quality_weight[pkd_only_mask] * F.smooth_l1_loss(
                diff, torch.zeros_like(diff), reduction='none'
            )).mean()
        else:
            losses["L_pkd_fallback"] = torch.tensor(0.0, device=pkd_pred.device)

        # ── 监控指标（不参与梯度，仅用于日志） ──
        if pkd_mask.any():
            diff = pkd_pred[pkd_mask] - batch["pkd_target"][pkd_mask]
            losses["L_pkd_monitor"] = F.l1_loss(diff, torch.zeros_like(diff))
        else:
            losses["L_pkd_monitor"] = torch.tensor(0.0, device=pkd_pred.device)

        if kcat_mask.any():
            diff = log_kcat_pred[kcat_mask] - batch["log_kcat_target"][kcat_mask]
            losses["L_kcat_monitor"] = F.l1_loss(diff, torch.zeros_like(diff))
        else:
            losses["L_kcat_monitor"] = torch.tensor(0.0, device=log_kcat_pred.device)

        # ── 总损失 ──
        total = losses["L_score"] + losses["L_pkd_fallback"]
        losses["total"] = total

        return total, losses

    def predict_activation_barrier(self, outputs: dict) -> torch.Tensor:
        """辅助函数：从模型输出计算预测的 ΔG‡ (kJ/mol)。

        使用 Marcus 方程 ΔG‡ = (λ + ΔG°)² / (4λ)。
        注意：此函数仅供分析参考，Marcus 约束已从训练损失中移除
        （经验证 kcat_true / kcat_marcus ≈ 10⁻⁶）。
        """
        delta_g_kj = DELTA_G_FACTOR * outputs["pkd"]     # kJ/mol
        lambda_ev = outputs["lambda"]                      # eV
        lambda_kj = lambda_ev * 23.0605 * 4.184            # eV → kcal → kJ
        numerator = (lambda_kj + delta_g_kj) ** 2
        denominator = 4 * lambda_kj + 1e-8
        return numerator / denominator

    def predict_catalytic_efficiency(self, outputs: dict) -> torch.Tensor:
        """辅助函数：计算 log10(kcat/KM) from pKd + kcat"""
        # KM = Kd (简化假设; 实际中 KM ≠ Kd 但不影响数量级估计)
        pkd = outputs["pkd"]
        log_kcat = outputs["log_kcat"]
        # kcat/KM ≈ kcat * Kd⁻¹ = kcat * 10^pKd
        log_kcat_km = log_kcat + pkd
        return log_kcat_km


# ─────────────────────────────────────────────────────────────
# 训练辅助
# ─────────────────────────────────────────────────────────────

def create_optimizer(model: MarcusPINN, lr: float = 1e-4, weight_decay: float = 1e-5):
    """创建分组优化器（编码器 vs 预测头不同学习率）"""
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    return torch.optim.AdamW([
        {"params": encoder_params, "lr": lr},
        {"params": head_params, "lr": lr * 2},
    ], weight_decay=weight_decay)


def warmup_schedule(step: int, warmup_steps: int = 1000) -> float:
    """物理损失 warmup: 先让数据损失收敛，再逐步增加物理约束"""
    if step < warmup_steps:
        return float(step) / warmup_steps
    return 1.0


# ─────────────────────────────────────────────────────────────
# 测试入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  酶挖掘排序模型 — 结构验证")
    print("=" * 60)

    model = MarcusPINN(hidden_dim=256)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量:    {total_params:,}")
    print(f"  可训练参数:  {trainable_params:,}")

    print(f"\n  辅因子先验表 ({len(COFACTOR_PRIORS)} 种):")
    for cf, prior in sorted(COFACTOR_PRIORS.items()):
        print(f"    {cf:12s}  λ={prior.lambda_mean:.2f}±{prior.lambda_std:.2f} eV  "
              f"机制={prior.mechanism:8s}  "
              f"{'δ='+str(prior.delta) if prior.delta else ''}"
              f"{'λ_p='+str(prior.lambda_p) if prior.lambda_p else ''}")

    print(f"  编码器:")
    print(f"    LigandEncoder:  GATv2Conv × {model.ligand_encoder.convs.__len__()}")
    print(f"    ProteinEncoder: ESM-2 1280-dim → {model.hidden_dim}-dim")
    print(f"    CofactorEncoder: {model.cofactor_encoder.num_types} types → {model.cofactor_embed_dim}-dim")
    print(f"    KcatPredictor: ESM-2 1280-dim + cofactor 64-dim → MLP → log_kcat (独立路径)")

    print(f"\n  输出头:")
    print(f"    pKd:      结合亲和力 (−log10 Kd)，约束 [2, 15] (InteractionModule → head)")
    print(f"    log_kcat: log10 催化速率常数 (s⁻¹) (KcatPredictor, 蛋白级独立路径)")
    print(f"    λ:        重组能 (eV), 辅因子先验 + 可学习偏移 (保留，不参与损失)")

    print(f"\n  损失函数:")
    print(f"    L_total = L_score(kcat/KM) + L_pkd_fallback(无kcat样本)")
    print(f"    - L_score:  直接优化 pKd + log_kcat = log10(kcat/KM)")
    print(f"    - L_pkd_fallback: 无kcat标签时回退为纯pKd损失")
    print(f"    - 监控: L_pkd_monitor, L_kcat_monitor (不参与梯度)")

    print("\n  ✓ 模型结构验证通过")
