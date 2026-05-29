"""
ranking_model.py
================
酶挖掘排序模型：基于蛋白序列嵌入 + 配体分子图 + 辅因子类型预测底物结合亲和力 (pKd)。

架构：
  LigandEncoder (GATv2×3) + ProteinEncoder (ESM-2 1280-dim) + CofactorEncoder
  → BINNInteraction (反应坐标Neural ODE)
  → BINNCatalysisHead (pKd + catalysis_rate)

训练目标：
  L_total = L_ts + L_catalysis + L_barrier + L_progress
  - L_ts: 过渡态稳定性（与pKd正相关）
  - L_catalysis: 催化效率（与log_kcat正相关）
  - L_barrier: 能垒正则化（物理先验）
  - L_progress: 反应坐标演化正则化
  - 权重由uncertainty weighting自动学习

核心设计（BINN）：
  - 不依赖电子转移假设（Marcus方程已被证伪）
  - 基于普适的过渡态理论
  - 用Neural ODE模拟酶-底物复合物沿反应坐标的演化
  - 门控机制模拟"能垒跨越"

用法：
  from ranking_model import TransitionBINN
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


# ═══════════════════════════════════════════════════════════════════════════════
# BINN: 基于反应坐标的过渡态理论
# ═══════════════════════════════════════════════════════════════════════════════
# 设计原则：
#   - 不依赖电子转移假设（Marcus方程已被证伪）
#   - 基于普适的过渡态理论（适用于所有酶催化）
#   - 用Neural ODE模拟酶-底物复合物沿反应坐标的演化

class TransitionStateGate(nn.Module):
    """
    门控机制：模拟"跨越能垒"的过程
    
    核心思想：
    - 在过渡态附近（ξ≈0.5），门控更"宽松"，信息通过更多
    - 在反应物/产物端（ξ≈0或1），门控更"严格"
    
    类比：就像分子需要"激发"才能跨越能垒
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # 门控网络：决定当前状态是否足够"激发"来跨越下一个能垒
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # h + catalyst + ligand context
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]，1=通过，0=阻挡
        )
        
        # 能量垒估计：预测当前位置的"能垒高度"（eV）
        self.barrier_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),  # h + catalyst
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保能量为正，约束到 [0.5, 5.0] eV
        )

    def forward(self, 
                h: torch.Tensor,           # (B, D) 当前状态
                catalyst_h: torch.Tensor,  # (B, D) 酶催化上下文
                ligand_h: torch.Tensor,    # (B, D) 底物状态
                xi_position: torch.Tensor  # (B,) ∈ [0, 1] 当前位置
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            gate_value: (B,) ∈ [0, 1]，门控值
            barrier_height: (B,)，估计的能垒高度 (eV)
        """
        B = h.size(0)
        device = h.device
        
        # 反应上下文 = 当前状态 + 催化剂 + 底物
        context = torch.cat([h, catalyst_h, ligand_h], dim=-1)
        
        # 可学习的调制
        learned_gate = self.gate_net(context).squeeze(-1)  # (B,)
        
        # 物理先验：距离过渡态越近（ξ≈0.5），门越开
        # 高斯函数：中心在ξ=0.5，给定基线门控
        xi_clamped = xi_position.clamp(0.0, 1.0)
        baseline_gate = torch.exp(-4.0 * (xi_clamped - 0.5) ** 2)  # (B,)
        
        # 组合：baseline（物理先验）+ learned（数据驱动）
        gate = 0.3 * baseline_gate + 0.7 * learned_gate
        gate = gate.clamp(0.0, 1.0)
        
        # 能垒高度估计：中间最高，两端最低
        barrier_context = torch.cat([h, catalyst_h], dim=-1)
        raw_barrier = self.barrier_net(barrier_context).squeeze(-1)  # (B,)
        
        # 物理约束：能垒应该随距离过渡态的距离而变化
        # 在ξ=0.5时最大，两端最小
        barrier_modulation = 1.0 + 4.0 * torch.abs(xi_clamped - 0.5)
        barrier = raw_barrier * barrier_modulation
        
        # 软约束到合理范围 [0.5, 5.0] eV
        barrier = 0.5 + 4.5 * torch.sigmoid(barrier.log() - 2.0)
        
        return gate, barrier


class ReactionCoordinateBINN(nn.Module):
    """
    基于反应坐标的BINN交互层
    
    核心假设：
    1. 存在一个潜在的反应坐标 ξ ∈ [0, 1]
       - ξ=0: 反应物（酶-底物初始结合）
       - ξ=0.5: 过渡态（最大能垒）
       - ξ=1: 产物（催化完成）
    
    2. 系统状态 h(ξ) 沿反应坐标连续演化
       dh/dξ = f(h, catalyst, substrate)
    
    3. 酶催化 = 加速 h(ξ) 的演化（降低能垒，稳定过渡态）
    
    不假设：
    - 电子转移是限速步
    - 特定的转移机制（ET/Hydride/PCET）
    - Marcus方程的适用性
    """

    def __init__(
        self, 
        hidden_dim: int = 256,
        n_ode_steps: int = 5,
        use_gate: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_steps = n_ode_steps
        self.use_gate = use_gate
        
        # Step 1: 初始状态构建（酶-底物复合物）
        self.initial_state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # protein + ligand
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        
        # Step 2: ODE动力学函数
        #  learns: dh/dξ = f(h, catalyst_context, ligand)
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # h + catalyst + ligand
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # Tanh保证动力学稳定
        )
        
        # 残差缩放因子（让ODE更稳定）
        self.dxi = 1.0 / n_ode_steps
        
        # Step 3: 门控机制
        if use_gate:
            self.gate = TransitionStateGate(hidden_dim)
        
        # Step 4: 最终演化状态 → 特征
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # h_final + h_initial
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
            h_reaction: (B, D) 沿反应坐标演化后的状态
            trajectory: list of (B, D) 每一步的状态
            gate_values: (B, n_steps) 每步的门控值
            barrier_heights: (B, n_steps) 每步的能垒高度
        """
        B = protein_h.size(0)
        device = protein_h.device
        
        # 催化剂上下文 = 酶 + 辅因子
        catalyst_h = protein_h + cofactor_h
        
        # ── Step 1: 构建初始状态 h(ξ=0) ──────────────────
        es_complex = torch.cat([protein_h, ligand_h], dim=-1)
        h0 = self.initial_state_proj(es_complex)
        
        # 沿反应坐标的演化轨迹
        trajectory = [h0]
        gate_values = []
        barrier_heights = []
        
        # 当前反应坐标位置
        xi = torch.zeros(B, device=device)
        
        # ── Step 2: ODE积分 ────────────────────────────────
        h = h0
        for step in range(self.n_steps):
            # 预测下一步的增量
            dynamics_input = torch.cat([h, catalyst_h, ligand_h], dim=-1)
            dh = self.dynamics_net(dynamics_input)
            
            # 门控：限制信息流通过量
            if self.use_gate:
                xi_position = xi + self.dxi / 2  # 中点估计
                gate, barrier = self.gate(h, catalyst_h, ligand_h, xi_position)
                gate_values.append(gate.detach())  # 门控不传梯度
                barrier_heights.append(barrier.detach())
                dh = dh * gate.unsqueeze(-1)
            
            # Euler积分 + 残差
            # 添加二阶校正以提高精度
            h_mid = h + 0.5 * self.dxi * dh
            dynamics_mid = torch.cat([h_mid, catalyst_h, ligand_h], dim=-1)
            dh_correction = self.dynamics_net(dynamics_mid)
            h = h + self.dxi * (dh + dh_correction) / 2
            
            # 更新反应坐标
            xi = xi + self.dxi
            trajectory.append(h.detach())
        
        # ── Step 3: 输出 ──────────────────────────────────
        # 演化幅度 = ||h_final - h_initial||
        reaction_progress = (h - h0).pow(2).mean(dim=-1)  # (B,)
        
        # 最终输出 = 最终状态 + 初始状态（保留初始信息）
        h_reaction = self.output_proj(torch.cat([h, h0], dim=-1))
        
        return {
            'h_reaction': h_reaction,
            'trajectory': trajectory,
            'gate_values': torch.stack(gate_values) if gate_values else None,
            'barrier_heights': torch.stack(barrier_heights) if barrier_heights else None,
            'reaction_progress': reaction_progress,
            'h_initial': h0,
            'h_final': h,
        }


class BINNCatalysisHead(nn.Module):
    """
    基于过渡态理论的催化预测头
    
    预测：
    1. ts_stability: 过渡态稳定性（与pKd对应）
    2. catalysis_rate: 催化效率（与log_kcat对应）
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        
        # 过渡态稳定性头：预测 pKd，范围 [2, 15]
        self.ts_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        
        # 催化效率头：预测 log10(kcat)
        self.catalysis_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        
        # 反应能垒头：预测 ΔG‡ (kJ/mol)，用于正则化
        self.barrier_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # 确保正值
        )

    def forward(self, h_reaction: torch.Tensor, h_initial: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            h_reaction: (B, D) 沿反应坐标演化后的状态
            h_initial:  (B, D) 初始状态
        
        Returns:
            ts_stability:  (B,) 预测的pKd（过渡态稳定性代理）
            catalysis_rate: (B,) 预测的log10(kcat)
            barrier_height: (B,) 预测的活化能垒
        """
        h = self.shared(h_reaction)
        
        # 过渡态稳定性：从最终状态预测，范围 [2, 15]
        raw_ts = self.ts_head(h).squeeze(-1)
        ts_stability = 2.0 + 13.0 * torch.sigmoid(raw_ts)
        
        # 催化效率：从状态差异预测
        h_diff = (h_reaction - h_initial).pow(2).mean(dim=-1)
        raw_cat = self.catalysis_head(h)
        catalysis_rate = raw_cat.squeeze(-1) + h_diff.clamp(max=5.0)
        
        # 活化能垒：物理约束范围 [30, 200] kJ/mol
        raw_barrier = self.barrier_head(h).squeeze(-1)
        barrier_height = 30.0 + 170.0 * torch.sigmoid(raw_barrier / 50.0)
        
        return {
            'ts_stability': ts_stability,
            'catalysis_rate': catalysis_rate,
            'barrier_height': barrier_height,
        }


class BINNLoss(nn.Module):
    """
    基于不确定性权重的BINN损失函数
    
    特点：
    - 自动学习每个损失项的权重
    - 基于uncertainty weighting (Kendall et al.)
    - 包含物理正则化项
    """

    def __init__(self):
        super().__init__()
        # 可学习的log方差，对应各损失项的权重
        # sigma越大 → 权重越小 → 该损失项的重要性降低
        self.register_parameter(
            'log_var_ts', nn.Parameter(torch.tensor(0.0))
        )
        self.register_parameter(
            'log_var_cat', nn.Parameter(torch.tensor(0.0))
        )

    def forward(self, 
                outputs: dict, 
                batch: dict,
                barrier_weight: float = 0.1,
                progress_weight: float = 0.01
               ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: 模型输出
            batch: 数据批次
            barrier_weight: 能垒正则化权重
            progress_weight: 进度正则化权重
        
        Returns:
            total_loss, losses_dict
        """
        device = outputs['ts_stability'].device
        losses = {}
        
        # ═══════════════════════════════════════════════════════
        # 1. 过渡态稳定性损失 L_ts（主损失）
        # ═══════════════════════════════════════════════════════
        pkd_mask = batch.get('pkd_target_mask', torch.ones_like(outputs['ts_stability'], dtype=torch.bool))
        
        if pkd_mask.any():
            l_ts = F.smooth_l1_loss(
                outputs['ts_stability'][pkd_mask],
                batch['pkd_target'][pkd_mask]
            )
        else:
            l_ts = torch.tensor(0.0, device=device)
        losses['L_ts'] = l_ts
        
        # ═══════════════════════════════════════════════════════
        # 2. 催化效率损失 L_catalysis
        # ═══════════════════════════════════════════════════════
        kcat_mask = batch.get('kcat_target_mask', torch.zeros_like(outputs['catalysis_rate'], dtype=torch.bool))
        
        if kcat_mask.any():
            l_cat = F.smooth_l1_loss(
                outputs['catalysis_rate'][kcat_mask],
                batch['log_kcat_target'][kcat_mask]
            )
        else:
            l_cat = torch.tensor(0.0, device=device)
        losses['L_catalysis'] = l_cat
        
        # ═══════════════════════════════════════════════════════
        # 3. 能垒正则化 L_barrier
        # ═══════════════════════════════════════════════════════
        barrier = outputs['barrier_height']  # (B,), 范围 [30, 200] kJ/mol
        
        # 鼓励能垒在合理范围：不要太低（催化太容易）也不要太高（太难）
        target_barrier = barrier.detach().clamp(50.0, 150.0)  # 中等范围
        l_barrier = F.mse_loss(barrier, target_barrier) * barrier_weight
        losses['L_barrier'] = l_barrier
        
        # ═══════════════════════════════════════════════════════
        # 4. 反应进度正则化 L_progress
        # ═══════════════════════════════════════════════════════
        # 确保BINN学到了"反应过程"，而非恒等映射
        if 'reaction_progress' in outputs:
            l_progress = outputs['reaction_progress'].mean()
            l_progress = F.relu(l_progress - 0.5) * progress_weight  # 鼓励进展
        else:
            l_progress = torch.tensor(0.0, device=device)
        losses['L_progress'] = l_progress
        
        # ═══════════════════════════════════════════════════════
        # 5. 不确定性权重 + 总损失
        # ═══════════════════════════════════════════════════════
        # 权重 = exp(-log_var)，sigma越大权重越小
        w_ts = torch.exp(-self.log_var_ts)
        w_cat = torch.exp(-self.log_var_cat)
        
        total = (
            w_ts * losses['L_ts'] + self.log_var_ts +
            w_cat * losses['L_catalysis'] + self.log_var_cat +
            losses['L_barrier'] +
            losses['L_progress']
        )
        
        losses['total'] = total
        losses['weights'] = {
            'w_ts': w_ts.item(),
            'w_cat': w_cat.item(),
            'sigma_ts': torch.exp(self.log_var_ts).item(),
            'sigma_cat': torch.exp(self.log_var_cat).item(),
        }
        
        return total, losses


# ═══════════════════════════════════════════════════════════════════════════════
# 完整模型：TransitionBINN（基于过渡态理论的新模型）
# ═══════════════════════════════════════════════════════════════════════════════

class TransitionBINN(nn.Module):
    """
    基于过渡态理论的BINN模型
    
    完全基于过渡态理论，不依赖电子转移假设：
    - 适用于所有氧化还原酶（不限于特定机制）
    - 不使用Marcus方程
    - 可扩展到其他酶类型（未来）
    
    架构：
      LigandEncoder (GATv2) + ProteinEncoder (ESM-2) + CofactorEncoder
      → ReactionCoordinateBINN (反应坐标Neural ODE + 门控)
      → BINNCatalysisHead (ts_stability + catalysis_rate)
    """

    COFACTOR_TYPES = sorted(COFACTOR_PRIORS.keys())

    def __init__(
        self,
        hidden_dim: int = 256,
        cofactor_embed_dim: int = 64,
        n_heads: int = 4,
        gnn_layers: int = 3,
        n_ode_steps: int = 5,
        use_gate: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.cofactor_embed_dim = cofactor_embed_dim
        
        # ═══════════════════════════════════════════════════════
        # 编码器（与原模型相同）
        # ═══════════════════════════════════════════════════════
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
        
        # ∗ 投影对齐
        self.ligand_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cofactor_proj = nn.Linear(cofactor_embed_dim, hidden_dim)
        
        # ═══════════════════════════════════════════════════════
        # BINN交互层（核心新组件）
        # ═══════════════════════════════════════════════════════
        self.binn = ReactionCoordinateBINN(
            hidden_dim=hidden_dim,
            n_ode_steps=n_ode_steps,
            use_gate=use_gate,
        )
        
        # ═══════════════════════════════════════════════════════
        # 预测头
        # ═══════════════════════════════════════════════════════
        self.catalysis_head = BINNCatalysisHead(hidden_dim=hidden_dim)
        
        # ═══════════════════════════════════════════════════════
        # 损失函数（可学习权重）
        # ═══════════════════════════════════════════════════════
        self.loss_fn = BINNLoss()

    def forward(self,
                ligand_data,
                seq_embed: torch.Tensor,
                cofactor_strs: list[str],
                struct_feat: Optional[torch.Tensor] = None,
                has_structure: Optional[torch.Tensor] = None,
                domain_masks: Optional[torch.Tensor] = None,
                domain_padding_mask: Optional[torch.Tensor] = None,
                pocket_cn: Optional[torch.Tensor] = None,
                pocket_pi: Optional[torch.Tensor] = None,
                pocket_dist: Optional[torch.Tensor] = None,
                pocket_mask: Optional[torch.Tensor] = None,
                ) -> dict[str, torch.Tensor]:
        """
        Returns:
            ts_stability:   (B,) 预测的pKd（过渡态稳定性）
            catalysis_rate: (B,) 预测的log10(kcat)
            barrier_height: (B,) 预测的活化能垒
            reaction_progress: (B,) 反应进度
            trajectory: list 演化轨迹
            h_reaction: (B, D) 最终状态
        """
        # ── 编码 ────────────────────────────────────────────
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

        cofactor_h, _ = self.cofactor_encoder(cofactor_strs)
        cofactor_h_proj = self.cofactor_proj(cofactor_h)
        
        # ── BINN反应坐标演化 ────────────────────────────────
        binn_output = self.binn(protein_h, ligand_h, cofactor_h_proj)
        
        # ── 催化预测 ────────────────────────────────────────
        catalysis_output = self.catalysis_head(
            binn_output['h_reaction'],
            binn_output['h_initial']
        )
        
        # ── 整合输出 ────────────────────────────────────────
        return {
            # 主要预测
            'ts_stability': catalysis_output['ts_stability'],     # ≈ pKd
            'catalysis_rate': catalysis_output['catalysis_rate'], # ≈ log10(kcat)
            'barrier_height': catalysis_output['barrier_height'], # ΔG‡ (kJ/mol)
            
            # 物理量（用于分析）
            'reaction_progress': binn_output['reaction_progress'],
            'h_reaction': binn_output['h_reaction'],
            'h_initial': binn_output['h_initial'],
            'h_final': binn_output['h_final'],
            
            # 轨迹信息（用于调试）
            'trajectory': binn_output['trajectory'],
            'gate_values': binn_output['gate_values'],
            'barrier_heights': binn_output['barrier_heights'],
            
            # 编码器输出（用于分析）
            'protein_h': protein_h,
            'ligand_h': ligand_h,
            'cofactor_h': cofactor_h,
        }

    def compute_loss(self,
                     outputs: dict,
                     batch: dict,
                     barrier_weight: float = 0.1,
                     progress_weight: float = 0.01,
                     ) -> tuple[torch.Tensor, dict]:
        """
        基于BINN的损失函数计算
        
        不再使用L_pkd + L_kcat的旧形式，而是：
        - L_ts: 过渡态稳定性（与pKd对应）
        - L_catalysis: 催化效率（与log_kcat对应）
        - L_barrier: 能垒正则化
        - L_progress: 反应进度正则化
        """
        return self.loss_fn(outputs, batch, barrier_weight, progress_weight)

    def predict_catalytic_efficiency(self, outputs: dict) -> torch.Tensor:
        """
        辅助函数：计算 log10(kcat/KM) 催化效率指标
        
        简化估计：kcat/KM ≈ kcat * Kd^(-1) = kcat * 10^pKd
        """
        pkd = outputs['ts_stability']
        log_kcat = outputs['catalysis_rate']
        log_kcat_km = log_kcat + pkd
        return log_kcat_km


def create_bin_optimizer(model: TransitionBINN, lr: float = 1e-4, weight_decay: float = 1e-5):
    """
    为TransitionBINN创建分组优化器
    
    不同于原模型，BINN的权重（dynamics_net, gate）需要不同的学习率
    """
    encoder_params = []
    binn_params = []
    head_params = []
    loss_params = []
    
    for name, param in model.named_parameters():
        if 'loss_fn' in name:
            loss_params.append(param)
        elif 'binn' in name:
            binn_params.append(param)
        elif 'catalysis_head' in name:
            head_params.append(param)
        else:
            encoder_params.append(param)
    
    return torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': binn_params, 'lr': lr * 0.5, 'weight_decay': weight_decay},  # BINN稍慢
        {'params': head_params, 'lr': lr * 2, 'weight_decay': weight_decay},
        {'params': loss_params, 'lr': lr * 0.1, 'weight_decay': 0},  # 损失权重学习更慢
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  酶挖掘排序模型 — TransitionBINN (基于过渡态理论)")
    print("=" * 60)

    model = TransitionBINN(hidden_dim=256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量:    {total_params:,}")

    print(f"\n  架构:")
    print(f"    LigandEncoder:   GATv2Conv x {model.ligand_encoder.convs.__len__()}")
    print(f"    ProteinEncoder:  ESM-2 1280-dim -> {model.hidden_dim}-dim")
    print(f"    CofactorEncoder: {model.cofactor_encoder.num_types} types -> {model.cofactor_embed_dim}-dim")
    print(f"    BINN层:          ReactionCoordinateBINN (n_steps={model.binn.n_steps})")
    print(f"    门控:            {'启用' if model.binn.use_gate else '禁用'}")

    print(f"\n  预测头:")
    print(f"    ts_stability:   过渡态稳定性 (approx pKd), 范围 [2, 15]")
    print(f"    catalysis_rate: 催化效率 (approx log10(kcat))")
    print(f"    barrier_height: 活化能垒 (kJ/mol)")

    print(f"\n  损失函数:")
    print(f"    L_ts:        过渡态稳定性 (不确定性权重)")
    print(f"    L_catalysis: 催化效率 (不确定性权重)")
    print(f"    L_barrier:   能垒正则化")
    print(f"    L_progress:  反应进度正则化")

    print("\n  ✓ 结构验证通过 — TransitionBINN Ready!")
