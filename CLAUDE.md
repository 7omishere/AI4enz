# AI4enz — 项目约定（面向 AI 助手）

1.每次对话前都叫我，多米
2.不要不懂装懂，我允许你不懂和否决/质疑我
3.架构设计决策记录见 [learning.md](learning.md)，包含 cross-attention 方案选型、ResidueQueryAttention 动机、损失函数审核、多任务 vs 单任务文献调研等，后继所有这种对架构设计工作的修改和决策，在learning.md里面更新而不是在这里.
4.claude.md更新不需要涉及测试的命令，需要修改的话你应该在README.md中更改（claude主要承担你对项目理解的任务）
5.给你布置训练任务时候，以README.md为参考


## 项目概述

酶-底物结合亲和力 (pKd)、催化效率 (kcat) 和米氏常数 (Km) 的深度预测模型。
核心是 Trenzition v5 架构——全酶交叉注意力 + 残基级查询注意力 + Eyring 物理硬编码。

## 关键文件

| 文件 | 内容 |
|------|------|
| `model/ranking_model.py` | **Trenzition v5 完整模型** — 编码器、CrossAttentionFusion、ResidueQueryAttention、预测头、损失函数 |
| `model/train.py` | 训练入口，数据加载，训练循环（含分层采样/不确定性加权/PCGrad） |
| `model/evaluate.py` | 评估脚本 (PCC/SCC/R²) |
| `model/predict.py` | 批量推理脚本 |
| `model/compute_token_embeddings.py` | ESM-2 token embedding 预计算 |
| `dataset_building/processed/proteins_token.h5` | ESM-2 per-token embedding (18,929 蛋白) |

## 当前架构（v5）

```
蛋白序列 → ESM-2 (离线) → (L, 1280) → H5 文件
                                          ↓
                                    ProteinEncoder
                                 token_proj (1280→512)
                                    → token_h (B, L, 512)
                                          ↓
LigandEncoder (GATv2×3) ──────────────┐    │
  → per_atom_h (N, 512)               │    │
  → to_dense → (B, A, 512)            │    │
                                      ▼    ▼
CofactorEncoder ───────────┐    CrossAttentionFusion:
  → multi-hot → embed (64) │      Q = [token_h; cofactor_token]
  → proj (512)             │      K/V = ligand_atoms
  → unsqueeze (B, 1, 512) ─┘      → 残差+FFN → h_protein_only (B, L, 512)
                                          ↓
TemperatureEncoder               ResidueQueryAttention × 3
  → (T-298.15)/50                (各头独立学习查询向量)
  → Linear → GELU → proj               │    │    │
                                          ▼    ▼    ▼
                                  Binding  Kcat   Km
                                  BINN(1步) │     MLP
                                  Kd/Ki双分 │
                                 支         │
                                   BINN(多步ODE) → ΔG‡ → Eyring → kcat
```

## 核心组件

### CrossAttentionFusion
- Q = \[蛋白 L 个残基; 辅因子 1 个虚拟 token\]  — 全酶（holoenzyme）
- K/V = 配体原子（来自 GATv2 per-atom 输出）
- 辅因子作为酶的一部分调制每个残基对底物的注意力模式
- 输出 `h_protein_only (B, L, 512)` — 保留残基级，不在 fusion 后池化

### ResidueQueryAttention（×3，各头独立）
- 每个预测头有一个可学习查询向量 `(1, 1, 512)`
- 查询向量对全长残基序列做 **cross-attention** → 加权求和
- 各头自然学会关注不同残基：结合位点 / 催化位点 / 底物解离路径
- MultiheadAttention 的 4 个头中分裂为 4 × 128 维子空间，共 3 × 4 = 12 种注意力模式

### 预测头
| 头 | 架构 | 输入 | 输出 |
|----|------|------|------|
| BindingDualHead | BINN(1步) + Kd/Ki双分支 | protein_h + ligand_h + catalyst_context | pKd (sigmoid归一化) |
| EyringKcatHead | BINN(多步ODE) → dG_predictor → Eyring | protein_h + ligand_h + catalyst_context + T(K) | log₁₀(kcat) |
| KmHead | MLP | protein_h + ligand_h + catalyst_context | log₁₀(Km) |

- Eyring 公式硬编码不可学习：`log₁₀(kcat) = log₁₀(κ·k_B·T/h) − ΔG‡ / (R·T·ln10)`
- 模型只预测 ΔG‡ (\[5, 300\] kJ/mol)，由 `ELU + x/(x+1)` 映射保证有界
- 可学习透射系数 `log_kappa`（全局标量）
- 温度双重身份：Eyring 公式物理参数 + TemperatureEncoder 可学习特征向量

## 损失函数

```python
L_total = L_binding + L_kcat + L_km + 0.1·L_joint + 0.05·L_dG_prior
```

| 项 | 内容 |
|----|------|
| L_binding | Kd分支(SmoothL1) + 0.3·Ki分支(SmoothL1) |
| L_kcat | kcat回归 (SmoothL1) |
| L_km | Km回归 (SmoothL1) |
| L_joint | kcat/Km比值回归 + 扩散极限约束 ReLU(kcat/Km − 9) |
| L_dG_prior | ΔG‡ 高斯先验 NLL(μ=70, σ=20) |

各损失通过 `pkd_target_mask` / `kcat_target_mask` / `km_target_mask` 布尔掩码自动跳过无标签样本。

### 多任务平衡（可选）

| 策略 | 机制 | CLI 参数 |
|------|------|----------|
| **分层采样** | 每 batch 从 binding/kcat/Km 三组等量采样 (1:1:1) | `--stratified-sampling` |
| **不确定性加权** | 3 个可学习 log_σ 参数，`L_i / (2σ²) + log(σ)` | `--uncertainty-weighting` |
| **PCGrad** | 梯度投影消除共享层冲突 | `--pcgrad` |

## 数据流细节

- 蛋白：从 `proteins_token.h5` 按 `protein_seq_hash` 加载 ESM-2 token embed → `(L, 1280)` → `ProteinEncoder.token_proj` → `(B, L, 512)`
- 配体：从 `processed/ligands/{inchikey}.pt` 加载 PyG 图 → `LigandEncoder(GATv2×3)` → per-atom `(B, A, 512)`
- 辅因子：parquet 中 `cofactors` 列（`|` 分隔）→ `CofactorEncoder` multi-hot 加权求和 → `(B, 512)`
- 温度：每样本独立 `temperature_K` → `(T-298.15)/50` → `TemperatureEncoder` → `(B, 512)`
- 催化剂上下文：`cofactor_h + temp_h`（残差相加），传入三个预测头

## hidden_dim=512 的选择依据

ESM-2 embedding 压缩失真评估（PCA + 局部拓扑保持分析，2026-06-23）：

| 指标 | 256 | 384 | **512** |
|------|-----|-----|---------|
| 解释方差 | 95.96% | 98.47% | **99.46%** |
| KNN Accuracy@15 | 93.65% | 97.16% | **98.68%** |
| Trustworthiness | 0.9998 | 0.9999 | **1.0000** |
| Sammon Stress | 0.00066 | 0.00010 | **0.00001** |

512 维保留 99.5% 方差，KNN 保持 98.7%，ESM-2 有效秩仅 ~135 维。

## 常见陷阱

- **dG_prior 在 Trenzition.compute_loss 中处理**，不在 ThreeHeadLoss 里
- **ligand_data 必须总是合法的 PyG batch** — 即使 `use_cross_attn=False` 也不能传 None
- **use_residue_attn 不依赖 use_cross_attn** — 无 cross-attn 时直接用 ProteinEncoder 的 token_h
- **ResidueQueryAttention 额外显存约 60MB**（B=32, L=500），对 H100/H200 可忽略
- **PCGrad + AMP 不兼容**，同时启用时自动禁用 PCGrad 并警告
- **PCGrad 只对共享层（encoder/fusion）做梯度投影**，预测头参数不受影响

## 参数量概览（v5，当前未使用不确定性加权时）

| 模块 | 参数量 | 占比 |
|------|--------|------|
| ProteinEncoder | 920,064 | 6.4% |
| LigandEncoder (GATv2×3) | 2,940,929 | 20.5% |
| CofactorEncoder | 5,377 | <0.1% |
| TemperatureEncoder | 8,736 | 0.06% |
| CrossAttentionFusion | 1,315,840 | 9.2% |
| BindingDualHead (含ResAttn) | 3,483,138 | 24.3% |
| EyringKcatHead (含ResAttn) | 3,713,666 | 25.9% |
| KmHead (含ResAttn) | 1,971,713 | 13.7% |
| **总计** | **14,359,463** | 100% |

每个 `ResidueQueryAttention` 增加 ~1.05M 参数（query + MultiheadAttention + LayerNorm），三个共增 ~3.16M。
