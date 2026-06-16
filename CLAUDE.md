# CLAUDE.md — AI4enz

AI 辅助酶挖掘项目。基于过渡态理论构建酶-底物结合亲和力预测模型，服务于"给定目标底物 → 推荐酶序列"的酶挖掘场景。

每次回复前都叫我：**"多米"**
原则是：我允许你不懂，你绝对不要不懂装懂，任何建议都需要有真实的理论支撑

## 核心架构

### Trenzition (v3, 三头 + Eyring 物理硬编码)

**三头独立预测**：BindingDualHead (Kd/Ki 双分支) + EyringKcatHead (BINN+ODE+Eyring硬编码) + KmHead (MLP)。
kcat 不再通过 MLP 黑箱回归，而是让模型预测 ΔG‡，用 Eyring 公式硬编码转换为 kcat。

**文件位置**: `model/ranking_model.py`

### 架构图

```
shared encoders (LigandEncoder GATv2×3 + ProteinEncoder ESM-2 + CofactorEncoder)
    │
    ├── BindingDualHead (BINN 1步, 无ODE)
    │     ├── Kd_branch → pKd_Kd (底物结合)
    │     └── Ki_branch → pKd_Ki (抑制剂结合, 损失权重0.7)
    │     + 测量类型 one-hot [is_Kd, is_Ki] 作为输入特征
    │
    ├── EyringKcatHead (BINN + 多步ODE + Eyring 公式硬编码)
    │     ├── dG_predictor (256→64→1) → ΔG‡ [5, 300] kJ/mol
    │     ├── log_kappa (可学习, 初始 log₁₀(0.5))  ← 透射系数
    │     ├── Eyring 公式:  ΔG‡ + T(K)  →  log₁₀(kcat)   ← 不可学习变换
    │     └── 无需 L_eyring: 物理规律已硬编码在模型内部
    │
    └── KmHead (简单 MLP, 归一化 [0,1])
          └── sigmoid → log₁₀(Km) [0,1], 由 ThreeHeadLoss 反归一化到 [-13, 3]
```

### 设计原则

1. **三头独立**：每个任务独立预测路径，避免负迁移
2. **Eyring 物理硬编码**：kcat 由 ΔG‡ 和温度通过 Eyring 公式直接算出，模型只能通过改变 ΔG‡ 影响 kcat，规避黑箱过拟合
3. **每样本独立温度**：温度 T(K) 作为输入传给 EyringKcatHead，覆盖 276-562K
4. **物理约束联合**：kcat/Km 扩散极限 (≤10⁹ M⁻¹s⁻¹) + 热力学循环
5. **测量类型感知**：Kd（底物结合）和 Ki（抑制剂结合）双分支，类型 one-hot 作为输入
6. **Kcat 保留 ODE**：仅 kcat 使用多步中点 Euler 积分（物理反应坐标动力学）

### 组件

| 组件 | 类名 | 功能 |
|------|------|------|
| `LigandEncoder` | GATv2×3 + GlobalAttention | 配体分子图 → 256-dim |
| `ProteinEncoder` | ESM-2 1280→256 + AA属性备选 | 蛋白序列 → 256-dim |
| `CofactorEncoder` | Embedding + Attention | 辅因子 → 64-dim → 256-dim |
| `LatentPathwayBINN` | 中点Euler积分 (ODE) | kcat 头的多步特征变换 |
| `BindingDualHead` | BINN(1步) + Kd/Ki双分支 | 结合亲和力 pKd 预测 |
| `EyringKcatHead` | BINN(ODE) + dG_predictor + Eyring硬编码 | dG_predictor→ΔG‡→Eyring→kcat (不可学习变换) |
| `KmHead` | 简单MLP | log₁₀(Km) 预测 |

### 物理公式 (Eyring)

```
kcat = κ · (k_B·T/h) · exp(-ΔG‡ / RT)

log₁₀(kcat) = log₁₀(κ) + log₁₀(k_B·T/h) - ΔG‡ / (R·T·ln10)

其中:
  κ (透射系数): 可学习全局标量, 初始 log₁₀(0.5)
  T (温度): 每样本独立 (K)
  ΔG‡ (活化自由能): 模型预测, 钳位 [5, 300] kJ/mol
  k_B, h, R: 物理常数
```

### 损失函数

```python
L_total = L_binding + L_kcat + L_km + 0.1*L_joint

# ── 1. 结合亲和力损失 L_binding ──
# Kd 分支: SmoothL1(kd_pred, pkd_target) 权重 1.0
# Ki 分支: SmoothL1(ki_pred, pkd_target) 权重 0.7
L_binding = L_kd + 0.7 * L_ki

# ── 2. 催化效率损失 L_kcat (Eyring 已硬编码在模型内) ──
L_kcat = SmoothL1(kcat_pred, log_kcat_target)    # 均 [0,1], 仅 kcat 标签有效

# ── 3. Km 损失 L_km ──
L_km = SmoothL1(log_km_pred, log_km_target_norm) # 均归一化 [0,1]

# ── 4. 联合约束 L_joint (kcat/Km 物理自洽) ──
# kcat/Km = log₁₀(kcat) - log₁₀(Km) ≈ 真实 kcat/Km
L_joint = SmoothL1(log_kcatKm_pred, log_kcatKm_true)
        + 0.01 * ReLU(log_kcatKm_pred - 9)       # 扩散极限 ≤ 10⁹ M⁻¹s⁻¹
```

> 注意：Eyring 约束已从损失中移除，因为 Eyring 公式已硬编码在 `EyringKcatHead` 中。模型输出的 kcat 直接由 ΔG‡ 通过 Eyring 方程算出，不可违背。

### 激活函数

| 位置 | 激活 | 说明 |
|------|------|------|
| ODE 动力学层 + 共享层 | **GELU** | 与 ESM-2 一致 |
| 输出头中间层 | **SiLU** | 比 ReLU 更平滑 |
| 输出头最后一层 | **sigmoid** | 压缩到 [0,1]，匹配归一化 |
| 动力学最后一层 | **Tanh** | 有界输出，保证稳定性 |

### 训练策略

| 策略 | 实现 | 说明 |
|------|------|------|
| 优化器 | **AdamW + 分组LR** | Encoder×1, BINN×0.5, Head×2 |
| LR 调度 | **Warmup → CosineAnnealingWarmRestarts** | 1000步预热 → T₀=5000, T_mult=2 |
| 梯度裁剪 | max_norm=**1.0** | 防止 ODE 梯度爆炸 |
| 参数初始化 | **Xavier 定制 gain** | 输出头 gain=0.3, 动力学 gain=0.67 |

### 参数初始化

在 `ranking_model.py:Trenzition._init_weights` 中实现：

| 层级 | shape | gain | 原因 |
|------|-------|------|------|
| 输出头最后一层 → sigmoid (Binding/Km) | Linear(64,1) | **0.3** | 防 sigmoid 初始饱和 |
| dG_predictor 最后一层 | Linear(64,1) | **0.3** | 防 ΔG‡ 初始发散 |
| BINN 动力学层 | Linear(256,256) | **0.67** | 防 ODE 多步积分爆炸 |
| 其他所有 Linear | 任意 | **1.0** | 标准 Xavier |
| LayerNorm | — | 保持默认 | 最优 |

## Min-Max 归一化

| 目标 | 原始范围 | 归一化参数 | 输出范围 |
|------|----------|------------|----------|
| pKd | [0, 12] | pkd_min=0.0, pkd_max=12.0 | [0, 1] |
| log₁₀(kcat) | [-7, 8] | kcat_min=-7.0, kcat_max=8.0 | [0, 1] |
| log₁₀(Km) | [-13, 3] | km_min=-13.0, km_max=3.0 | [0, 1] |

## 数据与脚本组织原则

1. **数据按来源分文件夹**：下载新数据集时，在 `dataset_building/` 下创建以数据库名命名的文件夹（如 `BindingDB/`、`BRENDA/`、`OED/`），原始文件直接放入。禁止将多个来源的数据混放在同一目录。
2. **同类脚本同目录**：同一子项目的脚本位于同一文件夹（如 `scripts/crawlers/` 放爬虫、`scripts/processors/` 放特征工程、`pipeline/` 放处理流水线）。新增子项目时创建对应文件夹。

## 项目结构

```
AI4enz/
├── model/                            # 模型核心目录 (训练/推理/评估)
│   ├── ranking_model.py              # Trenzition v3 模型定义 (三头+Eyring硬编码)
│   ├── train.py                      # 训练脚本 (OxidoreductaseDataset + Trainer)
│   ├── predict.py                    # 推理脚本 (单样本+批量)
│   ├── benchmark_enhanced.py         # 基准测试 (vs XGBoost/MLP baselines)
│   ├── checkpoints/
│   │   └── best.ckpt                 # 预训练权重
│   └── tst/                          # TST 模型 (实验性)
│       ├── __init__.py
│       └── tst_model.py
├── dataset_building/
│   ├── golddata/                     # 高质量金标子集
│   │   ├── gold.parquet              # 38,489 条 (双标签+辅因子+温度)
│   │   ├── proteins.h5 -> ../processed/proteins.h5
│   │   ├── ligands/ -> ../processed/ligands/
│   │   └── README.md
│   ├── processed/                    # V5 核心数据 (训练用)
│   │   ├── metadata.parquet          # 104,021 条元数据 (含温度列)
│   │   ├── proteins.h5               # 18,929 个蛋白 ESM-2 嵌入
│   │   └── ligands/                  # 14,888 个配体 GNN 图 (.pt)
│   ├── models_v1_obsolete/           # 旧版 v1/v2 模型归档
│   ├── pipeline/                     # 数据处理流水线 (01-08)
│   ├── scripts/                      # 数据构建/补充脚本
│   │   ├── fill_missing_ligands.py   # 补齐缺失配体图
│   │   └── ...
│   ├── analysis/                     # 图表与分析
│   ├── evaluation/                   # 模型评估
│   ├── release/                      # 旧版本发布
│   ├── BindingDB/                    # BindingDB 原始数据
│   ├── BRENDA/                       # BRENDA 原始数据
│   ├── CataPro/                      # CatPred-DB 源码+数据
│   ├── OED/                          # OED 动力学数据
│   ├── SABIO-RK/                     # SABIO-RK 数据
│   └── SKiD/                         # SKiD 数据
├── README.md
└── CLAUDE.md
```

## 虚拟环境

```bash
source /home/domi/BINN/.venv/bin/activate
```

## 训练命令

### 三头模型训练

```bash
cd /home/domi/AI4enz/model

# 快速验证（CPU, 小样本）
python train.py --kcat-ode-steps 10 \
  --epochs 10 --batch-size 32 --max-samples 5000 \
  --device cpu --num-workers 0

# GPU 全优化训练（推荐，默认三头模式）
python train.py --kcat-ode-steps 10 \
  --epochs 100 --batch-size 256 \
  --device cuda --num-workers 8 --amp
```

## 数据集 — trenzition V5（改良版, 2026-06-16）

### 全局统计

| 指标 | 值 |
|------|-----|
| 总记录 | **104,021** |
| pKd有效 | 77,867 (74.9%) |
| kcat有效 | 90,008 (86.5%) |
| **Km有效** | **94,279 (90.6%)** |
| 三标签(pKd+kcat+Km) | **63,672 (61.2%)** |
| 唯一蛋白 | 18,929 |
| 唯一配体 | 14,888 |
| 唯一 EC 号 | 2,848 |

### 测量类型分布

| 类型 | 数量 | 可信度 | 损失权重 |
|------|------|--------|---------|
| Kd | 20,690 | 高（直接结合） | 1.0 |
| Ki | 56,928 | 中（抑制常数近似） | 0.7 (Ki分支) |
| IC50_approx | 249 | 低 | 不参与binding损失 |
| (仅kcat数据) | 26,154 | — | 无binding损失 |

### Split（蛋白层级，零泄漏）

| 切分 | 样本数 | 蛋白数 | pKd有效 | kcat有效 | Km有效 |
|------|--------|--------|---------|---------|-------|
| train | 73,261 | 13,251 | 55,035 | 63,262 | 66,420 |
| val | 15,552 | 2,839 | 11,695 | 13,306 | 14,064 |
| test | 15,208 | 2,839 | 11,137 | 13,440 | 13,795 |

### 编码状态

| 组件 | 覆盖率 | 编码方式 |
|------|--------|----------|
| 蛋白 ESM-2 | 18,929/18,929 (100%) | esm2_t33_650M, 1280-dim mean-pool |
| 配体 GNN | 14,888/14,888 (100%) | GATv2×3, 79-dim 原子特征 + 10-dim 键特征 |
| 辅因子 | 56.7% 非空 | 310 种组合，可学习 Embedding |
| 温度 | 100% (67.9%非默认) | 每样本独立 T(K), 范围 276-562K |
| BindingDB 扩充 | 14,640行/426蛋白 | 新增 9,336 条高质量 Kd/Ki 数据 |

### 温度分布

| 范围 | 占比 | 说明 |
|------|------|------|
| 270-290K | 0.4% | 近室温 |
| 290-310K | **69.8%** | 生理温度 |
| 310-330K | **25.6%** | 中温（耐热酶） |
| 330-400K | 4.2% | 高温 |
| 400K+ | 4条 | 极端 |

## 已知问题

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🟡 中 | proteins.h5 含历史数据 | 53,133 个蛋白中仅 18,929 用于当前训练集 |
| 🟡 中 | 辅因子覆盖下降 | 56.7%（新增 BindingDB 行无辅因子信息），mask 机制跳过 |
| 🟡 中 | Ki 标签噪声 | Ki pKd 中位数仅 1.68，远低于 Kd 的 5.09，Ki 分支权重 0.7 |
| 🟢 低 | pKd-kcat 相关性差 (ρ≈-0.18) | 数据源固有差异，需训练后验证多任务收益 |

## 数据来源

| 来源 | 提供数据 | 规模 |
|------|----------|------|
| CatPred-DB (Nature Comms 2025) | kcat, Ki | ~31k |
| OED (NAR 2025) | kcat, Km, kcat/Km | ~36k |
| SKiD (Scientific Data 2025) | kcat, Km + 3D结构 | ~13k |
| BindingDB | Kd, Ki | ~153k |

## 当前任务进度 (2026-06-16)

### 已完成
- [x] **三头独立预测架构** — BindingDualHead(Kd/Ki) + EyringKcatHead(dG→Eyring→kcat) + KmHead(MLP)
- [x] **Eyring 公式硬编码** — 模型预测 ΔG‡，Eyring 公式不可学习变换为 kcat，无需 L_eyring
- [x] **每样本独立温度** — temperature_K 传入 EyringKcatHead，覆盖 276-562K
- [x] **测量类型修复** — `fillna("Kd")` bug 修复，零假 Kd
- [x] **无效数据过滤** — 4,608 条 kcat=-10 占位符 mask；3,832 条弱 Ki (pKd<0.3) mask
- [x] **Km 数据整合** — km_M 编入 metadata，归一化 [0,1] 与 Kd/kcat 一致
- [x] **BindingDB 扩充** — 5,304 条/179 蛋白 → **14,640 条/426 蛋白** (+9,336)
- [x] **配体图全覆盖** — 14,888/14,888 (100%)，补齐 177 个缺失配体
- [x] **Gold 数据更新** — 25,950 条双标签+辅因子+实测温度
- [x] **分类头/Gate 移除** — 纯回归模型，删除 gate/classification_head 引用
- [x] **温度补充** — OED + BRENDA 三层匹配，覆盖 100%
- [x] **辅因子补充** — EC + UniProt 双来源，覆盖率 56.7%
- [x] **模型目录重构** — 旧模型归档至 models_v1_obsolete/
- [x] **训练脚本清理** — 移除 Gate/L_eyring/未使用字段，统一归一化

### 待完成
- [ ] **三头模型 GPU 训练** — 在改良数据集上训练 v3
- [ ] **与已发表模型对比** — DLKcat/CatPred 在同数据集上比较
