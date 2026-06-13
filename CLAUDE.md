# CLAUDE.md — AI4enz

AI 辅助酶挖掘项目。基于过渡态理论构建酶-底物结合亲和力预测模型，服务于"给定目标底物 → 推荐酶序列"的酶挖掘场景。

每次回复前都叫我：**"多米"**

## 核心架构：Trenzition

**回归模式**：pKd 路径 + kcat 独立路径，直接优化 log₁₀(kcat/KM) 排序。Gate 作为酶-底物兼容性分类器，用于两阶段筛选（先判断"能不能反应"，再比较"反应效率"）。

### 设计原则

1. **过渡态理论**（普适，所有酶催化都满足）
2. **共享编码器 + 门控动力学**（信息流控制，非物理反应坐标）
3. **固定损失权重**（目标值归一化后量级一致）
4. **kcat 独立路径**（蛋白级表征，避免负迁移）

### 组件

| 组件 | 类名 | 功能 |
|------|------|------|
| `LigandEncoder` | GATv2×3 + GlobalAttention | 配体分子图 → 256-dim |
| `ProteinEncoder` | ESM-2 1280→256 (主力) + AA属性备选 | 蛋白序列 → 256-dim |
| `CofactorEncoder` | Embedding + Attention | 辅因子 → 64-dim → 256-dim |
| `LatentPathwayBINN` | 门控 + 中点Euler积分 (n_steps=1) | 特征变换 + 信息流控制 |
| `LatentGate` | MLP(h, catalyst, ligand) → sigmoid | 酶-底物兼容性分类器 |
| `TrenzitionCatalysisHead` | shared + 3个独立头 | ts_stability + catalysis_rate + dG_eyring |

### Gate 语义

Gate ∈ [0,1] 回答最基础的问题：**这个酶-底物对是否存在有意义的相互作用？**

```
gate ≈ 1 → "酶-底物对合理，动力学完全激活" → pKd/kcat 预测可信
gate ≈ 0 → "配对不合理，动力学被阻断"   → 预测值不可信（基于原始表示）
```

- Gate 不是物理能垒（那是 `dG_eyring` 的工作）
- Gate 是数据驱动的特异性分类器——类似"酶特异性"的软判断
- 实际用途：两阶段酶挖掘（Gate 过滤 → 催化效率排序）

### 损失函数

```python
L_total = L_ts + L_catalysis + 0.01 * L_eyring + gate_weight * L_gate

# ── 正样本（真实酶-底物对）──
L_ts:        SmoothL1(ts_stability, pkd_target)        # 仅 pKd 标签有效时
L_catalysis: SmoothL1(catalysis_rate, log_kcat_target) # 仅 kcat 标签有效时
L_eyring:    MSE(dG_pred, dG_from_kcat)                # 仅 kcat 标签有效时
L_gate:      (1 - gate)²                               # 推 gate → 1

# ── 负样本（跨 EC 随机配对）──
L_ts:        0  (mask=False, 无标签)
L_catalysis: 0  (mask=False, 无标签)
L_eyring:    0  (无催化反应，Eyring 方程不适用)
L_gate:      gate²                                     # 推 gate → 0（唯一梯度来源）

# 关键设计：
# - 缺失标签: mask 机制自动跳过，单标签样本仍贡献共享编码器梯度
# - Eyring 仅在 kcat 标签有效时计算（负样本跳过，避免噪声拉扯 log10_prefactor）
# - L_gate 使用非 detached gate_profile，梯度完整到达 gate_net + dynamics_net + encoders
```

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
| 输出头最后一层 → sigmoid | Linear(64,1) | **0.3** | 防 sigmoid 初始饱和 |
| BINN 动力学层 | Linear(256,256) | **0.67** | 防 ODE 多步积分爆炸 |
| 其他所有 Linear | 任意 | **1.0** | 标准 Xavier |
| LayerNorm | — | 保持默认 | 最优 |

## Min-Max 归一化

| 目标 | 原始范围 | 归一化参数 | 输出范围 |
|------|----------|------------|----------|
| pKd | [0, 12] | pkd_min=0.0, pkd_max=12.0 | [0, 1] |
| log₁₀(kcat) | [-7, 8] | kcat_min=-7.0, kcat_max=8.0 | [0, 1] |

## 数据与脚本组织原则

1. **数据按来源分文件夹**：下载新数据集时，在 `dataset_building/` 下创建以数据库名命名的文件夹（如 `BindingDB/`、`BRENDA/`、`OED/`），原始文件直接放入。禁止将多个来源的数据混放在同一目录。
2. **同类脚本同目录**：同一子项目的脚本位于同一文件夹（如 `scripts/crawlers/` 放爬虫、`scripts/processors/` 放特征工程、`pipeline/` 放处理流水线）。新增子项目时创建对应文件夹。

## 项目结构

```
AI4enz/
├── dataset_building/
│   ├── BindingDB/              # BindingDB 原始数据
│   ├── BRENDA/                 # BRENDA 原始数据
│   ├── CataPro/                # CataPro 源码+数据
│   ├── KEGG/                   # KEGG 酶/EC 数据
│   ├── OED/                    # OED 动力学数据
│   ├── SABIO-RK/               # SABIO-RK（API获取，暂空）
│   ├── SKiD/                   # SKiD 数据+kcat_archive
│   ├── turnup/                 # TurnUp 分子文件数据
│   ├── uniprotprot/            # UniProt/SwissProt 数据库
│   ├── models/                 # 模型定义+训练+权重
│   ├── pipeline/               # 数据处理流水线 (01-08)
│   ├── scripts/                # 数据构建/补充脚本
│   ├── analysis/               # 图表与分析
│   ├── evaluation/             # 模型评估
│   ├── checkpoints/            # 训练输出快照
│   ├── release/                # 最终训练数据集
│   └── processed/               # V5 核心数据 + 辅助文件
├── README.md                              # 项目简介
├── CLAUDE.md                              # 本文件
├── scripts/                               # 数据处理脚本
│   ├── crawlers/                          # 数据爬取/解析
│   │   ├── fetch_kegg_mapping.py          # KEGG API 爬取
│   │   ├── fetch_uniprot_ec_mapping.py    # UniProt API 爬取
│   │   ├── parse_swissprot.py             # SwissProt 解析
│   │   ├── parse_kegg_batch.py            # KEGG 批量查询
│   │   └── merge_bindingdb.py             # BindingDB 数据合并
│   └── processors/                        # 数据处理
│       ├── merge_external_data.py         # 外部数据整合
│       ├── extract_domains.py             # 结构域提取
│       ├── enrich_domains_pfam.py         # Pfam 富集
│       └── compute_esm2_embeddings.py     # ESM2 嵌入计算
└── dataset_building/
    ├── models/                            # 模型核心
    │   ├── ranking_model.py               # Trenzition 模型定义
    │   ├── train.py                       # 训练脚本
    │   └── __init__.py
    ├── evaluation/                        # 模型评估
    │   └── evaluate_model.py
    ├── analysis/                          # 分析与可视化
    │   ├── plot_*.py                      # 绘图脚本
    │   ├── analyze_metrics.py
    │   ├── ablation_study.py
    │   ├── baseline_comparison*.py
    │   └── diagnose_fluctuation.py
    ├── release/                           # 训练数据集
    ├── processed/                         # 处理后数据
    ├── external_data/                     # 原始数据源
    └── checkpoints/                       # 模型检查点
```

## 虚拟环境

```bash
source /home/domi/BINN/.venv/bin/activate
```

### 测量类型分布

| 类型 | 数量 | 可信度 | Loss 权重 |
|------|------|--------|-----------|
| Ki | 59,883 | 中-高 | 0.70 |
| Kd | 37,219 | 高 | 1.00 |
| IC50_approx | 249 | 低 | 0.15 |

> ⚠️ `THERMO_WEIGHT` 字典在 `train.py` 中定义并计算了 `quality_weight`，但当前 `TrenzitionLoss.forward()` 未使用这些权重（所有测量类型等权重训练）。Key 必须为 `'IC50_approx'`（与数据一致），不能使用 `'IC50'`。

## 训练命令

```bash
cd /home/domi/AI4enz/dataset_building/models

# 快速验证（CPU, 小样本）
python train.py --epochs 10 --batch-size 32 --max-samples 5000 --device cpu --num-workers 0

# 完整训练（CPU）
python train.py --epochs 100 --batch-size 128 --device cpu --num-workers 2

# GPU 全优化训练（推荐）
python train.py --epochs 100 --batch-size 256 --device cuda --num-workers 8 --amp --compile

# GPU 小显存 (< 16GB)
python train.py --epochs 100 --batch-size 64 --device cuda --num-workers 4 --amp --grad-accum 2

# 无 warmup（调试用）
python train.py --epochs 10 --batch-size 128 --warmup-steps 0
```

## 数据集 — trenzition V5

### 全局统计

| 指标 | 值 |
|------|-----|
| 总记录 | **97,351** |
| pKd有效 | 71,482 (73.4%) |
| kcat有效 | 92,743 (95.3%) |
| 双标签(pKd+kcat) | 66,874 (68.7%) |
| **EC号** | **97,351 (100%)** |
| 唯一蛋白 | 19,223 |
| 唯一配体 | 7,249 |

### 编码状态

| 组件 | 覆盖率 | 编码方式 |
|------|--------|----------|
| 蛋白 ESM-2 | 19,223/19,223 (100%) | esm2_t33_650M, 1280-dim mean-pool |
| 配体 GNN | 7,249/7,249 (100%) | GATv2×3, 79-dim 原子特征 + 10-dim 键特征 |
| 辅因子 | 57.8% 非空 | 310 种组合，可学习 Embedding |

### Split（蛋白层级，零泄漏）

| 切分 | 样本数 | 蛋白数 | pKd有效 | kcat有效 |
|------|--------|--------|---------|----------|
| train | 68,876 | 13,454 | 50,804 | 68,876 |
| val | 13,801 | 2,885 | 10,176 | 13,801 |
| test | 14,674 | 2,884 | 10,502 | 14,674 |

Split 按 **protein_seq_hash 层级**分配，test/val与train蛋白完全零重叠。

## 已知问题

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🟡 中 | 无 GPU | CPU 训练较慢，97k 样本 × 100 epoch ≈ 3 天 |
| 🟡 中 | proteins.h5 含历史数据 | 53,133 个蛋白中仅 19,223 用于当前训练集 |
| 🟡 中 | quality_weight 未使用 | `THERMO_WEIGHT` 计算了但 Loss 未加权，Ki/Kd/IC50 等权训练 |
| 🟢 低 | 辅因子覆盖 | 可后续补充稀有辅因子 |
| 🟢 低 | 结构/口袋特征未启用 | metadata 中 has_structure/has_binding_site 均为默认值 |
| 🟢 低 | 无早停/EMA | 辅助稳定性技巧，有空可加 |

## 数据来源

| 来源 | 提供数据 | 规模 |
|------|----------|------|
| CatPred-DB (Nature Comms 2025) | kcat, Ki | ~31k |
| OED (NAR 2025) | kcat, Km, kcat/Km | ~36k |
| SKiD (Scientific Data 2025) | kcat, Km + 3D结构 | ~13k |
| BindingDB | Kd, Ki | ~153k |

## 当前任务进度 (2026-06-14)

### 已完成
- [x] **GPU 训练** — best.ckpt (epoch 81, val_loss=0.0056)
- [x] **Benchmark** — pKd ρ=0.895, kcat ρ=0.671 (Trenzition)
- [x] **消融实验** — 配体 GNN 贡献最大 (R² 0.703→0.048)，ODE 多步可压缩至 1 步
- [x] **推理脚本** — `predict.py`: SMILES + 蛋白序列 → pKd + kcat + ΔG‡
- [x] **负样本生成** — `scripts/generate_negatives.py`: 跨 EC 负采样，1:1 (194k 样本)
- [x] **Gate 修复 (2026-06-14)** — 修复 4 个 bug：
  1. `gate.detach()` 移除 → L_gate 梯度正确回传 gate_net
  2. `dh_correction *= gate` → gate=0 时完全阻断动力学（修复 50% 泄漏）
  3. Eyring loss 仅对 kcat 标签样本计算 → 负样本不再噪声拉扯 log10_prefactor
  4. `pkd_target_mask` 默认值改为 zeros_like（安全默认）
- [x] **微调支持** — train.py 支持 `--finetune <ckpt_path>` + `--gate-weight`

### 待完成
- [ ] **负样本微调** (修复后重新训练)
- [ ] 微调后评估：Gate 分布（pos vs neg）、pKd/kcat 回归是否提升
- [ ] 与已发表模型 (DLKcat/CatPred) 在同数据集上对比

### 训练所需文件（发送给朋友）
放在 AI4enz 目录下：
1. `dataset_building/processed/metadata_with_negatives.parquet` — 含负样本的元数据
2. `dataset_building/processed/proteins.h5` — ESM-2 嵌入
3. `dataset_building/processed/ligands/` — 配体 GNN 图文件
4. `dataset_building/models/checkpoints/best.ckpt` — 预训练权重 (~25 MB)

### 朋友微调命令
```bash
cd AI4enz/dataset_building/models
python train.py \
  --unified-metadata ../processed/metadata_with_negatives.parquet \
  --finetune checkpoints/best.ckpt \
  --epochs 50 --batch-size 256 --lr 5e-5 \
  --gate-weight 0.02 \
  --device cuda --num-workers 8 --amp --compile
```
