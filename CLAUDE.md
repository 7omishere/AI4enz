# CLAUDE.md — AI4enz

AI 辅助酶挖掘项目。基于过渡态理论构建酶-底物结合亲和力预测模型，服务于"给定目标底物 → 推荐酶序列"的酶挖掘场景。

每次回复前都叫我：**"多米"**

## 核心架构：Hybrid TransitionBINN

**Hybrid 设计**：pKd 路径 + kcat 独立路径，直接优化 log₁₀(kcat/KM) 排序。

### 设计原则

1. **过渡态理论**（普适，所有酶催化都满足）
2. **反应坐标 ODE**（Neural ODE 模拟 ξ∈[0,1] 演化）
3. **固定损失权重**（目标值归一化后量级一致）
4. **kcat 独立路径**（蛋白级表征，避免负迁移）

### 反应坐标物理

```
ξ=0: 反应物（酶-底物初始结合）
ξ=0.5: 过渡态（最大能垒）
ξ=1: 产物（催化完成）

酶催化 = 降低能垒，稳定过渡态
```

### 组件

| 组件 | 功能 |
|------|------|
| `LigandEncoder` | GATv2×3 GNN |
| `ProteinEncoder` | ESM-2 + 结构特征 + 口袋 |
| `CofactorEncoder` | 辅因子 embedding |
| `ReactionCoordinateBINN` | ODE 积分，反应坐标演化 |
| `TransitionStateGate` | 门控，模拟能垒跨越 |
| `BINNCatalysisHead` | ts_stability + catalysis_rate（归一化到[0,1]） |

### 损失函数

```python
L_total = L_ts + L_catalysis + 0.01 * L_eyring

- L_ts: 结合亲和力（SmoothL1，归一化目标[0,1]）
- L_catalysis: 催化效率（SmoothL1，归一化目标[0,1]）
- L_eyring: Eyring 方程自洽约束（MSE，不依赖标签）
- 权重: pKd:kcat = 1:1（Min-Max归一化后量级一致）
- 缺失标签: mask 机制自动跳过，单标签样本仍贡献共享编码器梯度
```

### 激活函数

| 位置 | 激活 | 说明 |
|------|------|------|
| ODE 动力学层 + 共享层 | **GELU** | 与 ESM-2 一致 |
| 输出头中间层 | **SiLU** | 比 ReLU 更平滑 |
| 输出头最后一层 | **sigmoid** | 压缩到 [0,1]，匹配归一化 |

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
| BINN 动力学层 (nn×5) | Linear(256,256) | **0.67** | 防 ODE 多步积分爆炸 |
| 其他所有 Linear | 任意 | **1.0** | 标准 Xavier |
| LayerNorm | — | 保持默认 | 最优

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
    │   ├── ranking_model.py               # TransitionBINN 模型定义
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

> IC50_approx 的 Loss 权重在 `train.py` 的 `THERMO_WEIGHT` 字典中定义。Key 必须为 `'IC50_approx'`（与数据一致），不能使用 `'IC50'`，否则会 fallback 到默认值。

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

### 测量类型分布

| 类型 | 数量 | 可信度 | Loss 权重 |
|------|------|--------|-----------|
| Ki | 59,883 | 中-高 | 0.70 |
| Kd | 37,219 | 高 | 1.00 |
| IC50_approx | 249 | 低 | 0.15 |

## 已知问题

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🟡 中 | 无 GPU | CPU 训练较慢，97k 样本 × 100 epoch ≈ 3 天 |
| 🟡 中 | proteins.h5 含历史数据 | 53,133 个蛋白中仅 19,223 用于当前训练集 |
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

## 当前任务进度 (2026-06-13)

### 已完成
- [x] **GPU 训练** — best.ckpt (epoch 81, val_loss=0.0056)
- [x] **Benchmark** — pKd ρ=0.895, kcat ρ=0.671 (Trenzition); 增强基线 (ESM-2+MorganFP→MLP: pKd ρ=0.884)
- [x] **消融实验** — 配体 GNN 贡献最大 (R² 0.703→0.048)，ODE 多步可压缩至 1 步，Gate 已退化为 ~1.0
- [x] **推理脚本** — `predict.py`: SMILES + 蛋白序列 → pKd + kcat + ΔG‡
- [x] **负样本生成** — `scripts/generate_negatives.py`: 跨 EC 负采样，metadata_with_negatives.parquet (194k 样本, 1:1)
- [x] **Gate 正则化** — train.py 支持 `--gate-weight` (L_gate = pos→1, neg→0)
- [x] **微调支持** — train.py 支持 `--finetune <ckpt_path>`

### 待完成
- [ ] **负样本微调**: `--finetune best.ckpt --unified-metadata metadata_with_negatives.parquet --gate-weight 0.02`
- [ ] 微调后重新评估: gate 是否学会区分正负样本？pKd/kcat 回归是否提升？
- [ ] 与已发表模型 (DLKcat/CatPred) 在同数据集上对比

### 朋友训练所需文件
发送以下文件给朋友（放在 AI4enz 目录下解压）：
1. `training_data.tar.gz` → 解压到项目根目录
2. `checkpoints/best.ckpt` (25 MB) → 已有（git push 不含）
3. `processed/metadata_with_negatives.parquet` → 需单独发（gitignored）

朋友微调命令：
```bash
cd AI4enz/dataset_building/models
python train.py \
  --unified-metadata ../processed/metadata_with_negatives.parquet \
  --finetune checkpoints/best.ckpt \
  --epochs 50 --batch-size 256 --lr 5e-5 \
  --gate-weight 0.02 \
  --device cuda --num-workers 8 --amp --compile
```