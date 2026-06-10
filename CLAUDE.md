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

### 激活函数

- **GeLU**：用于ODE动力学层和共享层（与ESM-2一致）

### 损失函数

```python
L_total = L_ts + L_catalysis + 0.1*L_barrier + 0.01*L_progress

- L_ts: 过渡态稳定性（SmoothL1，归一化目标[0,1]）
- L_catalysis: 催化效率（SmoothL1，归一化目标[0,1]）
- L_barrier: 能垒正则化（MSE，建议先设为0测试）
- L_progress: 反应进度正则化
- 权重: pKd:kcat = 1:1（Min-Max归一化后量级一致）
```

## Min-Max 归一化

| 目标 | 原始范围 | 归一化参数 | 输出范围 |
|------|----------|------------|----------|
| pKd | [0, 14] | pk d_min=0.0, pkd_max=14.0 | [0, 1] |
| log₁₀(kcat) | [-6, 7] | kcat_min=-6.0, kcat_max=7.0 | [0, 1] |

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
│   └── processed/ → BINN/      # 中间产物（符号链接）
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

| 类型 | 数量 | 可信度 | 权重 |
|------|------|--------|------|
| Ki | 60,758 | 中-高 | 0.7 |
| Kd | 37,499 | 高 | 1.0 |
| IC50_approx | 249 | 低-中 | 0.4 |

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
```

## 数据集 — trenzition V5

### 全局统计

| 指标 | 值 |
|------|-----|
| 总记录 | **98,506** |
| pKd有效 | 72,361 (73.5%) |
| kcat有效 | 93,652 (95.1%) |
| 双标签(pKd+kcat) | 67,507 (68.5%) |
| **EC号** | **98,506 (100%)** |
| 唯一蛋白 | 19,278 |
| 唯一配体 | 7,273 |

### 编码状态

| 组件 | 覆盖率 | 编码方式 |
|------|--------|----------|
| 蛋白 ESM-2 | 19,278/19,278 (100%) | esm2_t33_650M, 1280-dim mean-pool |
| 配体 GNN | 7,273/7,278 (99.9%) | GATv2, 79-dim 原子特征 + 10-dim 键特征 |
| 无机离子 | 5 种 (Ag⁺, Co, S, NO, Na⁺) | 不可编码，已剔除 |

### Split（蛋白层级，零泄漏）

| 切分 | 样本数 | 蛋白数 | pKd | kcat |
|------|--------|--------|-----|------|
| train | 69,738 | 13,494 | 51,466 | 66,280 |
| val | 13,956 | 2,892 | 10,288 | 13,233 |
| test | 14,812 | 2,892 | 10,607 | 14,139 |

Split 按 **protein_seq_hash 层级**分配，test/val与train蛋白完全零重叠。

### 测量类型分布

| 类型 | 数量 | 可信度 | 权重 |
|------|------|--------|------|
| Ki | 60,758 | 中-高 | 0.7 |
| Kd | 37,499 | 高 | 1.0 |
| IC50_approx | 249 | 低 | 0.4 |

## 已知问题

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🟡 中 | 无 GPU | CPU 训练较慢（无 CUDA） |
| 🟡 中 | proteins.h5 含历史数据 | 53,133 个蛋白中仅 19,278 用于当前训练集 |
| 🟢 低 | 辅因子覆盖 | 可后续补充稀有辅因子 |
| 🟢 低 | 结构/口袋特征未启用 | metadata 中 has_structure/has_binding_site 均为默认值 |

## 数据来源

| 来源 | 提供数据 | 规模 |
|------|----------|------|
| CatPred-DB (Nature Comms 2025) | kcat, Ki | 31k |
| OED (NAR 2025) | kcat, Km, kcat/Km | 36k |
| SKiD (Scientific Data 2025) | kcat, Km + 3D结构 | 13k |
| BindingDB | Kd, Ki | 153k |
| **PDBbind v2020R1 (PDBbind+)** | **Kd, Ki, 3D结构** | **19k** |

## 当前任务进度 (2026-06-11)

### 已完成
- [x] trenzition V5 数据集 SMILES 100% 补齐
- [x] **metadata.parquet** — 98,506 条，train/val/test 70/15/15 蛋白级划分（零泄漏）
  - train: 69,738 / val: 13,956 / test: 14,812
- [x] **ESM-2 蛋白编码** — 19,278/19,278 (100%)，esm2_t33_650M，1280-dim mean-pooled
  - 总耗时 366 min (0.6 seq/s, CPU)，`proteins.h5` 18 GB
- [x] **配体 GNN 编码** — 7,273/7,278 (99.9%)，GATv2 图编码，79-dim 原子特征 + 10-dim 键特征
  - 5 个单原子/无机离子 (Ag⁺, Co, S, NO, Na⁺) 无法建图，已从 metadata 剔除（11 条记录）
- [x] Pantheon CLI 全线改为 minimax-m27 + key

### 待完成
- [ ] 验证管线: metadata + proteins.h5 + ligands forward pass + loss
- [ ] 开始训练: `python models/train.py --epochs 100 --batch-size 128`
  - 训练参数: `--unified-metadata processed/metadata.parquet --proteins-h5 processed/proteins.h5 --ligand-dir processed/ligands`

### 重要提醒
- `/tmp` 容易满（tmpfs 7.7G），后台任务日志写项目目录而非 /tmp
- 训练质量权重全平权 (quality_weight=1.0)
- 无 GPU，一切 CPU 运行