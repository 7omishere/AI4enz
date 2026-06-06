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

## 项目结构

```
AI4enz/
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
| Ki | ~150,000 | 中-高 | 0.7 |
| Kd | ~14,570 | 高 | 1.0 |
| kinetics | ~36,000 | 中 | 0.5 |
| kcat_only | ~69,000 | 中 | — |

## 训练命令

```bash
cd /home/domi/AI4enz/dataset_building/models

# 快速验证（CPU, 小样本）
python train.py --unified-metadata ../processed/oxidoreductase/unified_metadata.parquet \
  --proteins-h5 ../processed/proteins.h5 \
  --ligand-dir ../processed/ligands \
  --epochs 10 --batch-size 32 --max-samples 5000 --device cpu

# 完整训练（GPU）
python train.py --unified-metadata ../processed/oxidoreductase/unified_metadata.parquet \
  --proteins-h5 ../processed/proteins.h5 \
  --ligand-dir ../processed/ligands \
  --epochs 100 --batch-size 128 --device cuda
```

## 数据集

### 全局统计

| 指标 | 值 |
|------|-----|
| 总记录 | **233,134** |
| pKd有效 | 161,980 (69.5%) |
| kcat有效 | 74,697 (32.0%) |
| 双标签(pKd+kcat) | 5,495 (2.4%) |
| **EC号** | **140,897 (60.4%)** |
| 唯一蛋白 | 10,318 |
| 唯一配体 | 89,283 |

> [!NOTE]
> **数据增强 (2026-06-06)**:
> - EC号：36.9% → 60.4% (+23.5%)，通过UniProt REST API补充
> - pKd异常值清洗：消除1,947条(<0或>14)
> - 增强数据集：`release/recommended_training_set_enriched.parquet`

### Split

| 切分 | 样本数 | 总占比 | 有EC号 | 有pKd | 有kcat |
|------|--------|--------|--------|-------|--------|
| train | 187,511 | 80.4% | 110,354 | 130,261 | 59,485 |
| val | 20,693 | 8.9% | 13,712 | 14,303 | 6,634 |
| test | 24,930 | 10.7% | 16,831 | 17,416 | 8,578 |

Split 按 **UniProt ID 层级**分配，test/val与train蛋白重叠仅~4%，避免蛋白序列泄漏。

### 测量类型分布

| 类型 | 数量 | 可信度 | 权重 |
|------|------|--------|------|
| Ki | 149,357 | 中-高 | 0.7 |
| Kd | 14,570 | 高 | 1.0 |
| kinetics | 35,703 | 中 | 0.5 |
| kcat_only | 33,504 | 中 | 0.5 |

## 已知问题

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🟡 中 | kcat覆盖不足 | 仅32%，建议从SABIO-RK手动导出补充 |
| 🟡 中 | 双标签样本少 | 5,495个(2.4%)，限制log(kcat/KM)预测 |
| 🟡 中 | EC号补充瓶颈 | 剩余39.6%蛋白确实无UniProt催化注释 |
| 🟢 低 | 辅因子覆盖 | 可后续补充稀有辅因子 |

## 数据来源

| 来源 | 提供数据 | 规模 |
|------|----------|------|
| CatPred-DB (Nature Comms 2025) | kcat, Ki | 20k+11k |
| OED (NAR 2025) | kcat, Km, kcat/Km | 36k |
| SKiD (Scientific Data 2025) | kcat, Km + 3D结构 | 13k+18k |
| BindingDB | Kd, Ki | 14k+135k |

## GitHub

- **仓库**: https://github.com/7omishere/AI4enz
- 代码通过Git分发，数据集通过云盘传输