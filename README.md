# AI4enz — Enzyme Mining and Activity Prediction

基于过渡态理论的酶-底物结合亲和力（pKd）与催化效率（kcat/KM）预测模型。

## Quick Start

```bash
cd dataset_building

# 完整训练（GPU，推荐）
python train.py --unified-metadata release/recommended_training_set.parquet \
  --epochs 100 --batch-size 128 --device cuda

# 快速验证 (CPU)
python train.py --unified-metadata release/recommended_training_set.parquet \
  --epochs 10 --batch-size 32 --max-samples 5000 --device cpu
```

## Architecture

**TransitionBINN** — Hybrid 双路径设计：
- **pKd 路径**：Ligand GNN + Protein (ESM-2 + 口袋结构) + Cofactor → pKd [0,1]
- **kcat 路径**：Protein ESM-2 + Cofactor → log₁₀(kcat) [0,1]
- **Score**：pKd + log_kcat = log₁₀(kcat/KM)

### 技术改进
- **激活函数**：GeLU（与ESM-2一致）
- **目标归一化**：Min-Max到[0,1]（pkd:[0,14], kcat:[-6,7]）
- **损失函数**：固定权重1:1（归一化后量级一致）

## Dataset

| Metric | Value |
|--------|-------|
| 总样本 | **233,134** |
| pKd样本 | 163,927 (70%) |
| kcat样本 | 74,514 (32%) |
| 唯一蛋白 | 10,588 |
| 唯一配体 | 89,283 |

**数据集文件**（在 `release/` 目录）：
- `recommended_training_set.parquet` — 主训练集（完整50列）
- `recommended_training_set_compact.parquet` — 精简版（17列）
- `pkd_subset.parquet` — 仅K d/Ki (163K)
- `kcat_subset.parquet` — 仅kcat (75K)

数据来源：CatPred-DB, OED, SKiD, BindingDB

## Project Structure

```
AI4enz/
├── README.md                    # 本文件
├── CLAUDE.md                    # 详细项目文档
└── dataset_building/
    ├── ranking_model.py         # TransitionBINN 模型
    ├── train.py                 # 训练脚本
    ├── inference_enzyme_mining.py
    ├── release/                 # 训练数据集（需单独下载）
    ├── checkpoints/             # 模型检查点
    └── external_data/           # 原始数据源
```

## 数据集使用建议

| 阶段 | 数据集 | 权重 |
|------|--------|------|
| 预训练pKd | `pkd_subset.parquet` | L_ts=1, L_cat=0 |
| 联合训练 | `recommended_training_set.parquet` | L_ts=1, L_cat=1 |

## Requirements

- PyTorch ≥ 2.0 + PyTorch Geometric
- ESM-2 (transformers)
- RDKit