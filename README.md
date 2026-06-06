# AI4enz — Enzyme Mining and Activity Prediction

基于过渡态理论的酶-底物结合亲和力（pKd）与催化效率（kcat/KM）预测模型。

## Quick Start

```bash
cd /home/domi/AI4enz/dataset_building/models

# 快速验证 (CPU)
python train.py --unified-metadata ../processed/oxidoreductase/unified_metadata.parquet \
  --proteins-h5 ../processed/proteins.h5 \
  --ligand-dir ../processed/ligands \
  --epochs 10 --batch-size 32 --max-samples 5000 --device cpu --no-esm2

# 完整训练（GPU，需要ESM-2）
python train.py --unified-metadata ../processed/oxidoreductase/unified_metadata.parquet \
  --proteins-h5 ../processed/proteins.h5 \
  --ligand-dir ../processed/ligands \
  --epochs 100 --batch-size 128 --device cuda
```

## Architecture

**TransitionBINN** — Hybrid 双路径设计：
- **pKd 路径**：Ligand GNN + Protein (ESM-2 + 口袋结构) + Cofactor → pKd [0,1]
- **kcat 路径**：Protein ESM-2 + Cofactor → log₁₀(kcat) [0,1]
- **Score**：pKd + log_kcat = log₁₀(kcat/KM)

### 核心创新
- **过渡态理论**：替代Marcus方程，普适所有酶催化反应
- **Neural ODE**：模拟反应坐标 ξ∈[0,1] 演化
- **门控机制**：模拟"跨越能垒"过程
- **GeLU激活**：与ESM-2一致，梯度更流畅

### 损失函数
```python
L_total = L_ts + L_catalysis + 0.1*L_barrier + 0.01*L_progress
# 权重1:1配平（Min-Max归一化后量级一致）
```

## Dataset

| Metric | Value |
|--------|-------|
| 总样本 | **233,134** |
| pKd样本 | 161,980 (69.5%) |
| kcat样本 | 74,697 (32.0%) |
| **EC号样本** | **140,897 (60.4%)** |
| 双标签样本 | 5,495 (2.4%) |
| 唯一蛋白 | 10,318 |
| 唯一配体 | 89,283 |

> [!NOTE]
> **数据增强 (2026-06-06)**: EC号覆盖率 36.9% → 60.4%，通过UniProt REST API补充636个蛋白的EC号。增强数据集：`release/recommended_training_set_enriched.parquet`

### Split分布

| Split | 样本数 | 有EC号 | 有pKd | 有kcat |
|-------|--------|--------|-------|--------|
| train | 187,511 | 110,354 | 130,261 | 59,485 |
| val | 20,693 | 13,712 | 14,303 | 6,634 |
| test | 24,930 | 16,831 | 17,416 | 8,578 |

> [!NOTE]
> Split按UniProt ID层级分配，test/val与train蛋白重叠仅~4%，避免蛋白序列泄漏。

### 数据来源
- CatPred-DB (Nature Comms 2025): kcat, Ki
- OED (NAR 2025): kcat, Km, kcat/Km
- SKiD (Scientific Data 2025): kcat, Km + 3D结构
- BindingDB: Kd, Ki

## 测量类型

| 类型 | 可信度 | 权重 |
|------|--------|------|
| Kd | 高 | 1.0 |
| Ki | 中-高 | 0.7 |
| kinetics | 中 | 0.5 |

## Requirements

- PyTorch ≥ 2.0 + PyTorch Geometric
- ESM-2 (transformers, 可选，默认用AA属性)
- RDKit
- h5py