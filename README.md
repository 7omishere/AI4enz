# AI4enz — Enzyme Mining and Activity Prediction

基于过渡态理论的酶-底物结合亲和力（pKd）与催化效率（kcat/KM）预测模型。

## Quick Start

```bash
source /home/domi/BINN/.venv/bin/activate
cd /home/domi/AI4enz/dataset_building/models

# 快速验证 (CPU)
python train.py --unified-metadata ../processed/metadata.parquet \
  --proteins-h5 ../processed/proteins.h5 \
  --ligand-dir ../processed/ligands \
  --epochs 10 --batch-size 32 --max-samples 5000 --device cpu

# 完整训练 (CPU)
python train.py --unified-metadata ../processed/metadata.parquet \
  --proteins-h5 ../processed/proteins.h5 \
  --ligand-dir ../processed/ligands \
  --epochs 100 --batch-size 128 --device cpu
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

## Dataset — trenzition V5

| Metric | Value |
|--------|-------|
| 总样本 | **98,506** |
| pKd样本 | 72,361 (73.5%) |
| kcat样本 | 93,652 (95.1%) |
| **双标签样本** | **67,507 (68.5%)** |
| EC号样本 | 98,506 (100%) |
| 唯一蛋白 | 19,278 |
| 唯一配体 | 7,273 |

### Split分布（蛋白层级，零泄漏）

| Split | 样本数 | 蛋白数 | pKd | kcat |
|-------|--------|--------|-----|------|
| train | 69,738 | 13,494 | 51,466 | 66,280 |
| val | 13,956 | 2,892 | 10,288 | 13,233 |
| test | 14,812 | 2,892 | 10,607 | 14,139 |

### 编码状态

| 组件 | 覆盖率 | 方式 |
|------|--------|------|
| 蛋白 | 19,278/19,278 (100%) | ESM-2 (esm2_t33_650M, 1280-dim) |
| 配体 | 7,273/7,278 (99.9%) | GNN (GATv2, 79-dim atom + 10-dim bond) |
| 无机离子 | 5 种不可编码 → 已剔除 | Ag⁺, Co, S, NO, Na⁺ |

### 数据来源
- CatPred-DB (Nature Comms 2025): kcat, Ki
- OED (NAR 2025): kcat, Km, kcat/Km
- SKiD (Scientific Data 2025): kcat, Km + 3D结构
- BindingDB: Kd, Ki

## 测量类型

| 类型 | 数量 | 可信度 | 权重 |
|------|------|--------|------|
| Ki | 60,758 | 中-高 | 0.7 |
| Kd | 37,499 | 高 | 1.0 |
| IC50_approx | 249 | 低-中 | 0.4 |

## Requirements

- PyTorch ≥ 2.0 + PyTorch Geometric
- ESM-2 (esm2_t33_650M_UR50D)
- RDKit
- h5py

## 最新更新 (2026-06-11)

- ✅ ESM-2 蛋白编码完成：19,278 个蛋白，366 min (CPU)
- ✅ 配体 GNN 编码完成：7,273 个配体，5 个无机离子剔除
- ⬜ 待验证训练管线
- ⬜ 待开始 CPU 训练