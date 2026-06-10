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

## GPU 训练

### 环境配置

```bash
# 1. CUDA Toolkit (建议 ≥ 12.1)
nvidia-smi                          # 确认 GPU 驱动可用

# 2. 创建虚拟环境 (Python ≥ 3.10)
python -m venv .venv
source .venv/bin/activate

# 3. PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. PyTorch Geometric (匹配 PyTorch 版本)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

# 5. 其他依赖
pip install rdkit h5py pandas numpy tqdm transformers fair-esm
```

### 训练命令

```bash
cd dataset_building/models

# === 基础 GPU 训练 ===
python train.py --epochs 100 --batch-size 128 --device cuda

# === GPU 全优化（推荐） ===
python train.py \
  --epochs 100 --batch-size 256 \
  --device cuda --num-workers 8 \
  --amp --compile --grad-accum 1

# === 显存不足时 (VRAM < 16GB) ===
python train.py \
  --epochs 100 --batch-size 64 \
  --device cuda --num-workers 4 \
  --amp --grad-accum 2
```

### GPU 参数说明

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `--device cuda` | 使用 GPU 训练 | — |
| `--amp` | 自动混合精度，~2× 加速，显存减半 | 始终开启 |
| `--compile` | torch.compile，~30% 加速 (PyTorch ≥ 2.0) | 首次编译慢，后续生效 |
| `--grad-accum N` | 梯度累积，等效 batch = batch_size × N | VRAM 不足时用 |
| `--num-workers N` | DataLoader 并行加载 | GPU: 4-8, CPU: 2-4 |
| `--batch-size N` | 每步样本数 | 大显存: 256, 小显存: 64 |

### 性能估算

| 硬件 | batch_size | ~时间/epoch | 100 epochs |
|------|-----------|-------------|------------|
| CPU (i7) | 128 | ~40 min | ~67 h |
| RTX 3090 (24GB) | 128 | ~2 min | ~3.5 h |
| RTX 4090 (24GB) | 256 | ~1 min | ~1.7 h |
| A100 (40GB) | 512 | ~30 s | ~50 min |
| RTX 3060 (12GB) | 64 + grad_accum 2 | ~5 min | ~8 h |

## Requirements

- PyTorch ≥ 2.0 + PyTorch Geometric
- ESM-2 (esm2_t33_650M_UR50D)
- RDKit
- h5py
- **GPU 训练**: CUDA Toolkit ≥ 12.1, NVIDIA 驱动 ≥ 525

## 最新更新 (2026-06-11)

- ✅ ESM-2 蛋白编码完成：19,278 个蛋白，366 min (CPU)
- ✅ 配体 GNN 编码完成：7,273 个配体，5 个无机离子剔除
- ✅ 训练管线验证通过：端到端 forward + backward + loss，2 epochs
- ✅ GPU 训练支持：AMP 混合精度 + torch.compile + 梯度累积