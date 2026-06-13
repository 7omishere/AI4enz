# AI4enz — 基于过渡态理论的酶-底物结合预测

利用深度学习预测酶-底物结合亲和力 (pKd) 和催化效率 (log₁₀(kcat))，服务于 **"给定目标底物 → 推荐最优酶序列"** 的酶挖掘场景。

---

## 🚀 快速开始 — 训练

```bash
# 1. 激活环境
source /home/domi/BINN/.venv/bin/activate

# 2. 直接训练（默认参数使用 trenzition V5 数据集）
cd /home/domi/AI4enz/dataset_building/models

# 快速验证（10 分钟）
python train.py --epochs 5 --batch-size 32 --max-samples 2000 --device cpu --num-workers 0

# 完整训练（CPU, ~3 天）
python train.py --epochs 100 --batch-size 128 --device cpu --num-workers 2

# GPU 全优化（推荐，~1-2 小时）
python train.py --epochs 100 --batch-size 256 --device cuda --num-workers 8 --amp --compile
```

> 所有训练参数都有默认值，指定 `--unified-metadata` / `--proteins-h5` / `--ligand-dir` 可切换数据集。

### 训练参数说明

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 128 | 每步样本数（GPU 可设 256） |
| `--lr` | 1e-4 | 基础学习率 |
| `--warmup-steps` | 1000 | LR 预热步数（0 可关闭） |
| `--device` | auto | `cpu` 或 `cuda` |
| `--num-workers` | 4 | 数据加载并行数（CPU 用 0-2） |
| `--amp` | off | 混合精度加速（仅 CUDA） |
| `--compile` | off | torch.compile 加速（需 PyTorch ≥ 2.0） |
| `--grad-accum` | 1 | 梯度累积步数（小显存用） |
| `--max-samples` | ∞ | 限制样本数（调试用） |

---

## 📊 训练策略

Trenzition 模型采用一套完整的训练策略，在 `train.py` + `ranking_model.py` 中实现：

### 优化器：AdamW + 分组学习率

不同组件使用不同学习率，基于各自的学习需求：

```
Encoder（蛋白/配体/辅因子）:  lr × 1.0  = 1e-4  ← 正常学习
BINN（ODE 动力学）:          lr × 0.5  = 5e-5  ← 保守，防 ODE 不稳定
Head（3 个输出头）:          lr × 2.0  = 2e-4  ← 快速收敛
可学习 Loss 权重:            lr × 0.1  = 1e-5  ← 慢调（默认关闭）
```

### 学习率调度：Warmup → CosineAnnealingWarmRestarts

```
LR
↑
│         ╱╲      ╱╲      ╱╲
│        ╱  ╲    ╱  ╲    ╱  ╲
│       ╱    ╲  ╱    ╲  ╱    ╲
│      ╱      ╲╱      ╲╱      ╲
│  ━━╱                        ╲
│ ╱
│╱
└───────────────────────────────→ Steps
   ↑ warmup      T₀=5000    T₁=10000
   1000 steps    余弦退火    周期倍增
```

- **Warmup**：前 1000 步 LR 从 ≈0 线性升至目标值，防止初始 loss 爆炸
- **CosineAnnealingWarmRestarts**：T₀=5000 steps，T_mult=2，周期余弦退火，帮助跳出局部最优

### 梯度裁剪：max_norm=1.0

所有梯度裁剪到最大 L2 范数 1.0，防止梯度爆炸（尤其是 BINN ODE 路径）。

### 损失函数

```python
L_total = L_ts + L_catalysis + 0.01 × L_eyring

# L_ts: 结合亲和力（pKd）—— SmoothL1，归一化目标 [0,1]
# L_catalysis: 催化效率（log₁₀(kcat)）—— SmoothL1，归一化目标 [0,1]
# L_eyring: Eyring 方程自洽约束 —— MSE，不依赖标签
```

缺失标签处理：`pkd_target_mask` / `kcat_target_mask` 在 loss 计算中自动 mask 掉缺失值，单标签样本仍贡献共享编码器梯度。

### 参数初始化

针对非 ReLU 激活（GELU、SiLU）和 sigmoid 输出头的特性设计的初始化策略：

| 层级 | 初始化 | 原因 |
|------|--------|------|
| 输出头最后一层 `Linear(64,1)` → sigmoid | Xavier gain=**0.3** | 防 sigmoid 初始饱和 |
| BINN ODE 动力学层 | Xavier gain=**0.67** | 防多步积分放大 |
| 其他所有 Linear | Xavier gain=**1.0** | 标准 Xavier |
| LayerNorm | 保持默认 (1.0, 0.0) | |

---

## 📦 数据集 — trenzition V5

### 全局统计

| 指标 | 值 |
|------|-----|
| 总记录 | **97,351** |
| pKd 有效 | 71,482 (73.4%) |
| kcat 有效 | 92,743 (95.3%) |
| 双标签 (pKd + kcat) | 66,874 (68.7%) |
| EC 号覆盖 | 97,351 (100%) |
| 唯一蛋白 | 19,223 |
| 唯一配体 (InChIKey) | 7,249 |

### 数据来源

| 来源 | 期刊 | 数据 |
|------|------|------|
| CatPred-DB | Nature Comms 2025 | kcat, Ki |
| OED | NAR 2025 | kcat, Km, kcat/Km |
| SKiD | Scientific Data 2025 | kcat, Km + 3D 结构 |
| BindingDB | 月度更新 | Kd, Ki |

### Split 分布（蛋白层级，零泄漏）

| Split | 样本数 | 蛋白数 | pKd 有效 | kcat 有效 |
|-------|--------|--------|----------|-----------|
| train | 68,876 | 13,454 | 50,804 | 68,876 |
| val | 13,801 | 2,885 | 10,176 | 13,801 |
| test | 14,674 | 2,884 | 10,502 | 14,674 |

### 编码状态

| 组件 | 覆盖率 | 编码方式 |
|------|--------|----------|
| 蛋白 ESM-2 | 19,223/19,223 (100%) | esm2_t33_650M, 1280-dim mean-pool |
| 配体 GNN | 7,249/7,249 (100%) | GATv2×3, 79-dim 原子特征 |
| 辅因子 | 57.8% 非空 | 310 种组合，可学习 Embedding |

### 测量类型分布

| 类型 | 数量 | 可信度 | Loss 权重 |
|------|------|--------|-----------|
| Ki | 59,883 | 中-高 | 0.70 |
| Kd | 37,219 | 高 | 1.00 |
| IC50_approx | 249 | 低 | 0.15 |

### Min-Max 归一化

| 目标 | 原始范围 | 归一化到 | 依据 |
|------|----------|----------|------|
| pKd | [0, 12] | [0, 1] | P99.9=10.32，留余量 |
| log₁₀(kcat) | [-7, 8] | [0, 1] | P0.01=-6.98, P100=7.76 |

---

## 🏗 模型架构

```
输入: 蛋白序列 + 配体分子 + 辅因子
         │
    ┌────┴────┐
    │ 编码器层 │
    │ ─────── │
    │ Ligand  │  GATv2×3 GNN (79-dim → 256)
    │ Protein │  ESM-2 直通 + LayerNorm (1280 → 256)
    │ Cofactor│  Embedding lookup (310 types → 64)
    └────┬────┘
         │
    ┌────┴────┐
    │  BINN   │  LatentPathwayBINN
    │ ─────── │  Neural ODE, 5 步积分
    │ ξ→ξ+Δξ │  门控机制模拟"跨越能垒"
    │   ↻ ×5  │  GELU 激活
    └────┬────┘
         │
    ┌────┴──────────────┐
    │   TrenzitionCatalysisHead    │
    │ ────────────────── │
    │ ts_stability   [0,1]│  sigmoid → pKd
    │ catalysis_rate [0,1]│  sigmoid → log₁₀(kcat)
    │ dG_eyring  [20,200]│  sigmoid-scaling → ΔG‡ (kJ/mol)
    └────────────────────┘
```

### 设计亮点

- **过渡态理论**（普适所有酶催化）
- **反应坐标 ODE**（Neural ODE 模拟 ξ∈[0,1] 演化）
- **Hybrid 双路径**（pKd 路径 + kcat 独立路径）
- **Eyring 自洽约束**（两个输出头的物理一致性）

---

## 📁 项目结构

```
AI4enz/
├── dataset_building/
│   ├── models/               ← 模型定义 + 训练入口
│   │   ├── train.py           ← 训练脚本
│   │   ├── ranking_model.py   ← Trenzition 模型定义
│   │   ├── classification_head.py
│   │   ├── checkpoints/       ← 训练输出
│   │   └── __init__.py
│   ├── processed/
│   │   ├── metadata.parquet   ← 统一元数据（97,351 条）
│   │   ├── proteins.h5        ← ESM-2 嵌入（53,133 蛋白）
│   │   └── ligands/           ← 配体 GNN 图（7,249 个 .pt）
│   ├── pipeline/              ← 数据处理流水线
│   ├── scripts/               ← 爬虫与特征工程
│   ├── BindingDB/  BRENDA/  OED/  SKiD/  ...  ← 原始数据
│   └── release/               ← 旧版本发布
├── README.md
└── CLAUDE.md
```

---

## ⚙️ 环境配置

```bash
# 推荐：使用项目已有环境
source /home/domi/BINN/.venv/bin/activate

# 或者从头创建
python -m venv .venv
source .venv/bin/activate

# PyTorch (CPU)
pip install torch torchvision torchaudio

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

# 其他依赖
pip install rdkit h5py pandas numpy tqdm transformers fair-esm
```

---

## ⏱ 性能估算

| 硬件 | batch_size | ~时间/epoch | 100 epochs | 命令 |
|------|-----------|-------------|------------|------|
| CPU (i7, WSL2) | 128 | ~45 min | ~3 天 | `--device cpu --num-workers 2` |
| RTX 3060 (12GB) | 64+ga2 | ~5 min | ~8 h | `--amp --grad-accum 2` |
| RTX 3090 (24GB) | 128 | ~2 min | ~3.5 h | `--amp` |
| RTX 4090 (24GB) | 256 | ~1 min | ~1.7 h | `--amp --compile` |
| A100 (40GB) | 512 | ~30 s | ~50 min | `--amp --compile` |

---

## 📊 损失曲线解读

训练过程中日志格式：

```
Epoch  10/100 | train: 0.5234  val: 0.4981  (best: 0.4876) | L_ts: 0.0312  L_cat: 0.0254  L_barrier: 0.0000
```

- **train/val loss** 稳步下降 → 正常
- **val loss >> train loss** → 过拟合（减小 epoch 或增加正则化）
- **train loss 不降** → LR 太小或数据问题
- **L_barrier 始终=0** → 正常（此项已移除，保留为 0）

---

## 🔄 断点续训

```bash
python train.py --resume checkpoints/last.ckpt
```

自动恢复：模型权重、优化器状态、scheduler 状态、AMP scaler、global_step。

---

## 🧪 负样本训练 (2026-06-13)

Trenzition 支持跨 EC 大类负采样，让模型学习区分「能反应的酶-底物对」和「不能反应的酶-底物对」。

### 生成负样本

```bash
cd dataset_building/scripts
python generate_negatives.py --ratio 1.0 --strategy cross_ec --output metadata_with_negatives.parquet
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ratio` | 1.0 | 负:正比例 |
| `--strategy` | cross_ec | `cross_ec` / `random` / `hard` / `all` |

### 微调（从预训练权重开始）

```bash
cd dataset_building/models

python train.py \
  --unified-metadata ../processed/metadata_with_negatives.parquet \
  --finetune checkpoints/best.ckpt \
  --epochs 50 --batch-size 256 --lr 5e-5 \
  --gate-weight 0.02 \
  --device cuda --num-workers 8 --amp --compile
```

| 新参数 | 默认值 | 说明 |
|--------|--------|------|
| `--finetune` | None | 从已有 .ckpt 加载权重微调 |
| `--gate-weight` | 0.02 | Gate 正则化强度（0=关闭） |

### Gate 正则化

```
L_gate = 0.02 × [mean((1-gate_pos)²) + mean(gate_neg²)]

正样本: gate → 1  (完全信息流通)
负样本: gate → 0  (阻挡信息)
```

---

## 📝 最新更新 (2026-06-13)

- ✅ GPU 训练完成：best.ckpt (epoch 81, val loss 0.0056)
- ✅ Benchmark: pKd ρ=0.895, kcat ρ=0.671
- ✅ 消融实验：配体 GNN 贡献最大，ODE 多步可压缩至 1 步
- ✅ 负样本生成脚本 + Gate 正则化
- ✅ 推理脚本 `predict.py`：SMILES + 蛋白序列 → pKd + kcat
- ✅ 增强版 Benchmark：ESM-2+MorganFP baselines + 排序指标
