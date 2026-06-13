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
| `--finetune` | None | 从已有 .ckpt 加载权重微调 |
| `--gate-weight` | 0.02 | Gate 正则化强度（0=关闭，负样本训练用） |

---

## 📊 训练策略

### 优化器：AdamW + 分组学习率

| 参数组 | LR 倍率 | 说明 |
|--------|---------|------|
| Encoder（蛋白/配体/辅因子） | ×1.0 | 正常学习 |
| BINN（动力学 + Gate） | ×0.5 | 保守，防不稳定 |
| Head（输出头） | ×2.0 | 快速收敛 |

### 学习率调度：Warmup → CosineAnnealingWarmRestarts

前 1000 步 LR 从 ≈0 线性升至目标值，之后 T₀=5000 余弦退火（T_mult=2 周期倍增）。

### 梯度裁剪：max_norm=1.0

所有梯度裁剪到最大 L2 范数 1.0，防止 BINN 路径梯度爆炸。

### 参数初始化

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
    │ Protein │  ESM-2 直通 (1280 → 256)
    │ Cofactor│  Embedding (310 types → 64 → 256)
    └────┬────┘
         │
    ┌────┴──────────────┐
    │  LatentPathwayBINN │
    │ ───────────────── │
    │ h₀ = MLP(pro, lig) │  酶-底物复合物初始状态
    │                     │
    │ dh = dynamics(h)   │  特征变换
    │ dh *= gate         │  ← Gate 控制信息流 [0,1]
    │ h = h₀ + (dh+dh_c)/2 │  中点Euler积分 (1步)
    └────┬──────────────┘
         │
    ┌────┴──────────────────┐
    │ TrenzitionCatalysisHead│
    │ ────────────────────── │
    │ ts_stability   [0,1]   │  sigmoid → pKd
    │ catalysis_rate [0,1]   │  sigmoid → log₁₀(kcat)
    │ dG_eyring  [20,200]    │  ΔG‡ (kJ/mol)
    └────────────────────────┘
```

### Gate 门控机制

Gate ∈ [0,1] 是一个可学习的**酶-底物兼容性分类器**，回答最基本的问题："这个酶和底物之间是否存在有意义的相互作用？"

**推理流程**：
```
输入 (酶序列, 底物SMILES)
    │
    ▼
Gate 判断: "这个配对合理吗？"
    │
    ├─ gate ≈ 0 → "不合理，动力学阻断" → 预测值不可信
    │
    └─ gate ≈ 1 → "合理，动力学全开" → pKd + kcat → 排序推荐
```

**实用价值**：两阶段酶挖掘筛选
1. **Stage 1 (Gate 过滤)**：从候选酶池中筛掉明显不反应的配对
2. **Stage 2 (催化效率)**：对通过筛选的配对预测 pKd/kcat 并排序

**注意**：Gate 不是物理能垒（ΔG‡ 由 `dG_eyring` 预测），而是数据驱动的特异性判断。其可靠性取决于负样本质量——跨 EC 负采样学到的边界比随机负采样更有意义。

---

## 📐 损失函数

```python
L_total = L_ts + L_catalysis + 0.01 × L_eyring + gate_weight × L_gate

# ── 正样本（真实酶-底物对）──
L_ts:        SmoothL1(ts_stability, pkd_target)        # 仅 pKd 有效时
L_catalysis: SmoothL1(catalysis_rate, log_kcat_target) # 仅 kcat 有效时
L_eyring:    MSE(dG_pred, dG_from_kcat)                # 仅 kcat 有效时
L_gate:      (1 - gate)²                               # 推 gate → 1

# ── 负样本（跨 EC 随机配对）──
L_gate:      gate²   ← 负样本唯一梯度来源（其他 loss 自动 mask 掉）
```

| Loss 项 | 正样本 | 负样本 | 物理意义 |
|---------|--------|--------|----------|
| L_ts | ✅ | ❌ (mask) | 结合亲和力回归 |
| L_catalysis | ✅ | ❌ (mask) | 催化效率回归 |
| L_eyring | ✅ (仅kcat) | ❌ | 热力学自洽约束 |
| L_gate | → 1 | → 0 | 酶-底物兼容性 |

缺失标签通过 `pkd_target_mask` / `kcat_target_mask` 自动跳过，单标签样本仍贡献共享编码器梯度。

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

## 🔄 断点续训

```bash
python train.py --resume checkpoints/last.ckpt
```

自动恢复：模型权重、优化器状态、scheduler 状态、AMP scaler、global_step。

---

## 🧪 负样本训练

Trenzition 支持跨 EC 大类负采样，让 Gate 学会区分「能反应的酶-底物对」和「不能反应的酶-底物对」。

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

### Gate 正则化

```
L_gate = gate_weight × [mean((1-gate_pos)²) + mean(gate_neg²)]

正样本: gate → 1  (完全信息流通，动力学全开)
负样本: gate → 0  (完全阻断动力学，预测值不可信)
```

**设计要点（2026-06-14 修复）**：
- Gate 值不 detach → L_gate 梯度完整回传 Gate 网络
- 中点校正项 `dh_correction` 也被 Gate 控制 → gate=0 时完全阻断
- Eyring loss 仅对 kcat 标签样本计算 → 负样本不参与

---

## 📊 损失曲线解读

训练过程中日志格式：

```
Epoch  10/100 | train: 0.5234  val: 0.4981  (best: 0.4876) | L_ts: 0.0312  L_cat: 0.0254  L_barrier: 0.0000
```

| 观察 | 含义 |
|------|------|
| train/val loss 稳步下降 | 正常 |
| val loss >> train loss | 过拟合 |
| train loss 不降 | LR 太小或数据问题 |
| L_gate > 0 | Gate 正在学习区分正负样本 |
| L_gate → 0 | Gate 已收敛或 gate_weight 太小 |

---

## 👥 协作训练指南

### 发送给朋友的文件清单

```
朋友需要以下文件放在 AI4enz/ 目录下：

1. dataset_building/processed/metadata_with_negatives.parquet  (~50 MB, gitignored)
   — 含负样本的完整元数据 (194k 样本, 1:1 pos:neg)

2. dataset_building/processed/proteins.h5  (~2 GB)
   — ESM-2 预计算嵌入 (1280-dim)

3. dataset_building/processed/ligands/  (~80 MB, 7,249 个 .pt 文件)
   — 配体分子图 (GNN 输入)

4. dataset_building/models/checkpoints/best.ckpt  (~25 MB, gitignored)
   — 预训练权重 (epoch 81, val_loss=0.0056)
```

### 朋友环境搭建

```bash
# 1. Clone 项目
git clone <repo_url>
cd AI4enz

# 2. 解压数据文件到对应位置
tar xzf training_data.tar.gz

# 3. 放置 best.ckpt
mkdir -p dataset_building/models/checkpoints
cp /path/to/best.ckpt dataset_building/models/checkpoints/

# 4. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 5. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install rdkit h5py pandas numpy tqdm

# 6. 验证数据完整性
python -c "
import pandas as pd
df = pd.read_parquet('dataset_building/processed/metadata_with_negatives.parquet')
print(f'样本数: {len(df):,}')
print(f'正样本: {(~df.is_negative).sum():,}')
print(f'负样本: {df.is_negative.sum():,}')
print(f'Columns: {list(df.columns)}')
"
```

### 朋友微调命令

```bash
cd AI4enz/dataset_building/models

# 标准微调（推荐）
python train.py \
  --unified-metadata ../processed/metadata_with_negatives.parquet \
  --finetune checkpoints/best.ckpt \
  --epochs 50 --batch-size 256 --lr 5e-5 \
  --gate-weight 0.02 \
  --device cuda --num-workers 8 --amp --compile

# 小显存版本（< 16GB）
python train.py \
  --unified-metadata ../processed/metadata_with_negatives.parquet \
  --finetune checkpoints/best.ckpt \
  --epochs 50 --batch-size 64 --lr 5e-5 \
  --gate-weight 0.02 \
  --device cuda --num-workers 4 --amp --grad-accum 4

# CPU 版（慢，仅验证用）
python train.py \
  --unified-metadata ../processed/metadata_with_negatives.parquet \
  --finetune checkpoints/best.ckpt \
  --epochs 5 --batch-size 32 --max-samples 2000 \
  --gate-weight 0.02 \
  --device cpu --num-workers 2
```

### 微调后检查

```python
import torch
from ranking_model import Trenzition

ckpt = torch.load('checkpoints/best.ckpt', map_location='cpu', weights_only=False)
model = Trenzition()
state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state, strict=False)
model.eval()

# 用 evaluate_model.py 做完整评估
# python ../evaluation/evaluate_model.py --checkpoint checkpoints/best.ckpt
```

关注指标：
- **Gate 分布**：正样本 gate 应集中在 0.7-1.0，负样本 gate 应集中在 0-0.3
- **pKd/kcat 回归**：Spearman ρ 不应显著下降
- **L_gate**：应从高值下降并收敛

---

## 📝 最新更新

### 2026-06-14: Gate 修复
- 🐛 修复 4 个 bug（Gate 梯度截断、中点校正泄漏、Eyring 负样本噪声、pkd mask 默认值）
- 🔧 `gate_profile` 不再 detach → L_gate 梯度正确到达 gate_net
- 🔧 `dh_correction *= gate` → gate=0 完全阻断动力学
- 🔧 Eyring loss 仅对 kcat 标签样本计算
- 📝 更新 CLAUDE.md + README 反映真实架构

### 2026-06-13
- ✅ GPU 训练完成：best.ckpt (epoch 81, val loss 0.0056)
- ✅ Benchmark: pKd ρ=0.895, kcat ρ=0.671
- ✅ 消融实验：配体 GNN 贡献最大，ODE 1 步足够
- ✅ 负样本生成 + Gate 正则化 + 微调支持
- ✅ 推理脚本 `predict.py`：SMILES + 蛋白序列 → pKd + kcat + ΔG‡
