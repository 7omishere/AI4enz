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
└── dataset_building/
    ├── ranking_model.py                   # TransitionBINN 模型定义
    ├── train.py                           # 训练脚本
    ├── inference_enzyme_mining.py          # 酶挖掘推理
    ├── evaluate_model.py                   # 模型评估
    ├── release/                           # 训练数据集
    │   ├── recommended_training_set.parquet
    │   ├── recommended_training_set_compact.parquet
    │   ├── pkd_subset.parquet
    │   └── kcat_subset.parquet
    ├── checkpoints/                       # 模型检查点
    └── external_data/                     # 原始数据源
```

## 虚拟环境

```bash
source /home/domi/BINN/.venv/bin/activate
```

## 训练命令

```bash
cd /home/domi/AI4enz/dataset_building

# 快速验证（CPU, 小样本）
python train.py --epochs 10 --batch-size 32 --max-samples 5000 --device cpu

# 完整训练（GPU）
python train.py --epochs 100 --batch-size 128 --device cuda
```

## 数据集

### 全局统计

| 指标 | 值 |
|------|-----|
| 总记录 | **233,134** |
| pKd有效 | 163,927 (70.3%) |
| kcat有效 | 74,514 (32.0%) |
| 唯一蛋白 | 10,588 |
| 唯一配体 | 89,283 |

### Split

| 切分 | 样本数 | pKd | kcat | 双标签 |
|------|--------|-----|------|--------|
| train | 187,511 | 131,835 | 59,302 | 3,626 |
| val | 20,693 | 14,464 | 6,634 | 405 |
| test | 24,930 | 17,628 | 8,578 | 1,276 |

Split 按 UniProt ID 层级分配，避免蛋白序列泄漏。

### 测量类型分布

| 类型 | 数量 | 可信度 | 权重 |
|------|------|--------|------|
| Ki | 149,357 | 中-高 | 0.7 |
| kinetics | 35,703 | 中 | 0.5 |
| kcat_only | 33,504 | — | — |
| Kd | 14,570 | 高 | 1.0 |

## 训练策略建议

### 阶段1：预训练 pKd
```bash
python train.py --dataset release/pkd_subset.parquet \
  --epochs 50 --batch-size 64 --device cuda
```
目标：让编码器学习蛋白-配体结合模式

### 阶段2：联合训练
```bash
python train.py --dataset release/recommended_training_set.parquet \
  --epochs 100 --batch-size 128 --device cuda
```
目标：同时学习 kcat 预测

## 关键设计决策

1. **过渡态理论**：替代 Marcus 方程（后者在 100% 蛋白上失败）
2. **Hybrid 架构**：kcat 独立路径，避免底物级/蛋白级表征冲突
3. **Min-Max 归一化**：目标值统一到[0,1]，损失量级自然一致
4. **GeLU激活**：与ESM-2一致，梯度更流畅
5. **固定损失权重**：1:1配平，简化调参

## 已知问题

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🟡 中 | L_barrier权重 | 建议先设为0测试，看模型自然学习效果 |
| 🟡 中 | 双标签样本少 | 仅5,307个(2.3%)，但对score预测重要 |
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