# CLAUDE.md — AI4enz

AI 辅助酶挖掘项目。基于过渡态理论构建酶-底物结合亲和力预测模型，服务于"给定目标底物 → 推荐酶序列"的酶挖掘场景。

## 核心架构：Hybrid TransitionBINN

**Hybrid 设计**：pKd 双路径 + kcat 独立路径，直接优化 log₁₀(kcat/KM) 排序。

### 设计原则

1. **过渡态理论**（普适，所有酶催化都满足）
2. **反应坐标 ODE**（Neural ODE 模拟 ξ∈[0,1] 演化）
3. **不确定性权重**（自动学习损失权重）
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
| `BINNCatalysisHead` | ts_stability + catalysis_rate |
| `KcatPredictor` | 独立的 kcat 预测 MLP（Protein级） |

### 损失函数

```python
L_total = L_score(both_mask) + L_ts + L_cat + L_barrier + L_progress

- L_score: SmoothL1(pKd_pred + log_kcat_pred - pKd_true - log_kcat_true)
  直接优化 log₁₀(kcat/KM) 排序
- L_ts: 过渡态稳定性（与 pKd 正相关）
- L_cat: 催化效率（log10(kcat)）
- L_barrier: 能垒正则化
- w_ts, w_cat: 不确定性权重自动学习
```

## 项目结构

```
AI4enz/
├── CLAUDE.md                       # 本文件
├── README.md                       # 项目简介
├── .gitignore                      # Git 忽略规则
├── datepre/                        # 数据处理与模型定义
│   ├── ranking_model.py            # Hybrid TransitionBINN 模型
│   ├── build_unified_dataset.py    # 数据集构建 + 分层 split
│   └── supplement_cold_cofactors.py # ChEMBL 辅因子补充
└── dataset_building/
    ├── train.py                    # 训练脚本
    ├── inference_enzyme_mining.py  # 推理接口
    ├── extract_pocket_features.py  # 口袋特征提取
    ├── processed/                  # 已处理输出
    │   ├── proteins.h5             # 蛋白序列 + ESM-2 嵌入
    │   ├── ligands/                # 配体分子图 (.pt)
    │   └── oxidoreductase/         # 氧化还原酶子集
    │       ├── unified_metadata.parquet  # 统一数据集 (78,113)
    │       └── high_quality_kd_ki.parquet  # 高质量 Kd/Ki 子集 (6,184)
    └── checkpoints/                # 模型检查点
```

## 虚拟环境

```bash
source /home/domi/BINN/.venv/bin/activate
```

## 训练命令

```bash
cd /home/domi/BINN/AI4enz/dataset_building

# 快速验证（CPU, 小样本）
python train.py --epochs 10 --batch-size 32 --max-samples 5000 --device cpu

# 完整训练（GPU）
python train.py --epochs 100 --batch-size 128 --device cuda
```

## 数据集

### 全局统计

| 指标 | 值 |
|------|-----|
| 总记录 | 78,113 |
| 唯一蛋白 | 541 |
| 唯一配体 | 57,203 |

### Split（kcat 分层平衡）

| 切分 | kcat 覆盖率 |
|------|-------------|
| train | 78.7% |
| val | 78.7% |
| test | 78.7% |

### 测量类型分布

| 类型 | 数量 | 占比 | 可信度 |
|------|------|------|--------|
| IC50 | 71,929 | 92.1% | 低 (R²=0.437 校正) |
| Ki | 5,542 | 7.1% | 中 |
| Kd | 643 | 0.8% | 高 |

### 高质量子集（推荐）

- **路径**: `processed/oxidoreductase/high_quality_kd_ki.parquet`
- **样本数**: 6,184 (仅 Kd/Ki)
- **切分**: train 4,251 / val 968 / test 965

## 关键设计决策

1. **过渡态理论**：替代 Marcus 方程（后者在 100% 蛋白上失败）
2. **Hybrid 架构**：kcat 独立路径，避免底物级/蛋白级表征冲突
3. **直接优化 kcat/KM**：`score = pKd + log_kcat = log₁₀(kcat/KM)`
4. **热力学分层权重**：Kd=1.0, Ki=0.7, IC50=0.15
5. **kcat 分层 split**：所有 split 平衡 kcat 覆盖率（78.7%）

## 已知问题

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🟡 中 | IC50 过高 (92%) | 建议用 `high_quality_kd_ki.parquet` |
| 🟢 低 | 辅因子覆盖 | `supplement_cold_cofactors.py` 可补充稀有辅因子 |

## GitHub

- **仓库**: https://github.com/Domi-Joe/AI4enz
- 代码单独分发，数据集通过云盘传输