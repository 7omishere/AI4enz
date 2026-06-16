# AI4enz — 基于过渡态理论的酶-底物预测

利用深度学习预测酶-底物结合亲和力 (pKd)、催化效率 (log₁₀(kcat)) 和米氏常数 (log₁₀(Km))，
服务于 **"给定目标底物 → 推荐最优酶序列"** 的酶挖掘场景。

---

## 🧬 模型架构

### Trenzition v3 — 三头 + Eyring 物理硬编码

```
shared encoders (LigandEncoder GATv2×3 + ProteinEncoder ESM-2 + CofactorEncoder)
    │
    ├── BindingDualHead (BINN 1步, 无ODE)
    │     ├── Kd_branch → pKd_Kd (底物结合)
    │     └── Ki_branch → pKd_Ki (抑制剂结合, 权重0.7)
    │     + 测量类型 one-hot [is_Kd, is_Ki] 作为输入特征
    │
    ├── EyringKcatHead (BINN + 多步ODE + Eyring 公式硬编码)
    │     ├── dG_predictor (256→64→1) → ΔG‡ [5, 300] kJ/mol
    │     ├── log_kappa (可学习透射系数, 初始 κ=0.5)
    │     └── Eyring 公式:  ΔG‡ + T(K)  →  log₁₀(kcat)   ← 不可学习变换
    │
    └── KmHead (简单 MLP, 归一化 [0,1])
          └── sigmoid → log₁₀(Km)
```

**核心设计**：kcat 不再通过 MLP 黑箱回归，而是让模型预测 ΔG‡（活化自由能），
用 **Eyring 公式硬编码** 转换为 kcat。模型要改变 kcat 预测只能通过改变 ΔG‡，
大幅降低过拟合风险。

**每样本独立温度**：276–562K 温度范围直接传入 Eyring 公式。

### 损失函数

```
L_total = L_binding + L_kcat + L_km + 0.1·L_joint

L_binding = SmoothL1(pKd_Kd_pred, pKd_true) + 0.7·SmoothL1(pKd_Ki_pred, pKd_true)
L_kcat    = SmoothL1(kcat_pred, kcat_true)              # Eyring 已硬编码
L_km      = SmoothL1(Km_pred, Km_true)
L_joint   = SmoothL1(kcat/Km_pred, kcat/Km_true)        # 物理自洽
          + 0.01·ReLU(kcat/Km_pred - 9)                 # 扩散极限 ≤ 10⁹
```

---

## 🚀 快速开始

```bash
source /home/domi/BINN/.venv/bin/activate
```

### 训练

```bash
cd /home/domi/AI4enz/model

# GPU 全优化（默认三头模式）
python train.py --kcat-ode-steps 10 \
  --epochs 100 --batch-size 256 \
  --device cuda --num-workers 8 --amp

# CPU 快速验证
python train.py --kcat-ode-steps 10 \
  --epochs 5 --batch-size 32 --max-samples 5000 \
  --device cpu --num-workers 0
```

### 推理

```bash
cd /home/domi/AI4enz/model

# 单样本预测
python predict.py --smiles "CCO" --sequence "MKTVW..." --temperature 310.15

# 批量预测
python predict.py --csv candidates.csv --output results.csv
```

---

## 📦 数据集 — Trenzition V5

### 统计

| 指标 | 值 |
|------|-----|
| 总记录 | **104,021** |
| pKd 有效 | 77,867 (74.9%) |
| kcat 有效 | 90,008 (86.5%) |
| Km 有效 | 94,279 (90.6%) |
| 三标签 (pKd+kcat+Km) | 63,672 (61.2%) |
| 唯一蛋白 | 18,929 |
| 唯一配体 | 14,888 |
| 温度范围 | 276–562 K |

### 编码

| 组件 | 覆盖率 | 编码方式 |
|------|--------|----------|
| 蛋白 | 100% | ESM-2 1280-dim mean-pool |
| 配体 | 100% | GATv2 原子特征 (79+10-dim) |
| 辅因子 | 56.7% | 310 种 Embedding (mask 跳过) |
| 温度 | 100% | 每样本独立 K |

### Split（蛋白层级，零泄漏）

| Split | 样本 | 蛋白 | pKd有效 | kcat有效 | Km有效 |
|-------|------|------|---------|---------|-------|
| train | 73,261 | 13,251 | 55,035 | 63,262 | 66,420 |
| val | 15,552 | 2,839 | 11,695 | 13,306 | 14,064 |
| test | 15,208 | 2,839 | 11,137 | 13,440 | 13,795 |

---

## 📁 项目结构

```
AI4enz/
├── model/                           # ← 模型核心
│   ├── ranking_model.py             # Trenzition v3 定义
│   ├── train.py                     # 训练脚本
│   ├── predict.py                   # 推理脚本
│   ├── benchmark_enhanced.py        # 基准测试
│   ├── checkpoints/
│   └── tst/                         # TST 模型（实验性）
├── dataset_building/
│   ├── processed/                   # 训练用核心数据
│   │   ├── metadata.parquet         # 104,021 条元数据
│   │   ├── proteins.h5              # ESM-2 嵌入 (18,929)
│   │   └── ligands/                 # 配体 GNN 图 (14,888)
│   ├── golddata/                    # 高质量金标子集
│   ├── models_v1_obsolete/          # 旧版模型归档
│   ├── pipeline/                    # 数据处理流水线
│   ├── scripts/                     # 数据构建补充脚本
│   └── BindingDB/ BRENDA/ OED/ ... # 原始数据
├── README.md
└── CLAUDE.md
```

---

## 🔬 数据来源

| 来源 | 提供数据 | 规模 |
|------|----------|------|
| [CatPred-DB](https://www.nature.com/articles/s41467-025-57509-0) (Nature Comms 2025) | kcat, Ki | ~31k |
| [OED](https://academic.oup.com/nar/article/53/D1/D591/7775886) (NAR 2025) | kcat, Km, kcat/Km | ~36k |
| [SKiD](https://www.nature.com/articles/s41597-025-04734-5) (Scientific Data 2025) | kcat, Km + 3D结构 | ~13k |
| BindingDB | Kd, Ki | ~153k |

---

## 📝 更新历史

### 2026-06-16
- 🔬 **Trenzition v3**: Eyring 公式硬编码替代 kcat 黑箱回归
- 🌡️ **每样本温度**: temperature_K 传入 EyringKcatHead
- 📊 **Km 归一化**: 统一 [0,1] 与 Kd/kcat 一致
- 🧹 **代码清理**: 移除 Gate/L_eyring/未使用字段
- ✅ **配体图补齐**: 177 个缺失配体 → 100% 覆盖率

### 2026-06-15
- 🌡️ **温度补充**: OED + BRENDA 三层匹配覆盖 100%
- 📂 **模型目录重构**: 旧模型归档至 models_v1_obsolete/

### 2026-06-14
- ✅ GPU 训练完成：pKd ρ=0.895, kcat ρ=0.671
