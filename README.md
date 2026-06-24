# AI4enz — 基于过渡态理论的酶-底物预测

利用深度学习预测酶-底物结合亲和力 (pKd)、催化效率 (log₁₀(kcat)) 和米氏常数 (log₁₀(Km))，
服务于 **"给定目标底物 → 推荐最优酶序列"** 的酶挖掘场景。

---

## 🧬 模型架构

### Trenzition v5 — 全酶交叉注意力 + 残基级查询注意力 + Eyring 物理硬编码

```
LigandEncoder (GATv2×3, per-atom) + ProteinEncoder (ESM-2 token proj) + CofactorEncoder + TemperatureEncoder
    │
    ├── catalyst_context = cofactor_h + temp_h  (残差相加)
    │
    ├── CrossAttentionFusion (可选, 全酶↔配体原子, Q=[蛋白;辅因子])
    │     保留残基级表示 (B, L, 512)，不在此池化
    │
    ├── ResidueQueryAttention × 3 — 各头独立查询向量
    │     │        │          │
    │     ▼        ▼          ▼
    ├── BindingDualHead  │  KmHead
    │     BINN(1步)      │  MLP → log₁₀(Km)
    │     Kd/Ki双分支    │
    │     pKd (sigmoid)  │
    │               EyringKcatHead
    │               BINN(多步ODE) + dG_predictor
    │               → Eyring 硬编码 → log₁₀(kcat)
    │               + 可学习透射系数 log_kappa
    │
    └── 全头共享: catalyst_context, 可选异方差 (NLL)
```

**v5 新特性**:
- 🎯 **ResidueQueryAttention × 3**: 各预测头用独立查询向量从全长残基提取所需信息，告别全局池化
- 🔗 **全酶交叉注意力**: Q = [蛋白残基; 辅因子虚拟 token]，辅因子调制每个残基对底物的注意力
- 🌡️ **TemperatureEncoder**: 温度作为可学习特征嵌入全部三个预测头
- 🧹 **路径简化**: 始终使用 ESM-2 token 级 `(L, 1280)` 嵌入，无回退路径

**核心设计**: kcat 不再通过 MLP 黑箱回归，而是让模型预测 ΔG‡（活化自由能），
用 **Eyring 公式硬编码** 转换为 kcat。模型要改变 kcat 预测只能通过改变 ΔG‡，
大幅降低过拟合风险。

### 损失函数

```
L_total = L_binding + L_kcat + L_km + 0.1·L_joint + 0.05·L_dG_prior

默认（SmoothL1）:           可选（异方差 NLL）:
  SmoothL1(pred, target)     NLL = 0.5·log_var + 0.5·(y-μ)²/exp(log_var)

L_binding = L_kd + 0.3·L_ki
L_kcat    = Loss(kcat_pred, kcat_true)       # Eyring 已硬编码
L_km      = Loss(Km_pred, Km_true)
L_joint   = SmoothL1(kcat/Km_pred, kcat/Km_true)
          + 0.01·ReLU(kcat/Km_pred - 9)      # 扩散极限 ≤ 10⁹
L_dG_prior = NLL(ΔG‡ | μ=70, σ=20)           # ΔG‡ 高斯先验
```

各损失通过 `pkd_target_mask` / `kcat_target_mask` / `km_target_mask` 布尔掩码自动跳过无标签样本。

### 多任务平衡（可选）

| 策略 | 机制 | CLI 参数 | 覆盖层面 |
|------|------|----------|---------|
| **分层采样** | 每 batch 三任务等量采样 (1:1:1) | `--stratified-sampling` | 数据 |
| **不确定性加权** | 3 个可学习 log_σ 自动调权 | `--uncertainty-weighting` | 损失 |
| **PCGrad** | 梯度投影消除共享层冲突 | `--pcgrad` | 梯度 |

---



## 🚀 快速开始

```bash
source /home/domi/BINN/.venv/bin/activate
```

### 训练

```bash
cd /home/domi/AI4enz/model

# 基础配置（推荐起点）
python train.py --kcat-ode-steps 10 \
  --epochs 100 --batch-size 256 \
  --device cuda --num-workers 8 --amp

# 交叉注意力融合
python train.py --kcat-ode-steps 10 \
  --epochs 100 --batch-size 128 \
  --device cuda --num-workers 8 --amp \
  --use-cross-attn

# + 异方差 NLL 损失
python train.py --kcat-ode-steps 10 \
  --epochs 100 --batch-size 128 \
  --device cuda --num-workers 8 --amp \
  --use-cross-attn --heteroscedastic

# + 多任务平衡（推荐组合）
python train.py --kcat-ode-steps 10 \
  --epochs 100 --batch-size 128 \
  --device cuda --num-workers 8 --amp \
  --use-cross-attn --heteroscedastic \
  --stratified-sampling --uncertainty-weighting

# CPU 快速验证
python train.py --kcat-ode-steps 10 \
  --epochs 5 --batch-size 32 --max-samples 5000 \
  --device cpu --num-workers 0
```

### 推理

```bash
cd /home/domi/AI4enz/model

# 单样本预测（需安装 fair-esm）
python predict.py --smiles "CCO" --sequence "MKTVW..." --temperature 310.15

# 批量预测
python predict.py --csv candidates.csv --output results.csv
```

### 评估

```bash
cd /home/domi/AI4enz/model
python evaluate.py --checkpoint checkpoints/best.ckpt
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
| 蛋白 (per-token) | **100%** | ESM-2 `(L, 1280)` — **唯一蛋白输入** |
| 配体 | 100% | GATv2 原子特征 (79+10-dim) |
| 辅因子 | 56.7% | 310 种 Embedding (mask 跳过) |
| 温度 | **100%** (77.3% 实验值) | 每样本独立 K |

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
│   ├── ranking_model.py             # Trenzition v5 (cross-attn + ResidueQueryAttention)
│   ├── train.py                     # 训练脚本 (token-only)
│   ├── predict.py                   # 推理脚本 (ESM-2 token)
│   ├── evaluate.py                  # 评估脚本 (PCC/SCC/R²)
│   ├── benchmark_enhanced.py        # 基准测试 (vs XGBoost/MLP)
│   ├── compute_token_embeddings.py  # ESM-2 token 级嵌入预计算
│   ├── checkpoints/
│   └── tst/                         # TST 模型（实验性）
├── dataset_building/
│   ├── processed/                   # 训练用核心数据
│   │   ├── metadata.parquet         # 104,021 条元数据
│   │   ├── proteins_token.h5        # ESM-2 嵌入 (per-token, 18,929, 唯一输入)
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

### 2026-06-24
- 🧹 **移除 VIBLayer**: ResidueQueryAttention 已覆盖信息筛选功能，VIB 冗余
- ⚖️ **多任务平衡**: 新增分层采样 (`--stratified-sampling`)、不确定性加权 (`--uncertainty-weighting`)、PCGrad (`--pcgrad`)
- 📝 **文档重写**: CLAUDE.md + README.md 只记录当前架构，删除历史对比和已删除模块

### 2026-06-22
- 🎯 **Trenzition v5**: AttentionPooling 替代均值池化，残基级可学习注意力
- 🌡️ **TemperatureEncoder**: 温度编码器嵌入全部三头 (catalyst_context)
- 🔪 **删除 pooled 回退**: 始终使用 ESM-2 token 级 `(L, 1280)` 嵌入
- 🧹 **train.py 简化**: 删除 AA_PROPERTIES/--no-esm2/--strict-tokens
- 🐛 **ΔG‡ 映射修复**: sigmoid → ELU + x/(x+1)，梯度永不饱和
- 📉 **Ki 权重下调**: 0.7 → 0.3 (Ki pKd 噪声校正)
- 📊 **ΔG‡ 先验正则化**: Gaussian μ=70 kJ/mol, σ=20 kJ/mol
- 🔧 **kcat 数据清洗**: 4 条 log₁₀(kcat)=-10.0 伪样本修复
- 🌡️ **BRENDA 温度富集**: EC→median_temp 匹配 23,256 样本

### 2026-06-22 (VIB 已移除)
- 🧹 **移除 VIBLayer**: ResidueQueryAttention 已覆盖信息筛选功能，VIB 冗余
- 🧹 **dG_prior 逻辑修复**: 从 ThreeHeadLoss 移到 Trenzition.compute_loss

### 2026-06-17
- 🔗 **Trenzition v4**: 交叉注意力融合（蛋白残基 × 配体原子）
- 📊 **异方差 NLL 损失**: 预测 mean + variance，自动降噪
- 💾 **proteins_token.h5**: token-level ESM-2 嵌入预计算脚本

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
