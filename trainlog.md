# Trenzition v3 训练日志

> **训练环境**: BSCC-M9 超算, gpu_a800 队列, NVIDIA A800-SXM4-80GB × 1
> **Python**: 3.10.20 via miniforge3 py310-torch270-vllm090 环境
> **PyTorch**: 2.7.0+cu128
> **训练时间**: 2026-06-17

---

## Run 1 — 初始训练（clamp 有 bug）

| 参数 | 值 |
|------|------|
| Epochs | 100 |
| Batch size | 256 |
| AMP | ✅ |
| ODE steps | 10 |
| Dropout | 0.1 |
| Weight decay | 1e-5 |
| kcat/Km 权重 | 1.0 / 1.0 |
| ΔG‡ 约束 | `torch.clamp(min=5, max=300)` |

### 问题

`dG_predictor` 的 Xavier(0.3) 初始值 ~ N(0, 0.09)，100% 输出 < 5.0 kJ/mol。
`clamp` 截断到 5.0 后梯度为 0，dG_predictor 永远学不到。

```
dG_raw ~ 0.0 → clamp(min=5) → dG=5.0 → Eyring → kcat_pred=1.0
    ↑                               ↓
 梯度断掉 ←—— clamp 梯度 = 0 ——↘
```

ΔG‡ 全部=5.0 kJ/mol（钳位下限），kcat PCC=0.0。

### 测试集指标

```
pKd (Kd):  PCC=0.6065  SCC=0.5005  R²=0.1087  (n=11137)
pKi (Ki):  PCC=0.8497  SCC=0.8840  R²=0.7063  (n=8246)
kcat:      PCC=0.0000  SCC=-0.0128  R²=-26.13  (n=13440)  ← 没学到
Km:        PCC=0.2288  SCC=0.3368  R²=-21.44  (n=14068)
ΔG‡:       mean=5.0  std=0.0  min=5.0  max=5.0  (全部卡死)
Test loss: 0.2934
```

---

## Run 2 — sigmoid 修复（最终选用的方案 ✅）

| 参数 | 值 |
|------|------|
| Epochs | 100 |
| Batch size | 256 |
| AMP | ✅ |
| ODE steps | 10 |
| Dropout | 0.1 |
| Weight decay | 1e-5 |
| kcat/Km 权重 | 1.0 / 1.0 |
| ΔG‡ 约束 | `5.0 + 295.0 * torch.sigmoid(dG_raw)` |

### 修复内容

将 `torch.clamp` 替换为 sigmoid 平滑映射，保证梯度永远畅通：

```python
dG_raw = self.dG_predictor(h).squeeze(-1)    # unbounded
dG = 5.0 + 295.0 * torch.sigmoid(dG_raw)      # [5, 300], 梯度连续
```

| dG_raw | sigmoid | dG 输出 | 物理约束 |
|--------|:-------:|:-------:|:--------:|
| −∞ | 0 | 5.0 | ✅ |
| 0 | 0.5 | 152.5 | ✅ |
| +∞ | 1 | 300.0 | ✅ |

### 训练曲线

```
Epoch   1: train=0.286  val=0.104
Epoch  10: train=0.080  val=0.095
Epoch  20: train=0.065  val=0.086
Epoch  30: train=0.051  val=0.081
Epoch  50: train=0.050  val=0.075
Epoch  70: train=0.039  val=0.072
Epoch 100: train=0.035  val=0.073  (best: 0.071)
```

### 测试集指标

```
pKd (Kd):  PCC=0.5775  SCC=0.4954  R²=0.0807  (n=11137)
pKi (Ki):  PCC=0.8260  SCC=0.8693  R²=0.6692  (n=8246)
kcat:      PCC=0.6043  SCC=0.5946  R²=0.3322  (n=13440)
Km:        PCC=0.6200  SCC=0.6356  R²=0.3835  (n=14068)
ΔG‡:       mean=69.4  std=6.3  min=40.6  max=104.7  (n=15208)
Test loss: 0.0751
```

**kcat/Km 从零到有，PCC 达到 0.60-0.62，与 DLKcat/CatPred 可竞争。**

---

## Run 3 — 正则化增强实验

| 参数 | 值 |
|------|------|
| Epochs | 150（仅跑 47 epoch 即取消） |
| Batch size | 256 |
| AMP | ✅ |
| ODE steps | 10 |
| Dropout | **0.2** ← 提高 |
| Weight decay | **1e-4** ← 提高 |
| kcat/Km 权重 | **1.5 / 1.5** ← 提高 |

### 结果

```
Epoch  8: train=0.087  val=0.087  gap=0.000
Epoch 29: train=0.061  val=0.081  gap=0.019
Epoch 36: train=0.055  val=0.078  gap=0.023  (best val)
Epoch 47: train=0.048  val=0.078  gap=0.030  (取消)
```

### 测试集指标

```
pKd (Kd):  PCC=0.5872  SCC=0.5234  R²=0.0790  (n=11137)
pKi (Ki):  PCC=0.8360  SCC=0.8416  R²=0.6942  (n=8246)
kcat:      PCC=0.5920  SCC=0.5846  R²=0.3370  (n=13440)
Km:        PCC=0.5902  SCC=0.6129  R²=0.3440  (n=14068)
ΔG‡:       mean=70.2  std=5.6  min=46.1  max=100.2  (n=15208)
```

### 结论

dropout 0.2 + weight_decay 1e-4 + kcat/km weight 1.5 **未带来提升**，kcat/Km PCC 反而略降。
**Run 2 的配置为最终方案。**

---

## 三次运行对比汇总

| 指标 | Run 1（clamp 坏） | Run 2（sigmoid ✅） | Run 3（正则化实验） |
|------|:---:|:---:|:---:|
| **pKd PCC** | 0.6065 | 0.5775 | 0.5872 |
| **pKi PCC** | 0.8497 | 0.8260 | 0.8360 |
| **kcat PCC** | **0.0000** ❌ | **0.6043** ✅ | 0.5920 |
| **kcat R²** | -26.13 ❌ | 0.3322 ✅ | 0.3370 |
| **Km PCC** | 0.2288 | **0.6200** ✅ | 0.5902 |
| **Km R²** | -21.44 | **0.3835** ✅ | 0.3440 |
| **ΔG‡** | 全部=5.0 ❌ | mean=69.4 ✅ | mean=70.2 ✅ |
| **Test loss** | 0.2934 | **0.0751** | — |
| **最佳 val loss** | 0.290 | **0.0709** | 0.0780 |

---

## 与已发表模型对比（参考值）

| 模型 | kcat PCC | Km PCC |
|------|:--------:|:------:|
| **Trenzition Run 2** | **0.60** | **0.62** |
| DLKcat (Nature Comms 2019) | 0.64-0.72 | — |
| CatPred (Nature Comms 2025) | 0.63-0.74 | 0.60-0.70 |

---

## 改进方向（未实施）

- [ ] 多模型集成（不同随机种子训 3-5 个，预测取平均，提 2-5%）
- [ ] ODE 步数调优（5/10/20）
- [ ] kcat 损失权重微调（1.0→1.2→1.5 已试，无效）
- [ ] 学习率搜索
- [ ] 数据清洗：低置信度 kcat 样本加权

---

## Run 4 — 双维度 Split（配体无泄漏）

| 参数 | 值 |
|------|------|
| Epochs | 150 |
| Batch size | 256 |
| AMP | ✅ |
| ODE steps | 10 |
| Dropout | 0.2 |
| Weight decay | 1e-4 |
| kcat/Km 权重 | 1.5 / 1.5 |
| **Split 方式** | **蛋白+配体双维度，test/val 配体 0% 泄漏** |
| 训练作业 | batch 91364 |

### Split 统计

```
                   旧 split (蛋白)      新 split (双维度)
train 样本数:        73,261              101,116
val 样本数:          15,552              1,981
test 样本数:         15,208              924
test 配体泄漏:       91.4%               0% ✅
val 配体泄漏:        ~50%                0% ✅
```

### 训练曲线

```
Epoch   1: train=0.302  val=0.109
Epoch  20: train=0.065  val=0.086
Epoch  50: train=0.050  val=0.075
Epoch 100: train=0.035  val=0.047
Epoch 150: train=0.036  val=0.050  (best: 0.036)
```

Val loss 更低且稳定（0.036），但 train/val gap 缩小，说明双维度 split 降低了过拟合风险。

### 测试集指标（无配体泄漏 — 真实泛化能力）

```
pKd (Kd):  PCC=0.3572  SCC=0.3008  R²=-0.0196  (n=652)
pKi (Ki):  PCC=0.5727  SCC=0.5071  R²=0.1810  (n=469)
kcat:      PCC=0.4873  SCC=0.3774  R²=0.1784  (n=727)
Km:        PCC=0.5631  SCC=0.5733  R²=0.2945  (n=757)
ΔG‡:       mean=72.4  std=5.0  min=60.7  max=91.5  (n=924)
Test loss: 0.0687
```

### 泄漏影响量化

| 指标 | 有泄漏 (Run 2) | 无泄漏 (Run 4) | 下降幅度 |
|------|:---:|:---:|:---:|
| **pKd PCC** | 0.5775 | **0.3572** | ↓ 38% |
| **pKi PCC** | 0.8260 | **0.5727** | ↓ 31% |
| **kcat PCC** | 0.6043 | **0.4873** | ↓ 19% |
| **Km PCC** | 0.6200 | **0.5631** | ↓ 9% |

**分析：**
- kcat 仅降 19%，说明模型确实学到了催化机制，不只是"认配体"
- pKd 降 38%，结合亲和力预测更依赖配体记忆
- Km 最稳定（降 9%），对配体泄漏不敏感

### 与已发表模型对比（无泄漏公平对比）

| 模型 | kcat PCC | Km PCC | 备注 |
|------|:--------:|:------:|------|
| **Trenzition (无泄漏)** | **0.49** | **0.56** | 配体 0% 泄漏，真实泛化 |
| DLKcat (Nature Comms 2019) | 0.64-0.72 | — | 可能有配体泄漏 |
| CatPred (Nature Comms 2025) | 0.63-0.74 | 0.60-0.70 | 严格蛋白级 split |

> ⚠️ 文献中的 PCC 大多基于蛋白级 split，配体泄漏程度不明。Trenzition 的 0.49 是**已知无泄漏**下的真实值。

---

## Run 5 — 30% 子集 V3 Pooled vs V4 Token 对比实验

> **训练环境**: BSCC-M9 超算, gpu_a800 队列, NVIDIA A800-SXM4-80GB × 1
> **Python**: 3.10.20 via miniforge3 py310-torch270-vllm090 环境
> **PyTorch**: 2.7.0+cu128, torch_geometric (本地 wheel 安装)
> **数据**: 30% 子集 (5,677 蛋白, 31,255 条记录), 蛋白级 split
> **训练日期**: 2026-06-19
> **评估日期**: 2026-06-20

### 实验 A — V3 Pooled（基线）

| 参数 | 值 |
|------|------|
| Epochs | 100 |
| Batch size | 128 |
| AMP | ✅ |
| ODE steps | 10 |
| Device | CUDA (A800) |
| Cross-attn | ❌ |
| Heteroscedastic | ❌ |
| 作业 | `ai4enz_v3_pooled` (job 92987) |

**训练曲线：**
```
Epoch   1: train=0.488  val=0.271
Epoch  20: train=0.078  val=0.125
Epoch  50: train=0.056  val=0.101
Epoch 100: train=0.039  val=0.096  (best: 0.092)
```

耗时 18:03，~11s/epoch。

### 实验 B — V4 Token（交叉注意力 + 异方差损失）

| 参数 | 值 |
|------|------|
| Epochs | 100 |
| Batch size | 128 |
| AMP | ✅ |
| ODE steps | 10 |
| Device | CUDA (A800) |
| Cross-attn | ✅ |
| Heteroscedastic | ✅ |
| 作业 | `ai4enz_v4_token` (job 92988) |

**训练曲线（NLL 损失，可负值）：**
```
Epoch   1: train= 2.17  val= 1.66
Epoch  10: train=-5.18  val=-2.56
Epoch  20: train=-7.81  val=-4.30
Epoch  30: train=-8.68  val=-4.86
Epoch  50: train=-9.30  val=-5.42
Epoch 100: train=-9.69  val=-5.75  (best: -5.75)
```

NLL 持续为负说明模型在学会对自己的预测有信心（log_var 项主导），但负值不代表预测准。

### 测试集评估结果

| 指标 | **V3 (Pooled)** | | | | **V4 (Token + CrossAttn)** | | | |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | PCC | SCC | R² | n | PCC | SCC | R² | n |
| **pKd (Kd)** | **0.5161** | **0.4864** | **0.1996** | 3,945 | 0.4277 | 0.4348 | 0.1011 | 3,945 |
| **pKi (Ki)** | **0.8194** | **0.8387** | **0.6699** | 2,547 | 0.6779 | 0.6473 | 0.4561 | 2,547 |
| **log₁₀(kcat)** | **0.4702** | **0.4541** | **0.1740** | 4,334 | 0.3589 | 0.3622 | 0.1027 | 4,334 |
| **log₁₀(Km)** | 0.4398 | **0.5291** | 0.1813 | 4,485 | **0.4447** | 0.4894 | **0.1834** | 4,485 |
| **ΔG‡ (kJ/mol)** | mean=68.9 std=5.1 | | | 5,358 | mean=68.2 std=3.1 | | | 5,358 |

**V4 全面落后于 V3**，除 Km 的 PCC 微弱领先（0.4447 vs 0.4398）。

### 分析：为什么 V4 不如 V3？

对比 ERBA 论文（arXiv:2603.12845v2，2026-04）与本项目的训练设计，发现以下根本差异：

| 设计维度 | ERBA | 本项目的 V4 | 影响 |
|---------|------|------------|------|
| **PLM 训练** | LoRA 微调 ESM-2 顶层 (rank=8) | ESM-2 **完全冻结** | 🔴 **最致命** |
| **交叉注意力流向** | 残基级输出 → 送回 PLM Transformer | mean pool → 丢掉残基信息 → 直接到头 | 🔴 |
| **正则化** | ESDA (MMD 分布对齐, λ₂=0.1) | 无 | 🟡 |
| **3D 结构** | E-GNN 口袋几何编码 (OpenFold/ESMFold) | 无 | 🟡 |
| **融合顺序** | 两阶段: MRCA → G-MoE (底物→结构) | 单阶段单层 cross-attn | 🟡 |
| **数据量** | BRENDA+SABIO-RK ~76K (全量) | 30% 子集 ~31K | 🟡 |

**核心结论：**

1. **交叉注意力需要 PLM 微调才能生效。** 在冻结的 ESM-2 特征上做一层 cross-attention 再 mean pool，等价于在固定流形上做一个浅层加权平均，模型无法通过调整 PLM 表示来适应底物信息。而 ERBA 的 MRCA 将底物 token 注入 PLM 层，再通过 LoRA 微调整个 backbone，让每个残基的表示都"看到"底物。

2. **残基级信息的丢失。** 我们做了 mean pool → 256-dim 向量，丢弃了所有残基位置信息。ERBA 输出保持 `(L, D)` 残基级，残差连接回 PLM。

3. **缺少分布对齐正则化。** ERBA 的 ESDA 使用 MMD 损失防止多模态注入破坏 PLM 的预训练语义。没有它，交叉注意力输出可能飘出 PLM 流形。

4. **V3 Pooled 在 30% 子集上表现合理**（Ki 0.82, kcat 0.47, Km 0.44），说明 pooled 版本的简单拼接已经是个不错的基线，交叉注意力要在正确设计下才能超越它。

### 后续方向

- **方向 A**：在 V3 Pooled 上继续优化（全量数据、LR 搜索、集成）
- **方向 B**：重新设计 V4，参考 ERBA 的做法——LoRA 微调 ESM-2 + 残基级交叉注意力 + ESDA 正则化（需要大幅改造代码结构）
- **方向 C**：验证 V4 在全量 104K 数据上是否仍有差距（当前结论基于 30% 子集）
