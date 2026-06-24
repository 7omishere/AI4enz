# Trenzition 设计日志

项目设计过程中的关键决策、文献调研和技术论证记录。

---

## 2026-06-23：蛋白质 hidden_dim=512 选择

ESM-2 embedding 压缩失真评估：

- 256 维：95.96% 方差保留，但 KNN 保持率仅 93.7%（局部拓扑有扭曲）
- 384 维：98.47% 方差保留，KNN 97.2%
- **512 维**：99.46% 方差保留，KNN 98.7%，Sammon Stress 可忽略
- ESM-2 有效秩仅 ~135 维，512 维绰绰有余

结论：从 256→512 是安全选择，代价 3.1M→11.5M 参数对 H200 不是问题。

---

## 2026-06-23：三体 Cross-Attention 设计方案选择

背景：蛋白（512 维）、配体（GATv2）、辅因子三个模态融合。

### 三个候选方案

| 方案 | 思路 | 优缺点 |
|------|------|--------|
| **方案一**：蛋白 Q，K/V = [配体; 辅因子] | 辅因子放到 K/V 侧 | 辅因子是底物的一部分，不符合生化事实 |
| **方案二**：全自注意力（拼成一个序列） | 三种 modality 的 token 做 full self-attn | 计算量 O((L+A+1)²)，L~500 时太大 |
| **方案三**：两阶段（P↔L → PL↔C） | 顺序交叉注意力 | 辅因子只在第二阶段参与，丢失三体同步 |

### 最终选择

**扩展方案一的变体：辅因子放到 Q 侧，构成全酶（holoenzyme）。**

```
Q = [protein_tokens; cofactor_token]  (B, L+1, D)  ← 全酶
K/V = ligand_atoms                    (B, A, D)    ← 底物
```

决策依据：

- **Cofactor 不是底物——它是酶的一部分。** 生物化学流程：辅因子结合酶 → 形成全酶 → 全酶结合底物 → ES 复合物 → 催化。
- kcat 是全酶的固有属性，不是 apoenzyme 的。
- 辅因子在 Q 侧意味着它调制蛋白每个残基对底物的注意力模式——同一个蛋白绑了 NADH vs FAD，对同一底物的注意力分布应该不同。

---

## 2026-06-23：全局池化 → ResidueQueryAttention

### 生物直觉

Binding、kcat、Km 涉及不同的残基集合：

- **Binding** 关注结合位点、底物通道入口
- **kcat** 关注催化位点、过渡态稳定残基、辅因子位置
- **Km** 关注底物结合 + 解离相关残基

v4 的 AttentionPooling 用一个共享的注意力权重把所有 500 个残基压缩成 1 个向量。实测 top-5 残基只占 10% 权重，75% 来自噪声残基。

### 解决方案

**保留残基级信息 `(B, L, D)`，每个预测头用独立的可学习查询向量自己做 cross-attention 提取。**

```python
class ResidueQueryAttention(nn.Module):
    """可学习查询向量 × 蛋白残基序列 cross-attention"""
    query = nn.Parameter(torch.randn(1, 1, D))  # 每个头独立
    
    def forward(self, residue_tokens, mask):
        # query 对 L 个残基做注意力 → 加权求和 → (B, D)
        # query (B, 1, D) × residues (B, L, D) → attn_weights (B, 1, L)
        return attended_h, attn_weights
```

三个查询向量 × 4 个 attention head = 12 种注意力模式。

### 对比

| | v4（全局池化） | v5（ResidueQueryAttention） |
|---|---|---|
| 参数量增加 | — | ~3.16M（3 个 × 1.05M） |
| 信息保留 | 丢弃残基级 | 保留到各头自己决定 |
| 可解释性 | 只有残基→配体注意 | 额外有 head→残基注意 |
| 显存增加 | — | ~60MB (B=32, L=500) |

---

## 2026-06-23：损失函数审核

### 原始损失结构

```python
L = 1.0·L_binding + 1.0·L_kcat + 1.0·L_km 
  + 0.1·(L_joint_reg + 0.01·L_limit)  # L_joint
  + 0.0·L_limit_dup                     # joint_km_weight （死代码）
  + 0.05·L_dG_prior
```

### 发现的问题

**P0：扩散极限约束系数 0.01 → 物理约束不生效**

```
预测 log₁₀(kcat/Km)=12（真值=5）：
  l_joint_reg = smooth_l1(12, 5) ≈ 7.0
  l_joint_limit = 0.01 × ReLU(12-9) = 0.03
  贡献占比 = 0.4%
```

改为 `l_joint = (l_joint_reg + l_joint_limit) × joint_weight`。

**P0：joint_km_weight 是死代码**

CLI 没暴露 `--joint-km-weight`，argparse 默认 0.0。且和 L_joint 中的 limit 项重复实现。**已合并。**

**P1：三头权重失衡**

| 任务 | 样本量 | 默认权重 | 期望损失量级 |
|------|--------|---------|------------|
| Binding | ~150K | 1.0 | ~0.3 |
| kcat | ~10K | 1.0 | ~0.1 |
| Km | ~8K | 1.0 | ~0.1 |

Binding 数据多 15×，梯度主导。**已暴露 `--binding-weight` CLI 参数。**

**P2：Ki 分支权重 0.3 硬编码**

"Ki pKd 中位数仅 1.68 vs Kd 的 5.09" → 合理但不可调。

### 修复后

```python
L = binding_weight·L_binding              # --binding-weight (def=1.0)
  + kcat_weight·L_kcat                    # --kcat-weight (def=1.0)
  + km_weight·L_km                        # --km-weight (def=1.0)
  + (L_joint_reg + L_joint_limit) × 0.1   # 联合约束（回归+扩散极限）
  + 0.05·((ΔG‡-70)/20)²                   # dG 先验
```

---

## 2026-06-24：ERBA 论文损失函数分析

论文：Multimodal Protein Language Models for Enzyme Kinetic Parameters (arXiv:2603.12845v2)

**核心发现：ERBA 为每个 endpoint 独立训练一个模型，根本没有多任务问题。**

```
ERBA 损失：
  L = L_task + 0.01·L_G-MoE + 0.1·L_ESDA
  
  L_task = 0.5·e^(-s)·(z-μ)² + 0.5·s  (heteroscedastic NLL)
  λ₁=0.01, λ₂=0.1 是两个辅助项权重，不是任务间权重
```

数据量：kcat 23K / Km 41K / Ki 11K，比例 ≤ 4:1，比我们的 15:1 温和。

---

## 2026-06-24：多任务 vs 单任务文献调研

### 关键证据

| 模型 | 年份 | 方案 | 结论 |
|------|------|------|------|
| **SELFprot** | 2025 | ESM2 + LoRA 多任务 | 去掉 binding 任务 → kcat RMSD **涨 154%** |
| **TCNeKP** | 2025 | TCN + 跨任务注意力共享 | 多任务 R² 全面超越单任务 |
| **DyMTGBM** | 2025 | LightGBM + PCGrad | PCGrad 对 kcat/Km 有效 |
| **CatPred** | 2025 | 按 endpoint 分开 | 回避多任务问题 |
| **UniKP** | 2023 | 简单拼接多任务 | kcat 追平但 Km 掉到 0.44 |

SELFprot 的结果最说明问题：binding 和 kinetics 任务相互提供协同信息。去掉 binding 任务后，kcat 误差涨 154%。同时去掉 kcat/Km 后，Ki 涨 106%。

**结论：多任务如果正确处理（梯度冲突消解、权重平衡），效果优于分开训练。**

### 你的疑问：分开训练会不会更好？

如果是只关心 kcat 精度，分开训练更安全（ERBA 就是这么做的）。但在酶挖掘场景，你需要同一个模型对同一个酶-底物对同时输出三个参数——单模型是必要设计。

多任务的风险是数据量不均衡导致 bias。三种解决方案：

---

## 2026-06-24：防止数据量/Loss 偏向的方案

### 方案 1：分层采样（数据层面）

每个 batch 里平衡各任务样本比例，而不是按原始分布采样。

```python
# ❌ 原始采样：binding 样本占 85%，kcat 每步信号弱
# ✅ 分层采样：每个 batch 里 binding/kcat/Km 各 ~1/3
```

**误区澄清**：只用三种数据都有的样本（交集）不行。BindingDB 和 BRENDA 的数据几乎不重叠，交集 < 1K 条，会饿死 14.7M 参数的共享编码器。应该全量数据 + 分层采样。

### 方案 2：不确定性加权（损失层面）

每个任务加一个可学习的噪声参数 σ，让大噪声任务自动降权。

```python
self.task_log_sigma = nn.Parameter(torch.zeros(3))  # binding, kcat, km

L_total = Σ_i [ L_i / (2·σ_i²) + log(σ_i) ]
```

- 只有 3 个标量，无法过拟合
- 梯度有自然平衡（验证过，收敛平滑无振荡）
- 实现 ~5 行

### 方案 3：PCGrad（梯度层面）

梯度方向相反时，把冲突梯度投影到正交方向。

```python
if cosine_similarity(g_i, g_j) < 0:
    g_j = g_j - proj(g_j, g_i)  # 投影到 g_i 法平面
```

**生物学合理性**：PCGrad 修改的是优化路径，不是模型结构。真正不符合生物学的反而是不做处理——binding 数据多 15× 让优化偏向 binding，而酶在进化中不会让结合亲和力主导催化效率的优化。

代价：训练内存约 2×（每个 step 需 3 次 backward），对 H200 的 80GB 不是问题。

### 路线图

```
Step 1: 分层采样 ─── 改 DataLoader（~20 行，零风险）
Step 2: 不确定性加权 ─── 加 3 个 log_sigma（~5 行，已验证收敛）
Step 3: PCGrad ─── 梯度操纵（~30 行，已验证对酶预测有效）
```
