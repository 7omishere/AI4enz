# 酶挖掘排序模型 — 架构设计文档

> 对应代码：`datepre/ranking_model.py` + `dataset_building/train.py`。
> 最后更新：2026-05-22

---

## 一、背景

### 1.1 Marcus 约束诊断结果

对 329 个有 kcat 数据的蛋白，计算 `kcat_true / kcat_marcus(λ_prior, pkd_aligned)` 比值：

| 比值区间 | 蛋白数 | 占比 |
|---------|--------|------|
| < 0.001 | 317 | 96.4% |
| 0.001 ~ 0.01 | 5 | 1.5% |
| 0.01 ~ 0.1 | 7 | 2.1% |
| > 0.1 | 0 | 0.0% |

**结论**：100% 的蛋白 kcat 远低于 Marcus 方程预测的 ET 步速率（差距 10⁴~10⁶ 倍）。Marcus 方程预测的是电子转移步的理论速率上限，而实际 kcat 由产物释放、构象变化等其他限速步决定。`L_physics` 对所有样本产生系统性错误梯度，已移除。

### 1.2 kcat 数据局限性

对 329 个有 kcat 的蛋白分析发现：同一蛋白内所有底物共享同一个 kcat 值。kcat 是蛋白级别的聚合值（来自 BRENDA/UniProt），不能用于底物级别的排序学习。pKd 在同一蛋白内不同底物间有丰富变化，是当前唯一可用的底物级排序信号。

### 1.3 pKd 排序原型验证

用 3 个蛋白（低/中/高 kcat）× 3000 底物，纯 ESM-2+GNN 排序模型验证核心能力：Spearman 从 0.34 (epoch 1) 提升到 0.53 (epoch 20)，随机基线 ≈ 0.0。核心能力成立。

---

## 二、损失函数

### 当前设计

```
L_total = L_pkd + L_kcat

L_pkd  = SmoothL1(pKd_pred, pKd_true) × thermo_weight
L_kcat = SmoothL1(log_kcat_pred, log_kcat_true) × kcat_source_weight × thermo_weight

thermo_weight: Kd=1.0, Ki=0.7, IC50=0.15 （反映不同测量类型对热力学平衡常数的可信度）
kcat_source_weight: BRENDA+SABIO=1.0, SABIO-only=0.9, BindingDB-only=0.7
```

### 已移除的部分

- **L_physics**：Marcus 方程对 100% 蛋白不成立，产生系统性错误梯度
- **L_OT**：Sinkhorn 输运正则化，依赖已移除的 λ 分布匹配
- **warmup 调度**：不再需要（原来用于 L_physics 的渐进式激活）
- **λ 预测头**：保留在模型中但不参与损失计算

---

## 三、模型架构

```
输入:
  酶: ESM-2 (1280-dim) + struct_feat (contact_number_mean, protrusion_index_mean, has_structure)
      + pocket_cn/pocket_pi/pocket_dist (口袋残基几何特征)
      + domain_masks (15种辅因子域 × 序列长度, 可选)
  底物: SMILES → GATv2×3 GNN (79-dim原子 + 10-dim键)
  辅因子: 字符串列表 → CofactorEncoder (15种embedding + λ 预测头)

编码器:
  LigandEncoder:  GATv2×3 → GlobalAttention → 256-dim
  ProteinEncoder: ESM-2 (1280→256) + struct_feat (3→64, 拼接pocket 64-dim) 
                  + domain_mask 注意力池化
  PocketEncoder:  口袋残基特征 (cn, pi) → 距离加权高斯核消息传递 → 64-dim
  CofactorEncoder: 辅因子类型 embedding (64-dim) + λ 重组能预测 (保留但不参与损失)

交互:
  InteractionModule:
    Cross-Attention (配体↔蛋白)
    Cross-Attention (蛋白↔辅因子)
    ↓
  融合特征 (蛋白 + 配体投影 + 辅因子投影)

预测头 (MultiTaskHead):
  pkd_raw → sigmoid → [2, 15]   (底物结合亲和力)
  log_kcat                        (蛋白级催化效率，辅助任务)
  λ_offset                        (重组能偏移，不参与损失)
```

### 参数量：~2.24M

---

## 四、数据完整度（541 蛋白）

| 特征 | 覆盖 | 来源 |
|------|------|------|
| ESM-2 嵌入 | 541/541 (100%) | ESM-2 650M 预计算 |
| 结构特征 | 541/541 (100%) | AlphaFold DB + ESMFold 补充 |
| 口袋特征 | 541/541 (100%) | contact_map + binding_site_mask (519) + 结构推断 (22) |
| 辅因子注释 | 382/541 (70.6%) | Pfam域 (164) + UniProt API (92) + domain_masks修复 (31) + DISCODE预测 (95) |
| kcat 数据 | 329/541 (60.8%) | BRENDA + SABIO-RK + UniProt + BindingDB |

### 辅因子类型分布

HEME (95) > FAD (77) > FMN (37) > FES (32) > NAD (22) > TPP (21) > COQ (12) > NADP (9) > CU (7) > MPT (5)

### 剩余缺口说明

159 个蛋白（29.4%）无辅因子注释，其中约 100 个是**真正不需要有机辅因子**的酶（EC 1.14.11 Fe/2OG双加氧酶、1.11.1 过氧化物酶、1.13.11 非血红素铁双加氧酶），氧化还原活性来自氨基酸残基或直接配位金属离子。剩余约 60 个可能用 FAD 但缺乏预测工具。

---

## 五、训练数据

| 指标 | 值 |
|------|-----|
| 总记录 | 78,113 |
| 唯一蛋白序列 | 541 |
| 唯一配体 (InChIKey) | 57,203 |
| Split | train 55,388 / val 9,951 / test 12,775（蛋白级哈希分割） |
| pKd 范围 | [2.0, 11.7]，均值 6.41 |
| 测量类型 | IC50 71,929 / Ki 5,542 / Kd 643 |
| kcat 范围 | [1.33e-3, 7.29e+3] s⁻¹ |

---

## 六、推理接口

`inference_enzyme_mining.py`：给定底物 SMILES → 从 541 个氧化还原酶库中按预测 pKd 排序推荐 Top-K。

```
python inference_enzyme_mining.py --smiles "O=C1CCCCC1" --top-k 10
python inference_enzyme_mining.py --smiles-file candidates.txt --cofactor HEME --top-k 5
python inference_enzyme_mining.py --smiles "c1ccccc1O" --output results.json
```

---

## 七、已完成的改动

| 改动 | 状态 |
|------|------|
| attn_pc bug 修复（输出误加到 ligand_h 改为 protein_h） | ✅ |
| pkd_head sigmoid 约束到 [2, 15] | ✅ |
| 移除 L_physics、L_OT、OTRegularizer、MarcusPhysicsLoss | ✅ |
| thermo_weight 分层权重 (Kd=1.0, Ki=0.7, IC50=0.15) | ✅ |
| PocketEncoder 子模块（Gaussian-kernel 距离加权消息传递） | ✅ |
| 口袋特征提取 + 22个蛋白结构推断补全 | ✅ |
| 辅因子注释扩充（30%→70.6%）| ✅ |
| 推理脚本 inference_enzyme_mining.py | ✅ |
| train.py 适配口袋特征和辅因子变更 | ✅ |

---

## 八、验证标准

```bash
cd /home/domi/BINN/AI4enz/dataset_building
source /home/domi/BINN/.venv/bin/activate

# 快速冒烟测试
python train.py --epochs 2 --batch-size 16 --max-samples 500 --device cpu --num-workers 0

# 推理接口
python inference_enzyme_mining.py --smiles "O=C1CCCCC1" --top-k 5
```

## 九、与 EZSpecificity 对比

| | 本模型 | EZSpecificity (Nature 2025) |
|---|---|---|
| 任务 | pKd 连续值回归 | 底物特异性二分类 |
| 训练标签 | 实验 pKd（PDBbind+BindingDB+ChEMBL） | AutoDock Vina 对接打分 |
| 酶编码 | ESM-2 + 距离矩阵口袋 + 结构特征 + 辅因子域 | ESM-2 + SE(3)-等变GNN（口袋Cα坐标） |
| 辅因子 | 15种辅因子编码 + λ预测头 | 不考虑 |
| kcat | 蛋白级辅助回归 | 无 |
| 损失 | SmoothL1 × thermo_weight + kcat | Binary Cross-Entropy |
| 输出 | 连续 pKd 值（可排序） | 特异性分（能/不能） |
| 酶范围 | 氧化还原酶专项 (541) | 通用 (8,124) |
| 训练集 | 78K 对（实验数据） | 323K 对（对接数据） |
