# CLAUDE.md — AI4enz

AI 辅助酶挖掘项目。基于过渡态理论构建酶-底物结合亲和力预测模型，服务于"给定目标底物 → 推荐酶序列"的酶挖掘场景。
每次回复前都叫我：“多米”

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
├── CLAUDE.md                              # 本文件
├── README.md                              # 项目简介
├── .gitignore                             # Git 忽略规则
└── dataset_building/                      # 所有代码与数据
    ├── ranking_model.py                   # TransitionBINN 模型定义
    ├── train.py                           # 训练脚本
    ├── inference_enzyme_mining.py          # 酶挖掘推理接口
    ├── evaluate_model.py                   # 模型评估
    ├── extract_pocket_features.py          # 口袋特征提取
    ├── extract_domains.py                  # UniProt 域提取
    ├── enrich_domains_pfam.py              # Pfam 域富集
    ├── compute_esm2_embeddings.py           # ESM-2 嵌入计算
    ├── merge_external_data.py              # 外部数据集合并
    ├── merge_bindingdb.py                  # BindingDB 合并
    ├── pipeline/                           # 模块化数据处理流水线
    │   ├── 01_parse_pdbbind.py
    │   ├── 02_parse_bindingdb.py
    │   ├── 03_align_distributions.py
    │   ├── 04_build_ligand_graphs.py
    │   ├── 05_write_storage.py
    │   ├── 06_dataset.py
    │   ├── 07_run_pipeline.py
    │   └── 08_make_splits.py
    ├── external_data/                      # 外部数据源
    │   ├── BindingDB_All_202605_tsv.zip
    │   ├── oed_kinetics.json
    │   ├── SKiD_Main_dataset_v1.xlsx
    │   └── kcat_archive/                   # SKiD 3D 结构 (12,866 复合物)
    ├── processed/                          # 处理后输出
    │   └── oxidoreductase/
    │       ├── unified_metadata_v2.parquet    # 主数据集 (322,763)
    │       └── high_quality_kd_ki_v2.parquet  # Ki/Kd 子集 (163,927)
    ├── checkpoints/                        # 模型检查点
    └── ablation_results/                   # 消融实验结果
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

### 全局统计 (v2)

| 指标 | 值 |
|------|-----|
| 总记录 | **322,763** |
| 唯一蛋白 (UniProt) | **11,044** |
| 唯一配体 (InChIKey) | **141,941** |

### Split

| 切分 | 占比 |
|------|------|
| train | ~80% |
| val | ~10% |
| test | ~10% |

Split 按 UniProt ID 层级分配，避免蛋白序列泄漏。

### 测量类型分布

| 类型 | 数量 | 可信度 |
|------|------|--------|
| Ki | 149,357 | 中-高 (BindingDB+CatPred) |
| IC50 | 71,929 | 低 (R²=0.437 校正) |
| kcat_only | 69,207 | — (动力学参数) |
| km_only | 17,700 | — |
| Kd | 14,570 | 高 (BindingDB) |

### 高质量子集（推荐）

- **路径**: `processed/oxidoreductase/high_quality_kd_ki_v2.parquet`
- **样本数**: **163,927** (仅 Kd/Ki，不含 IC50)

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

## 数据来源与渠道

### 已整合

| 来源 | 官方 URL | 期刊 | 提供数据 | 规模 |
|------|----------|------|----------|------|
| **CatPred-DB** | `github.com/maranasgroup/CatPred` (AWS S3) | Nature Comms 2025 | kcat, Ki | 20k kcat + 11k Ki |
| **OED** | `openenzymedb-api.platform.moleculemaker.org/api/v1/data` | NAR 2025 | kcat, Km, kcat/Km | 36k |
| **SKiD** | `doi.org/10.5281/zenodo.15355031` | Scientific Data 2025 | kcat, Km + 3D结构 | 13k kcat + 18k Km |
| **BindingDB** | `bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp` | 月度更新 | Kd, Ki | 14k Kd + 135k Ki |

### 可补充

| 渠道 | 说明 |
|------|------|
| **BRENDA** | 酶动力学金标准，需学术许可，SOAP API |
| **SABIO-RK** | 反应动力学，REST API 导出 |
| **UniProt** | 序列/功能注释，REST API |
| **PDBbind** | 晶体结构 + Kd/Ki，精炼集 ~5k 条 |

### EZSpecificity (参考架构)

Nature 2025，赵惠民组。ESIbank 数据库 323,783 酶-底物对（binary 标签，无 kcat/Kd）。ESM-2 + SE(3)-GNN + Cross-Attention → 底物特异性预测。与我们互补。

## GitHub

- **仓库**: https://github.com/Domi-Joe/AI4enz
- 代码单独分发，数据集通过云盘传输