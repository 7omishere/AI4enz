# CLAUDE.md — AI4enz

AI 辅助酶挖掘项目。构建酶-底物结合亲和力数据集，训练排序模型预测氧化还原酶的底物偏好，服务于"给定目标底物→推荐酶序列"的酶挖掘场景。架构参见 `dataset_building/ARCHITECTURE_REVISION.md`。

## 项目结构

```
BINN/
├── .venv/                              # Python 3.13 虚拟环境
├── AI4enz/
│   ├── CLAUDE.md                       # 本文件
│   ├── .gitignore                      # Git 忽略规则（数据文件不入库）
│   ├── datepre/                        # 数据处理与模型定义脚本
│   │   ├── prepare_oxidoreductase.py   #   EC 1.x 筛选 + 辅因子注释
│   │   ├── supplement_kinetics.py      #   UniProt 动力学补充
│   │   ├── parse_brenda.py             #   BRENDA flat file 解析
│   │   ├── enrich_domains_pfam.py      #   pyhmmer Pfam HMM 本地域扫描
│   │   ├── compute_esm2_embeddings.py  #   ESM-2 650M 嵌入预计算
│   │   ├── update_domain_metadata.py   #   从 proteins.h5 同步域注释
│   │   ├── build_unified_dataset.py    #   统一数据集组装 + 分层 split
│   │   ├── supplement_chembl_activities.py  # ChEMBL IC50/Ki/Kd 查询 + 校正模型
│   │   ├── supplement_sabio.py         #   SABIO-RK 动力学补充 + kcat 交叉验证
│   │   ├── apply_ic50_correction.py    #   IC50→Ki 校正 + Kd 补充合并
│   │   └── ranking_model.py            #   酶挖掘排序模型定义
│   └── dataset_building/
│       ├── train.py                    # 训练脚本
│       ├── ARCHITECTURE_REVISION.md    # 正式架构设计文档
│       ├── inference_enzyme_mining.py   # 推理接口：底物→酶排序
│       ├── extract_pocket_features.py   # 口袋特征提取（AlphaFold结构→几何特征）
│       ├── data/                       # 原始数据 (PDBbind + BindingDB)
│       │   ├── pdbbind/                #   PDBbind v2020R1 结构+索引
│       │   └── bindingdb/              #   BindingDB_All.tsv (~8.8 GB)
│       ├── pipeline/                   # 数据处理脚本 (01-08，原始 pipeline)
│       ├── processed/                  # 已处理输出
│       │   ├── proteins.h5             #   蛋白序列 + ESM-2 嵌入 + domain_masks + 结构特征
│       │   ├── Pfam_cofactor_full.hmm  #   220 个辅因子相关 Pfam HMM
│       │   ├── metadata.parquet        #   完整元数据 (2.19M 行)
│       │   ├── ligands/                #   1,276,832 配体分子图 (.pt)
│       │   └── oxidoreductase/         #   氧化还原酶子集
│       │       ├── unified_metadata.parquet      # 统一数据集 (78,113 × 47)
│       │       ├── ic50_ki_correction.json       # IC50→Ki 校正模型
│       │       ├── chembl_paired_data.parquet    # ChEMBL IC50-Ki 配对 (1,464 对)
│       │       ├── sabio_summary.parquet         # SABIO-RK 汇总 (830 UniProt)
│       │       ├── sabio_aligned.parquet         # SABIO-RK 对齐到蛋白 (146 蛋白)
│       │       ├── sabio_kinetics.parquet        # SABIO-RK 详细条目 (14,906 条)
│       │       ├── brenda_aligned.parquet        # BRENDA 动力学对齐
│       │       ├── kinetics.parquet              # UniProt 动力学
│       │       └── cache/                        # API 查询缓存
│       └── checkpoints/               # 模型检查点
└── BINN/                               # 上游 BINN 包 (只读参考)
```

## 虚拟环境

```bash
source /home/domi/BINN/.venv/bin/activate
```

已安装核心依赖：PyTorch, PyTorch Geometric, RDKit, BioPython, h5py, pandas, scikit-learn, numpy, pyhmmer, transformers, chembl_webresource_client, fair-esm。

## Marcus-PINN 训练管线

### 数据处理流程

| 步骤 | 脚本 | 功能 |
|------|------|------|
| 1 | `prepare_oxidoreductase.py` | EC 1.x 筛选 + 辅因子注释 |
| 2 | `parse_brenda.py` | BRENDA 动力学解析 (KM, kcat, kcat/KM) |
| 3 | `supplement_kinetics.py` | UniProt 动力学补充 |
| 4 | `enrich_domains_pfam.py` | pyhmmer 本地 Pfam 域扫描 (220 HMM) |
| 5 | `compute_esm2_embeddings.py` | ESM-2 650M 嵌入预计算 |
| 6 | `update_domain_metadata.py` | 域注释同步到 metadata |
| 7 | `build_unified_dataset.py` | 统一数据集组装 + 分层 split |
| 8 | `supplement_chembl_activities.py` | ChEMBL IC50/Ki/Kd 查询 → IC50→Ki 校正模型 |
| 9 | `supplement_sabio.py` | SABIO-RK 动力学补充 + BRENDA 交叉验证 |
| 10 | `apply_ic50_correction.py` | IC50 校正 + Kd 补充 → 最终 unified_metadata.parquet |

### 训练命令

```bash
cd /home/domi/BINN/AI4enz/dataset_building
source /home/domi/BINN/.venv/bin/activate

# 快速测试 (CPU, 500 样本, 2 epochs)
python train.py \
    --epochs 2 --batch-size 16 --max-samples 500 \
    --device cpu --num-workers 0

# 完整训练 (GPU, ESM-2 默认)
python train.py \
    --epochs 100 --batch-size 128 --device cuda

```

### 模型架构概要

- **酶挖掘排序模型** (1.2M~2.2M 参数)
- 编码器：LigandEncoder (GATv2×3) + ProteinEncoder (ESM-2 1280-dim + 结构特征 + 域掩码) + CofactorEncoder (15 种辅因子 embedding)
- 交互层：Cross-Attention(蛋白↔配体, 蛋白↔辅因子) → 融合特征
- 预测头：pKd (底物亲和力排序) + kcat (蛋白级催化效率，辅助任务)
- λ 预测头保留但不再参与主损失计算
- 损失：`L = L_pkd(带 thermo_weight) + L_kcat(带 source_weight)` — 无 Marcus 物理约束，无 OT 正则化
- thermo_weight: Kd=1.0, Ki=0.7, IC50=0.15（反映不同测量类型对热力学平衡常数的可信度）
- pkd_head: sigmoid 约束到 [2, 15]；λ 预测头保留但不参与损失

## 氧化还原酶数据集关键参数（2026-05-20 更新后）

### 全局统计

- 总记录：78,113（PDBbind 216 + BindingDB 77,898）
- 唯一蛋白序列：541（全 TrEMBL，无 Swiss-Prot reviewed）
- 唯一配体 (InChIKey)：57,203
- Split：train 55,388 (70.9%) / val 9,951 (12.7%) / test 12,775 (16.4%)，蛋白级哈希分割

### 结合亲和力 (pKd)

| 指标 | 值 |
|------|-----|
| 有 pKd 标签 | 78,113 (100%) |
| 测量类型 | IC50 71,929 / Ki 5,542 / Kd 643 |
| pkd_aligned 均值 | 6.41 (中位数 6.27) |
| pkd_aligned 范围 | [2.0, 11.7] |
| IC50→Ki 校正 | `pKi = 0.7172 × pIC50 + 2.2939`, R²=0.437, n=1464 对 |
| 校正来源 | chembl_ic50_ki_model (71,929 条), none (6,185 条) |

### 催化效率 (kcat)

| 来源 | 蛋白数 | 记录数 | 说明 |
|------|--------|--------|------|
| multi_source (BRENDA+SABIO) | 77 | 35,307 | 双源交叉验证，最可信 |
| bdb (BindingDB/BRENDA) | 252 | 26,191 | 单一来源 |
| sabio_only | 5 | 2,305 | SABIO-RK 独有，BindingDB 无 |
| 无 kcat | 208 | 14,311 | 38% 蛋白缺乏催化数据 |
| **合计有 kcat** | **334** | **63,803 (82%)** | |

kcat 范围: [1.33e-3, 7.29e+3] s⁻¹, log_kcat: [-2.88, 3.86]
kcat 异常值: 378 条 (0.5%)，标记为 kcat_outlier=True

### 结构特征

| 来源 | 记录数 | 蛋白数 |
|------|--------|--------|
| AlphaFold DB | 76,721 | 388 |
| PDBbind 实验 | ~1,088 | 148 |
| ESMFold 补充 | 305 | 5 |
| **合计有结构** | **78,113 (100%)** | **541/541 (100%)** |

5 个之前缺失的蛋白（P33072, P36969, P45845, Q9NNW7, Q9Z0J5）已于 2026-05-21 补充结构特征。

### 域注释与辅因子

- 辅因子覆盖：382/541 蛋白（70.6%），来源：Pfam域扫描 (164) + UniProt API (92) + domain_masks修复 (31) + DISCODE预测 (95)
- 15 种辅因子类型（按频率）：HEME (95) > FAD (77) > FMN (37) > FES (32) > NAD (22) > TPP (21) > COQ (12) > NADP (9) > CU (7) > MPT (5)
- CofactorEncoder PRIORS 仅 12 种 — B12, THF, COA 缺少 λ 先验值
- 约 100 个蛋白是不需要有机辅因子的氧化还原酶（Fe/2OG双加氧酶、过氧化物酶等）

## 关键设计决策

1. **ESM-2 预计算**：嵌入离线计算存储于 proteins.h5（每蛋白 5KB），训练时直接加载，无需加载 650M 模型
2. **双路径蛋白编码**：自动检测输入维度（1280=ESM-2, 6=AA属性），fallback 到 AA 物化性质
3. **去掉 Marcus 物理约束**：诊断证实 kcat_true / kcat_marcus ≈ 10⁻⁶（100% 蛋白），L_physics 产生系统性错误梯度，已移除
4. **pKd 排序替代 kcat 回归**：kcat 无底物级变化，pKd 有——当前用 pKd 作为底物偏好排序信号
5. **热力学分层权重**：thermo_weight = Kd(1.0) / Ki(0.7) / IC50(0.15)，乘入 quality_weight（已在 train.py 实现）
6. **蛋白级 split**：按蛋白序列哈希分层，辅因子类型覆盖验证
7. **kcat 异常值标记**：378 条记录标记为异常（kcat > 1e3 或 < 1e-3）
8. **结构特征全覆盖**：541/541 (100%) 蛋白有结构特征（contact_number + protrusion_index），包括 5 个新补充的蛋白
9. **口袋几何特征已实现**：PocketEncoder (Gaussian-kernel 距离加权消息传递) 已集成，541/541 (100%) 蛋白有口袋特征
10. **pKd 范围约束**：`pkd = 2.0 + 13.0 * sigmoid(raw)`，输出约束到 [2, 15]（与数据集 pKd 范围一致）
11. **损失函数简化**：`L_total = L_pkd + L_kcat`，无物理约束/OT正则化，λ 预测头保留但不参与损失

## 已知问题与待办

| 优先级 | 问题 | 说明 |
|--------|------|------|
| 🔴 高 | kcat 无底物级变化 | 同一蛋白的所有底物共享一个 kcat，无法做底物级 kcat 排序。pKd 有底物级变化，当前用它作排序信号 |
| 🟡 中 | ChEMBL Kd 补充为 0 | 分子 InChIKey 解析失败，Kd 补充缺失 |
| 🟡 中 | 208 蛋白无 kcat (38%) | 训练时需 mask kcat 损失，共享层仍可受益 |
| 🟢 低 | 159 蛋白无辅因子注释 (29.4%) | 其中~100 真正不需要有机辅因子，~60 可能漏检 |
| 🟡 中 | CofactorEncoder PRIORS | B12/THF/COA 缺少 λ 先验值，当前用默认值 0.70 eV |

### 已修复 (2026-05-21)

| 问题 | 修复方式 |
|------|---------|
| InteractionModule attn_pc bug | `attn_pc` 改加到 `protein_h`（原误加到 `ligand_h`） |
| Marcus 物理约束不成立 | 移除 L_physics、L_OT、OTRegularizer、MarcusPhysicsLoss |
| pkd_head 无范围约束 | 加 `pkd = 2.0 + 13.0 * torch.sigmoid(pkd)`，约束到 [2, 15] |
| IC50 权重过高 | thermo_weight 降为 0.15（Kd=1.0, Ki=0.7），乘入 quality_weight |

## 关键数据发现（2026-05-21 诊断）

- **kcat 无底物级变化**：329 个蛋白每个只有 1 个 kcat 值，重复于所有底物。kcat 是蛋白级属性，不能做底物级排序
- **Marcus 方程不成立**：100% 蛋白的 `kcat_true / kcat_marcus ≈ 10⁻⁶`。ET 步速率远高于实际 kcat，限速步在别处（产物释放、构象变化等）
- **pKd 是底物级排序信号**：同一蛋白内不同底物的 pKd 有丰富变化（50~1700 unique per protein）
- **基线验证**：3 蛋白 × 3000 底物，纯 ESM-2+GNN 排序模型 Spearman=0.53（随机=0），核心能力成立

## 原始 Pipeline 概述（dataset_building/pipeline/）

8 个脚本构成端到端数据处理流水线，将 PDBbind + BindingDB 合并为统一的酶-底物结合亲和力数据集：

| 脚本 | 功能 |
|------|------|
| `01_parse_pdbbind.py` | 解析 PDBbind：索引→标签表、蛋白质序列、结合位点、接触图、配体 SMILES |
| `02_parse_bindingdb.py` | 解析 BindingDB：TSV 过滤、亲和力类型选择、UniProt 注释、去重 |
| `03_align_distributions.py` | GMM 分布分析 → 计算 w_multiplier |
| `04_build_ligand_graphs.py` | SMILES → PyG 分子图（79 维原子特征 + 10 维键特征） |
| `05_write_storage.py` | 最终存储：HDF5 (蛋白) + Parquet (元数据) + split |
| `06_dataset.py` | PyTorch Dataset + DataLoader |
| `07_run_pipeline.py` | 端到端编排器，支持 --resume |
| `08_make_splits.py` | 按蛋白序列哈希划分 80/10/10 |

运行命令：

```bash
cd /home/domi/BINN/AI4enz/dataset_building
python pipeline/07_run_pipeline.py --workers 8       # 完整运行
python pipeline/07_run_pipeline.py --resume           # 断点续跑
python pipeline/07_run_pipeline.py --no-uniprot       # 跳过 UniProt API
```

## 代码仓库与数据分发

- **GitHub 私有仓库**：https://github.com/Domi-Joe/AI4enz（仅代码，不含数据集）
- **数据集**：`dataset_building/ai4enz_dataset.tar.gz` (~86MB)，包含训练所需全部数据
  - proteins.h5 (541蛋白特征) + 57,202 配体 .pt + unified_metadata.parquet (78,113条)
  - 通过云盘/线下传输发给协作者，解压到 `processed/` 即可训练
- BINN/BINN/ 是上游包，不要直接修改
- Pipeline 默认自动检测 `data/` 目录（相对脚本位置 `../data/`）
- ESM-2 模型通过 HF mirror 下载：`HF_ENDPOINT=https://hf-mirror.com`
- proteins.h5 被 h5py 以读写模式打开时会独占锁定，避免同时多进程写入
- 原始数据文件极大（BindingDB_All.tsv ~8.8 GB），已在 .gitignore 排除
