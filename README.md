# AI4enz — Enzyme Mining and Activity Prediction

基于过渡态理论的酶-底物结合亲和力（pKd）与催化效率（kcat/KM）预测模型。

## Quick Start

```bash
cd dataset_building

# 使用高质量 Kd/Ki 数据训练（推荐）
python train.py --unified-metadata processed/oxidoreductase/high_quality_kd_ki_v2.parquet \
  --epochs 50 --batch-size 64 --device cuda

# 快速验证 (CPU)
python train.py --unified-metadata processed/oxidoreductase/high_quality_kd_ki_v2.parquet \
  --epochs 10 --batch-size 32 --max-samples 5000 --device cpu
```

## Architecture

**TransitionBINN** — Hybrid 双路径设计：
- **pKd 路径**：Ligand GNN + Protein (ESM-2 + 口袋结构) + Cofactor → pKd
- **kcat 路径**：Protein ESM-2 + Cofactor → log₁₀(kcat)
- **Score**：pKd + log_kcat = log₁₀(kcat/KM)

## Dataset (v2)

| Metric | Value |
|--------|-------|
| Total samples | **322,763** |
| High-quality (Kd/Ki) | **163,927** |
| Proteins | **11,044** |
| Ligands | **141,941** |

数据来源：CatPred-DB, OED, SKiD, BindingDB（详见 CLAUDE.md）

## Project Structure

```
AI4enz/
├── CLAUDE.md                    # 项目文档
├── README.md                    # 本文件
└── dataset_building/            # 所有代码与数据
    ├── ranking_model.py         # TransitionBINN 模型
    ├── train.py                 # 训练
    ├── inference_enzyme_mining.py # 推理
    ├── external_data/           # 外部数据源
    └── processed/               # 处理输出
```

## Requirements

- PyTorch ≥ 2.0 + PyTorch Geometric
- ESM-2 (transformers)
- RDKit
