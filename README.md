# AI4enz - Enzyme Mining and Activity Prediction

Machine learning models for predicting enzyme-substrate binding affinity (pKd) and catalytic efficiency (kcat/KM).

## Quick Start

```bash
cd dataset_building

# Train with high-quality Kd/Ki data (recommended)
python train.py --unified-metadata processed/oxidoreductase/high_quality_kd_ki.parquet \
  --epochs 50 --batch-size 64 --device cuda

# Quick test (CPU)
python train.py --unified-metadata processed/oxidoreductase/high_quality_kd_ki.parquet \
  --epochs 10 --batch-size 32 --max-samples 5000 --device cpu
```

## Project Structure

```
AI4enz/
├── datepre/
│   └── ranking_model.py   # Hybrid TransitionBINN model
└── dataset_building/
    ├── train.py           # Training script
    ├── inference_enzyme_mining.py  # Inference
    └── processed/         # Dataset (get separately)
```

## Architecture

TransitionBINN with dual-pathway design:
- **pKd pathway**: Ligand GNN + Protein structure + Cofactor → pKd
- **kcat predictor**: ESM-2 + Cofactor → log₁₀(kcat)
- **score**: pKd + log_kcat = log₁₀(kcat/KM)

## Dataset

| Metric | Value |
|--------|-------|
| Total samples | 78,113 |
| High-quality (Kd/Ki) | 6,184 |
| Proteins | 541 |
| Ligands | 57,203 |

## Requirements

- PyTorch ≥ 2.0
- PyTorch Geometric
- ESM-2 (transformers)
- RDKit

See [CLAUDE.md](CLAUDE.md) for details.