# AI4enz - Enzyme Mining and Activity Prediction

Machine learning models for predicting enzyme-substrate binding affinity (pKd) and catalytic efficiency (kcat).

## Project Structure

```
AI4enz/
├── dataset_building/       # Dataset preparation and model training
│   ├── pipeline/          # Data processing pipeline scripts
│   └── train.py           # Main training script
└── datepre/               # Data preprocessing and feature engineering
    └── ranking_model.py   # MarcusPINN model architecture
```

## Getting Started

1. **Prepare data**: Use the pipeline scripts in `dataset_building/pipeline/` to prepare your dataset
2. **Train model**: Run `python dataset_building/train.py` to train the model
3. **Inference**: Use `inference_enzyme_mining.py` for inference

## Requirements

- PyTorch
- PyTorch Geometric (PyG)
- Pandas, NumPy
- Scikit-learn
- Other dependencies in `CLAUDE.md`

## Reference

This is a research project for enzyme mining and activity prediction.
