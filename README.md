# GNN Mule Detection

Detects money mule accounts in transaction networks using a GraphSAGE neural network, ensembled with LightGBM.

# How it works

Transaction data is modelled as a graph accounts are nodes, transactions are edges. A two-layer GraphSAGE model learns a mule probability per node by aggregating each account's own features with those of its transaction neighbours. The GNN output is then blended with LightGBM predictions, with the optimal blend weight found by grid search on the validation set.

# Usage
# Train the GNN
python train_gnn.py

# Find best ensemble blend and save final predictions
python ensemble.py

# Requirements

torch  , torch-geometric , polars , pandas , scikit-learn ,lightgbm , numpy

# Key config

| Parameter | Default |
|-----------|---------|
| Max edges | 2,000,000 |
| Hidden dims | 128 → 64 |
| Dropout | 0.3 |
| Epochs | 500 |
| Optimizer | Adam lr=0.001 |

## Outputs

| File | Description |
|------|-------------|
| `models/gnn_model.pt` | Saved model weights |
| `outputs/gnn_probs.npy` | GNN scores per node |
| `outputs/ensemble_probs.npy` | Final blended predictions |
