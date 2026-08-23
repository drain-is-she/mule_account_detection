# GNN Mule Detection

Detects money mule accounts in a transaction network using a two-layer GraphSAGE
model ensembled with LightGBM, on RBI-sourced transaction data (~20 lakh nodes)

## How it works

Transactions are modelled as a graph: accounts are nodes, transactions are edges.
A GraphSAGE model learns a mule probability per node from its own features plus
its neighbours'. A LightGBM model is trained separately on the same tabular
features. The two probability vectors are blended with a weight found by grid
search on the validation split.

## Pipeline

Run in this order:

1. **Feature engineering** — three parallel feature tables built from raw transactions:
   - `src/txn_features.py` → per-account aggregates (txn count, total/avg/max
     amount, unique counterparties) → `features/txn_features.parquet`
   - `src/graph_features.py` → in/out-degree, PageRank, clustering coefficient,
     fan-in/out ratio → `features/graph_features.parquet`
   - `src/community_features.py` → Louvain community id + community size, a
     signal for dense fraud-ring-like clusters → `features/community_features.parquet`
   - `src/feature_engineering.py` → independent aggregate set (degree, sent/received
     amount stats, counterparty diversity, active-hour spread) → `features/node_features.npy`
2. **GraphSAGE training** — `src/gnn.py` builds the graph, trains with early
   stopping, saves `models/gnn_model.pt`.
3. **LightGBM training** — `src/train_lgb.py` trains on the tabular feature matrix.
4. **Ensembling** — `src/ensemble.py` grid-searches the GNN/LightGBM blend weight
   on the validation split.
5. **Evaluation & scoring** — `src/evaluate.py` reports AUC, best-F1 threshold,
   precision/recall, and a temporal-overlap check; `src/predict.py` writes the
   final ranked CSV.

> **Known gap:** `graph_features.parquet` and `community_features.parquet` are
> computed but not currently merged into either model's input — see
> [Known issues](#known-issues) before relying on results.

## Setup

```bash
pip install -r requirements.txt
```

Expected data layout:

```
DATA/
  transactions_batch-1/part_*.parquet   # account_id, counterparty_id, amount, transaction_timestamp
  transactions_batch-2/...
  transactions_batch-3/...
  transactions_batch-4/...
  train_labels.parquet                  # account_id, is_mule
```

## Usage

```bash
python src/txn_features.py
python src/graph_features.py
python src/community_features.py
python src/feature_engineering.py

python src/gnn.py
python src/train_lgb.py
python src/ensemble.py
python src/evaluate.py
python src/predict.py
```

## Key config

| Parameter | Value |
|---|---|
| Max edges sampled | 2,000,000 |
| GraphSAGE hidden dims | 128 → 64 |
| Dropout | 0.3 |
| Epochs (early-stop patience 20) | 500 |
| Optimizer | Adam, lr=0.001 |
| LightGBM | 1000 trees, lr=0.03, num_leaves=64, class_weight=balanced |
| Train/val split | 80/20 stratified random (see Known issues) |

## Outputs

| File | Description |
|---|---|
| `models/gnn_model.pt` | Saved GraphSAGE weights |
| `outputs/gnn_probs.npy` | GNN mule probability per node |
| `outputs/lgb_probs.npy` | LightGBM mule probability per node |
| `outputs/ensemble_probs.npy` | Final blended predictions |
| `outputs/final_predictions.csv` | Ranked, thresholded account list |

## Results

| Model | Validation AUC |
|---|---|
| GraphSAGE only | — |
| LightGBM only | — |
| Ensemble | — |

*Fill in from your latest `evaluate.py` run.*

## Known issues

- **Node ordering isn't shared across scripts.** `gnn.py` and
  `feature_engineering.py` each derive their own account→node-index mapping
  independently, in different orders. Loading their outputs as row-aligned
  arrays (as `train_lgb.py` and `ensemble.py` currently do) risks silently
  mismatching features/labels/predictions between accounts. Fix: persist one
  canonical `account_id → node_id` mapping and have every script join against it.
- **Missing save calls in `gnn.py`.** `train_lgb.py`, `ensemble.py`,
  `evaluate.py`, and `predict.py` expect `outputs/node_labels.npy`,
  `outputs/train_mask.npy`, `outputs/val_mask.npy`, `outputs/node_ids.npy`, and
  `outputs/gnn_probs.npy`, none of which `gnn.py` currently writes.
- **Graph/community features aren't merged into the model input.**
  `graph_features.py` and `community_features.py` compute real signal that
  neither `gnn.py` nor `train_lgb.py` currently consumes.
- **Dead code in `txn_features.py`.** Draft aggregations (std/median amount,
  temporal velocity, round-amount ratio) sit below the save call, never wired
  into the actual `.agg()` call.
- **Random split, not temporal.** The 80/20 split in `gnn.py` is stratified
  random. For mule detection this can let the model implicitly see
  future-dated behavior for accounts whose labels reflect later-discovered
  fraud, inflating validation AUC relative to a real deployment setting. A
  time-based split is more representative
