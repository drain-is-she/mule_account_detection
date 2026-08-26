# GNN Mule Detection

Detects money mule accounts in a transaction network using a two-layer GraphSAGE model ensembled with LightGBM, on RBI-sourced transaction data (~20 lakh nodes).

Built for Razorpay's **AI Risk Manager** buildathon track. This is a **defense-only** risk scorer — it flags accounts for human review. It does not freeze, block, or take any autonomous action on an account.

## The problem

Money mule accounts move illicit funds through accounts that, on the surface, look like anyone else's — plausible balances, unremarkable transaction sizes, no single flag that gives them away. Catching them means separating a small, camouflaged population from a much larger population of ordinary customers, without punishing the ordinary customers in the process.

## What we tested and rejected

Before building the model, we ran a systematic audit of 25 candidate behavioral indicators against the transaction data, to find out which commonly-assumed mule "tells" actually hold up. Most did not:

| Indicator | Verdict | Why it fails |
|---|---|---|
| Round-number transactions | Rejected | Legitimate ATM withdrawals produce the same pattern |
| Night-time activity | Rejected | Both groups transact overwhelmingly during daytime hours |
| Branch hotspots (raw counts) | Rejected | Reflects branch size, not risk — needs to be a percentage |
| Fan-in / fan-out transaction counts | Rejected | Mules are fully camouflaged inside the legitimate-user cloud |
| Dormancy-before-activation | Rejected | Both groups cluster in the same 0–100 day window |
| Rapid pass-through holding time | Rejected | Legitimate accounts show *higher* density of instant transfers |

Three signals survived scrutiny and form the backbone of the feature set:

1. **10-day burnout trajectory** — legitimate accounts show steady, log-linear volume growth; mule accounts show a near-vertical spike in the first ~10 days followed by near-total dormancy.
2. **Daily credit/debit synchronicity ("mirror effect")** — measured via Pearson correlation between daily credit and debit aggregates, isolating accounts that behave as pass-through pipes rather than stores of value.
3. **Threshold capping** — mule activity drops off sharply near regulatory reporting thresholds, rather than flowing past them the way legitimate high-value activity does.

## How it works

Transactions are modelled as a graph: accounts are nodes, transactions are edges. A GraphSAGE model learns a mule probability per node from its own features plus its neighbours'. A LightGBM model is trained separately on the same tabular features. The two probability outputs are blended with a weight found by grid search on the validation split.

## Pipeline

Run in this order:

1. **Feature engineering** — three parallel feature tables built from raw transactions:
   - `src/txn_features.py` → per-account aggregates (txn count, total/avg/max amount, unique counterparties) → `features/txn_features.parquet`
   - `src/graph_features.py` → in/out-degree, PageRank, clustering coefficient, fan-in/out ratio → `features/graph_features.parquet`
   - `src/community_features.py` → Louvain community id + size, a signal for dense fraud-ring-like clusters → `features/community_features.parquet`
   - `src/feature_engineering.py` → independent aggregate set (degree, sent/received amount stats, counterparty diversity, active-hour spread) → `features/node_features.npy`
2. **GraphSAGE training** — `src/gnn.py` builds the graph, trains with early stopping, saves `models/gnn_model.pt`.
3. **LightGBM training** — `src/train_lgb.py` trains on the tabular feature matrix.
4. **Ensembling** — `src/ensemble.py` grid-searches the GNN/LightGBM blend weight on the validation split.
5. **Evaluation & scoring** — `src/evaluate.py` reports AUC, best-F1 threshold, precision/recall; `src/predict.py` writes the final ranked CSV.

> Graph and community features are computed but not yet merged into either model's input — see **Limitations** below.

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
| Train/val split | 80/20 stratified random (see Limitations) |

## Results

**Validation summary**

| Model | AUC | Best F1 | Optimal threshold |
|---|---|---|---|
| GraphSAGE only | 0.9085 | 0.6778 | 0.352 |
| LightGBM only | 0.9113 | 0.7206 | 0.508 |
| Ensemble (saved / baseline) | 0.9113 | 0.7206 | 0.508 |
| **Ensemble (5% GNN + 95% LGB)** | **0.9172** | **0.7239** | **0.500** |

**Detailed metrics, best model (5% GNN + 95% LGB)**

| Metric | @ Best-F1 threshold | @ 0.5 |
|---|---|---|
| Precision | 0.834 | 0.834 |
| Recall | 0.639 | 0.639 |
| F1 | 0.724 | 0.724 |

**What this means in practice:** at this threshold, the model catches ~64% of mule accounts, and roughly 83% of flagged accounts are correctly identified — meaning about 1 in 6 flags would be a legitimate account sent for review, not blocked outright. The GraphSAGE component contributes a modest, consistent lift (+0.0059 AUC over LightGBM alone) rather than carrying detection on its own — tabular behavioral features currently do most of the separating work; graph structure adds an independent signal on top.

**Caveat:** the validation split above is random, not temporal. For a fraud-detection task this can inflate AUC relative to a live deployment, where the model only ever has the past to predict the future. Treat these numbers as an upper bound pending re-validation on a time-based split (see below).

## Limitations & next steps

- **Split is random, not temporal.** A time-based split is needed to get an honest read on deployment performance.
- **Node ordering is not yet shared across scripts.** `gnn.py` and `feature_engineering.py` each derive their own account→node-index mapping independently; row-aligning their outputs (as `train_lgb.py` and `ensemble.py` currently do) risks silently mismatching features/labels/predictions. Fix: persist one canonical `account_id → node_id` mapping and join against it everywhere.
- **Graph and community features aren't merged into model input yet.** `graph_features.py` and `community_features.py` compute real signal (PageRank, clustering coefficient, Louvain community) that neither model currently consumes.
- **Planned feature additions:** transaction entropy, in-out time-delta variance, node degree centrality, velocity of exhaustion, first-transaction magnitude ratio, window-based burstiness score.

## Scope

This system produces a risk score for human review. It does not freeze, block, or take any automated action against an account.
