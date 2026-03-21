import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob

# define data paths and ensure output directory exists
DATA_DIR = "DATA"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRANSACTION_BATCHES = [
    "transactions_batch-1",
    "transactions_batch-2",
    "transactions_batch-3",
    "transactions_batch-4"
]

# load all transaction parquet files from multiple batches into a single dataframe
print("Loading transactions...")
dfs = []

for batch in TRANSACTION_BATCHES:
    batch_path = os.path.join(DATA_DIR, batch)

    # recursively find all parquet files in the batch folder
    parquet_files = glob.glob(os.path.join(batch_path, "**/*.parquet"), recursive=True)

    print(f"Loading {batch} ({len(parquet_files)} files)")

    for file in parquet_files:
        df_part = pd.read_parquet(file)
        dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

print("Total transactions loaded:", len(df))
print("Columns:", df.columns)

# rename columns for consistency and easier graph interpretation (source → destination)
df = df.rename(columns={
    "account_id": "src_id",
    "counterparty_id": "dst_id",
    "transaction_timestamp": "timestamp"
})

# compute in-degree and out-degree as basic graph connectivity features
print("Computing degree features...")
out_degree = df.groupby("src_id").size().rename("out_degree")
in_degree = df.groupby("dst_id").size().rename("in_degree")
degree_features = pd.concat([out_degree, in_degree], axis=1).fillna(0)

# aggregate transaction amount statistics separately for sent and received money
print("Computing amount statistics...")
sent_amount = df.groupby("src_id")["amount"].agg(["sum", "mean", "max", "min", "std"])
sent_amount.columns = ["sent_sum", "sent_mean", "sent_max", "sent_min", "sent_std"]

recv_amount = df.groupby("dst_id")["amount"].agg(["sum", "mean", "max", "min", "std"])
recv_amount.columns = ["recv_sum", "recv_mean", "recv_max", "recv_min", "recv_std"]

amount_features = pd.concat([sent_amount, recv_amount], axis=1).fillna(0)

# count unique counterparties to capture diversity of interactions (fan-in / fan-out behavior)
print("Computing counterparty features...")
unique_receivers = df.groupby("src_id")["dst_id"].nunique().rename("unique_receivers")
unique_senders = df.groupby("dst_id")["src_id"].nunique().rename("unique_senders")

counterparty_features = pd.concat(
    [unique_receivers, unique_senders], axis=1
).fillna(0)

# extract temporal behavior by counting distinct active hours for each account
print("Computing time features...")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
active_hours = df.groupby("src_id")["hour"].nunique().rename("active_hours")

# merge all engineered features into a single feature matrix
print("Combining features...")
features = pd.concat(
    [
        degree_features,
        amount_features,
        counterparty_features,
        active_hours
    ],
    axis=1
).fillna(0)

# create a continuous node_id mapping required for graph-based models (e.g., GNN indexing)
print("Creating node ids...")
nodes = pd.Index(features.index)
node_to_id = {node: i for i, node in enumerate(nodes)}

features["node_id"] = features.index.map(node_to_id)
features = features.sort_values("node_id")

# normalize features to standard scale for stable model training
print("Normalizing features...")
scaler = StandardScaler()
feature_cols = features.columns.drop("node_id")
features[feature_cols] = scaler.fit_transform(features[feature_cols])

# save features in both numpy and parquet formats for downstream model consumption
print("Saving features...")
np.save(
    os.path.join(OUTPUT_DIR, "node_features.npy"),
    features[feature_cols].values
)

np.save(
    os.path.join(OUTPUT_DIR, "node_ids.npy"),
    features["node_id"].values
)

features.to_parquet(
    os.path.join(OUTPUT_DIR, "node_features.parquet"),
    index=False
)

print("Feature engineering complete.")
