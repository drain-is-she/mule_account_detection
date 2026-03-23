import glob
import polars as pl
from tqdm import tqdm
import os

print("Loading transaction files...")

files = glob.glob("DATA/transactions_batch-*/part_*.parquet")

print("Total files found:", len(files))
print("Example files:", files[:3])

if len(files) == 0:
    print("ERROR: No parquet files found.")
    exit()

feature_tables = []

print("Processing transaction files...")

for f in tqdm(files):

    txn = pl.read_parquet(f)

    features = txn.group_by("account_id").agg([
        pl.len().alias("txn_count"),
        pl.col("amount").sum().alias("total_amount"),
        pl.col("amount").mean().alias("avg_amount"),
        pl.col("amount").max().alias("max_txn"),
        pl.col("counterparty_id").n_unique().alias("unique_counterparties")
    ])

    feature_tables.append(features)

print("Combining feature tables...")

combined = pl.concat(feature_tables)

txn_features = combined.group_by("account_id").agg([
    pl.col("txn_count").sum(),
    pl.col("total_amount").sum(),
    pl.col("avg_amount").mean(),
    pl.col("max_txn").max(),
    pl.col("unique_counterparties").max()
])

os.makedirs("features", exist_ok=True)

print("Saving features...")

txn_features.write_parquet("features/txn_features.parquet")

print("Transaction features saved successfully!")


# In txn_features.py — add these aggregations:
pl.col("amount").std().alias("std_amount"),
pl.col("amount").median().alias("median_amount"),
pl.col("transaction_type").n_unique().alias("unique_txn_types"),

# Temporal features (critical for mule detection):
pl.col("timestamp").max().alias("last_txn_time"),
pl.col("timestamp").min().alias("first_txn_time"),
(pl.col("timestamp").max() - pl.col("timestamp").min()).alias("account_lifetime"),
pl.col("txn_count") / pl.col("account_lifetime"),  # velocity

# Round-amount ratio (mules often transact in round numbers):
(pl.col("amount") % 100 == 0).sum().alias("round_amount_count"),
