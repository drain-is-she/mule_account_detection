import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import glob
import os

# define input data directory and output file path for storing generated community features
DATA_DIR = "DATA"
OUTPUT_FILE = "features/community_features.parquet"

print("Step 1: Loading transaction edges...")

# collect all parquet files containing batched transaction data
files = glob.glob(f"{DATA_DIR}/transactions_batch-*/*.parquet")

print("Found files:", len(files))

# ensure dataset is not empty to avoid silent pipeline failures
if len(files) == 0:
    raise ValueError("No parquet files found")

dfs = []

# load only relevant columns to reduce memory footprint
for f in files:
    df = pd.read_parquet(f)[["account_id", "counterparty_id"]]
    dfs.append(df)

# combine all transaction batches into a single edge list
edges = pd.concat(dfs, ignore_index=True)

print("Total edges loaded:", len(edges))

# edges 

# limit graph size for computational feasibility (Louvain is expensive on large graphs)
MAX_EDGES = 2_000_000

# randomly sample edges if dataset exceeds threshold
if len(edges) > MAX_EDGES:
    edges = edges.sample(MAX_EDGES, random_state=42)

print("Edges used for graph:", len(edges))

# building the graph 
print("\nStep 2: Building graph...")

# initialize undirected graph where nodes = accounts and edges = transactions
G = nx.Graph()

# add edges efficiently from dataframe (tuple format avoids overhead)
G.add_edges_from(
    edges[["account_id", "counterparty_id"]].itertuples(index=False, name=None)
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# community detection 

print("\nStep 3: Running Louvain...")

# apply Louvain algorithm to detect densely connected account clusters (potential fraud rings)
partition = community_louvain.best_partition(G, resolution=2.5)

# count distinct communities formed
num_communities = len(set(partition.values()))

print("Communities detected:", num_communities)

# build feature map 

print("\nStep 4: Building community feature table...")

# convert partition dictionary into structured dataframe
community_df = pd.DataFrame({
    "account_id": list(partition.keys()),
    "community_id": list(partition.values())
})

# compute size of each community (useful fraud signal: large dense clusters)
sizes = community_df.groupby("community_id").size()

# map community size back to each account
community_df["community_size"] = community_df["community_id"].map(sizes)



# ensure output directory exists before saving
os.makedirs("features", exist_ok=True)

# save features in parquet format for efficient downstream processing
community_df.to_parquet(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print(community_df.head())
