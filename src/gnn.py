import os
import polars as pl
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# configuration for paths and device
DATA_DIR = "../DATA"
FEATURE_DIR = "../features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EDGES = 2_000_000

print("Device:", DEVICE)
print("Working directory:", os.getcwd())

# load transaction data to construct graph edges
print("Loading transactions")

txn = pl.scan_parquet(f"{DATA_DIR}/transactions_batch-*/*.parquet")

edges = txn.select([
    "account_id",
    "counterparty_id"
]).collect()

if len(edges) > MAX_EDGES:
    edges = edges.sample(MAX_EDGES)

edges = edges.to_pandas()

print("Transactions loaded:", edges.shape)

# create node list and index mapping
nodes = pd.unique(
    edges[["account_id", "counterparty_id"]].values.ravel()
)

node_map = {n: i for i, n in enumerate(nodes)}

edges["src"] = edges["account_id"].map(node_map)
edges["dst"] = edges["counterparty_id"].map(node_map)

# create edge tensor for pytorch geometric
edge_index = torch.tensor(
    edges[["src", "dst"]].values.T,
    dtype=torch.long
)

print("Total nodes:", len(nodes))
print("Total edges:", edge_index.shape)

# load node feature table
print("Loading node features")

features = pd.read_parquet(f"{FEATURE_DIR}/txn_features.parquet")
features = features.set_index("account_id")

feature_cols = features.columns

# create feature matrix
X = torch.zeros((len(nodes), len(feature_cols)))

# map features to node indices
for acc, row in features.iterrows():
    if acc in node_map:
        X[node_map[acc]] = torch.tensor(row.values)

# normalize feature matrix
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X), dtype=torch.float)

print("Feature matrix:", X.shape)

# load mule labels
print("Loading labels")

labels = pd.read_parquet(f"{DATA_DIR}/train_labels.parquet")

y = torch.full((len(nodes),), -1)

# assign labels to nodes
for _, row in labels.iterrows():

    acc = row["account_id"]

    if acc in node_map:
        y[node_map[acc]] = row["is_mule"]

# select labeled nodes
labeled_nodes = torch.where(y >= 0)[0]

# split labeled nodes into train and validation
train_idx, val_idx = train_test_split(
    labeled_nodes.numpy(),
    test_size=0.2,
    stratify=y[labeled_nodes],
    random_state=42
)

train_mask = torch.zeros(len(nodes), dtype=torch.bool)
val_mask = torch.zeros(len(nodes), dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True

print("Train nodes:", train_mask.sum().item())
print("Validation nodes:", val_mask.sum().item())

# create pytorch geometric graph object
data = Data(
    x=X,
    edge_index=edge_index,
    y=y
)

data.train_mask = train_mask
data.val_mask = val_mask

data = data.to(DEVICE)

# define graphsage based gnn model
class GNN(torch.nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.conv1 = SAGEConv(in_channels, 128)
        self.conv2 = SAGEConv(128, 64)

        self.lin = torch.nn.Linear(64, 1)

    # forward pass through graph layers
    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.lin(x)

        return x.squeeze()

model = GNN(X.shape[1]).to(DEVICE)

# compute class imbalance weight
pos = (y == 1).sum().item()
neg = (y == 0).sum().item()

pos_weight = torch.tensor(neg / pos).to(DEVICE)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# initialize early stopping variables
best_auc = 0
patience = 20
counter = 0

print("Training started")

for epoch in range(500):

    # train model
    model.train()

    optimizer.zero_grad()

    logits = model(data)

    train_loss = criterion(
        logits[data.train_mask],
        data.y[data.train_mask].float()
    )

    train_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()

    # evaluate on validation nodes
    model.eval()

    with torch.no_grad():

        logits = model(data)

        val_logits = logits[data.val_mask]
        val_labels = data.y[data.val_mask].float()

        val_loss = criterion(val_logits, val_labels)

        probs = torch.sigmoid(val_logits).cpu().numpy()
        labels_np = val_labels.cpu().numpy()

        try:
            auc = roc_auc_score(labels_np, probs)
        except:
            auc = 0

    # check improvement in auc
    if auc > best_auc:

        best_auc = auc
        counter = 0

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/gnn_model.pt")

    else:
        counter += 1

    # stop training if auc does not improve
    if counter >= patience:
        print("Early stopping triggered")
        break

    # print training and validation loss
    print(
        f"Epoch {epoch} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}"
    )

print("Training finished")
print("Best model saved at models/gnn_model.pt")

