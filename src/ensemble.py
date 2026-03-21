
import numpy as np
from sklearn.metrics import roc_auc_score

print("Loading predictions")

gnn = np.load("outputs/gnn_probs.npy")
lgb = np.load("outputs/lgb_probs.npy")

y = np.load("outputs/node_labels.npy")
val_mask = np.load("outputs/val_mask.npy")

y_val = y[val_mask]

best_auc = 0
best_w = 0

print("Searching best ensemble weight...")

for w in np.linspace(0,1,21):

    ensemble = w * gnn + (1-w) * lgb

    auc = roc_auc_score(y_val, ensemble[val_mask])

    if auc > best_auc:
        best_auc = auc
        best_w = w

print("\nBest weight for GNN:", best_w)
print("Best weight for LGB:", 1-best_w)
print("Best Ensemble AUC:", best_auc)

final = best_w * gnn + (1-best_w) * lgb

np.save("outputs/ensemble_probs.npy", final)

print("Saved outputs/ensemble_probs.npy")
