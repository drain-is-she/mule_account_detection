import numpy as np
import pandas as pd
import glob

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve


print("Loading predictions...")

probs = np.load("outputs/ensemble_probs.npy")
y = np.load("outputs/node_labels.npy")
val_mask = np.load("outputs/val_mask.npy")
node_ids = np.load("outputs/node_ids.npy")  # account_id list saved from training


y_val = y[val_mask]
probs_val = probs[val_mask]
#auc score for matrix evaluation 
#how well your model separates two classes

auc = roc_auc_score(y_val, probs_val)

print("\nAUC Score:", round(auc,4))


#f1 score 
#is my model good at catching fraud without falsely accusing too many normal users
precision, recall, thresholds = precision_recall_curve(y_val, probs_val)

f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

best_idx = np.argmax(f1_scores)

best_threshold = thresholds[best_idx]

print("Best Threshold:", round(best_threshold,4))



flags = probs >= best_threshold

flags_val = flags[val_mask]

precision_val = precision_score(y_val, flags_val)
recall_val = recall_score(y_val, flags_val)
f1_val = f1_score(y_val, flags_val)

print("\nPrecision:", round(precision_val,4))
print("Recall:", round(recall_val,4))
print("F1 Score:", round(f1_val,4))




print("\nComputing Temporal IoU...")
print("Loading transaction files...")

files = glob.glob("DATA/transactions_batch-*/*.parquet")

print("Files found:", len(files))

dfs = []

# sample subset to avoid huge memory usage
for f in files[:20]:

    df = pd.read_parquet(
        f,
        columns=["account_id", "transaction_timestamp"]
    )

    dfs.append(df)

transactions = pd.concat(dfs, ignore_index=True)

print("Transactions loaded:", len(transactions))


# ------------------------------------------------
# CONVERT TIMESTAMP
# ------------------------------------------------

transactions["transaction_timestamp"] = pd.to_datetime(
    transactions["transaction_timestamp"]
)



true_accounts = set(node_ids[y == 1])

true_tx = transactions[
    transactions["account_id"].isin(true_accounts)
]

#predicted mule accounts 


pred_accounts = set(node_ids[flags == 1])

pred_tx = transactions[
    transactions["account_id"].isin(pred_accounts)
]


true_start = true_tx["transaction_timestamp"].min()
true_end = true_tx["transaction_timestamp"].max()

pred_start = pred_tx["transaction_timestamp"].min()
pred_end = pred_tx["transaction_timestamp"].max()


intersection_start = max(true_start, pred_start)
intersection_end = min(true_end, pred_end)

intersection = (intersection_end - intersection_start).total_seconds()


union_start = min(true_start, pred_start)
union_end = max(true_end, pred_end)

union = (union_end - union_start).total_seconds()


temporal_iou = intersection / union if union > 0 else 0

print("\nTemporal IoU:", round(temporal_iou,4))

print("\nEvaluation Complete.")


