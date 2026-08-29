import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import os

print("Loading ensemble data...")

X = np.load("outputs/node_features.npy")
y = np.load("outputs/node_labels.npy")
train_mask = np.load("outputs/train_mask.npy")
val_mask = np.load("outputs/val_mask.npy")

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

print("Training LightGBM model...")

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc"
)

val_probs = model.predict_proba(X_val)[:,1]

auc = roc_auc_score(y_val, val_probs)

print("\nLightGBM Validation AUC:", auc)

all_probs = model.predict_proba(X)[:,1]

os.makedirs("outputs", exist_ok=True)

np.save("outputs/lgb_probs.npy", all_probs)

print("Saved outputs/lgb_probs.npy")
