import numpy as np
import pandas as pd

probs = np.load("outputs/ensemble_probs.npy")

# load account ids
node_features = np.load("outputs/node_features.npy")

# we saved mapping inside gnn training earlier
# but we can reconstruct account ids from labels file

labels = pd.read_parquet("DATA/train_labels.parquet")

accounts = labels["account_id"].values

# build dataframe
df = pd.DataFrame({
    "account_id": accounts,
    "mule_probability": probs[:len(accounts)]
})

# sort highest risk first
df = df.sort_values("mule_probability", ascending=False)

# flag accounts above threshold
threshold = 0.5
df["is_predicted_mule"] = (df["mule_probability"] > threshold).astype(int)

df.to_csv("outputs/final_predictions.csv", index=False)

print("Saved outputs/final_predictions.csv")
print(df.head(10))
