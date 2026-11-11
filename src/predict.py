"""Export scores only with an explicit, matching account ordering."""
import argparse
import csv
from pathlib import Path

import numpy as np


def build_predictions(accounts, probabilities, threshold=0.5):
    accounts = np.asarray(accounts)
    probabilities = np.asarray(probabilities, dtype=float)
    if accounts.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Account IDs and probabilities must be one-dimensional.")
    if len(accounts) != len(probabilities) or not len(accounts):
        raise ValueError("Account IDs and probabilities must have equal, nonzero lengths.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between 0 and 1.")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    if accounts.dtype.kind not in "iuU":
        raise ValueError("Account IDs must be integer or string arrays, without pickle objects.")
    if any(not str(account).strip() for account in accounts):
        raise ValueError("Account IDs must not be blank.")
    if len(np.unique(accounts)) != len(accounts):
        raise ValueError("Account IDs must be unique.")
    return [
        {"account_id": accounts[i].item(), "mule_probability": float(probabilities[i]),
         "is_predicted_mule": int(probabilities[i] >= threshold)}
        for i in np.argsort(-probabilities, kind="stable")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-ids", required=True,
                        help="NPY account IDs in exactly the same order as the probability array.")
    parser.add_argument("--probabilities", default="outputs/ensemble_probs.npy")
    parser.add_argument("--output", default="outputs/final_predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Flag scores greater than or equal to this value (default: 0.5).")
    args = parser.parse_args()
    rows = build_predictions(np.load(args.account_ids, allow_pickle=False),
                             np.load(args.probabilities, allow_pickle=False), args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["account_id", "mule_probability", "is_predicted_mule"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
