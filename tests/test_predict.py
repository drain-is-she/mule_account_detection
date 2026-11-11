import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "src/predict.py"
spec = importlib.util.spec_from_file_location("predict", SCRIPT)
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


class PredictionTests(unittest.TestCase):
    def test_identity_survives_ranking(self):
        rows = predict.build_predictions(["account-z", "account-a", "account-b"], [0.2, 0.9, 0.5])
        self.assertEqual([r["account_id"] for r in rows], ["account-a", "account-b", "account-z"])
        self.assertEqual([r["is_predicted_mule"] for r in rows], [1, 1, 0])

    def test_equal_scores_preserve_input_order(self):
        self.assertEqual([r["account_id"] for r in predict.build_predictions([9, 2], [0.8, 0.8])], [9, 2])

    def test_threshold_equality_matches_evaluation(self):
        for threshold in [0.0, 0.5, 0.8, 1.0]:
            with self.subTest(threshold=threshold):
                scores = [0.0, 0.49, 0.5, 0.79, 0.8, 1.0]
                rows = predict.build_predictions(list(range(len(scores))), scores, threshold)
                flags = {row["account_id"]: row["is_predicted_mule"] for row in rows}
                self.assertEqual([flags[i] for i in range(len(scores))],
                                 [int(score >= threshold) for score in scores])

    def test_invalid_inputs(self):
        cases = [([], []), ([1], [0.1, 0.2]), ([[1]], [0.2]), ([1], [[0.2]]),
                 ([1, 1], [0.2, 0.3]), ([""], [0.2]), ([1.5], [0.2]), ([b"007"], [0.2]),
                 ([1], [float("nan")]), ([1], [float("inf")]), ([1], [-0.1]), ([1], [1.1])]
        for accounts, scores in cases:
            with self.subTest(accounts=accounts, scores=scores), self.assertRaises(ValueError):
                predict.build_predictions(accounts, scores)
        for threshold in [-1, 2, float("nan")]:
            with self.assertRaises(ValueError):
                predict.build_predictions([1], [0.2], threshold)

    def test_cli_exports_explicit_mapping_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "ids.npy", np.array(["007", "002"]))
            np.save(root / "scores.npy", [0.1, 0.8])
            output = root / "nested/predictions.csv"
            args = [sys.executable, str(SCRIPT), "--account-ids", str(root / "ids.npy"),
                    "--probabilities", str(root / "scores.npy"), "--output", str(output), "--threshold", "0.8"]
            subprocess.run(args, check=True, capture_output=True)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["account_id"] for r in rows], ["002", "007"])
            self.assertEqual([r["is_predicted_mule"] for r in rows], ["1", "0"])
            original = output.read_bytes()
            np.save(root / "scores.npy", [0.8])
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_requires_mapping(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--account-ids", result.stderr)


if __name__ == "__main__":
    unittest.main()
