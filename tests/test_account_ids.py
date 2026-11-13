import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("account_ids", Path(__file__).resolve().parents[1] / "src/account_ids.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AccountIdTests(unittest.TestCase):
    def test_integer_ids_preserve_order_and_identity(self):
        np.testing.assert_array_equal(module.account_id_array([900, 12, 65]), [900, 12, 65])

    def test_string_object_index_roundtrips_without_pickle(self):
        original = np.array(["009", "002", "account-z"], dtype=object)
        ids = module.account_id_array(original)
        self.assertEqual(ids.dtype.kind, "U")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "node_ids.npy"
            np.save(path, ids, allow_pickle=False)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), original)

    def test_invalid_ids_are_rejected(self):
        cases = [[], [[1, 2]], [1, 1], [" "], [1.5], [float("nan")],
                 np.array(["one", None], dtype=object), np.array(["one", 2], dtype=object),
                 np.array([b"009"])]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                module.account_id_array(values)


if __name__ == "__main__":
    unittest.main()
