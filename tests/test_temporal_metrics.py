import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "temporal_metrics", Path(__file__).resolve().parents[1] / "src/temporal_metrics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
iou = module.temporal_interval_iou


def time(hour):
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)


class TemporalIoUTests(unittest.TestCase):
    def test_disjoint_intervals_are_zero_in_both_orders(self):
        self.assertEqual(iou(time(0), time(1), time(3), time(4)), 0)
        self.assertEqual(iou(time(3), time(4), time(0), time(1)), 0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(iou(time(0), time(4), time(2), time(6)), 1 / 3)

    def test_identical_and_contained_intervals(self):
        self.assertEqual(iou(time(0), time(4), time(0), time(4)), 1)
        self.assertEqual(iou(time(0), time(4), time(1), time(3)), 0.5)

    def test_touching_and_zero_duration(self):
        self.assertEqual(iou(time(0), time(2), time(2), time(4)), 0)
        self.assertEqual(iou(time(1), time(1), time(1), time(1)), 0)
        self.assertEqual(iou(time(0), time(2), time(1), time(1)), 0)

    def test_each_missing_boundary(self):
        for index in range(4):
            values = [time(0), time(4), time(2), time(6)]
            values[index] = None
            with self.subTest(index=index):
                self.assertEqual(iou(*values), 0)

    def test_reversed_interval_is_invalid(self):
        for values in [(2, 1, 0, 4), (0, 4, 2, 1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                iou(*(time(value) for value in values))

    def test_symmetry_and_bounds(self):
        for a in range(4):
            for b in range(a, 4):
                for c in range(4):
                    for d in range(c, 4):
                        result = iou(time(a), time(b), time(c), time(d))
                        self.assertGreaterEqual(result, 0)
                        self.assertLessEqual(result, 1)
                        self.assertEqual(result, iou(time(c), time(d), time(a), time(b)))


if __name__ == "__main__":
    unittest.main()
