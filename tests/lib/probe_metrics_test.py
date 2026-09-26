import math
import unittest

from src.lib import probe_metrics as pm


class TestAuroc(unittest.TestCase):
    """auroc: positive class = unfaithful (label 0), score = p_unfaithful."""

    def test_perfect_separation(self):
        labels = [0, 0, 1, 1]
        self.assertEqual(pm.auroc(labels, [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertEqual(pm.auroc(labels, [0.1, 0.2, 0.8, 0.9]), 0.0)

    def test_ties_count_half(self):
        self.assertAlmostEqual(pm.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_hand_computed_value(self):
        labels = [0, 0, 1, 1, 1]
        scores = [0.9, 0.3, 0.5, 0.2, 0.1]
        # pairs (unf, fai): 0.9 beats all three; 0.3 beats 0.2 and 0.1 -> 5 of 6
        self.assertAlmostEqual(pm.auroc(labels, scores), 5 / 6)

    def test_single_class_is_nan(self):
        self.assertTrue(math.isnan(pm.auroc([1, 1], [0.2, 0.3])))


class TestRecallAtFpr(unittest.TestCase):
    """threshold_at_fpr / recall_at_fpr."""

    labels = [1, 1, 1, 1, 0, 0]
    scores = [0.1, 0.2, 0.3, 0.4, 0.35, 0.9]

    def test_budget_respected(self):
        self.assertAlmostEqual(pm.threshold_at_fpr(self.labels, self.scores, 0.25), 0.3)
        self.assertAlmostEqual(pm.recall_at_fpr(self.labels, self.scores, 0.25), 1.0)

    def test_zero_fpr(self):
        self.assertAlmostEqual(pm.threshold_at_fpr(self.labels, self.scores, 0.0), 0.4)
        self.assertAlmostEqual(pm.recall_at_fpr(self.labels, self.scores, 0.0), 0.5)

    def test_fpr_one_allows_everything(self):
        self.assertEqual(pm.threshold_at_fpr(self.labels, self.scores, 1.0), -math.inf)
        self.assertEqual(pm.recall_at_fpr(self.labels, self.scores, 1.0), 1.0)

    def test_no_faithful_is_nan(self):
        self.assertTrue(math.isnan(pm.threshold_at_fpr([0, 0], [0.1, 0.2], 0.1)))


class TestCalibration(unittest.TestCase):
    """expected_calibration_error."""

    def test_confident_and_correct_is_near_zero(self):
        ece = pm.expected_calibration_error([0, 0, 1, 1], [0.99, 0.99, 0.01, 0.01])
        self.assertAlmostEqual(ece, 0.01, places=6)

    def test_confident_and_wrong_is_large(self):
        ece = pm.expected_calibration_error([0, 0, 1, 1], [0.01, 0.01, 0.99, 0.99])
        self.assertAlmostEqual(ece, 0.99, places=6)


class TestSummarize(unittest.TestCase):
    """summarize."""

    def test_summary_keys_and_counts(self):
        s = pm.summarize([0, 1, 0, 1], [0.8, 0.3, 0.6, 0.7], loss=0.42)
        for k in ("n", "n_unfaithful", "n_faithful", "loss", "accuracy", "balanced_accuracy", "auroc",
                  "precision_unfaithful", "recall_unfaithful", "precision_faithful", "recall_faithful",
                  "recall_at_1pct_fpr", "recall_at_5pct_fpr", "threshold_at_1pct_fpr", "ece"):
            self.assertIn(k, s)
        self.assertEqual((s["n"], s["n_unfaithful"], s["n_faithful"]), (4, 2, 2))
        self.assertEqual(s["loss"], 0.42)

    def test_precision_recall_convention(self):
        # predicted unfaithful: p > 0.5 -> rows 0, 2, 3; true unfaithful: rows 0, 2
        s = pm.summarize([0, 1, 0, 1], [0.8, 0.3, 0.6, 0.7])
        self.assertAlmostEqual(s["precision_unfaithful"], 2 / 3)
        self.assertAlmostEqual(s["recall_unfaithful"], 1.0)
        self.assertAlmostEqual(s["recall_faithful"], 0.5)
        self.assertAlmostEqual(s["accuracy"], 0.75)
        self.assertAlmostEqual(s["balanced_accuracy"], 0.75)

    def test_empty_input(self):
        s = pm.summarize([], [])
        self.assertEqual(s["n"], 0)
        self.assertNotIn("auroc", s)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pm.auroc([0, 1], [0.5])


if __name__ == "__main__":
    unittest.main()
