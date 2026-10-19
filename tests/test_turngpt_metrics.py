import math
import unittest

from turngpt_metrics import aggregate_classification_metrics, classification_metrics


class ClassificationMetricsTest(unittest.TestCase):
    def test_threshold_ties_and_confusion(self):
        # The two tied scores at 0.5 must both count as positive.
        metrics, counts = classification_metrics(
            [0.9, 0.5, 0.5, 0.2, 0.1, 0.0], [1, 1, 0, 1, 0, 0], 0.5
        )
        self.assertEqual(counts, {"tp": 2, "fp": 1, "tn": 2, "fn": 1})
        for value in metrics.values():
            self.assertAlmostEqual(value, 2 / 3)

    def test_no_predicted_positive(self):
        metrics, counts = classification_metrics([0.1, 0.2, 0.3], [1, 0, 0], 0.5)
        self.assertEqual(counts, {"tp": 0, "fp": 0, "tn": 2, "fn": 1})
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)
        self.assertEqual(metrics["bacc"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_perfect_predictions(self):
        metrics, _ = classification_metrics([0.8, 0.9, 0.1], [1, 1, 0], 0.5)
        self.assertTrue(all(value == 1 for value in metrics.values()))

    def test_macro_weighted_and_pooled_are_distinct(self):
        summary = aggregate_classification_metrics([
            {"tp": 9, "fp": 1, "tn": 9, "fn": 1},
            {"tp": 1, "fp": 0, "tn": 90, "fn": 9},
        ])
        self.assertAlmostEqual(summary["macro_test_bacc"], 0.725)
        self.assertAlmostEqual(summary["position_weighted_test_bacc"], 73 / 120)
        self.assertAlmostEqual(summary["pooled_test_metrics"]["bacc"], 0.745)
        self.assertAlmostEqual(summary["macro_test_f1"], (0.9 + 2 / 11) / 2)
        self.assertAlmostEqual(summary["pooled_test_metrics"]["f1"], 20 / 31)
        self.assertEqual(summary["test_prediction_positions"], 120)
        self.assertEqual(summary["pooled_test_confusion_matrix"],
                         {"tp": 10, "fp": 1, "tn": 99, "fn": 10})

    def test_invalid_inputs_fail_instead_of_reporting_misleading_metrics(self):
        for scores, labels, threshold in [
            ([], [], 0.5), ([0.1], [0, 1], 0.5),
            ([0.1, math.nan], [0, 1], 0.5),
            ([0.1, 0.9], [0, 2], 0.5),
            ([0.1, 0.9], [0, 1], math.inf),
            ([0.1, 0.9], [0, 0], 0.5),
        ]:
            with self.subTest(scores=scores, labels=labels):
                with self.assertRaises(ValueError):
                    classification_metrics(scores, labels, threshold)


if __name__ == "__main__":
    unittest.main()
