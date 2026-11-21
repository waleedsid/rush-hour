import unittest
from unittest.mock import patch
import numpy as np

from rush_hour.evaluation import asymmetric_cost, day_bootstrap, interval_bounds, interval_offsets, metrics
from rush_hour.models import CONFIGS, Forecaster
from rush_hour.features import NUMERIC_FEATURES
from rush_hour.study import select


def row(date, target, lag=10):
    return {**{key: 0 for key in NUMERIC_FEATURES},
            "origin": date+"T00:00:00", "target_timestamp": date+"T00:00:00",
            "target": target, "lag_24": lag, "lag_168": lag+1, "previous_day_mean": 7,
            "previous_week_mean": 8}


class EvaluationTests(unittest.TestCase):
    def test_metric_known_values_and_alignment(self):
        r = metrics([1, 3], [2, 1])
        self.assertEqual(r["mae"], 1.5)
        self.assertAlmostEqual(r["rmse"], np.sqrt(2.5))
        self.assertEqual(r["wape"], 0.75)
        self.assertIsNone(metrics([0, 0], [1, 2])["wape"])
        with self.assertRaises(ValueError):
            metrics([1, 2], [1])
        with self.assertRaises(ValueError):
            metrics([1, np.nan], [1, 2])

    def test_bootstrap_identical_predictions_zero_difference(self):
        r = day_bootstrap([1, 2, 3], [2, 4, 4], [2, 4, 4], ["a", "a", "b"], 200)
        self.assertEqual((r["low"], r["high"], r["delta_mae"]), (0, 0, 0))

    def test_day_bootstrap_weighted_units_and_reproducibility(self):
        a = day_bootstrap([0, 0, 0], [1, 1, 4], [0, 0, 0], ["a", "a", "b"], 500)
        b = day_bootstrap([0, 0, 0], [1, 1, 4], [0, 0, 0], ["a", "a", "b"], 500)
        self.assertEqual(a, b)
        self.assertEqual(a["delta_mae"], 2)
        self.assertGreaterEqual(a["low"], 1)
        self.assertLessEqual(a["high"], 4)

    def test_calibration_only_errors_and_nonnegative_intervals(self):
        offsets = interval_offsets([0, 10, 20], [5, 15, 25])
        self.assertEqual(offsets, [-5, -5])
        low, high = interval_bounds([1, 10], offsets)
        np.testing.assert_array_equal(low, [0, 5])
        np.testing.assert_array_equal(high, [0, 5])

    def test_asymmetric_cost_known_values(self):
        self.assertEqual(asymmetric_cost([5, 5], [3, 8], 4), 5.5)
        with self.assertRaises(ValueError):
            asymmetric_cost([1], [1], -1)

    def test_baseline_fallback_uses_history(self):
        model = Forecaster(CONFIGS["last_week"]).fit([row("2011-01-15", 20)])
        example = row("2012-10-01", 9999, lag=12)
        example["lag_168"] = None
        self.assertEqual(model.predict([example])[0], 12)
        example["lag_24"] = None
        self.assertEqual(model.predict([example])[0], 7)
        example["previous_day_mean"] = example["previous_week_mean"] = None
        self.assertEqual(model.predict([example])[0], 20)

    def test_transform_is_fitted_only_on_training(self):
        training = [row("2011-01-15", n, lag=n) for n in [1, 3, 5, 7]]
        model = Forecaster(CONFIGS["ridge_1"]).fit(training)
        before = model.estimator[0].statistics_.copy()
        model.predict([row("2012-10-01", 0, lag=1e8)])
        np.testing.assert_array_equal(before, model.estimator[0].statistics_)

    def test_selection_ignores_calibration_and_test_labels(self):
        rows = [row("2011-01-15", 5), row("2012-07-01", 7),
                row("2012-08-01", 8), row("2012-09-01", 100),
                row("2012-10-01", 9999)]
        with patch("rush_hour.study.CONFIGS", {k: CONFIGS[k] for k in ["yesterday", "last_week", "ridge_1"]}):
            first = select(rows)
            rows[-1]["target"] = rows[-2]["target"] = -1e6
            self.assertEqual(first, select(rows))

    def test_boosting_disables_random_validation(self):
        model = Forecaster(CONFIGS["boost_15"]).fit([row("2011-01-15", i) for i in range(30)])
        self.assertFalse(model.estimator.early_stopping)
        self.assertTrue(np.all(model.predict([row("2012-10-01", 0)]) >= 0))


if __name__ == "__main__":
    unittest.main()
