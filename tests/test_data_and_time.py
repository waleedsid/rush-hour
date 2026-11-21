import unittest
from datetime import datetime, timedelta

from rush_hour.data import Observation, audit, parse_hourly
from rush_hour.features import NUMERIC_FEATURES, forecast_day, supervised_rows


class DataTests(unittest.TestCase):
    def test_components_and_duplicates_fail(self):
        header = "dteday,hr,cnt,casual,registered\n"
        for rows in ["2012-01-01,0,4,1,2\n",
                     "2012-01-01,0,3,1,2\n2012-01-01,0,3,1,2\n",
                     "2012-01-01,24,3,1,2\n",
                     "2012-01-01,0,-1,0,-1\n"]:
            with self.assertRaises(ValueError):
                parse_hourly(header+rows)

    def test_missing_hours_are_not_zero_rows(self):
        rows = parse_hourly("dteday,hr,cnt,casual,registered\n2012-01-01,0,3,1,2\n2012-01-01,2,0,0,0\n")
        report = audit(rows)
        self.assertEqual(report["observed_rows"], 2)
        self.assertEqual(report["missing_calendar_hours"], 1)
        self.assertEqual(report["observed_zero_count_rows"], 1)

    def test_sorting_and_schema(self):
        rows = parse_hourly("dteday,hr,cnt,casual,registered\n2012-01-01,2,3,1,2\n2012-01-01,0,3,1,2\n")
        self.assertLess(rows[0].timestamp, rows[1].timestamp)
        with self.assertRaises(ValueError):
            parse_hourly("date,value\nx,1")


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.origin = datetime(2012, 10, 1)
        self.past = [Observation(self.origin-timedelta(hours=i), i) for i in range(1, 400)]

    def test_future_outcomes_cannot_change_features(self):
        future = [Observation(self.origin+timedelta(hours=i), 1_000_000) for i in range(72)]
        self.assertEqual(forecast_day(self.past, self.origin),
                         forecast_day(self.past+future, self.origin))

    def test_lags_are_clock_based_with_gaps(self):
        history = [row for row in self.past if row.timestamp != self.origin-timedelta(hours=24)]
        features = forecast_day(history, self.origin)
        self.assertIsNone(features[0]["lag_24"])
        self.assertEqual(features[1]["lag_24"], 23)
        self.assertEqual(features[0]["lag_168"], 168)
        self.assertEqual(features[0]["previous_day_observed_hours"], 23)

    def test_all_horizons_use_prior_day_aggregate(self):
        features = forecast_day(self.past, self.origin)
        self.assertEqual(len(features), 24)
        for row in features:
            self.assertEqual(row["previous_day_mean"], 12.5)
            self.assertEqual(row["previous_day_observed_hours"], 24)
        self.assertEqual(features[23]["lag_24"], 1)

    def test_forecast_does_not_require_target_rows(self):
        output = forecast_day([], self.origin)
        self.assertEqual(len(output), 24)
        self.assertIsNone(output[0]["lag_168"])
        self.assertIsNone(output[0]["previous_day_mean"])

    def test_invalid_origin_and_duplicate_history(self):
        with self.assertRaises(ValueError):
            forecast_day(self.past, self.origin+timedelta(hours=1))
        with self.assertRaises(ValueError):
            forecast_day(self.past+self.past[:1], self.origin)

    def test_target_components_and_weather_excluded(self):
        forbidden = {"casual", "registered", "cnt", "temp", "atemp", "hum", "windspeed", "weathersit"}
        self.assertFalse(forbidden.intersection(NUMERIC_FEATURES))
        self.assertFalse(forbidden.intersection(forecast_day(self.past, self.origin)[0]))

    def test_only_observed_target_labels_are_returned(self):
        target = Observation(self.origin+timedelta(hours=3), 123)
        rows = supervised_rows(self.past+[target], self.origin, self.origin+timedelta(days=1))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target"], 123)
        self.assertEqual(rows[0]["lag_24"], 21)


if __name__ == "__main__":
    unittest.main()

