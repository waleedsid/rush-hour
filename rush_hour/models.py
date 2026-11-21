"""Small, prespecified model set with transformations fitted on training only."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import NUMERIC_FEATURES


CONFIGS = {
    "yesterday": {"kind": "baseline", "lag": "lag_24"},
    "last_week": {"kind": "baseline", "lag": "lag_168"},
    "ridge_1": {"kind": "ridge", "alpha": 1.0},
    "ridge_100": {"kind": "ridge", "alpha": 100.0},
    "boost_15": {"kind": "boost", "leaves": 15},
    "boost_31": {"kind": "boost", "leaves": 31},
}
CALENDAR_FEATURES = NUMERIC_FEATURES[:8]


def matrix(rows, features):
    return np.array([[np.nan if row[name] is None else row[name] for name in features]
                     for row in rows], dtype=float)


class Forecaster:
    def __init__(self, config, calendar_only=False):
        self.config = dict(config)
        self.features = CALENDAR_FEATURES if calendar_only else NUMERIC_FEATURES
        self.estimator = None
        self.fallback = None

    def fit(self, rows):
        if not rows:
            raise ValueError("Training rows must not be empty.")
        y = np.array([r["target"] for r in rows], dtype=float)
        if not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError("Targets must be nonnegative and finite.")
        self.fallback = float(np.median(y))
        kind = self.config["kind"]
        if kind == "baseline":
            return self
        if kind == "ridge":
            self.estimator = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(), Ridge(alpha=self.config["alpha"]))
        elif kind == "boost":
            # Random early-stopping validation would violate the temporal design.
            self.estimator = HistGradientBoostingRegressor(
                loss="absolute_error", max_leaf_nodes=self.config["leaves"],
                learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                l2_regularization=1.0, early_stopping=False, random_state=42)
        else:
            raise ValueError("Unknown model kind.")
        self.estimator.fit(matrix(rows, self.features), y)
        return self

    def predict(self, rows):
        if self.fallback is None:
            raise ValueError("Fit the forecaster first.")
        if not rows:
            return np.array([], dtype=float)
        if self.config["kind"] == "baseline":
            primary = self.config["lag"]
            backup = "lag_168" if primary == "lag_24" else "lag_24"
            values = []
            for row in rows:
                choices = [row[primary], row[backup], row["previous_day_mean"],
                           row["previous_week_mean"], self.fallback]
                values.append(next(value for value in choices if value is not None))
            prediction = np.asarray(values, dtype=float)
        else:
            prediction = self.estimator.predict(matrix(rows, self.features))
        if not np.isfinite(prediction).all():
            raise ValueError("Predictions must be finite.")
        return np.maximum(0, prediction)
