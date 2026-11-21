"""Evaluation on common rows and descriptive paired day-bootstrap uncertainty."""

import numpy as np


def aligned(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or not len(y) or prediction.shape != y.shape:
        raise ValueError("Expected equally sized nonempty vectors.")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Values must be finite.")
    return y, prediction


def metrics(y, prediction):
    y, prediction = aligned(y, prediction)
    error = prediction-y
    denominator = float(np.abs(y).sum())
    return {"n": len(y), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "wape": float(np.abs(error).sum()/denominator) if denominator else None,
            "bias": float(error.mean())}


def interval_offsets(y, prediction, alpha=0.1):
    y, prediction = aligned(y, prediction)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one.")
    return np.quantile(y-prediction, [alpha/2, 1-alpha/2]).tolist()


def interval_bounds(prediction, offsets):
    low, high = offsets
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Interval offsets must be finite and ordered.")
    return np.maximum(0, np.asarray(prediction)+low), np.maximum(0, np.asarray(prediction)+high)


def day_bootstrap(y, prediction, reference, day_labels, repetitions=2000, seed=42):
    y, prediction = aligned(y, prediction)
    _, reference = aligned(y, reference)
    if len(day_labels) != len(y) or repetitions < 1:
        raise ValueError("Require aligned days and positive repetitions.")
    days = np.asarray(day_labels)
    unique = sorted(set(day_labels))
    difference = np.abs(y-prediction)-np.abs(y-reference)
    totals = np.array([difference[days == day].sum() for day in unique])
    counts = np.array([np.sum(days == day) for day in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    distribution = totals[draws].sum(axis=1)/counts[draws].sum(axis=1)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return {"delta_mae": float(difference.mean()), "low": float(lower), "high": float(upper),
            "days": len(unique), "repetitions": repetitions,
            "interpretation": "Negative favors the model. Days resampled with replacement; between-day dependence is not preserved."}


def asymmetric_cost(y, prediction, under_cost):
    y, prediction = aligned(y, prediction)
    if not np.isfinite(under_cost) or under_cost <= 0:
        raise ValueError("Underforecast cost must be positive.")
    return float(np.mean(under_cost*np.maximum(y-prediction, 0)+np.maximum(prediction-y, 0)))
