"""Forecast all 24 target hours from information available at the forecast origin."""

from datetime import datetime, timedelta
import math

NUMERIC_FEATURES = (
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "weekend", "days_since_2011",
    "lag_24", "lag_168", "lag_336",
    "previous_day_mean", "previous_day_observed_hours",
    "previous_week_mean", "previous_week_observed_hours",
)


def forecast_day(records, origin):
    """Create feature rows without reading any outcomes at or after midnight.

    The same function handles historical replay and future dates. Timestamp lags
    are lookups in clock time, never positional shifts over an incomplete CSV.
    Observations absent from the source stay None. Models must fit any imputer on
    training data only. This assumes data through 23:00 are available at midnight.
    """
    if not isinstance(origin, datetime) or origin.tzinfo is not None:
        raise ValueError("origin must be a naive datetime matching source clock labels.")
    if origin.hour or origin.minute or origin.second or origin.microsecond:
        raise ValueError("Daily forecast origins must be exactly midnight.")
    if any(row.timestamp.tzinfo is not None for row in records):
        raise ValueError("Source timestamps must be naive clock labels.")
    historical = [row for row in records if row.timestamp < origin]
    history = {row.timestamp: row.count for row in historical}
    if len(history) != len(historical):
        raise ValueError("Historical timestamps must be unique.")
    if any(row.timestamp.minute or row.timestamp.second or row.timestamp.microsecond
           or row.count < 0 for row in historical):
        raise ValueError("History must contain nonnegative counts at exact hours.")
    previous_day = [row.count for row in historical
                    if origin-timedelta(days=1) <= row.timestamp]
    previous_week = [row.count for row in historical
                     if origin-timedelta(days=7) <= row.timestamp]
    result = []
    for hour in range(24):
        target = origin + timedelta(hours=hour)
        values = {
            "origin": origin.isoformat(), "target_timestamp": target.isoformat(),
            "hour_sin": math.sin(2*math.pi*hour/24),
            "hour_cos": math.cos(2*math.pi*hour/24),
            "weekday_sin": math.sin(2*math.pi*target.weekday()/7),
            "weekday_cos": math.cos(2*math.pi*target.weekday()/7),
            "month_sin": math.sin(2*math.pi*(target.month-1)/12),
            "month_cos": math.cos(2*math.pi*(target.month-1)/12),
            "weekend": int(target.weekday() >= 5),
            "days_since_2011": (target-datetime(2011, 1, 1)).days,
            "previous_day_mean": sum(previous_day)/len(previous_day) if previous_day else None,
            "previous_day_observed_hours": len(previous_day),
            "previous_week_mean": sum(previous_week)/len(previous_week) if previous_week else None,
            "previous_week_observed_hours": len(previous_week),
        }
        for lag in (24, 168, 336):
            values[f"lag_{lag}"] = history.get(target-timedelta(hours=lag))
        result.append(values)
    return result


def supervised_rows(records, start, stop):
    """Build labeled historical rows on [start, stop); never fabricate missing labels."""
    if stop <= start:
        raise ValueError("stop must be after start.")
    labels = {row.timestamp.isoformat(): row.count for row in records}
    output = []
    origin = start
    while origin < stop:
        for features in forecast_day(records, origin):
            timestamp = features["target_timestamp"]
            if timestamp in labels:
                output.append({**features, "target": labels[timestamp]})
        origin += timedelta(days=1)
    return output

