"""Prespecified rolling selection, independent calibration and final evaluation."""

import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import platform

import numpy as np
import sklearn

from .data import ARCHIVE_SHA256, audit
from .evaluation import asymmetric_cost, day_bootstrap, interval_bounds, interval_offsets, metrics
from .features import forecast_day
from .models import CONFIGS, Forecaster

START = datetime(2011, 1, 15)
JULY, AUGUST, SEPTEMBER, OCTOBER, END = [datetime.fromisoformat(s) for s in
    ["2012-07-01", "2012-08-01", "2012-09-01", "2012-10-01", "2013-01-01"]]


def all_forecasts(records, start=START, stop=END):
    labels = {r.timestamp.isoformat(): r.count for r in records}
    rows = []
    day = start
    while day < stop:
        for row in forecast_day(records, day):
            rows.append({**row, "target": labels.get(row["target_timestamp"])})
        day += timedelta(days=1)
    return rows


def window(rows, start, stop, observed=True):
    a, b = start.isoformat(), stop.isoformat()
    return [r for r in rows if a <= r["target_timestamp"] < b
            and (not observed or r["target"] is not None)]


def select(rows):
    fold_results = []
    for start, stop in [(JULY, AUGUST), (AUGUST, SEPTEMBER)]:
        training, validation = window(rows, START, start), window(rows, start, stop)
        for name, config in CONFIGS.items():
            model = Forecaster(config).fit(training)
            result = metrics([r["target"] for r in validation], model.predict(validation))
            fold_results.append({"model": name, "training_stop_exclusive": start.isoformat(),
                                 "validation_stop_exclusive": stop.isoformat(),
                                 "training_n": len(training), **result})
    scores = {name: float(np.mean([r["mae"] for r in fold_results if r["model"] == name]))
              for name in CONFIGS}
    winner = min(scores, key=lambda key: (scores[key], key))
    learned = min((k for k in scores if CONFIGS[k]["kind"] != "baseline"),
                  key=lambda key: (scores[key], key))
    return {"configs": CONFIGS, "folds": fold_results, "mean_fold_mae": scores,
            "selected": winner, "best_learned": learned,
            "criterion": "Unweighted mean of July and August MAE; alphabetical tie break."}


def finish(rows, selection, repetitions=2000):
    training = window(rows, START, SEPTEMBER)
    calibration = window(rows, SEPTEMBER, OCTOBER)
    testing = window(rows, OCTOBER, END, observed=False)
    observed = np.array([r["target"] is not None for r in testing])
    y = np.array([r["target"] for r in testing if r["target"] is not None], dtype=float)
    yc = np.array([r["target"] for r in calibration], dtype=float)
    names = ["yesterday", "last_week", selection["selected"], selection["best_learned"]]
    predictions, calibration_predictions, scores, models = {}, {}, {}, {}
    for name in dict.fromkeys(names):
        models[name] = Forecaster(CONFIGS[name]).fit(training)
        calibration_predictions[name] = models[name].predict(calibration)
        predictions[name] = models[name].predict(testing)
        scores[name] = metrics(y, predictions[name][observed])
    learned = selection["best_learned"]
    ablation = Forecaster(CONFIGS[learned], calendar_only=True).fit(training)
    predictions["calendar_only"] = ablation.predict(testing)
    scores["calendar_only"] = metrics(y, predictions["calendar_only"][observed])
    winner = selection["selected"]
    offsets = interval_offsets(yc, calibration_predictions[winner])
    lower, upper = interval_bounds(predictions[winner], offsets)
    cover = (y >= lower[observed]) & (y <= upper[observed])
    days = [r["origin"][:10] for r in testing if r["target"] is not None]
    paired = day_bootstrap(y, predictions[winner][observed], predictions["last_week"][observed], days, repetitions)
    slices = []
    observed_rows = [r for r in testing if r["target"] is not None]
    for kind, groups in [("hour", range(24)), ("weekend", range(2)), ("month", (10, 11, 12))]:
        for group in groups:
            mask = np.array([(datetime.fromisoformat(r["target_timestamp"]).hour if kind == "hour"
                              else datetime.fromisoformat(r["target_timestamp"]).month if kind == "month"
                              else r["weekend"]) == group for r in observed_rows])
            if not mask.any():
                continue
            result = {"slice": kind, "group": int(group), "selected": metrics(y[mask], predictions[winner][observed][mask]),
                      "last_week": metrics(y[mask], predictions["last_week"][observed][mask]),
                      "coverage": float(cover[mask].mean())}
            slices.append(result)
    cost_rows = []
    for ratio in [1, 2, 4, 8]:
        for name in dict.fromkeys(["last_week", winner]):
            # Shift each forecast using only its September errors for fair comparison.
            offset = float(np.quantile(yc-calibration_predictions[name], ratio/(ratio+1)))
            forecast = np.maximum(0, predictions[name][observed]+offset)
            cost_rows.append({"under_cost": ratio, "over_cost": 1, "model": name,
                              "calibration_shift": offset, "mean_loss": asymmetric_cost(y, forecast, ratio)})
    output_rows = []
    for i, row in enumerate(testing):
        output_rows.append({"origin": row["origin"], "timestamp": row["target_timestamp"],
                            "observed": bool(observed[i]), "actual": row["target"],
                            **{name: float(value[i]) for name, value in predictions.items()},
                            "lower90": float(lower[i]), "upper90": float(upper[i])})
    return {
        "training_stop_exclusive": SEPTEMBER.isoformat(),
        "calibration_window": [SEPTEMBER.isoformat(), OCTOBER.isoformat()],
        "test_window": [OCTOBER.isoformat(), END.isoformat()],
        "planned_hours": len(testing), "observed_hours": int(observed.sum()),
        "missing_targets": int((~observed).sum()), "metrics": scores,
        "interval": {"nominal": 0.9, "signed_error_offsets": offsets,
                     "observed_coverage": float(cover.mean()),
                     "mean_width_observed": float((upper[observed]-lower[observed]).mean()),
                     "warning": "Empirical September intervals. Temporal dependence and distribution shift preclude an IID coverage guarantee."},
        "selected_vs_last_week": paired, "slices": slices,
        "cost_scenarios": cost_rows, "ablation_parent": learned,
    }, output_rows


def run(records, output, repetitions=2000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = all_forecasts(records)
    chosen = select(rows)
    # Persist selection before any final-window prediction or scoring.
    (output/"selection.json").write_text(json.dumps(chosen, indent=2)+"\n", encoding="utf-8")
    result, predictions = finish(rows, chosen, repetitions)
    report = {"source_archive_sha256": ARCHIVE_SHA256, "data_audit": audit(records),
              "selection": chosen, "evaluation": result,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scikit_learn": sklearn.__version__}}
    (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (output/"forecasts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return report, predictions
