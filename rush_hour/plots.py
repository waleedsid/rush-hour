"""Figures from saved predictions and metrics; no model fitting during rendering."""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

LABELS = {"yesterday": "Yesterday", "last_week": "Last week", "ridge_1": "Ridge (1)",
          "ridge_100": "Ridge (100)", "boost_15": "Boosting (15)", "boost_31": "Boosting (31)",
          "calendar_only": "Calendar only"}
TEAL, GRAY, RED = "#087f8c", "#667085", "#c65b42"


def render(report, rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fafbfc", "axes.facecolor": "#fafbfc",
                         "svg.hashsalt": "rush-hour"})
    winner = report["selection"]["selected"]
    evaluation = report["evaluation"]

    def save(fig, name):
        fig.savefig(directory/f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(directory/f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, values, title in [
        (axes[0], report["selection"]["mean_fold_mae"], "Choose on July and August"),
        (axes[1], {k: v["mae"] for k, v in evaluation["metrics"].items()}, "Evaluate on October through December")]:
        ordered = sorted(values, key=values.get)
        ax.barh([LABELS[k] for k in ordered], [values[k] for k in ordered],
                color=[TEAL if k == winner else GRAY for k in ordered])
        ax.invert_yaxis()
        ax.set(title=title, xlabel="Mean absolute error (rentals per hour)")
        ax.bar_label(ax.containers[0], fmt="%.1f", padding=3)
        ax.set_xlim(0, max(values.values())*1.18)
    fig.suptitle("The scorecard: accuracy must survive a later season")
    save(fig, "scorecard")

    week = [r for r in rows if "2012-10-01" <= r["timestamp"][:10] < "2012-10-08"]
    dates = [datetime.fromisoformat(r["timestamp"]) for r in week]
    actual = [r["actual"] if r["actual"] is not None else np.nan for r in week]
    fig, ax = plt.subplots(figsize=(12, 4.7), layout="constrained")
    ax.fill_between(dates, [r["lower90"] for r in week], [r["upper90"] for r in week],
                    color=TEAL, alpha=0.12, label="Empirical 90% interval")
    ax.plot(dates, actual, color="#202c3d", label="Observed", linewidth=1.8)
    ax.plot(dates, [r[winner] for r in week], color=TEAL, label=LABELS[winner], linewidth=1.5)
    ax.plot(dates, [r["last_week"] for r in week], color=RED, linestyle=":", label="Last-week baseline")
    ax.set(title="The first evaluation week, chosen before inspecting errors", ylabel="Rentals per hour", xlabel="October 2012")
    ax.legend(ncol=2, fontsize=9)
    save(fig, "forecast-week")

    observed = [r for r in rows if r["actual"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    hourly = [r for r in evaluation["slices"] if r["slice"] == "hour"]
    axes[0].plot([r["group"] for r in hourly], [r["selected"]["mae"] for r in hourly], "o-", color=TEAL, label="Selected")
    axes[0].plot([r["group"] for r in hourly], [r["last_week"]["mae"] for r in hourly], "o:", color=GRAY, label="Last week")
    axes[0].set(title="Where do errors concentrate?", xlabel="Hour of target day", ylabel="Mean absolute error", xticks=range(0, 24, 3))
    axes[0].legend()
    days = sorted({r["origin"] for r in rows})
    for name, color in [(winner, TEAL), ("last_week", GRAY)]:
        daily = [np.mean([abs(r["actual"]-r[name]) for r in observed if r["origin"] == day])
                 if any(r["origin"] == day for r in observed) else np.nan for day in days]
        axes[1].plot([datetime.fromisoformat(day) for day in days], daily, color=color,
                     label=LABELS[name], alpha=0.85, linewidth=1)
    axes[1].set(title="Daily errors expose difficult dates", xlabel="Evaluation date", ylabel="Mean absolute error")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[1].legend()
    save(fig, "error-patterns")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    months = [r for r in evaluation["slices"] if r["slice"] == "month"]
    axes[0].bar(["Oct", "Nov", "Dec"], [r["coverage"] for r in months], color=TEAL)
    axes[0].axhline(0.9, color=RED, linestyle="--", label="Nominal target: 90%")
    axes[0].set(title="Does September's uncertainty transfer?", ylabel="Observed interval coverage", ylim=(0, 1))
    axes[0].legend(fontsize=9)
    width = [r["upper90"]-r["lower90"] for r in observed]
    axes[1].scatter([r[winner] for r in observed], width, s=8, alpha=0.3, color=TEAL)
    axes[1].set(title="Residual intervals have limited adaptability", xlabel="Point forecast (rentals per hour)", ylabel="Interval width (rentals)")
    save(fig, "uncertainty")

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for name in dict.fromkeys(["last_week", winner]):
        subset = [r for r in evaluation["cost_scenarios"] if r["model"] == name]
        ax.plot([r["under_cost"] for r in subset], [r["mean_loss"] for r in subset],
                "o-", label=LABELS[name], color=TEAL if name == winner else GRAY)
    ax.set(title="What if underforecasting costs more?", xlabel="Assumed underforecast cost / overforecast cost",
           ylabel="Mean hypothetical loss per observed hour", xticks=[1, 2, 4, 8])
    ax.legend()
    save(fig, "cost-sensitivity")

    missing = [datetime.fromisoformat(t) for t in report["data_audit"]["missing_timestamps"]]
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    counts = np.array([[sum(t.year == year and t.month == month for t in missing) for month in range(1, 13)]
                       for year in [2011, 2012]])
    im = ax.imshow(counts, cmap="YlOrRd", aspect="auto")
    for i in range(2):
        for j in range(12):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                    color="white" if counts[i, j] > counts.max()*0.6 else "#333")
    ax.set(title="No missing cells does not mean no missing hours", xticks=range(12),
           xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           yticks=[0, 1], yticklabels=["2011", "2012"])
    fig.colorbar(im, ax=ax, label="Absent calendar hours")
    save(fig, "missing-hours")
