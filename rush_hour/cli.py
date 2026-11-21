"""Public CLI for audit, study reproduction, and offline plot rendering."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rush Hour: tomorrow's demand, tonight's information.")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["audit", "run"]:
        cmd = sub.add_parser(command)
        cmd.add_argument("--archive", type=Path, default=Path("data/raw/bike-sharing.zip"))
        cmd.add_argument("--output", type=Path, default=Path("reports"))
        if command == "run":
            cmd.add_argument("--bootstrap", type=int, default=2000)
    plot = sub.add_parser("plot")
    plot.add_argument("--report", type=Path, default=Path("reports"))
    args = parser.parse_args()
    try:
        if args.command == "plot":
            from .plots import render
            report = json.loads((args.report/"results.json").read_text(encoding="utf-8"))
            with (args.report/"forecasts.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                for key in row:
                    if key not in ("origin", "timestamp", "observed"):
                        row[key] = float(row[key]) if row[key] else None
            render(report, rows, args.report/"figures")
            print("Rendered six figures from saved results.")
            return
        from .data import audit, download, read_hourly
        records = read_hourly(download(args.archive))
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command == "audit":
            result = audit(records)
            (args.output/"data-audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            print(json.dumps({k: v for k, v in result.items() if k != "missing_timestamps"}, indent=2))
        else:
            if args.bootstrap < 100:
                raise ValueError("Use at least 100 bootstrap resamples.")
            from .study import run
            from .plots import render
            report, rows = run(records, args.output, args.bootstrap)
            render(report, rows, args.output/"figures")
            print(json.dumps({"selected": report["selection"]["selected"],
                              "evaluation": {k: v for k, v in report["evaluation"].items()
                                             if k not in ("slices", "cost_scenarios")}}, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
