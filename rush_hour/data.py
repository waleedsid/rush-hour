"""Pinned public data retrieval and strict hourly record validation."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

SOURCE = "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"
ARCHIVE_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
REQUIRED = {"dteday", "hr", "cnt", "casual", "registered"}


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    count: int


def download(destination):
    """Verify the exact archive before atomically caching it. Never extract paths."""
    destination = Path(destination)
    if destination.exists():
        payload = destination.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            payload = response.read(5_000_001)
    if len(payload) > 5_000_000:
        raise ValueError("Archive exceeds the permitted download size.")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"Archive checksum mismatch: {actual}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_suffix(".download")
        staging.write_bytes(payload)
        staging.replace(destination)
    return destination


def read_hourly(archive_path):
    """Read counts only; target components and observed weather never become features."""
    payload = Path(archive_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Unrecognized source archive. Use the pinned UCI archive.")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_bytes = archive.read("hour.csv")
    return parse_hourly(csv_bytes.decode("utf-8-sig"))


def parse_hourly(text):
    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing required columns: {sorted(REQUIRED)}")
    records, seen = [], set()
    for line, row in enumerate(reader, start=2):
        try:
            date = datetime.strptime(row["dteday"], "%Y-%m-%d")
            hour, count = int(row["hr"]), int(row["cnt"])
            casual, registered = int(row["casual"]), int(row["registered"])
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid field at CSV line {line}") from error
        if not 0 <= hour <= 23 or min(count, casual, registered) < 0:
            raise ValueError(f"Invalid hour or negative count at CSV line {line}")
        if count != casual + registered:
            raise ValueError(f"Count components disagree at CSV line {line}")
        timestamp = date.replace(hour=hour)
        if timestamp in seen:
            raise ValueError(f"Duplicate hour: {timestamp.isoformat()}")
        seen.add(timestamp)
        records.append(Observation(timestamp, count))
    if not records:
        raise ValueError("No hourly records found.")
    return sorted(records, key=lambda row: row.timestamp)


def audit(records):
    if not records:
        raise ValueError("Cannot audit empty records.")
    observed = {row.timestamp for row in records}
    if len(observed) != len(records):
        raise ValueError("Duplicate timestamps in audit.")
    first, last = min(observed), max(observed)
    expected = int((last - first).total_seconds() / 3600) + 1
    missing = [first + timedelta(hours=i) for i in range(expected)
               if first + timedelta(hours=i) not in observed]
    by_year = {}
    for row in records:
        key = str(row.timestamp.year)
        by_year[key] = by_year.get(key, 0) + 1
    return {
        "observed_rows": len(records), "calendar_hours": expected,
        "missing_calendar_hours": len(missing),
        "missing_timestamps": [t.isoformat() for t in missing],
        "first_timestamp": first.isoformat(), "last_timestamp": last.isoformat(),
        "rows_by_year": by_year,
        "observed_zero_count_rows": sum(row.count == 0 for row in records),
        "maximum_observed_hourly_count": max(row.count for row in records),
        "timestamp_policy": "Naive source clock labels. No DST resolution is inferred.",
        "missingness_policy": "Absent hours are unobserved, not automatically zero demand.",
    }

