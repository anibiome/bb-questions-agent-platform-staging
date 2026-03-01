from __future__ import annotations

from datetime import date, datetime, timezone


def date_to_start_iso(d: date) -> str:
    return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def date_to_end_iso_exclusive(d: date) -> str:
    # End exclusive: next day at 00:00Z
    d2 = d.toordinal() + 1
    next_day = date.fromordinal(d2)
    return (
        datetime(next_day.year, next_day.month, next_day.day, 0, 0, 0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_date(d: str) -> date:
    return date.fromisoformat(d)


def parse_iso_datetime(ts: str) -> datetime:
    # Supports "...Z" or "+00:00"
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def iso_to_date(ts: str) -> date:
    return parse_iso_datetime(ts).date()
