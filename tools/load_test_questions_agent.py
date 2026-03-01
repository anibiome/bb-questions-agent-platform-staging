from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Load test daily selection + answers + snapshot reads.")
    parser.add_argument("--base-url", type=str, required=True, help="API base URL, e.g. http://localhost:8080")
    parser.add_argument("--api-key", type=str, required=True, help="X-API-Key value")
    parser.add_argument("--users", type=int, default=25)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--date-param-name", type=str, default="date_param", help="Use date for reference stack, date_param for prod stack.")
    parser.add_argument("--out", type=str, default=None, help="Optional report JSON path.")
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args(argv)

    users = max(1, int(args.users))
    days = max(1, int(args.days))
    concurrency = max(1, int(args.concurrency))
    base_url = str(args.base_url).rstrip("/")
    date_param_name = str(args.date_param_name).strip() or "date_param"
    api_key = str(args.api_key)

    end_day = date.today()
    all_days = [end_day - timedelta(days=offset) for offset in reversed(range(days))]

    tasks: List[Tuple[str, date]] = []
    for user_index in range(users):
        user_id = f"load-user-{user_index + 1}"
        for day in all_days:
            tasks.append((user_id, day))

    timings: Dict[str, List[float]] = {"daily_select_ms": [], "answer_ingest_ms": [], "state_read_ms": [], "circle_read_ms": []}
    errors: List[Dict[str, Any]] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _run_single_day,
                base_url=base_url,
                api_key=api_key,
                user_id=user_id,
                day=day,
                date_param_name=date_param_name,
            )
            for (user_id, day) in tasks
        ]
        for fut in as_completed(futures):
            result = fut.result()
            if result.get("ok"):
                for key in timings.keys():
                    value = result.get(key)
                    if value is not None:
                        timings[key].append(float(value))
            else:
                errors.append(result)

    total_ms = (time.perf_counter() - started) * 1000.0
    report = {
        "base_url": base_url,
        "users": users,
        "days": days,
        "concurrency": concurrency,
        "requests_scenarios": len(tasks),
        "duration_ms": total_ms,
        "latency": {
            name: _summary(vals)
            for name, vals in timings.items()
        },
        "errors": {"count": len(errors), "examples": errors[:20]},
        "status": "ok" if not errors else "degraded",
    }

    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
        print(str(out_path))

    if args.fail_on_errors and errors:
        return 2
    return 0


def _run_single_day(*, base_url: str, api_key: str, user_id: str, day: date, date_param_name: str) -> Dict[str, Any]:
    day_str = day.isoformat()
    headers = {"X-API-Key": api_key}

    daily_url = f"{base_url}/v1/users/{urllib.parse.quote(user_id)}/daily-questions?{urllib.parse.quote(date_param_name)}={urllib.parse.quote(day_str)}"
    t0 = time.perf_counter()
    daily = _http_json("GET", daily_url, headers=headers)
    daily_ms = (time.perf_counter() - t0) * 1000.0
    session_id = str(daily.get("session_id") or "")
    questions = daily.get("questions") or []
    if not session_id or not questions:
        return {"ok": False, "stage": "daily_questions", "user_id": user_id, "day": day_str, "error": "missing session/questions"}

    answers = [
        {
            "client_event_id": f"{user_id}::{day_str}::{q['item_id']}",
            "item_id": q["item_id"],
            "value": 2,
            "answered_at": f"{day_str}T12:00:00Z",
        }
        for q in questions
    ]
    t1 = time.perf_counter()
    _ = _http_json(
        "POST",
        f"{base_url}/v1/users/{urllib.parse.quote(user_id)}/answers",
        headers={**headers, "Content-Type": "application/json"},
        payload={"session_id": session_id, "answers": answers},
    )
    answer_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    _ = _http_json(
        "GET",
        f"{base_url}/v1/users/{urllib.parse.quote(user_id)}/state?{urllib.parse.quote(date_param_name)}={urllib.parse.quote(day_str)}&window=30d",
        headers=headers,
    )
    state_ms = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    _ = _http_json(
        "GET",
        f"{base_url}/v1/users/{urllib.parse.quote(user_id)}/circle?{urllib.parse.quote(date_param_name)}={urllib.parse.quote(day_str)}&window=30d",
        headers=headers,
    )
    circle_ms = (time.perf_counter() - t3) * 1000.0

    return {
        "ok": True,
        "daily_select_ms": daily_ms,
        "answer_ingest_ms": answer_ms,
        "state_read_ms": state_ms,
        "circle_read_ms": circle_ms,
    }


def _http_json(method: str, url: str, *, headers: Dict[str, str], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, method=method, headers=headers, data=body)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTP {exc.code} for {method} {url}: {detail}") from exc


def _summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    sorted_vals = sorted(float(v) for v in values)
    return {
        "count": len(sorted_vals),
        "p50_ms": float(median(sorted_vals)),
        "p95_ms": float(_percentile(sorted_vals, 95.0)),
        "max_ms": float(sorted_vals[-1]),
    }


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = max(0.0, min(100.0, float(q))) / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


if __name__ == "__main__":
    raise SystemExit(main())

