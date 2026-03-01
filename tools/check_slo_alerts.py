from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from questions_agent_platform.prod.db import make_engine, make_session_factory, session_scope
from questions_agent_platform.prod.migrations import ensure_schema_ready
from questions_agent_platform.prod.service_pg import build_slo_report
from questions_agent_platform.prod.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Questions Agent production SLO thresholds.")
    parser.add_argument("--days", type=int, default=None, help="Lookback window in days (default: SLO_WINDOW_DAYS env)")
    parser.add_argument(
        "--fail-on-degraded",
        action="store_true",
        help="Exit non-zero when SLO status is degraded.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    ensure_schema_ready(
        database_url=settings.database_url,
        mode=settings.db_migration_mode,
        baseline_existing=settings.db_migration_baseline_existing,
    )
    engine = make_engine(settings.database_url)
    SessionLocal = make_session_factory(engine)
    lookback_days = settings.slo_window_days if args.days is None else max(1, int(args.days))
    try:
        with session_scope(SessionLocal) as session:
            report = build_slo_report(
                session,
                day=date.today(),
                window_days=lookback_days,
                safe_fallback_rate_max=settings.slo_safe_fallback_rate_max,
                answer_ingestion_p95_ms_max=settings.slo_answer_ingestion_p95_ms_max,
                snapshot_write_failure_rate_max=settings.slo_snapshot_write_failure_rate_max,
            )
    finally:
        engine.dispose()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_degraded and str(report.get("status")) != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

