from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from questions_agent_platform.pipeline.time_utils import parse_date
from questions_agent_platform.prod.db import make_engine, make_session_factory, session_scope
from questions_agent_platform.prod.migrations import ensure_schema_ready
from questions_agent_platform.prod.service_pg import build_clinical_audit_export
from questions_agent_platform.prod.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export clinical audit payload with configurable PHI redaction.")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD (default: end_date - 29d)")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--user-id", type=str, default=None, help="Optional single-user export")
    parser.add_argument("--redact-user-ids", type=str, default=None, help="true|false override")
    parser.add_argument("--redact-raw-payloads", type=str, default=None, help="true|false override")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args(argv)

    settings = load_settings()
    ensure_schema_ready(
        database_url=settings.database_url,
        mode=settings.db_migration_mode,
        baseline_existing=settings.db_migration_baseline_existing,
    )

    end_day = parse_date(args.end_date) if args.end_date else date.today()
    start_day = parse_date(args.start_date) if args.start_date else (end_day - timedelta(days=29))
    if start_day > end_day:
        raise ValueError("start-date must be <= end-date")

    redact_user_ids = (
        settings.phi_export_redact_user_ids
        if args.redact_user_ids is None
        else _bool(args.redact_user_ids)
    )
    redact_raw_payloads = (
        settings.phi_export_redact_raw_payloads
        if args.redact_raw_payloads is None
        else _bool(args.redact_raw_payloads)
    )

    engine = make_engine(settings.database_url)
    SessionLocal = make_session_factory(engine)
    try:
        with session_scope(SessionLocal) as session:
            payload = build_clinical_audit_export(
                session,
                start_day=start_day,
                end_day=end_day,
                user_id=args.user_id,
                redact_user_ids=redact_user_ids,
                redact_raw_payloads=redact_raw_payloads,
                user_hash_salt=settings.phi_export_user_hash_salt,
            )
    finally:
        engine.dispose()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


def _bool(v: str) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())

