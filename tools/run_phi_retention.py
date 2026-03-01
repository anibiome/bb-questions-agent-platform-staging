from __future__ import annotations

import argparse
import json
from datetime import date

from questions_agent_platform.prod.db import make_engine, make_session_factory, session_scope
from questions_agent_platform.prod.migrations import ensure_schema_ready
from questions_agent_platform.prod.service_pg import run_phi_retention
from questions_agent_platform.prod.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PHI retention/redaction job.")
    parser.add_argument("--dry-run", action="store_true", help="Show affected rows without mutating data.")
    args = parser.parse_args(argv)

    settings = load_settings()
    ensure_schema_ready(
        database_url=settings.database_url,
        mode=settings.db_migration_mode,
        baseline_existing=settings.db_migration_baseline_existing,
    )
    engine = make_engine(settings.database_url)
    SessionLocal = make_session_factory(engine)
    try:
        with session_scope(SessionLocal) as session:
            result = run_phi_retention(
                session,
                day=date.today(),
                answer_raw_retention_days=settings.phi_retention_answer_raw_days,
                observation_raw_retention_days=settings.phi_retention_observation_raw_days,
                policy_context_retention_days=settings.phi_retention_policy_context_days,
                dry_run=bool(args.dry_run),
            )
    finally:
        engine.dispose()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

