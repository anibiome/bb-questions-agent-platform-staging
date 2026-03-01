from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Set

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine


REQUIRED_BASELINE_TABLES = {
    "registry_versions",
    "users",
    "daily_sessions",
    "answer_events",
    "scale_scores",
    "policy_decisions",
    "policy_outcomes",
}
LEGACY_BASELINE_STAMP_REVISION = "20260214_0001"


def ensure_schema_ready(
    *,
    database_url: str,
    mode: str = "auto",
    baseline_existing: bool = True,
) -> Dict[str, Any]:
    """
    Ensure DB schema is aligned to Alembic head.

    mode:
      - auto: baseline existing legacy schema if needed, then upgrade to head
      - require: fail if DB is not already at head
      - off: skip checks/migrations (not recommended)
    """
    mode_norm = str(mode or "auto").strip().lower()
    if mode_norm not in ("auto", "require", "off"):
        raise ValueError("mode must be one of: auto, require, off")

    if mode_norm == "off":
        return {"mode": mode_norm, "status": "skipped"}

    cfg = _build_alembic_config(database_url=database_url)
    state = _get_revision_state(database_url=database_url, cfg=cfg)
    head = str(state["head_revision"])
    current = state["current_revision"]
    tables = set(state["tables"])

    was_stamped = False
    if current is None and bool(baseline_existing) and _looks_like_legacy_schema(tables):
        from alembic import command

        command.stamp(cfg, LEGACY_BASELINE_STAMP_REVISION)
        was_stamped = True
        state = _get_revision_state(database_url=database_url, cfg=cfg)
        current = state["current_revision"]

    if mode_norm == "auto":
        from alembic import command

        command.upgrade(cfg, "head")
        state = _get_revision_state(database_url=database_url, cfg=cfg)
        return {
            "mode": mode_norm,
            "status": "upgraded",
            "head_revision": str(state["head_revision"]),
            "current_revision": str(state["current_revision"]),
            "stamped_legacy_schema": bool(was_stamped),
        }

    # require mode
    if str(current or "") != head:
        raise RuntimeError(
            "Database schema is not at Alembic head. "
            f"current={current}, head={head}. "
            "Run: python -m questions_agent_platform.prod.manage upgrade"
        )
    return {
        "mode": mode_norm,
        "status": "ok",
        "head_revision": head,
        "current_revision": str(current),
        "stamped_legacy_schema": bool(was_stamped),
    }


def _looks_like_legacy_schema(tables: Set[str]) -> bool:
    return REQUIRED_BASELINE_TABLES.issubset(set(tables))


def _get_revision_state(*, database_url: str, cfg) -> Dict[str, Any]:
    from alembic.script import ScriptDirectory
    from alembic.runtime.migration import MigrationContext

    engine: Engine = create_engine(database_url, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            current = ctx.get_current_revision()
            tables = set(inspect(conn).get_table_names())
        script = ScriptDirectory.from_config(cfg)
        head = script.get_current_head()
        return {
            "current_revision": current,
            "head_revision": head,
            "tables": tables,
        }
    finally:
        engine.dispose()


def _build_alembic_config(*, database_url: str):
    from alembic.config import Config

    prod_dir = Path(__file__).resolve().parent
    cfg = Config(str(prod_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(prod_dir / "alembic"))
    cfg.set_main_option("sqlalchemy.url", str(database_url))
    return cfg
