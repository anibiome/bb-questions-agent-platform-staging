from __future__ import annotations

from datetime import date
from pathlib import Path

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.pipeline.registry import load_registry
from questions_agent_platform.pipeline.service_monolith import (
    ensure_registry_active,
    get_or_create_daily_session,
    get_user_profile,
    submit_answers,
    upsert_user_profile,
)


def _bootstrap(tmp_path: Path):
    db_path = tmp_path / "qa.sqlite"
    registry_root = tmp_path / "registry"
    init_db(str(db_path))
    seed_demo_registry(str(registry_root))
    cfg = QuestionsAgentConfig(
        database_path=str(db_path),
        registry_root=str(registry_root),
        host="127.0.0.1",
        port=0,
    )
    return db_path, registry_root, cfg


def test_upsert_user_profile_hardens_string_booleans_and_scalar_domains(tmp_path: Path) -> None:
    db_path, registry_root, cfg = _bootstrap(tmp_path)

    with connect(str(db_path)) as conn:
        registry_version = ensure_registry_active(conn, str(registry_root))
        registry = load_registry(str(registry_root), registry_version)

        profile = upsert_user_profile(
            conn,
            user_id="u1",
            patch={
                "mode": "trial",
                "onboarding_complete": "false",
                "active_domains": "sleep",
                "queued_domains": "anxiety",
                "promoted_domains": "sleep",
            },
            registry=registry,
            cfg=cfg,
        )

        assert profile["mode"] == "trial"
        assert profile["onboarding_complete"] is False
        assert profile["active_domains"] == ["cardiometabolic"]
        assert "anxiety" in profile["queued_domains"]
        assert "sleep" in profile["promoted_domains"]


def test_service_monolith_hardens_context_answer_and_metadata_false_strings(tmp_path: Path) -> None:
    db_path, registry_root, cfg = _bootstrap(tmp_path)

    with connect(str(db_path)) as conn:
        session = get_or_create_daily_session(
            conn,
            cfg=cfg,
            registry_root=str(registry_root),
            user_id="u2",
            day=date(2026, 3, 6),
            policy_context={"safety_trigger_active": "false"},
        )
        assert conn.execute("SELECT COUNT(*) AS cnt FROM safety_events").fetchone()["cnt"] == 0

        item_id = session.core_item_ids[0]
        result = submit_answers(
            conn,
            registry_root=str(registry_root),
            user_id="u2",
            session_id=session.session_id,
            answers=[
                {
                    "client_event_id": "u2::2026-03-06::item-1",
                    "item_id": item_id,
                    "value": 2,
                    "answered_at": "2026-03-06T12:00:00Z",
                    "raw": {
                        "permanently_declined": "false",
                        "safety_trigger": "false",
                    },
                    "metadata": {
                        "response_latency_ms": 1200,
                        "edit_count": 1,
                        "was_skipped": "false",
                        "was_declined": "false",
                    },
                }
            ],
        )

        profile = get_user_profile(conn, "u2")
        metadata_row = conn.execute(
            "SELECT was_skipped, was_declined FROM response_metadata WHERE user_id=? ORDER BY created_at DESC LIMIT 1;",
            ("u2",),
        ).fetchone()
        safety_count = conn.execute("SELECT COUNT(*) AS cnt FROM safety_events").fetchone()["cnt"]

        assert result["inserted_answer_events"] == 1
        assert profile["permanently_declined_item_ids"] == []
        assert safety_count == 0
        assert metadata_row is not None
        assert int(metadata_row["was_skipped"]) == 0
        assert int(metadata_row["was_declined"]) == 0
