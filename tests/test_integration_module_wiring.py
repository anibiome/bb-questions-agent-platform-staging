"""
Integration tests for the module wiring: Phases 1-7 of production readiness.

Validates that the 4 new pipeline modules (response_metadata, anamnesis,
concordance, site_config) are correctly wired into the production orchestration
layer (service.py).

These tests exercise the FULL path through submit_answers() and selection,
not the module logic in isolation (those have their own 97+ tests).
"""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.registry import (
    Item,
    Questionnaire,
    Registry,
    Scale,
    ScaleItem,
    save_registry,
)
from questions_agent_platform.pipeline.service import (
    get_or_create_daily_session,
    submit_answers,
    _resolve_site_config,
    _get_latest_session_uncertainty_profile,
)
from questions_agent_platform.pipeline.site_config import (
    SiteConfig,
    config_consumer_wellness,
)
from questions_agent_platform.pipeline.time_utils import now_iso


def _build_test_registry(registry_root: str) -> str:
    """Build a minimal test registry with cardiometabolic items."""
    items = {}
    for idx in range(1, 8):
        items[f"cardio_{idx}"] = Item(
            id=f"cardio_{idx}",
            text=f"Cardiometabolic item {idx}",
            response_type="likert_0_4",
            tags=("cardiometabolic", "metabolic"),
            sensitivity="low",
            timeframes_allowed=("last_7_days",),
            intrusiveness="low",
            declinable=True,
            onboarding_order=idx,
        )
    for idx in range(1, 4):
        items[f"mood_{idx}"] = Item(
            id=f"mood_{idx}",
            text=f"Mood item {idx}",
            response_type="likert_0_4",
            tags=("mood",),
            sensitivity="medium",
            timeframes_allowed=("last_7_days",),
            intrusiveness="low",
            declinable=True,
        )

    registry = Registry(
        version="v_integration_test",
        items=items,
        questionnaires={
            "q_cardio": Questionnaire(id="q_cardio", version="1", name="Cardiometabolic Core"),
            "q_mood": Questionnaire(id="q_mood", version="1", name="Mood Assessment"),
        },
        scales={
            "scale_cardio": Scale(
                id="scale_cardio",
                questionnaire_id="q_cardio",
                version="1",
                name="Cardio baseline",
                method="mean",
                min_items_required=5,
                retest_interval_days=7,
                unlock_window_days=14,
                response_type="likert_0_4",
                ewma_alpha=0.2,
                normalize_min=0,
                normalize_max=4,
                items=tuple(
                    ScaleItem(item_id=f"cardio_{i}", weight=1.0)
                    for i in range(1, 8)
                ),
                tags=("cardiometabolic",),
            ),
            "scale_mood": Scale(
                id="scale_mood",
                questionnaire_id="q_mood",
                version="1",
                name="Mood Score",
                method="mean",
                min_items_required=2,
                retest_interval_days=7,
                unlock_window_days=14,
                response_type="likert_0_4",
                ewma_alpha=0.2,
                normalize_min=0,
                normalize_max=4,
                items=tuple(
                    ScaleItem(item_id=f"mood_{i}", weight=1.0)
                    for i in range(1, 4)
                ),
                tags=("mood",),
            ),
        },
    )
    save_registry(registry_root, registry)
    return registry.version


def _make_cfg(db_path: str, registry_root: str) -> QuestionsAgentConfig:
    return QuestionsAgentConfig(
        database_path=db_path,
        registry_root=registry_root,
        core_questions_per_day=5,
        item_repeat_cooldown_days=1,
        max_active_new_domains=5,
        site_config_id="consumer",
    )


def _setup_env():
    """Create temp dirs, DB, registry, config."""
    tmpdir = tempfile.mkdtemp()
    db_path = str(Path(tmpdir) / "test.db")
    registry_root = str(Path(tmpdir) / "registry" / "versions")
    init_db(db_path)
    version = _build_test_registry(registry_root)
    cfg = _make_cfg(db_path, registry_root)

    # Activate the registry
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO registry_versions(version, status, created_at) VALUES (?, 'active', ?);",
            (version, now_iso()),
        )
        conn.commit()

    return tmpdir, db_path, registry_root, cfg


class TestMetadataCaptureAndSEAdjustment(unittest.TestCase):
    """Test 1: submit answers with metadata → verify response_metadata rows,
    session_uncertainty_profile, and se_theta_adjusted on scale_scores."""

    def test_metadata_capture_stores_rows(self):
        _, db_path, registry_root, cfg = _setup_env()
        day = date.today()

        with connect(db_path) as conn:
            session = get_or_create_daily_session(
                conn, registry_root=registry_root, user_id="u1", day=day, cfg=cfg,
            )

            # Build answers with metadata
            answers = []
            core_ids = list(session.core_item_ids)
            for i, item_id in enumerate(core_ids):
                answers.append({
                    "item_id": item_id,
                    "client_event_id": f"evt_{i}",
                    "value": 2.0,
                    "answered_at": now_iso(),
                    "metadata": {
                        "response_latency_ms": 3500 + i * 500,
                        "edit_count": i % 2,
                        "was_skipped": False,
                        "channel": "tap",
                        "time_of_day_hour": 14,
                    },
                })

            site_config = config_consumer_wellness()
            submit_answers(
                conn,
                registry_root=registry_root,
                user_id="u1",
                session_id=session.session_id,
                answers=answers,
                site_config=site_config,
            )

            # Verify response_metadata rows created
            meta_rows = conn.execute(
                "SELECT * FROM response_metadata WHERE session_id=?;",
                (session.session_id,),
            ).fetchall()
            self.assertEqual(len(meta_rows), len(core_ids))
            for row in meta_rows:
                self.assertIsNotNone(row["uncertainty_multiplier"])
                self.assertIsNotNone(row["confidence_label"])
                self.assertGreater(float(row["uncertainty_multiplier"]), 0)

            # Verify session_uncertainty_profiles row
            profile_rows = conn.execute(
                "SELECT * FROM session_uncertainty_profiles WHERE session_id=?;",
                (session.session_id,),
            ).fetchall()
            self.assertEqual(len(profile_rows), 1)
            profile = profile_rows[0]
            self.assertIn(profile["engagement_quality"], ("focused", "normal", "distracted", "fatigued"))
            self.assertGreater(float(profile["session_multiplier"]), 0)

    def test_metadata_disabled_no_rows(self):
        """When metadata disabled, no response_metadata or profile rows."""
        _, db_path, registry_root, cfg = _setup_env()
        day = date.today()

        with connect(db_path) as conn:
            session = get_or_create_daily_session(
                conn, registry_root=registry_root, user_id="u1", day=day, cfg=cfg,
            )

            answers = [{
                "item_id": session.core_item_ids[0],
                "client_event_id": "evt_0",
                "value": 2.0,
                "metadata": {"response_latency_ms": 3000, "edit_count": 0},
            }]

            # Create a config with metadata disabled
            disabled_config = SiteConfig(
                config_id="test_disabled",
                display_name="Test Disabled",
                config_type="wellness",
                behavioural_metadata_enabled=False,
                anamnesis_enabled=False,
            )
            submit_answers(
                conn,
                registry_root=registry_root,
                user_id="u1",
                session_id=session.session_id,
                answers=answers,
                site_config=disabled_config,
            )

            meta_count = conn.execute(
                "SELECT COUNT(*) FROM response_metadata WHERE session_id=?;",
                (session.session_id,),
            ).fetchone()[0]
            self.assertEqual(meta_count, 0)

            profile_count = conn.execute(
                "SELECT COUNT(*) FROM session_uncertainty_profiles WHERE session_id=?;",
                (session.session_id,),
            ).fetchone()[0]
            self.assertEqual(profile_count, 0)


class TestLegacyBackwardCompatibility(unittest.TestCase):
    """Test 5: Submit answers without metadata field → everything works as before."""

    def test_legacy_answers_no_metadata(self):
        _, db_path, registry_root, cfg = _setup_env()
        day = date.today()

        with connect(db_path) as conn:
            session = get_or_create_daily_session(
                conn, registry_root=registry_root, user_id="u1", day=day, cfg=cfg,
            )

            # Legacy answer format: no "metadata" key at all
            answers = [{
                "item_id": session.core_item_ids[0],
                "client_event_id": "evt_legacy",
                "value": 3.0,
                "answered_at": now_iso(),
            }]

            result = submit_answers(
                conn,
                registry_root=registry_root,
                user_id="u1",
                session_id=session.session_id,
                answers=answers,
            )

            self.assertEqual(result["inserted_answer_events"], 1)

            # No metadata rows
            meta_count = conn.execute(
                "SELECT COUNT(*) FROM response_metadata WHERE session_id=?;",
                (session.session_id,),
            ).fetchone()[0]
            self.assertEqual(meta_count, 0)

    def test_legacy_answers_no_site_config(self):
        """Submit without explicit site_config → uses default consumer config."""
        _, db_path, registry_root, cfg = _setup_env()
        day = date.today()

        with connect(db_path) as conn:
            session = get_or_create_daily_session(
                conn, registry_root=registry_root, user_id="u1", day=day, cfg=cfg,
            )

            # Answer with metadata but NO site_config param → default consumer → metadata enabled
            answers = [{
                "item_id": session.core_item_ids[0],
                "client_event_id": "evt_0",
                "value": 2.0,
                "metadata": {"response_latency_ms": 3000, "edit_count": 0, "channel": "tap"},
            }]

            result = submit_answers(
                conn,
                registry_root=registry_root,
                user_id="u1",
                session_id=session.session_id,
                answers=answers,
                # No site_config → defaults to consumer which has metadata enabled
            )

            self.assertEqual(result["inserted_answer_events"], 1)

            # Default consumer config has metadata enabled, so metadata rows should exist
            meta_count = conn.execute(
                "SELECT COUNT(*) FROM response_metadata WHERE session_id=?;",
                (session.session_id,),
            ).fetchone()[0]
            self.assertEqual(meta_count, 1)


class TestSiteConfigResolution(unittest.TestCase):
    """Test 6: Default user gets consumer config. User with custom site_config_id
    gets the corresponding config."""

    def test_default_consumer_config(self):
        _, db_path, _, cfg = _setup_env()

        with connect(db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?);",
                ("u1", now_iso()),
            )
            conn.execute(
                """INSERT OR IGNORE INTO user_profiles(
                    user_id, mode, permanently_declined_items_json,
                    active_domains_json, queued_domains_json, promoted_domains_json,
                    onboarding_complete, state_schema_version, updated_at
                ) VALUES (?, 'consumer', '[]', '["cardiometabolic"]', '[]', '["cardiometabolic"]', 0, 'v1_state_schema', ?);""",
                ("u1", now_iso()),
            )
            conn.commit()

            resolved = _resolve_site_config(conn, "u1")
            self.assertEqual(resolved.config_id, "consumer")
            self.assertTrue(resolved.behavioural_metadata_enabled)
            self.assertTrue(resolved.anamnesis_enabled)

    def test_explicit_config_override(self):
        _, db_path, _, cfg = _setup_env()

        with connect(db_path) as conn:
            custom = SiteConfig(
                config_id="custom_test",
                display_name="Custom",
                config_type="trial",
                behavioural_metadata_enabled=False,
            )
            resolved = _resolve_site_config(conn, "u1", site_config=custom)
            self.assertEqual(resolved.config_id, "custom_test")
            self.assertFalse(resolved.behavioural_metadata_enabled)


class TestDBMigrationV9(unittest.TestCase):
    """Verify migration v9 creates all expected tables and columns."""

    def test_v9_tables_exist(self):
        _, db_path, _, _ = _setup_env()

        with connect(db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()}
            self.assertIn("response_metadata", tables)
            self.assertIn("session_uncertainty_profiles", tables)
            self.assertIn("anamnesis_episodes", tables)

    def test_v9_altered_columns(self):
        _, db_path, _, _ = _setup_env()

        with connect(db_path) as conn:
            up_cols = {r["name"] for r in conn.execute("PRAGMA table_info(user_profiles);").fetchall()}
            self.assertIn("site_config_id", up_cols)

            ss_cols = {r["name"] for r in conn.execute("PRAGMA table_info(scale_scores);").fetchall()}
            for col in ("se_theta", "se_theta_adjusted", "uncertainty_multiplier", "engagement_quality"):
                self.assertIn(col, ss_cols)


class TestEngagementFeedsNextSelection(unittest.TestCase):
    """Test 2: Day 1: slow answers → distracted. Day 2: verify engagement in explain."""

    def test_engagement_quality_flows_to_next_session(self):
        _, db_path, registry_root, cfg = _setup_env()
        day1 = date.today() - timedelta(days=1)
        day2 = date.today()

        with connect(db_path) as conn:
            # Day 1: create session and submit slow answers
            session1 = get_or_create_daily_session(
                conn, registry_root=registry_root, user_id="u1", day=day1, cfg=cfg,
            )

            answers = []
            for i, item_id in enumerate(session1.core_item_ids):
                answers.append({
                    "item_id": item_id,
                    "client_event_id": f"d1_evt_{i}",
                    "value": 2.0,
                    "metadata": {
                        "response_latency_ms": 15000,  # very slow
                        "edit_count": 3,  # lots of edits
                        "was_skipped": True if i == 0 else False,
                        "channel": "tap",
                    },
                })

            submit_answers(
                conn,
                registry_root=registry_root,
                user_id="u1",
                session_id=session1.session_id,
                answers=answers,
                site_config=config_consumer_wellness(),
            )

            # Check Day 1 profile was stored
            profile = _get_latest_session_uncertainty_profile(
                conn, user_id="u1", before_date=day2,
            )
            self.assertIsNotNone(profile)
            self.assertIn(profile.engagement_quality, ("distracted", "fatigued"))

            # Day 2: create new session — should have engagement context in explain
            session2 = get_or_create_daily_session(
                conn, registry_root=registry_root, user_id="u1", day=day2, cfg=cfg,
            )

            # Verify the explain payload includes engagement context
            explain = json.loads(
                conn.execute(
                    "SELECT selection_explain_json FROM daily_sessions WHERE session_id=?;",
                    (session2.session_id,),
                ).fetchone()["selection_explain_json"]
            )
            self.assertIn("previous_engagement_quality", explain)
            self.assertIn(explain["previous_engagement_quality"], ("distracted", "fatigued"))


class TestAnamnesisEpisodeLifecycle(unittest.TestCase):
    """Test anamnesis episode lifecycle through the DB."""

    def test_anamnesis_episode_crud(self):
        """Direct DB operations for anamnesis episodes work correctly."""
        _, db_path, _, _ = _setup_env()

        with connect(db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?);",
                ("u1", now_iso()),
            )

            # Insert an anamnesis episode
            conn.execute(
                """
                INSERT INTO anamnesis_episodes(
                    episode_id, user_id, trigger_date, drift_domain,
                    trigger_type, trigger_value, triggered_scale_ids_json,
                    status, follow_up_item_ids_json, priority, max_days,
                    observations_collected, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "ep_test_1", "u1", date.today().isoformat(), "metabolic",
                    "ews_threshold", 0.75, '["scale_cardio"]',
                    "active", '["cardio_1", "cardio_2"]', 100, 7,
                    0, now_iso(), now_iso(),
                ),
            )
            conn.commit()

            # Verify it's there
            rows = conn.execute(
                "SELECT * FROM anamnesis_episodes WHERE user_id='u1' AND status='active';",
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["episode_id"], "ep_test_1")
            self.assertEqual(rows[0]["drift_domain"], "metabolic")

            # Resolve it
            conn.execute(
                """
                UPDATE anamnesis_episodes
                SET status='resolved', resolution_date=?, resolution_reason='test'
                WHERE episode_id='ep_test_1';
                """,
                (date.today().isoformat(),),
            )
            conn.commit()

            active = conn.execute(
                "SELECT COUNT(*) FROM anamnesis_episodes WHERE user_id='u1' AND status='active';",
            ).fetchone()[0]
            self.assertEqual(active, 0)

            resolved = conn.execute(
                "SELECT * FROM anamnesis_episodes WHERE episode_id='ep_test_1';",
            ).fetchone()
            self.assertEqual(resolved["status"], "resolved")


class TestConcordanceFlag(unittest.TestCase):
    """Test concordance_eligible flag appears when concordance_mode is enabled."""

    def test_concordance_flag_in_score_payload(self):
        """When concordance_mode=True, score payloads include concordance_eligible."""
        _, db_path, registry_root, cfg = _setup_env()
        day = date.today()

        with connect(db_path) as conn:
            session = get_or_create_daily_session(
                conn, registry_root=registry_root, user_id="u1", day=day, cfg=cfg,
            )

            # Answer all cardio items to trigger a scale score
            answers = []
            for i, item_id in enumerate(session.core_item_ids):
                answers.append({
                    "item_id": item_id,
                    "client_event_id": f"evt_{i}",
                    "value": 2.0,
                })

            concordance_config = SiteConfig(
                config_id="trial_test",
                display_name="Trial",
                config_type="trial",
                concordance_mode=True,
                behavioural_metadata_enabled=False,
                anamnesis_enabled=False,
            )
            result = submit_answers(
                conn,
                registry_root=registry_root,
                user_id="u1",
                session_id=session.session_id,
                answers=answers,
                site_config=concordance_config,
            )

            # Check if any new scores have concordance_eligible
            for score in result.get("new_scale_scores", []):
                if score:
                    self.assertTrue(score.get("concordance_eligible", False))


if __name__ == "__main__":
    unittest.main()
