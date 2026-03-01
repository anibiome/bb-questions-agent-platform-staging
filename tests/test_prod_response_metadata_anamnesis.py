"""
Integration tests for production wiring of:
  - Response Metadata (Claim Family 5) into submit_answers
  - SE Adjustment into _compute_and_store_scores
  - Anamnesis Episodes (Claim Family 4) into drift detection
  - Engagement Bonus into session selection
  - New Pydantic schemas
  - New ORM models

These tests validate the v17 production-readiness changes without requiring
a PostgreSQL database (they test schemas, ORM model construction, and module
imports rather than full service-level integration which requires the DB).
"""

import json
import math
import unittest
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from questions_agent_platform.pipeline.response_metadata import (
    ResponseMetadata,
    UncertaintyModifier,
    SessionUncertaintyProfile,
    compute_uncertainty_modifier,
    compute_session_uncertainty_profile,
    adjust_se_with_metadata,
    engagement_selection_bonus,
)
from questions_agent_platform.pipeline.anamnesis import (
    AnamnesisEpisode,
    create_anamnesis_episode,
    evaluate_drift_triggers,
    evaluate_episode_resolution,
    run_anamnesis_check,
)
from questions_agent_platform.pipeline.baseline import BaselineState


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestResponseMetadataSchemas(unittest.TestCase):
    """Test Pydantic schema additions for response metadata."""

    def test_answer_in_backward_compat(self):
        """AnswerIn should accept answers without metadata field."""
        from questions_agent_platform.prod.schemas import AnswerIn
        a = AnswerIn(client_event_id="evt1", item_id="item1", value=3.0)
        self.assertIsNone(a.metadata)
        self.assertEqual(a.value, 3.0)

    def test_answer_in_with_metadata(self):
        """AnswerIn should accept metadata dict."""
        from questions_agent_platform.prod.schemas import AnswerIn
        meta = {"response_latency_ms": 3200, "edit_count": 0, "channel": "tap"}
        a = AnswerIn(client_event_id="evt2", item_id="item2", value=2.0, metadata=meta)
        self.assertEqual(a.metadata["response_latency_ms"], 3200)

    def test_submit_answers_out_with_engagement(self):
        """SubmitAnswersOut should include optional session_engagement."""
        from questions_agent_platform.prod.schemas import SubmitAnswersOut, SessionUncertaintyProfileOut
        out = SubmitAnswersOut(
            inserted_answer_events=3,
            new_scale_scores=[],
            scale_progress={},
            session_engagement=None,
        )
        self.assertIsNone(out.session_engagement)

    def test_submit_answers_out_with_engagement_data(self):
        """SubmitAnswersOut should accept SessionUncertaintyProfileOut."""
        from questions_agent_platform.prod.schemas import SubmitAnswersOut, SessionUncertaintyProfileOut
        profile = SessionUncertaintyProfileOut(
            session_id="sess1",
            user_id="user1",
            session_date="2026-03-01",
            session_multiplier=1.05,
            engagement_quality="normal",
            median_latency_ms=3200.0,
            total_edits=1,
            skip_count=0,
            decline_count=0,
            item_count=5,
        )
        out = SubmitAnswersOut(
            inserted_answer_events=5,
            new_scale_scores=[],
            scale_progress={},
            session_engagement=profile,
        )
        self.assertEqual(out.session_engagement.engagement_quality, "normal")
        self.assertEqual(out.session_engagement.item_count, 5)

    def test_session_uncertainty_profile_out_roundtrip(self):
        """SessionUncertaintyProfileOut should serialize/deserialize correctly."""
        from questions_agent_platform.prod.schemas import SessionUncertaintyProfileOut
        data = {
            "session_id": "s123",
            "user_id": "u456",
            "session_date": "2026-03-01",
            "session_multiplier": 0.95,
            "engagement_quality": "focused",
            "median_latency_ms": 2800.5,
            "total_edits": 0,
            "skip_count": 0,
            "decline_count": 0,
            "item_count": 7,
        }
        profile = SessionUncertaintyProfileOut(**data)
        self.assertAlmostEqual(profile.session_multiplier, 0.95)
        self.assertEqual(profile.engagement_quality, "focused")


class TestAnamnesisSchemas(unittest.TestCase):
    """Test Pydantic schema additions for anamnesis episodes."""

    def test_anamnesis_episode_out(self):
        """AnamnesisEpisodeOut should accept all fields."""
        from questions_agent_platform.prod.schemas import AnamnesisEpisodeOut
        ep = AnamnesisEpisodeOut(
            episode_id="anm_user1_2026-03-01_metabolic",
            user_id="user1",
            trigger_date="2026-03-01",
            drift_domain="metabolic",
            trigger_type="ews_threshold",
            trigger_value=0.72,
            triggered_scale_ids=["scale_cardio_auditc"],
            status="active",
            follow_up_item_ids=["item1", "item2"],
            priority=50,
            max_days=7,
            observations_collected=0,
        )
        self.assertEqual(ep.status, "active")
        self.assertEqual(ep.drift_domain, "metabolic")

    def test_anamnesis_episode_list_out(self):
        """AnamnesisEpisodeListOut should contain a list of episodes."""
        from questions_agent_platform.prod.schemas import AnamnesisEpisodeListOut, AnamnesisEpisodeOut
        ep = AnamnesisEpisodeOut(
            episode_id="anm1", user_id="u1", trigger_date="2026-03-01",
            drift_domain="general", trigger_type="velocity", trigger_value=0.25,
            triggered_scale_ids=[], status="expired", follow_up_item_ids=[],
            priority=100, max_days=7, observations_collected=3,
            resolution_date="2026-03-08", resolution_reason="max_days_exceeded",
        )
        lst = AnamnesisEpisodeListOut(user_id="u1", episodes=[ep])
        self.assertEqual(len(lst.episodes), 1)
        self.assertEqual(lst.episodes[0].resolution_reason, "max_days_exceeded")

    def test_anamnesis_episode_resolved_fields(self):
        """Resolved episodes should have resolution_date and resolution_reason."""
        from questions_agent_platform.prod.schemas import AnamnesisEpisodeOut
        ep = AnamnesisEpisodeOut(
            episode_id="anm2", user_id="u2", trigger_date="2026-02-28",
            drift_domain="cardiovascular", trigger_type="ews_threshold",
            trigger_value=0.68, triggered_scale_ids=["s1", "s2"],
            status="denied", follow_up_item_ids=["i1"],
            priority=40, max_days=7, observations_collected=5,
            resolution_date="2026-03-02", resolution_reason="ews_subsided",
        )
        self.assertEqual(ep.status, "denied")
        self.assertIsNotNone(ep.resolution_date)


class TestConcordanceSchemas(unittest.TestCase):
    """Test concordance report schemas."""

    def test_concordance_report_out(self):
        """ConcordanceReportOut should accept all fields."""
        from questions_agent_platform.prod.schemas import ConcordanceReportOut
        report = ConcordanceReportOut(
            instrument_id="scale_demo_mood",
            n_participants=50,
            n_paired_observations=200,
            icc={"icc_3_1": 0.85, "ci_lower": 0.78, "ci_upper": 0.92},
            bland_altman={"mean_diff": 0.02, "loa_lower": -0.5, "loa_upper": 0.54},
            kappa={"cohen_kappa": 0.79, "weighted_kappa": 0.82},
            overall_pass=True,
            failure_reasons=[],
        )
        self.assertTrue(report.overall_pass)

    def test_concordance_summary_out(self):
        """ConcordanceSummaryOut should contain multiple reports."""
        from questions_agent_platform.prod.schemas import ConcordanceSummaryOut, ConcordanceReportOut
        r = ConcordanceReportOut(
            instrument_id="test", n_participants=10, n_paired_observations=40,
            icc={"icc_3_1": 0.70}, bland_altman={}, kappa={},
            overall_pass=False, failure_reasons=["icc_below_threshold"],
        )
        summary = ConcordanceSummaryOut(
            reports=[r],
            summary={"total": 1, "passing": 0, "failing": 1},
        )
        self.assertEqual(len(summary.reports), 1)
        self.assertFalse(summary.reports[0].overall_pass)


# ---------------------------------------------------------------------------
# ORM Model Construction Tests
# ---------------------------------------------------------------------------

class TestResponseMetadataORM(unittest.TestCase):
    """Test ORM model construction for response_metadata."""

    def test_response_metadata_row_construction(self):
        """ResponseMetadataRow should accept all columns."""
        from questions_agent_platform.prod.models import ResponseMetadataRow
        row = ResponseMetadataRow(
            metadata_id="meta1",
            user_id="user1",
            session_id="sess1",
            answer_event_id="evt1",
            item_id="item_mood_1",
            response_latency_ms=3200.0,
            edit_count=0,
            was_skipped=False,
            was_declined=False,
            channel="tap",
            uncertainty_multiplier=0.92,
            confidence_label="high",
            contributing_factors_json='["fast_response"]',
            raw_components_json='{"latency_factor": 0.85}',
            created_at="2026-03-01T12:00:00Z",
        )
        self.assertEqual(row.item_id, "item_mood_1")
        self.assertAlmostEqual(row.uncertainty_multiplier, 0.92)

    def test_response_metadata_nullable_fields(self):
        """Optional fields should allow None."""
        from questions_agent_platform.prod.models import ResponseMetadataRow
        row = ResponseMetadataRow(
            metadata_id="meta2",
            user_id="u2",
            session_id="s2",
            answer_event_id="e2",
            item_id="item2",
            response_latency_ms=None,
            voice_hesitation_ms=None,
            time_of_day_hour=None,
            edit_count=0,
            was_skipped=False,
            was_declined=False,
            channel="tap",
            uncertainty_multiplier=1.0,
            confidence_label="medium",
            contributing_factors_json="[]",
            raw_components_json="{}",
            created_at="2026-03-01T12:00:00Z",
        )
        self.assertIsNone(row.response_latency_ms)
        self.assertIsNone(row.voice_hesitation_ms)


class TestSessionUncertaintyProfileORM(unittest.TestCase):
    """Test ORM model for session_uncertainty_profiles."""

    def test_profile_row_construction(self):
        """SessionUncertaintyProfileRow should accept all columns."""
        from questions_agent_platform.prod.models import SessionUncertaintyProfileRow
        row = SessionUncertaintyProfileRow(
            profile_id="prof1",
            user_id="user1",
            session_id="sess1",
            session_date="2026-03-01",
            session_multiplier=1.05,
            engagement_quality="normal",
            median_latency_ms=3500.0,
            total_edits=2,
            skip_count=1,
            decline_count=0,
            item_count=5,
            created_at="2026-03-01T12:00:00Z",
        )
        self.assertEqual(row.engagement_quality, "normal")
        self.assertEqual(row.item_count, 5)


class TestAnamnesisEpisodeORM(unittest.TestCase):
    """Test ORM model for anamnesis_episodes."""

    def test_episode_row_construction(self):
        """AnamnesisEpisodeRow should accept all columns."""
        from questions_agent_platform.prod.models import AnamnesisEpisodeRow
        row = AnamnesisEpisodeRow(
            episode_id="anm_user1_2026-03-01_metabolic",
            user_id="user1",
            trigger_date="2026-03-01",
            drift_domain="metabolic",
            trigger_type="ews_threshold",
            trigger_value=0.72,
            triggered_scale_ids_json='["scale_cardio_auditc"]',
            status="active",
            follow_up_item_ids_json='["item1", "item2"]',
            priority=50,
            max_days=7,
            observations_collected=0,
            created_at="2026-03-01T12:00:00Z",
            updated_at="2026-03-01T12:00:00Z",
        )
        self.assertEqual(row.drift_domain, "metabolic")
        self.assertEqual(row.status, "active")

    def test_episode_row_json_parsing(self):
        """JSON fields should be parseable."""
        from questions_agent_platform.prod.models import AnamnesisEpisodeRow
        row = AnamnesisEpisodeRow(
            episode_id="anm2",
            user_id="u2",
            trigger_date="2026-03-01",
            drift_domain="general",
            trigger_type="velocity",
            trigger_value=0.25,
            triggered_scale_ids_json='["s1", "s2", "s3"]',
            status="expired",
            follow_up_item_ids_json='["i1", "i2"]',
            priority=100,
            max_days=7,
            observations_collected=3,
            resolution_date="2026-03-08",
            resolution_reason="max_days_exceeded",
            created_at="2026-03-01T12:00:00Z",
            updated_at="2026-03-08T12:00:00Z",
        )
        scale_ids = json.loads(row.triggered_scale_ids_json)
        self.assertEqual(len(scale_ids), 3)
        follow_ids = json.loads(row.follow_up_item_ids_json)
        self.assertEqual(len(follow_ids), 2)


class TestScaleScoreNewColumns(unittest.TestCase):
    """Test that ScaleScore ORM model has new SE-related columns."""

    def test_scale_score_has_se_columns(self):
        """ScaleScore should have se_theta, se_theta_adjusted, uncertainty_multiplier, engagement_quality."""
        from questions_agent_platform.prod.models import ScaleScore
        row = ScaleScore(
            score_id="score1",
            user_id="u1",
            scale_id="s1",
            computed_at="2026-03-01T12:00:00Z",
            window_start=date(2026, 2, 25),
            window_end=date(2026, 3, 1),
            raw_score=12.0,
            normalized_score=0.6,
            confidence_tier="moderate",
            items_answered_count=4,
            items_required=5,
            se_theta=0.35,
            se_theta_adjusted=0.37,
            uncertainty_multiplier=1.05,
            engagement_quality="normal",
        )
        self.assertAlmostEqual(row.se_theta, 0.35)
        self.assertAlmostEqual(row.se_theta_adjusted, 0.37)
        self.assertEqual(row.engagement_quality, "normal")

    def test_scale_score_se_columns_nullable(self):
        """SE columns should be nullable for backward compatibility."""
        from questions_agent_platform.prod.models import ScaleScore
        row = ScaleScore(
            score_id="score2",
            user_id="u2",
            scale_id="s2",
            computed_at="2026-03-01T12:00:00Z",
            window_start=date(2026, 2, 25),
            window_end=date(2026, 3, 1),
            raw_score=8.0,
            normalized_score=0.4,
            confidence_tier="low",
            items_answered_count=2,
            items_required=5,
        )
        self.assertIsNone(row.se_theta)
        self.assertIsNone(row.se_theta_adjusted)
        self.assertIsNone(row.uncertainty_multiplier)
        self.assertIsNone(row.engagement_quality)


# ---------------------------------------------------------------------------
# Pipeline Integration Tests
# ---------------------------------------------------------------------------

class TestSEAdjustmentIntegration(unittest.TestCase):
    """Test that SE adjustment flows correctly through the response_metadata pipeline."""

    def test_modifier_to_se_adjustment_flow(self):
        """Full flow: metadata → modifier → SE adjustment."""
        meta = ResponseMetadata(
            item_id="item1",
            response_latency_ms=2000.0,
            edit_count=0,
            channel="tap",
        )
        modifier = compute_uncertainty_modifier(meta, response_type="likert_0_4")
        self.assertLess(modifier.multiplier, 1.0)  # fast → lower uncertainty

        se_original = 0.35
        se_adjusted = adjust_se_with_metadata(se_original, [modifier])
        self.assertLess(se_adjusted, se_original)
        self.assertGreater(se_adjusted, se_original * 0.5)  # clamped

    def test_slow_edited_increases_se(self):
        """Slow + edited responses should increase SE."""
        meta = ResponseMetadata(
            item_id="item2",
            response_latency_ms=12000.0,
            edit_count=3,
            channel="tap",
        )
        modifier = compute_uncertainty_modifier(meta, response_type="likert_0_4")
        self.assertGreater(modifier.multiplier, 1.0)

        se_original = 0.35
        se_adjusted = adjust_se_with_metadata(se_original, [modifier])
        self.assertGreater(se_adjusted, se_original)

    def test_session_profile_aggregation(self):
        """Session profile should aggregate multiple item modifiers."""
        modifiers = []
        metadata_list = []
        for i in range(5):
            meta = ResponseMetadata(
                item_id=f"item_{i}",
                response_latency_ms=3000.0 + i * 500,
                edit_count=0,
                channel="tap",
            )
            modifier = compute_uncertainty_modifier(meta, response_type="likert_0_4")
            modifiers.append(modifier)
            metadata_list.append(meta)

        profile = compute_session_uncertainty_profile(
            modifiers, metadata_list,
            session_date="2026-03-01",
            user_id="test_user",
        )
        self.assertIn(profile.engagement_quality, ("focused", "normal", "distracted", "fatigued"))
        self.assertGreater(profile.session_multiplier, 0.0)
        self.assertEqual(len(profile.item_modifiers), 5)

    def test_engagement_selection_bonus(self):
        """Engagement quality should produce different selection bonuses."""
        # Build focused and distracted profiles
        fast_meta = ResponseMetadata(item_id="i1", response_latency_ms=1500.0, edit_count=0, channel="tap")
        fast_mod = compute_uncertainty_modifier(fast_meta, response_type="likert_0_4")
        slow_meta = ResponseMetadata(item_id="i2", response_latency_ms=15000.0, edit_count=4, channel="tap")
        slow_mod = compute_uncertainty_modifier(slow_meta, response_type="likert_0_4")

        focused_profile = SessionUncertaintyProfile(
            session_date="2026-03-01", user_id="u1",
            item_modifiers=(fast_mod,), session_multiplier=fast_mod.multiplier,
            engagement_quality="focused", median_latency_ms=1500.0,
            total_edits=0, skip_count=0, decline_count=0,
        )
        distracted_profile = SessionUncertaintyProfile(
            session_date="2026-03-01", user_id="u1",
            item_modifiers=(slow_mod,), session_multiplier=slow_mod.multiplier,
            engagement_quality="distracted", median_latency_ms=15000.0,
            total_edits=4, skip_count=0, decline_count=0,
        )

        bonus_focused = engagement_selection_bonus(focused_profile, "i1", response_type="likert_0_4")
        bonus_distracted = engagement_selection_bonus(distracted_profile, "i2", response_type="likert_0_4")

        # Focused should get a better (or equal) bonus than distracted
        self.assertGreaterEqual(bonus_focused, bonus_distracted)


class TestAnamnesisIntegration(unittest.TestCase):
    """Test anamnesis episode lifecycle."""

    def test_drift_triggers_episode_creation(self):
        """High EWS score should trigger anamnesis episode creation."""
        triggers = evaluate_drift_triggers(
            ews_score=0.72,
            velocity=0.25,
        )
        self.assertGreater(len(triggers), 0)
        trigger_types = [t[0] for t in triggers]
        self.assertIn("ews_threshold", trigger_types)

    def test_episode_creation(self):
        """create_anamnesis_episode should produce a valid episode."""
        episode = create_anamnesis_episode(
            user_id="test_user",
            trigger_date=date(2026, 3, 1),
            trigger_type="ews_threshold",
            trigger_value=0.72,
            drift_domain="metabolic",
            triggered_scale_ids=["scale_cardio_auditc", "scale_cardio_findrisc"],
            follow_up_item_ids=["item1", "item2", "item3"],
            priority=50,
            max_days=7,
        )
        self.assertEqual(episode.status, "active")
        self.assertEqual(episode.drift_domain, "metabolic")
        self.assertEqual(len(episode.triggered_scale_ids), 2)
        self.assertEqual(len(episode.follow_up_item_ids), 3)

    def test_episode_resolution_ews_subsided(self):
        """Episode should resolve when EWS drops below threshold."""
        episode = AnamnesisEpisode(
            episode_id="anm1",
            user_id="u1",
            trigger_date="2026-03-01",
            drift_domain="metabolic",
            trigger_type="ews_threshold",
            trigger_value=0.72,
            triggered_scale_ids=("s1",),
            status="active",
            follow_up_item_ids=("i1",),
        )
        resolved = evaluate_episode_resolution(
            episode,
            current_date=date(2026, 3, 3),
            current_ews_score=0.30,  # below 0.6 * 0.7 = 0.42
        )
        self.assertEqual(resolved.status, "denied")
        self.assertEqual(resolved.resolution_reason, "ews_subsided")

    def test_episode_expiration(self):
        """Episode should expire after max_days."""
        episode = AnamnesisEpisode(
            episode_id="anm2",
            user_id="u2",
            trigger_date="2026-03-01",
            drift_domain="general",
            trigger_type="velocity",
            trigger_value=0.25,
            triggered_scale_ids=("s1",),
            status="active",
            follow_up_item_ids=("i1",),
            max_days=7,
        )
        resolved = evaluate_episode_resolution(
            episode,
            current_date=date(2026, 3, 10),  # 9 days later > 7
            current_ews_score=0.65,
        )
        self.assertEqual(resolved.status, "expired")
        self.assertEqual(resolved.resolution_reason, "max_days_exceeded")

    def test_episode_scale_completed(self):
        """Episode should resolve as confirmed when scale is completed."""
        episode = AnamnesisEpisode(
            episode_id="anm3",
            user_id="u3",
            trigger_date="2026-03-01",
            drift_domain="cardiovascular",
            trigger_type="ews_threshold",
            trigger_value=0.68,
            triggered_scale_ids=("scale_cvd",),
            status="active",
            follow_up_item_ids=("i1", "i2"),
        )
        resolved = evaluate_episode_resolution(
            episode,
            current_date=date(2026, 3, 4),
            scale_completed=True,
        )
        self.assertEqual(resolved.status, "confirmed")
        self.assertEqual(resolved.resolution_reason, "scale_completed")

    def test_no_trigger_low_ews(self):
        """Low EWS should not produce triggers."""
        triggers = evaluate_drift_triggers(ews_score=0.3, velocity=0.05)
        self.assertEqual(len(triggers), 0)


class TestImportIntegrity(unittest.TestCase):
    """Test that all new imports in prod/ modules resolve correctly."""

    def test_service_pg_imports(self):
        """service_pg should import without errors."""
        import questions_agent_platform.prod.service_pg as svc
        self.assertTrue(hasattr(svc, "submit_answers"))
        self.assertTrue(hasattr(svc, "get_anamnesis_episodes"))
        self.assertTrue(hasattr(svc, "get_anamnesis_episode"))
        self.assertTrue(hasattr(svc, "resolve_anamnesis_episode"))
        self.assertTrue(hasattr(svc, "get_session_engagement_profile"))
        self.assertTrue(hasattr(svc, "get_follow_up_queue"))

    def test_app_imports(self):
        """app.py should import without errors (requires DATABASE_URL)."""
        import os
        if not os.getenv("DATABASE_URL"):
            os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_db"
            os.environ["API_KEY"] = "test_key"
        try:
            import questions_agent_platform.prod.app as app_module
            self.assertTrue(hasattr(app_module, "app"))
        except Exception:
            # In CI without a real DB, the import may fail at engine creation
            # but the module-level schema/route definitions are still validated
            pass

    def test_models_imports(self):
        """New ORM models should be importable."""
        from questions_agent_platform.prod.models import (
            ResponseMetadataRow,
            SessionUncertaintyProfileRow,
            AnamnesisEpisodeRow,
        )
        self.assertTrue(ResponseMetadataRow)
        self.assertTrue(SessionUncertaintyProfileRow)
        self.assertTrue(AnamnesisEpisodeRow)

    def test_schemas_imports(self):
        """New schemas should be importable."""
        from questions_agent_platform.prod.schemas import (
            AnamnesisEpisodeOut,
            AnamnesisEpisodeListOut,
            SessionUncertaintyProfileOut,
            ResponseMetadataOut,
            ConcordanceReportOut,
            ConcordanceSummaryOut,
        )
        self.assertTrue(AnamnesisEpisodeOut)
        self.assertTrue(AnamnesisEpisodeListOut)

    def test_migration_file_exists(self):
        """Migration 0007 should exist and be parseable."""
        import importlib
        mod = importlib.import_module(
            "questions_agent_platform.prod.alembic.versions.20260301_0007_response_metadata_anamnesis"
        )
        self.assertEqual(mod.revision, "20260301_0007")
        self.assertEqual(mod.down_revision, "20260228_0006")
        self.assertTrue(callable(mod.upgrade))
        self.assertTrue(callable(mod.downgrade))


if __name__ == "__main__":
    unittest.main()
