"""Integration tests for production wiring of N-of-1, coherence, and cardio risk modules.

Tests the Pydantic schemas, ORM models, and service functions added in v15.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from questions_agent_platform.prod.schemas import (
    CalibrationListOut,
    CalibrationOut,
    CardioRiskHistoryOut,
    CardioRiskOut,
    CoherenceHistoryOut,
    CoherenceOut,
)


class TestCalibrationSchemas(unittest.TestCase):
    """Validate N-of-1 calibration Pydantic schemas."""

    def test_calibration_out_minimal(self) -> None:
        c = CalibrationOut(
            user_id="u1",
            scale_id="scale_phq9",
            phase="warming",
            n_observations=3,
            theta_personal=0.45,
            se_personal=0.22,
            shrinkage=0.1,
        )
        self.assertEqual(c.user_id, "u1")
        self.assertEqual(c.phase, "warming")
        self.assertIsNone(c.reliable_change)
        self.assertIsNone(c.sufficiency)

    def test_calibration_out_with_rci(self) -> None:
        c = CalibrationOut(
            user_id="u1",
            scale_id="scale_phq9",
            phase="calibrated",
            n_observations=15,
            theta_personal=1.2,
            se_personal=0.15,
            theta_baseline=0.5,
            se_baseline=0.18,
            within_person_sd=0.3,
            shrinkage=0.7,
            reliable_change={
                "rci": 2.45,
                "p_value": 0.014,
                "significant": True,
                "direction": "improvement",
                "magnitude": "moderate",
                "delta_theta": 0.7,
                "mid_exceeded": True,
            },
            sufficiency={
                "se_current": 0.15,
                "se_target": 0.25,
                "precision_ratio": 0.6,
                "sufficient": True,
                "reliability": 0.92,
            },
        )
        self.assertEqual(c.phase, "calibrated")
        self.assertTrue(c.reliable_change["significant"])
        self.assertTrue(c.sufficiency["sufficient"])

    def test_calibration_list_out(self) -> None:
        lst = CalibrationListOut(
            user_id="u1",
            calibrations=[
                CalibrationOut(
                    user_id="u1",
                    scale_id="scale_a",
                    phase="warming",
                    n_observations=1,
                    theta_personal=0.0,
                    se_personal=1.0,
                    shrinkage=0.0,
                ),
            ],
        )
        self.assertEqual(len(lst.calibrations), 1)
        d = lst.model_dump()
        self.assertEqual(d["user_id"], "u1")
        self.assertEqual(len(d["calibrations"]), 1)


class TestCoherenceSchemas(unittest.TestCase):
    """Validate coherence detection Pydantic schemas."""

    def test_coherence_out_valid(self) -> None:
        c = CoherenceOut(
            session_id="s1",
            user_id="u1",
            date="2026-02-27",
            coherence_score=0.85,
            tier="valid",
            n_flagged=1,
            n_items=5,
            signals=[
                {"name": "longstring", "flagged": False, "value": 0.2, "threshold": 0.5},
                {"name": "response_time", "flagged": True, "value": 0.8, "threshold": 0.6},
            ],
        )
        self.assertEqual(c.tier, "valid")
        self.assertEqual(c.n_flagged, 1)
        self.assertEqual(len(c.signals), 2)

    def test_coherence_history_out(self) -> None:
        h = CoherenceHistoryOut(
            user_id="u1",
            assessments=[
                CoherenceOut(
                    session_id="s1",
                    user_id="u1",
                    date="2026-02-26",
                    coherence_score=0.9,
                    tier="valid",
                    n_flagged=0,
                    n_items=5,
                    signals=[],
                ),
                CoherenceOut(
                    session_id="s2",
                    user_id="u1",
                    date="2026-02-27",
                    coherence_score=0.6,
                    tier="suspect",
                    n_flagged=3,
                    n_items=5,
                    signals=[],
                ),
            ],
        )
        self.assertEqual(len(h.assessments), 2)
        self.assertEqual(h.assessments[1].tier, "suspect")


class TestCardioRiskSchemas(unittest.TestCase):
    """Validate cardiometabolic risk Pydantic schemas."""

    def test_cardio_risk_out(self) -> None:
        r = CardioRiskOut(
            user_id="u1",
            date="2026-02-27",
            composite_risk=0.35,
            composite_risk_pct=42.0,
            risk_tier="moderate",
            instruments_available=4,
            instruments_total=6,
            coverage=0.667,
            confidence="moderate",
            components=[
                {"instrument": "findrisc", "raw_score": 12.0, "weight": 0.25, "contribution": 0.03},
            ],
        )
        self.assertEqual(r.risk_tier, "moderate")
        self.assertEqual(r.instruments_available, 4)
        self.assertEqual(len(r.components), 1)

    def test_cardio_risk_history_out(self) -> None:
        h = CardioRiskHistoryOut(
            user_id="u1",
            snapshots=[
                CardioRiskOut(
                    user_id="u1",
                    date="2026-02-20",
                    composite_risk=0.3,
                    composite_risk_pct=38.0,
                    risk_tier="moderate",
                    instruments_available=3,
                    instruments_total=6,
                    coverage=0.5,
                    confidence="low",
                    components=[],
                ),
            ],
        )
        self.assertEqual(len(h.snapshots), 1)
        d = h.model_dump()
        self.assertEqual(d["user_id"], "u1")
        self.assertEqual(d["snapshots"][0]["confidence"], "low")


class TestCalibrationPipelineIntegration(unittest.TestCase):
    """Test N-of-1 pipeline functions used by the production service."""

    def test_initial_calibration_and_update(self) -> None:
        from questions_agent_platform.pipeline.n_of_1 import (
            initial_calibration,
            update_calibration,
            calibration_to_dict,
            calibration_from_dict,
            assess_reliable_change,
            assess_measurement_sufficiency,
        )

        cal = initial_calibration("scale_phq9")
        self.assertEqual(cal.scale_id, "scale_phq9")
        self.assertEqual(cal.phase, "warming")
        self.assertEqual(cal.n_observations, 0)

        # Simulate 10 observations (positional args: theta_observed, se_observed)
        for i in range(10):
            cal = update_calibration(cal, 0.5 + i * 0.05, 0.3, days_since_last=1.0)

        self.assertGreater(cal.n_observations, 0)
        self.assertIn(cal.phase, ("warming", "calibrating", "calibrated"))

        # Serialisation roundtrip
        d = calibration_to_dict(cal)
        self.assertIsInstance(d, dict)
        cal2 = calibration_from_dict(d)
        self.assertEqual(cal2.n_observations, cal.n_observations)
        self.assertAlmostEqual(cal2.theta_personal, cal.theta_personal, places=6)

        # Measurement sufficiency
        suff = assess_measurement_sufficiency(cal)
        self.assertIsNotNone(suff.se_current)
        self.assertIsInstance(suff.sufficient, bool)

    def test_calibration_json_roundtrip(self) -> None:
        from questions_agent_platform.pipeline.n_of_1 import (
            initial_calibration,
            update_calibration,
            calibration_to_dict,
            calibration_from_dict,
        )

        cal = initial_calibration("scale_gad7")
        cal = update_calibration(cal, 1.0, 0.25, days_since_last=1.0)
        cal = update_calibration(cal, 0.8, 0.2, days_since_last=1.0)

        # Simulate what the production service does
        cal_json = json.dumps(calibration_to_dict(cal), ensure_ascii=False)
        cal_back = calibration_from_dict(json.loads(cal_json))
        self.assertEqual(cal_back.scale_id, "scale_gad7")
        self.assertEqual(cal_back.n_observations, cal.n_observations)


class TestCoherencePipelineIntegration(unittest.TestCase):
    """Test coherence pipeline functions used by the production service."""

    def test_assess_coherence_valid_session(self) -> None:
        from questions_agent_platform.pipeline.coherence import assess_coherence, ItemResponse

        # Simulated consistent answers with varied values and moderate response times
        responses = [
            ItemResponse(
                item_id=f"item_{i}",
                scale_id="scale_a",
                response_value=i % 4,  # varied responses
                response_time_ms=3000 + i * 500,
                item_position=i,
                n_response_options=5,
            )
            for i in range(5)
        ]
        assessment = assess_coherence(responses)
        self.assertIsNotNone(assessment)
        self.assertIn(assessment.tier, ("valid", "suspect", "invalid"))
        self.assertGreaterEqual(assessment.coherence_score, 0.0)
        self.assertLessEqual(assessment.coherence_score, 1.0)
        self.assertIsInstance(assessment.n_items, int)

    def test_assess_coherence_straightlining(self) -> None:
        from questions_agent_platform.pipeline.coherence import assess_coherence, ItemResponse

        # All same value = straightlining pattern
        responses = [
            ItemResponse(
                item_id=f"item_{i}",
                scale_id="scale_a",
                response_value=4,  # always the same
                response_time_ms=1000,  # fast
                item_position=i,
                n_response_options=5,
            )
            for i in range(10)
        ]
        assessment = assess_coherence(responses)
        self.assertIsNotNone(assessment)
        # With 10 identical answers, should flag straightlining
        self.assertGreater(assessment.n_flagged, 0)


class TestCardioRiskPipelineIntegration(unittest.TestCase):
    """Test cardio risk index pipeline functions used by the production service."""

    def test_compute_cardio_risk_with_instruments(self) -> None:
        from questions_agent_platform.pipeline.cardio_risk_index import (
            compute_cardio_risk_index,
            cardio_risk_to_dict,
        )
        from questions_agent_platform.pipeline.scoring import ScaleScoreResult

        scores = {
            "instrument_findrisc": ScaleScoreResult(
                raw_score=12.0,
                normalized_score=0.48,
                answered_count=8,
                items_required=8,
                risk_tier="moderate",
            ),
            "instrument_ez_cvd": ScaleScoreResult(
                raw_score=8.0,
                normalized_score=0.35,
                answered_count=7,
                items_required=7,
                risk_tier="low",
            ),
        }
        index = compute_cardio_risk_index(scores)
        self.assertIsNotNone(index)
        self.assertGreater(index.composite_risk, 0)
        self.assertEqual(index.instruments_available, 2)
        self.assertEqual(index.instruments_total, 6)
        self.assertIn(index.risk_tier, ("low", "moderate", "high", "very_high"))

        # Dict roundtrip
        d = cardio_risk_to_dict(index)
        self.assertIn("composite_risk", d)
        self.assertIn("components", d)

    def test_compute_cardio_risk_insufficient_data(self) -> None:
        from questions_agent_platform.pipeline.cardio_risk_index import compute_cardio_risk_index

        # No instruments → should return None (insufficient data)
        index = compute_cardio_risk_index({})
        self.assertIsNone(index)


class TestScaleIdToCardioMethod(unittest.TestCase):
    """Test the production helper that maps scale_ids to cardio method names."""

    def test_known_mappings(self) -> None:
        from questions_agent_platform.prod.service_pg import _scale_id_to_cardio_method

        self.assertEqual(_scale_id_to_cardio_method("scale_cm_findrisc"), "instrument_findrisc")
        self.assertEqual(_scale_id_to_cardio_method("scale_cm_ez_cvd"), "instrument_ez_cvd")
        self.assertEqual(_scale_id_to_cardio_method("scale_cm_ipaq_sf"), "instrument_ipaq_sf")
        self.assertEqual(_scale_id_to_cardio_method("scale_cm_lee_nafld"), "instrument_lee_nafld")
        self.assertEqual(_scale_id_to_cardio_method("scale_cm_scored"), "instrument_scored")
        self.assertEqual(_scale_id_to_cardio_method("scale_cm_audit_c"), "instrument_audit_c")

    def test_unknown_scale_returns_none(self) -> None:
        from questions_agent_platform.prod.service_pg import _scale_id_to_cardio_method

        self.assertIsNone(_scale_id_to_cardio_method("scale_phq9"))
        self.assertIsNone(_scale_id_to_cardio_method("unknown"))


class TestAlembicMigrationFile(unittest.TestCase):
    """Verify the new Alembic migration is syntactically valid."""

    def test_migration_0005_importable(self) -> None:
        import importlib
        mod = importlib.import_module(
            "questions_agent_platform.prod.alembic.versions.20260227_0005_calibration_coherence_cardio"
        )
        self.assertEqual(mod.revision, "20260227_0005")
        self.assertEqual(mod.down_revision, "20260215_0004")
        self.assertTrue(callable(mod.upgrade))
        self.assertTrue(callable(mod.downgrade))

    def test_migration_0006_importable(self) -> None:
        import importlib
        mod = importlib.import_module(
            "questions_agent_platform.prod.alembic.versions.20260228_0006_user_profile_site_config"
        )
        self.assertEqual(mod.revision, "20260228_0006")
        self.assertEqual(mod.down_revision, "20260227_0005")
        self.assertTrue(callable(mod.upgrade))
        self.assertTrue(callable(mod.downgrade))


class TestProductionModelsImportable(unittest.TestCase):
    """Verify the new ORM models are importable and have correct table names."""

    def test_personal_calibration_row(self) -> None:
        from questions_agent_platform.prod.models import PersonalCalibrationRow
        self.assertEqual(PersonalCalibrationRow.__tablename__, "personal_calibrations")

    def test_session_coherence(self) -> None:
        from questions_agent_platform.prod.models import SessionCoherence
        self.assertEqual(SessionCoherence.__tablename__, "session_coherence")

    def test_cardio_risk_snapshot(self) -> None:
        from questions_agent_platform.prod.models import CardioRiskSnapshot
        self.assertEqual(CardioRiskSnapshot.__tablename__, "cardio_risk_snapshots")

    def test_user_profile_has_site_config_id(self) -> None:
        from questions_agent_platform.prod.models import UserProfile
        self.assertTrue(hasattr(UserProfile, "site_config_id"))


class TestSiteConfigSchemas(unittest.TestCase):
    """Validate SiteConfig Pydantic schemas."""

    def test_site_config_out(self) -> None:
        from questions_agent_platform.prod.schemas import SiteConfigOut
        c = SiteConfigOut(
            config_id="consumer",
            display_name="ani.ai",
            config_type="wellness",
            score_display_mode="wellness",
        )
        self.assertEqual(c.config_id, "consumer")
        self.assertEqual(c.config_type, "wellness")
        self.assertFalse(c.anifold_enabled)
        self.assertTrue(c.behavioural_metadata_enabled)

    def test_site_config_list_out(self) -> None:
        from questions_agent_platform.prod.schemas import SiteConfigListOut, SiteConfigOut
        lst = SiteConfigListOut(
            configs=[
                SiteConfigOut(
                    config_id="consumer",
                    display_name="ani.ai",
                    config_type="wellness",
                ),
                SiteConfigOut(
                    config_id="samd_full",
                    display_name="ani medical AI",
                    config_type="samd",
                    anifold_enabled=True,
                ),
            ],
        )
        self.assertEqual(len(lst.configs), 2)
        self.assertFalse(lst.configs[0].anifold_enabled)
        self.assertTrue(lst.configs[1].anifold_enabled)

    def test_user_profile_patch_includes_site_config_id(self) -> None:
        from questions_agent_platform.prod.schemas import UserProfilePatchIn
        patch = UserProfilePatchIn(site_config_id="cds_default")
        self.assertEqual(patch.site_config_id, "cds_default")
        d = patch.model_dump(exclude_none=True)
        self.assertIn("site_config_id", d)


class TestSiteConfigServiceFunctions(unittest.TestCase):
    """Test SiteConfig service functions from prod/service_pg.py."""

    def test_list_site_configs(self) -> None:
        from questions_agent_platform.prod.service_pg import list_site_configs
        configs = list_site_configs()
        self.assertIsInstance(configs, list)
        self.assertGreaterEqual(len(configs), 4)
        ids = {c["config_id"] for c in configs}
        self.assertIn("consumer", ids)
        self.assertIn("samd_full", ids)

    def test_site_config_to_dict_fields(self) -> None:
        from questions_agent_platform.prod.service_pg import list_site_configs
        configs = list_site_configs()
        consumer = next(c for c in configs if c["config_id"] == "consumer")
        self.assertEqual(consumer["config_type"], "wellness")
        self.assertFalse(consumer["anifold_enabled"])
        self.assertTrue(consumer["wellness_language"])
        self.assertEqual(consumer["score_display_mode"], "wellness")
        self.assertIn("daily_question_budget", consumer)

    def test_format_score_wellness(self) -> None:
        from questions_agent_platform.pipeline.site_config import (
            config_consumer_wellness,
            format_score_display,
        )
        config = config_consumer_wellness()
        display = format_score_display(
            config=config,
            scale_name="PHQ-9",
            raw_score=12.0,
            normalized_score=65.0,
            risk_tier="moderate",
        )
        self.assertIn("score_display", display)
        self.assertIn("trend_label", display)
        # Wellness mode: no disease names, no raw_score field
        self.assertNotIn("raw_score", display)

    def test_format_score_clinical(self) -> None:
        from questions_agent_platform.pipeline.site_config import (
            config_clinician_cds,
            format_score_display,
        )
        config = config_clinician_cds()
        display = format_score_display(
            config=config,
            scale_name="PHQ-9",
            raw_score=12.0,
            normalized_score=65.0,
            risk_tier="moderate",
        )
        self.assertIn("raw_score", display)
        self.assertEqual(display["raw_score"], 12.0)
        self.assertIn("risk_tier", display)


if __name__ == "__main__":
    unittest.main()
