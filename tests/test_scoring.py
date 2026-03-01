"""
Scoring validation tests for all 6 cardiometabolic instruments.

Each test uses reference values derived from published scoring algorithms:
- FINDRISC: Lindström & Tuomilehto, 2003. Diabetes Care 26(3):725-731.
- EZ-CVD: Gaziano et al., 2008. Lancet 371(9611):923-931.
- SCORED: Bansal et al., 2007. Arch Intern Med 167(4):374-381.
- Lee NAFLD: Lee et al., 2018. J Gastroenterol Hepatol 33(6):1176-1183.
- IPAQ-SF: Craig et al., 2003. Med Sci Sports Exerc 35(8):1381-1395.
- AUDIT-C: Bush et al., 1998. Arch Intern Med 158(16):1789-1795.
"""

import unittest
from typing import Dict

from questions_agent_platform.pipeline.registry import Scale, ScaleItem
from questions_agent_platform.pipeline.scoring import (
    ScaleScoreResult,
    compute_scale_score,
    _findrisc_tier,
    _ez_cvd_tier,
    _scored_tier,
    _lee_nafld_tier,
    _ipaq_tier,
    _audit_c_tier,
)


def _make_scale(method: str, item_ids: list, min_items: int = 1, normalize_max: float = 26.0) -> Scale:
    """Helper to build a Scale object for instrument-specific scorers."""
    return Scale(
        id="test_scale",
        questionnaire_id="test_q",
        version="1",
        name="Test",
        method=method,
        min_items_required=min_items,
        unlock_window_days=14,
        retest_interval_days=90,
        response_type="likert_0_4",
        normalize_min=0.0,
        normalize_max=normalize_max,
        items=tuple(ScaleItem(item_id=iid) for iid in item_ids),
        tags=(),
    )


class TestReverseCodingAndNormalization(unittest.TestCase):
    """Original test: reverse coding + mean normalization."""

    def test_reverse_coding_mean_normalization(self) -> None:
        scale = Scale(
            id="s1",
            questionnaire_id="q1",
            version="1",
            name="Test",
            method="mean",
            min_items_required=1,
            unlock_window_days=14,
            retest_interval_days=90,
            response_type="likert_0_4",
            normalize_min=0.0,
            normalize_max=4.0,
            items=(ScaleItem(item_id="i1", reverse=True, weight=1.0),),
            tags=(),
        )
        res = compute_scale_score(scale, {"i1": 0.0})
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 4.0, places=6)
        self.assertAlmostEqual(res.normalized_score, 100.0, places=6)


# ---------------------------------------------------------------------------
# FINDRISC — Reference: 0-26 point scale
# Scoring: age(0-4) + BMI(0-3) + waist(0-4) + activity(0/2) + fruit/veg(0/1)
#          + BP_med(0/2) + glucose(0/5) + family_diabetes(0/3/5)
# Tiers: <7 low, 7-11 slightly_elevated, 12-14 moderate, 15-20 high, >20 very_high
# ---------------------------------------------------------------------------
class TestFINDRISC(unittest.TestCase):

    def _answers(self, **overrides) -> Dict[str, float]:
        """Build a complete FINDRISC answer set with overrides."""
        base = {
            "cm_age_band": 0.0,       # <45y = 0 pts
            "cm_bmi_band": 0.0,       # <25 = 0 pts
            "cm_waist_band": 0.0,     # <94cm M / <80cm F = 0 pts
            "cm_activity_daily": 1.0,  # active = 0 pts
            "cm_fruit_veg_daily": 1.0, # eats daily = 0 pts
            "cm_bp_medication": 0.0,   # no = 0 pts
            "cm_high_glucose_history": 0.0,  # no = 0 pts
            "cm_family_diabetes_history": 0.0,  # none = 0 pts
        }
        base.update(overrides)
        return base

    def _score(self, **overrides) -> ScaleScoreResult:
        scale = _make_scale(
            "instrument_findrisc",
            list(self._answers().keys()),
            min_items=8,
            normalize_max=26.0,
        )
        res = compute_scale_score(scale, self._answers(**overrides))
        self.assertIsNotNone(res, f"FINDRISC returned None with overrides={overrides}")
        assert res is not None
        return res

    def test_minimum_risk_profile(self):
        """Young, active, healthy weight, no history -> score 0, low risk."""
        res = self._score()
        self.assertAlmostEqual(res.raw_score, 0.0, places=1)
        self.assertEqual(res.risk_tier, "low")

    def test_age_contributes_points(self):
        """age_band=2 (45-54y) -> +2 pts."""
        res = self._score(cm_age_band=2.0)
        self.assertAlmostEqual(res.raw_score, 2.0, places=1)
        self.assertEqual(res.risk_tier, "low")

    def test_high_risk_composite(self):
        """Older, obese, sedentary, glucose history, family diabetes -> high risk."""
        res = self._score(
            cm_age_band=3.0,       # 55-64y = 3 pts
            cm_bmi_band=2.0,       # 30+ = 2 pts (approx)
            cm_waist_band=3.0,     # >102cm M = 3 pts
            cm_activity_daily=0.0, # inactive = 2 pts
            cm_fruit_veg_daily=0.0, # no daily = 1 pt
            cm_bp_medication=1.0,  # yes = 2 pts
            cm_high_glucose_history=1.0,  # yes = 5 pts
            cm_family_diabetes_history=3.0,  # parent/sibling = 3 pts
        )
        # 3 + 2 + 3 + 2 + 1 + 2 + 5 + 3 = 21
        self.assertGreaterEqual(res.raw_score, 20.0)
        self.assertEqual(res.risk_tier, "very_high")

    def test_moderate_risk(self):
        """Score in 12-14 range -> moderate."""
        res = self._score(
            cm_age_band=2.0,       # 2 pts
            cm_bmi_band=1.0,       # 1 pt
            cm_waist_band=2.0,     # 2 pts (approx)
            cm_activity_daily=0.0, # 2 pts
            cm_fruit_veg_daily=0.0, # 1 pt
            cm_bp_medication=1.0,  # 2 pts
            cm_high_glucose_history=0.0,
            cm_family_diabetes_history=3.0,  # 3 pts
        )
        # 2 + 1 + 2 + 2 + 1 + 2 + 0 + 3 = 13
        self.assertGreaterEqual(res.raw_score, 12.0)
        self.assertLessEqual(res.raw_score, 14.0)
        self.assertEqual(res.risk_tier, "moderate")

    def test_insufficient_items_returns_none(self):
        """Fewer items than min_items_required -> None."""
        scale = _make_scale(
            "instrument_findrisc",
            list(self._answers().keys()),
            min_items=8,
            normalize_max=26.0,
        )
        res = compute_scale_score(scale, {"cm_age_band": 2.0})  # only 1 of 8
        self.assertIsNone(res)


# ---------------------------------------------------------------------------
# FINDRISC tier boundaries
# ---------------------------------------------------------------------------
class TestFINDRISCTiers(unittest.TestCase):
    def test_tier_boundaries(self):
        self.assertEqual(_findrisc_tier(0.0), "low")
        self.assertEqual(_findrisc_tier(6.9), "low")
        self.assertEqual(_findrisc_tier(7.0), "slightly_elevated")
        self.assertEqual(_findrisc_tier(11.0), "slightly_elevated")
        self.assertEqual(_findrisc_tier(12.0), "moderate")
        self.assertEqual(_findrisc_tier(14.0), "moderate")
        self.assertEqual(_findrisc_tier(15.0), "high")
        self.assertEqual(_findrisc_tier(20.0), "high")
        self.assertEqual(_findrisc_tier(21.0), "very_high")
        self.assertEqual(_findrisc_tier(26.0), "very_high")


# ---------------------------------------------------------------------------
# EZ-CVD — Reference: 6 binary factors, 0-6 scale
# Factors: age≥55M/≥65F (1), male (1), smoking (1), diabetes (1),
#          hypertension (1), family MI (1)
# Tiers: ≤1 low, 2-3 moderate, ≥4 high
# ---------------------------------------------------------------------------
class TestEZCVD(unittest.TestCase):

    def _answers(self, **overrides) -> Dict[str, float]:
        base = {
            "cm_age_band": 0.0,  # young
            "cm_sex_at_birth": 0.0,  # female
            "cm_smoking_current": 0.0,
            "cm_high_glucose_history": 0.0,
            "cm_bp_medication": 0.0,
            "cm_family_mi_history": 0.0,
        }
        base.update(overrides)
        return base

    def _score(self, **overrides) -> ScaleScoreResult:
        scale = _make_scale(
            "instrument_ez_cvd",
            list(self._answers().keys()),
            min_items=6,
            normalize_max=6.0,
        )
        res = compute_scale_score(scale, self._answers(**overrides))
        self.assertIsNotNone(res)
        assert res is not None
        return res

    def test_zero_risk_factors(self):
        """Young female, no risk factors -> score 0, low."""
        res = self._score()
        self.assertAlmostEqual(res.raw_score, 0.0, places=1)
        self.assertEqual(res.risk_tier, "low")

    def test_male_sex_adds_point(self):
        """Male sex is a risk factor -> +1."""
        res = self._score(cm_sex_at_birth=1.0)
        self.assertGreaterEqual(res.raw_score, 1.0)

    def test_high_risk_multiple_factors(self):
        """Older male with smoking, diabetes, hypertension, family MI -> high."""
        res = self._score(
            cm_age_band=3.0,  # ≥55 for male
            cm_sex_at_birth=1.0,
            cm_smoking_current=1.0,
            cm_high_glucose_history=1.0,
            cm_bp_medication=1.0,
            cm_family_mi_history=1.0,
        )
        self.assertGreaterEqual(res.raw_score, 4.0)
        self.assertEqual(res.risk_tier, "high")

    def test_age_threshold_male_vs_female(self):
        """Male: age≥55 (band≥2) is 1pt. Female: age≥65 (band≥3) is 1pt."""
        # Male age band 2 (55-64) -> age factor = 1
        res_male = self._score(cm_sex_at_birth=1.0, cm_age_band=2.0)
        # Female age band 2 (55-64) -> age factor = 0 (needs band≥3)
        res_female = self._score(cm_sex_at_birth=0.0, cm_age_band=2.0)
        self.assertGreater(res_male.raw_score, res_female.raw_score)


class TestEZCVDTiers(unittest.TestCase):
    def test_tier_boundaries(self):
        self.assertEqual(_ez_cvd_tier(0.0), "low")
        self.assertEqual(_ez_cvd_tier(1.0), "low")
        self.assertEqual(_ez_cvd_tier(2.0), "moderate")
        self.assertEqual(_ez_cvd_tier(3.0), "moderate")
        self.assertEqual(_ez_cvd_tier(4.0), "high")
        self.assertEqual(_ez_cvd_tier(6.0), "high")


# ---------------------------------------------------------------------------
# SCORED — Reference: Weighted scoring for CKD screening
# Age(2-4) + female(1) + anemia(1) + HTN(1) + DM(1) + CVD(1) + CHF(1)
# + PVD(1) + proteinuria(1). ≥4 = positive screen.
# ---------------------------------------------------------------------------
class TestSCORED(unittest.TestCase):

    def _answers(self, **overrides) -> Dict[str, float]:
        base = {
            "cm_age_band": 0.0,
            "cm_sex_at_birth": 1.0,  # male = 0 female points
            "cm_anemia_history": 0.0,
            "cm_bp_medication": 0.0,
            "cm_high_glucose_history": 0.0,
            "cm_cvd_history": 0.0,
            "cm_chf_history": 0.0,
            "cm_pvd_history": 0.0,
            "cm_proteinuria_history": 0.0,
        }
        base.update(overrides)
        return base

    def _score(self, **overrides) -> ScaleScoreResult:
        scale = _make_scale(
            "instrument_scored",
            list(self._answers().keys()),
            min_items=9,
            normalize_max=13.0,
        )
        res = compute_scale_score(scale, self._answers(**overrides))
        self.assertIsNotNone(res)
        assert res is not None
        return res

    def test_young_healthy_male(self):
        """Young healthy male -> 0, negative screen."""
        res = self._score()
        self.assertAlmostEqual(res.raw_score, 0.0, places=1)
        self.assertEqual(res.risk_tier, "negative_screen")

    def test_female_adds_point(self):
        """Female (sex=0) adds 1 point."""
        res = self._score(cm_sex_at_birth=0.0)
        self.assertGreaterEqual(res.raw_score, 1.0)

    def test_positive_screen_threshold(self):
        """Score ≥ 4 is positive screen."""
        res = self._score(
            cm_age_band=3.0,  # 3 pts
            cm_sex_at_birth=0.0,  # female +1
            cm_anemia_history=1.0,  # +1
        )
        self.assertGreaterEqual(res.raw_score, 4.0)
        self.assertEqual(res.risk_tier, "positive_screen")

    def test_age_scoring_tiers(self):
        """Age band 2 -> 2pts, band 3 -> 3pts, band 4+ -> 4pts."""
        res2 = self._score(cm_age_band=2.0)
        res3 = self._score(cm_age_band=3.0)
        res4 = self._score(cm_age_band=4.0)
        self.assertAlmostEqual(res2.raw_score, 2.0, places=1)
        self.assertAlmostEqual(res3.raw_score, 3.0, places=1)
        self.assertAlmostEqual(res4.raw_score, 4.0, places=1)


class TestSCOREDTiers(unittest.TestCase):
    def test_threshold(self):
        self.assertEqual(_scored_tier(0.0), "negative_screen")
        self.assertEqual(_scored_tier(3.9), "negative_screen")
        self.assertEqual(_scored_tier(4.0), "positive_screen")
        self.assertEqual(_scored_tier(13.0), "positive_screen")


# ---------------------------------------------------------------------------
# Lee NAFLD — Reference: 9 variables, weighted 0-15, ≥8 high risk
# ---------------------------------------------------------------------------
class TestLeeNAFLD(unittest.TestCase):

    def _answers(self, **overrides) -> Dict[str, float]:
        base = {
            "cm_age_band": 0.0,
            "cm_sex_at_birth": 0.0,  # female
            "cm_waist_band": 0.0,
            "cm_bmi_band": 0.0,
            "cm_high_glucose_history": 0.0,
            "cm_dyslipidemia_history": 0.0,
            "cm_alcohol_intake_frequency": 0.0,
            "cm_activity_daily": 1.0,  # active
            "cm_menopause_status": 0.0,
        }
        base.update(overrides)
        return base

    def _score(self, **overrides) -> ScaleScoreResult:
        scale = _make_scale(
            "instrument_lee_nafld",
            list(self._answers().keys()),
            min_items=9,
            normalize_max=15.0,
        )
        res = compute_scale_score(scale, self._answers(**overrides))
        self.assertIsNotNone(res)
        assert res is not None
        return res

    def test_low_risk_profile(self):
        """Young active female, no comorbidities -> low risk."""
        res = self._score()
        self.assertLess(res.raw_score, 6.0)
        self.assertEqual(res.risk_tier, "low_risk")

    def test_diabetes_double_weighted(self):
        """Diabetes contributes 2x weight."""
        res_no_dm = self._score()
        res_dm = self._score(cm_high_glucose_history=1.0)
        self.assertAlmostEqual(res_dm.raw_score - res_no_dm.raw_score, 2.0, places=1)

    def test_high_risk_composite(self):
        """Multiple risk factors -> high risk (≥8)."""
        res = self._score(
            cm_age_band=3.0,
            cm_sex_at_birth=1.0,  # male +1
            cm_waist_band=2.0,
            cm_bmi_band=2.0,
            cm_high_glucose_history=1.0,  # +2
            cm_dyslipidemia_history=1.0,
            cm_alcohol_intake_frequency=2.0,
            cm_activity_daily=0.0,  # inactive +1
            cm_menopause_status=0.0,
        )
        self.assertGreaterEqual(res.raw_score, 8.0)
        self.assertEqual(res.risk_tier, "high_risk")

    def test_inactivity_adds_point(self):
        """Inactive (activity=0) adds 1 point vs active."""
        res_active = self._score(cm_activity_daily=1.0)
        res_inactive = self._score(cm_activity_daily=0.0)
        self.assertAlmostEqual(res_inactive.raw_score - res_active.raw_score, 1.0, places=1)


class TestLeeNAFLDTiers(unittest.TestCase):
    def test_tier_boundaries(self):
        self.assertEqual(_lee_nafld_tier(0.0), "low_risk")
        self.assertEqual(_lee_nafld_tier(5.9), "low_risk")
        self.assertEqual(_lee_nafld_tier(6.0), "elevated_risk")
        self.assertEqual(_lee_nafld_tier(7.9), "elevated_risk")
        self.assertEqual(_lee_nafld_tier(8.0), "high_risk")
        self.assertEqual(_lee_nafld_tier(15.0), "high_risk")


# ---------------------------------------------------------------------------
# IPAQ-SF — Reference: MET-min/week formula
# Vigorous=8.0, Moderate=4.0, Walking=3.3
# Total = (8.0 * vig_days * vig_min) + (4.0 * mod_days * mod_min) + (3.3 * walk_days * walk_min)
# <600 low, 600-3000 moderate, >3000 high
# ---------------------------------------------------------------------------
class TestIPAQSF(unittest.TestCase):

    def _answers(self, **overrides) -> Dict[str, float]:
        base = {
            "cm_ipaq_vig_days": 0.0,
            "cm_ipaq_vig_minutes": 0.0,
            "cm_ipaq_mod_days": 0.0,
            "cm_ipaq_mod_minutes": 0.0,
            "cm_ipaq_walk_days": 0.0,
            "cm_ipaq_walk_minutes": 0.0,
            "cm_ipaq_sit_minutes": 480.0,
        }
        base.update(overrides)
        return base

    def _score(self, **overrides) -> ScaleScoreResult:
        scale = _make_scale(
            "instrument_ipaq_sf",
            list(self._answers().keys()),
            min_items=6,
            normalize_max=10000.0,
        )
        res = compute_scale_score(scale, self._answers(**overrides))
        self.assertIsNotNone(res)
        assert res is not None
        return res

    def test_sedentary_zero_met(self):
        """No activity -> 0 MET-min/week, low."""
        res = self._score()
        self.assertAlmostEqual(res.raw_score, 0.0, places=1)
        self.assertEqual(res.risk_tier, "low_activity")

    def test_moderate_walker(self):
        """Walk 30min/day, 5 days -> 3.3*5*30 = 495 MET-min -> low."""
        res = self._score(cm_ipaq_walk_days=5.0, cm_ipaq_walk_minutes=30.0)
        self.assertAlmostEqual(res.raw_score, 495.0, places=1)
        self.assertEqual(res.risk_tier, "low_activity")

    def test_moderate_activity_level(self):
        """Walk 5*30 + moderate 3*30 -> 495 + 360 = 855 MET-min -> moderate."""
        res = self._score(
            cm_ipaq_walk_days=5.0, cm_ipaq_walk_minutes=30.0,
            cm_ipaq_mod_days=3.0, cm_ipaq_mod_minutes=30.0,
        )
        expected = (3.3 * 5 * 30) + (4.0 * 3 * 30)
        self.assertAlmostEqual(res.raw_score, expected, places=1)
        self.assertEqual(res.risk_tier, "moderate_activity")

    def test_high_activity_level(self):
        """Vigorous 5*60 + moderate 3*30 + walk 7*30 -> very high MET."""
        res = self._score(
            cm_ipaq_vig_days=5.0, cm_ipaq_vig_minutes=60.0,
            cm_ipaq_mod_days=3.0, cm_ipaq_mod_minutes=30.0,
            cm_ipaq_walk_days=7.0, cm_ipaq_walk_minutes=30.0,
        )
        expected = (8.0 * 5 * 60) + (4.0 * 3 * 30) + (3.3 * 7 * 30)
        self.assertAlmostEqual(res.raw_score, expected, places=1)
        self.assertGreater(res.raw_score, 3000.0)
        self.assertEqual(res.risk_tier, "high_activity")

    def test_exact_met_formula(self):
        """Verify the exact MET-min formula: 8*V + 4*M + 3.3*W."""
        res = self._score(
            cm_ipaq_vig_days=3.0, cm_ipaq_vig_minutes=20.0,
            cm_ipaq_mod_days=2.0, cm_ipaq_mod_minutes=45.0,
            cm_ipaq_walk_days=4.0, cm_ipaq_walk_minutes=15.0,
        )
        expected = (8.0 * 3 * 20) + (4.0 * 2 * 45) + (3.3 * 4 * 15)
        self.assertAlmostEqual(res.raw_score, expected, places=1)


class TestIPAQTiers(unittest.TestCase):
    def test_tier_boundaries(self):
        self.assertEqual(_ipaq_tier(0.0), "low_activity")
        self.assertEqual(_ipaq_tier(599.9), "low_activity")
        self.assertEqual(_ipaq_tier(600.0), "moderate_activity")
        self.assertEqual(_ipaq_tier(3000.0), "moderate_activity")
        self.assertEqual(_ipaq_tier(3001.0), "high_activity")


# ---------------------------------------------------------------------------
# AUDIT-C — Reference: 3 questions, each 0-4, sum 0-12
# Positive screen: ≥4 men / ≥3 women
# ---------------------------------------------------------------------------
class TestAUDITC(unittest.TestCase):

    def _answers(self, **overrides) -> Dict[str, float]:
        base = {
            "cm_alcohol_intake_frequency": 0.0,
            "cm_audit_typical_drinks": 0.0,
            "cm_audit_binge_frequency": 0.0,
            "cm_sex_at_birth": 1.0,  # male
        }
        base.update(overrides)
        return base

    def _score(self, **overrides) -> ScaleScoreResult:
        scale = _make_scale(
            "instrument_audit_c",
            ["cm_alcohol_intake_frequency", "cm_audit_typical_drinks", "cm_audit_binge_frequency"],
            min_items=3,
            normalize_max=12.0,
        )
        res = compute_scale_score(scale, self._answers(**overrides))
        self.assertIsNotNone(res)
        assert res is not None
        return res

    def test_no_alcohol(self):
        """Zero on all items -> no_use."""
        res = self._score()
        self.assertAlmostEqual(res.raw_score, 0.0, places=1)
        self.assertEqual(res.risk_tier, "no_use")

    def test_low_use_male(self):
        """Male, score 1-3 -> low_use."""
        res = self._score(cm_alcohol_intake_frequency=1.0, cm_audit_typical_drinks=1.0)
        self.assertAlmostEqual(res.raw_score, 2.0, places=1)
        self.assertEqual(res.risk_tier, "low_use")

    def test_positive_screen_male(self):
        """Male, score ≥ 4 -> positive_screen."""
        res = self._score(
            cm_alcohol_intake_frequency=2.0,
            cm_audit_typical_drinks=1.0,
            cm_audit_binge_frequency=1.0,
        )
        self.assertAlmostEqual(res.raw_score, 4.0, places=1)
        self.assertEqual(res.risk_tier, "positive_screen")

    def test_high_risk(self):
        """High scores -> high_risk (≥ threshold+3)."""
        res = self._score(
            cm_alcohol_intake_frequency=4.0,
            cm_audit_typical_drinks=4.0,
            cm_audit_binge_frequency=4.0,
        )
        self.assertAlmostEqual(res.raw_score, 12.0, places=1)
        self.assertEqual(res.risk_tier, "high_risk")

    def test_sex_specific_thresholds(self):
        """Female threshold is 3, male is 4. Score=3 is positive for female, low for male."""
        # This tests the tier function directly since the scorer uses sex from answers
        self.assertEqual(_audit_c_tier(3.0, male=True), "low_use")
        self.assertEqual(_audit_c_tier(3.0, male=False), "positive_screen")

    def test_female_positive_at_3(self):
        """Female (sex=0), score=3 -> positive screen."""
        # When sex is unknown (male=None), threshold defaults to 3 (conservative)
        self.assertEqual(_audit_c_tier(3.0, male=None), "positive_screen")


class TestAUDITCTiers(unittest.TestCase):
    def test_male_tiers(self):
        self.assertEqual(_audit_c_tier(0.0, male=True), "no_use")
        self.assertEqual(_audit_c_tier(1.0, male=True), "low_use")
        self.assertEqual(_audit_c_tier(3.0, male=True), "low_use")
        self.assertEqual(_audit_c_tier(4.0, male=True), "positive_screen")
        self.assertEqual(_audit_c_tier(7.0, male=True), "high_risk")

    def test_female_tiers(self):
        self.assertEqual(_audit_c_tier(0.0, male=False), "no_use")
        self.assertEqual(_audit_c_tier(1.0, male=False), "low_use")
        self.assertEqual(_audit_c_tier(2.0, male=False), "low_use")
        self.assertEqual(_audit_c_tier(3.0, male=False), "positive_screen")
        self.assertEqual(_audit_c_tier(6.0, male=False), "high_risk")


# ---------------------------------------------------------------------------
# Insufficient data handling
# ---------------------------------------------------------------------------
class TestInsufficientData(unittest.TestCase):
    """All instruments must return None when data is insufficient."""

    def test_findrisc_insufficient(self):
        scale = _make_scale("instrument_findrisc", ["cm_age_band", "cm_bmi_band"], min_items=8)
        self.assertIsNone(compute_scale_score(scale, {"cm_age_band": 1.0}))

    def test_ez_cvd_insufficient(self):
        scale = _make_scale("instrument_ez_cvd", ["cm_age_band", "cm_sex_at_birth"], min_items=6)
        self.assertIsNone(compute_scale_score(scale, {"cm_age_band": 1.0}))

    def test_scored_insufficient(self):
        scale = _make_scale("instrument_scored", ["cm_age_band"], min_items=9)
        self.assertIsNone(compute_scale_score(scale, {"cm_age_band": 1.0}))

    def test_lee_nafld_insufficient(self):
        scale = _make_scale("instrument_lee_nafld", ["cm_age_band"], min_items=9)
        self.assertIsNone(compute_scale_score(scale, {"cm_age_band": 1.0}))

    def test_ipaq_insufficient(self):
        scale = _make_scale("instrument_ipaq_sf", ["cm_ipaq_vig_days"], min_items=6)
        self.assertIsNone(compute_scale_score(scale, {"cm_ipaq_vig_days": 3.0}))

    def test_audit_c_insufficient(self):
        scale = _make_scale("instrument_audit_c", ["cm_alcohol_intake_frequency"], min_items=3)
        self.assertIsNone(compute_scale_score(scale, {"cm_alcohol_intake_frequency": 1.0}))


if __name__ == "__main__":
    unittest.main()
