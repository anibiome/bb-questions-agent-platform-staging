"""
Scoring validation tests for the 4 vitality battery instruments.

Each test uses reference values derived from published scoring algorithms:
- WHO-5: Topp et al., 2015. Psychother Psychosom 84(3):167-176.
- SF-36 VT: Ware & Sherbourne, 1992. Med Care 30(6):473-483; RAND v1.0 recoding.
- SVS: Ryan & Frederick, 1997. J Pers Soc Psychol 73(3):549-560.
- PROMIS Fatigue 7a: HealthMeasures, Northwestern University, Nov 2025.

The vitality battery captures the energy-fatigue axis from 4 complementary
psychometric perspectives — hedonic wellbeing (WHO-5), functional vitality
(SF-36 VT), eudaimonic aliveness (SVS), and calibrated fatigue severity
(PROMIS Fatigue 7a with IRT T-score lookup).
"""

import unittest
from typing import Dict, Optional

from questions_agent_platform.pipeline.registry import Scale, ScaleItem
from questions_agent_platform.pipeline.scoring import (
    ScaleScoreResult,
    compute_scale_score,
    infer_risk_tier_from_scale,
    _who5_tier,
    _sf36_vt_tier,
    _promis_fatigue_tier,
    _PROMIS_FATIGUE_7A_RAW_TO_T,
    ITEM_WHO5_1, ITEM_WHO5_2, ITEM_WHO5_3, ITEM_WHO5_4, ITEM_WHO5_5,
    ITEM_SF36_VT_PEP, ITEM_SF36_VT_ENERGY, ITEM_SF36_VT_WORN, ITEM_SF36_VT_TIRED,
    ITEM_SVS_1, ITEM_SVS_3, ITEM_SVS_4, ITEM_SVS_5, ITEM_SVS_6, ITEM_SVS_7,
    ITEM_PF7_TIRED, ITEM_PF7_EXHAUSTION, ITEM_PF7_RUN_OUT,
    ITEM_PF7_LIMIT_WORK, ITEM_PF7_THINK, ITEM_PF7_BATH, ITEM_PF7_EXERCISE,
    VITALITY_METHOD_WHO5,
    VITALITY_METHOD_SF36_VT,
    VITALITY_METHOD_PROMIS_FATIGUE_7A,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vitality_scale(
    method: str,
    item_ids: list,
    *,
    min_items: int = 1,
    response_type: str = "likert_0_4",
    normalize_min: float = 0.0,
    normalize_max: float = 100.0,
    tags: tuple = ("energy", "vitality"),
) -> Scale:
    """Build a Scale object for vitality instrument scorers."""
    return Scale(
        id=f"test_{method}",
        questionnaire_id="test_vitality_q",
        version="1",
        name=f"Test {method}",
        method=method,
        min_items_required=min_items,
        unlock_window_days=14,
        retest_interval_days=30,
        response_type=response_type,
        normalize_min=normalize_min,
        normalize_max=normalize_max,
        items=tuple(ScaleItem(item_id=iid) for iid in item_ids),
        tags=tags,
    )


WHO5_ITEMS = [ITEM_WHO5_1, ITEM_WHO5_2, ITEM_WHO5_3, ITEM_WHO5_4, ITEM_WHO5_5]
SF36_VT_ITEMS = [ITEM_SF36_VT_PEP, ITEM_SF36_VT_ENERGY, ITEM_SF36_VT_WORN, ITEM_SF36_VT_TIRED]
SVS_ITEMS = [ITEM_SVS_1, ITEM_SVS_3, ITEM_SVS_4, ITEM_SVS_5, ITEM_SVS_6, ITEM_SVS_7]
PF7_ITEMS = [
    ITEM_PF7_TIRED, ITEM_PF7_EXHAUSTION, ITEM_PF7_RUN_OUT,
    ITEM_PF7_LIMIT_WORK, ITEM_PF7_THINK, ITEM_PF7_BATH, ITEM_PF7_EXERCISE,
]


# ═══════════════════════════════════════════════════════════════════════════
# WHO-5 Well-Being Index
# ═══════════════════════════════════════════════════════════════════════════

class TestWHO5Scoring(unittest.TestCase):
    """WHO-5: 5 items, 0-5 scale, sum×4 → 0-100 percentage."""

    def _make_scale(self, min_items: int = 3) -> Scale:
        return _make_vitality_scale(
            VITALITY_METHOD_WHO5, WHO5_ITEMS,
            min_items=min_items,
            response_type="who5_0_5",
            normalize_min=0.0,
            normalize_max=100.0,
        )

    def _answers(self, value: float) -> Dict[str, float]:
        """All 5 items at the same value."""
        return {iid: value for iid in WHO5_ITEMS}

    def _score(self, answers: Dict[str, float], min_items: int = 3) -> Optional[ScaleScoreResult]:
        return compute_scale_score(self._make_scale(min_items), answers)

    def test_perfect_wellbeing(self):
        """All items at maximum (5) → raw sum=25, percentage=100."""
        res = self._score(self._answers(5.0))
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 25.0, places=2)
        self.assertAlmostEqual(res.original_metric_value, 100.0, places=2)
        self.assertAlmostEqual(res.normalized_score, 100.0, places=2)
        self.assertEqual(res.risk_tier, "high_wellbeing")
        self.assertEqual(res.answered_count, 5)

    def test_zero_wellbeing(self):
        """All items at minimum (0) → raw sum=0, percentage=0."""
        res = self._score(self._answers(0.0))
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 0.0, places=2)
        self.assertAlmostEqual(res.original_metric_value, 0.0, places=2)
        self.assertAlmostEqual(res.normalized_score, 0.0, places=2)
        self.assertEqual(res.risk_tier, "likely_depression")

    def test_depression_cutoff_at_28(self):
        """WHO-5 ≤28 screens for depression. Score exactly 28 → likely_depression."""
        # sum=7 → 7×4=28
        answers = {
            ITEM_WHO5_1: 2.0,  # 2
            ITEM_WHO5_2: 2.0,  # 2
            ITEM_WHO5_3: 1.0,  # 1
            ITEM_WHO5_4: 1.0,  # 1
            ITEM_WHO5_5: 1.0,  # 1  → sum=7, pct=28
        }
        res = self._score(answers)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.original_metric_value, 28.0, places=2)
        self.assertEqual(res.risk_tier, "likely_depression")

    def test_just_above_depression_cutoff(self):
        """Score of 32 (>28) → low_wellbeing, not depression."""
        # sum=8 → 8×4=32
        answers = {
            ITEM_WHO5_1: 2.0,
            ITEM_WHO5_2: 2.0,
            ITEM_WHO5_3: 2.0,
            ITEM_WHO5_4: 1.0,
            ITEM_WHO5_5: 1.0,  # sum=8, pct=32
        }
        res = self._score(answers)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.original_metric_value, 32.0, places=2)
        self.assertEqual(res.risk_tier, "low_wellbeing")

    def test_low_wellbeing_boundary_at_50(self):
        """WHO-5 ≤50 → low well-being. Score exactly 50 → low_wellbeing."""
        # sum=12.5 → 12.5×4=50
        # Can't get exactly 12.5 with integers, so:
        # sum=12 → 48, sum=13 → 52
        answers_48 = {
            ITEM_WHO5_1: 3.0, ITEM_WHO5_2: 3.0, ITEM_WHO5_3: 2.0,
            ITEM_WHO5_4: 2.0, ITEM_WHO5_5: 2.0,  # sum=12, pct=48
        }
        res48 = self._score(answers_48)
        assert res48 is not None
        self.assertEqual(res48.risk_tier, "low_wellbeing")

        answers_52 = {
            ITEM_WHO5_1: 3.0, ITEM_WHO5_2: 3.0, ITEM_WHO5_3: 3.0,
            ITEM_WHO5_4: 2.0, ITEM_WHO5_5: 2.0,  # sum=13, pct=52
        }
        res52 = self._score(answers_52)
        assert res52 is not None
        self.assertEqual(res52.risk_tier, "adequate_wellbeing")

    def test_high_wellbeing_boundary_at_72(self):
        """WHO-5 >72 → high well-being. 72 is adequate, 76 is high."""
        # sum=18 → 72 (adequate), sum=19 → 76 (high)
        answers_72 = {
            ITEM_WHO5_1: 4.0, ITEM_WHO5_2: 4.0, ITEM_WHO5_3: 4.0,
            ITEM_WHO5_4: 3.0, ITEM_WHO5_5: 3.0,  # sum=18, pct=72
        }
        res72 = self._score(answers_72)
        assert res72 is not None
        self.assertEqual(res72.risk_tier, "adequate_wellbeing")

        answers_76 = {
            ITEM_WHO5_1: 4.0, ITEM_WHO5_2: 4.0, ITEM_WHO5_3: 4.0,
            ITEM_WHO5_4: 4.0, ITEM_WHO5_5: 3.0,  # sum=19, pct=76
        }
        res76 = self._score(answers_76)
        assert res76 is not None
        self.assertEqual(res76.risk_tier, "high_wellbeing")

    def test_prorating_partial_answers(self):
        """With 3 of 5 items answered, pro-rate to 5-item equivalent."""
        # 3 items each at 4 → sum=12, pro-rated: 12*(5/3)=20, pct=80
        answers = {
            ITEM_WHO5_1: 4.0,
            ITEM_WHO5_2: 4.0,
            ITEM_WHO5_3: 4.0,
        }
        res = self._score(answers, min_items=3)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 20.0, places=2)
        self.assertAlmostEqual(res.original_metric_value, 80.0, places=2)
        self.assertEqual(res.answered_count, 3)
        self.assertEqual(res.risk_tier, "high_wellbeing")

    def test_prorating_4_of_5(self):
        """With 4 of 5 items → pro-rate correctly."""
        # 4 items each at 3 → sum=12, pro-rated: 12*(5/4)=15, pct=60
        answers = {
            ITEM_WHO5_1: 3.0,
            ITEM_WHO5_2: 3.0,
            ITEM_WHO5_3: 3.0,
            ITEM_WHO5_4: 3.0,
        }
        res = self._score(answers, min_items=3)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 15.0, places=2)
        self.assertAlmostEqual(res.original_metric_value, 60.0, places=2)
        self.assertEqual(res.answered_count, 4)

    def test_insufficient_items_returns_none(self):
        """Fewer than min_items_required → None."""
        answers = {ITEM_WHO5_1: 3.0, ITEM_WHO5_2: 4.0}  # only 2
        res = self._score(answers, min_items=3)
        self.assertIsNone(res)

    def test_values_clamped_to_0_5(self):
        """Values outside 0-5 should be clamped."""
        answers = {iid: 10.0 for iid in WHO5_ITEMS}  # all 10 → clamped to 5
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 25.0, places=2)  # 5*5=25

    def test_metric_label(self):
        """original_metric_label should be 'WHO-5 (0-100)'."""
        res = self._score(self._answers(3.0))
        assert res is not None
        self.assertEqual(res.original_metric_label, "WHO-5 (0-100)")

    def test_confidence_full_answers(self):
        """All 5 items answered → high confidence."""
        res = self._score(self._answers(3.0))
        assert res is not None
        self.assertEqual(res.confidence, "high")

    def test_confidence_partial_answers(self):
        """3 of 5 items → medium confidence."""
        answers = {ITEM_WHO5_1: 3.0, ITEM_WHO5_2: 3.0, ITEM_WHO5_3: 3.0}
        res = self._score(answers, min_items=3)
        assert res is not None
        self.assertEqual(res.confidence, "medium")


class TestWHO5Tiers(unittest.TestCase):
    """Direct tier function boundary tests."""

    def test_depression_boundary(self):
        self.assertEqual(_who5_tier(0.0), "likely_depression")
        self.assertEqual(_who5_tier(28.0), "likely_depression")
        self.assertEqual(_who5_tier(28.1), "low_wellbeing")

    def test_low_wellbeing_boundary(self):
        self.assertEqual(_who5_tier(29.0), "low_wellbeing")
        self.assertEqual(_who5_tier(50.0), "low_wellbeing")
        self.assertEqual(_who5_tier(50.1), "adequate_wellbeing")

    def test_adequate_boundary(self):
        self.assertEqual(_who5_tier(51.0), "adequate_wellbeing")
        self.assertEqual(_who5_tier(72.0), "adequate_wellbeing")
        self.assertEqual(_who5_tier(72.1), "high_wellbeing")

    def test_high_wellbeing(self):
        self.assertEqual(_who5_tier(73.0), "high_wellbeing")
        self.assertEqual(_who5_tier(100.0), "high_wellbeing")


# ═══════════════════════════════════════════════════════════════════════════
# SF-36 Vitality Subscale (RAND v1.0)
# ═══════════════════════════════════════════════════════════════════════════

class TestSF36VTScoring(unittest.TestCase):
    """SF-36 VT: 4 items (1-6), RAND recoding → 0-100 average."""

    def _make_scale(self, min_items: int = 2) -> Scale:
        return _make_vitality_scale(
            VITALITY_METHOD_SF36_VT, SF36_VT_ITEMS,
            min_items=min_items,
            response_type="sf36_vt_1_6",
            normalize_min=0.0,
            normalize_max=100.0,
        )

    def _score(self, answers: Dict[str, float], min_items: int = 2) -> Optional[ScaleScoreResult]:
        return compute_scale_score(self._make_scale(min_items), answers)

    def test_maximum_vitality(self):
        """Positive items=1 ('All of the time'), negative items=6 ('None') → 100.

        Recoding: pep=1→100, energy=1→100, worn=6→100, tired=6→100.
        Average = 100.
        """
        answers = {
            ITEM_SF36_VT_PEP: 1.0,     # positive: 1→100
            ITEM_SF36_VT_ENERGY: 1.0,   # positive: 1→100
            ITEM_SF36_VT_WORN: 6.0,     # negative: 6→100
            ITEM_SF36_VT_TIRED: 6.0,    # negative: 6→100
        }
        res = self._score(answers)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 100.0, places=2)
        self.assertAlmostEqual(res.normalized_score, 100.0, places=2)
        self.assertEqual(res.risk_tier, "high")

    def test_minimum_vitality(self):
        """Positive items=6 ('None'), negative items=1 ('All of the time') → 0.

        Recoding: pep=6→0, energy=6→0, worn=1→0, tired=1→0.
        Average = 0.
        """
        answers = {
            ITEM_SF36_VT_PEP: 6.0,     # positive: 6→0
            ITEM_SF36_VT_ENERGY: 6.0,   # positive: 6→0
            ITEM_SF36_VT_WORN: 1.0,     # negative: 1→0
            ITEM_SF36_VT_TIRED: 1.0,    # negative: 1→0
        }
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 0.0, places=2)
        self.assertEqual(res.risk_tier, "severely_low")

    def test_positive_item_recoding(self):
        """Verify RAND positive recoding: 1→100, 2→80, 3→60, 4→40, 5→20, 6→0."""
        for val, expected_recode in [(1, 100), (2, 80), (3, 60), (4, 40), (5, 20), (6, 0)]:
            answers = {ITEM_SF36_VT_PEP: float(val), ITEM_SF36_VT_ENERGY: float(val)}
            res = self._score(answers)
            assert res is not None, f"None for val={val}"
            self.assertAlmostEqual(
                res.raw_score, float(expected_recode), places=2,
                msg=f"Positive recode failed for val={val}: expected {expected_recode}, got {res.raw_score}",
            )

    def test_negative_item_recoding(self):
        """Verify RAND negative recoding: 1→0, 2→20, 3→40, 4→60, 5→80, 6→100."""
        for val, expected_recode in [(1, 0), (2, 20), (3, 40), (4, 60), (5, 80), (6, 100)]:
            answers = {ITEM_SF36_VT_WORN: float(val), ITEM_SF36_VT_TIRED: float(val)}
            res = self._score(answers)
            assert res is not None, f"None for val={val}"
            self.assertAlmostEqual(
                res.raw_score, float(expected_recode), places=2,
                msg=f"Negative recode failed for val={val}: expected {expected_recode}, got {res.raw_score}",
            )

    def test_mixed_items_average(self):
        """Mix of positive and negative at population midpoint.

        pep=3 (60), energy=4 (40), worn=3 (40), tired=4 (60).
        Average = (60+40+40+60)/4 = 50.
        """
        answers = {
            ITEM_SF36_VT_PEP: 3.0,     # 60
            ITEM_SF36_VT_ENERGY: 4.0,   # 40
            ITEM_SF36_VT_WORN: 3.0,     # 40
            ITEM_SF36_VT_TIRED: 4.0,    # 60
        }
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 50.0, places=2)
        self.assertEqual(res.risk_tier, "average")

    def test_partial_answers_2_of_4(self):
        """With only 2 items answered, averages over answered items."""
        answers = {
            ITEM_SF36_VT_PEP: 1.0,     # 100
            ITEM_SF36_VT_TIRED: 6.0,    # 100
        }
        res = self._score(answers, min_items=2)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 100.0, places=2)
        self.assertEqual(res.answered_count, 2)

    def test_insufficient_items_returns_none(self):
        """Fewer than min_items → None."""
        answers = {ITEM_SF36_VT_PEP: 3.0}  # only 1
        res = self._score(answers, min_items=2)
        self.assertIsNone(res)

    def test_snapping_to_nearest_response(self):
        """Non-integer values should snap to nearest 1-6."""
        answers = {
            ITEM_SF36_VT_PEP: 2.7,     # rounds to 3 → recode 60
            ITEM_SF36_VT_ENERGY: 4.3,   # rounds to 4 → recode 40
        }
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 50.0, places=2)

    def test_out_of_range_clamped(self):
        """Values outside 1-6 are clamped before snapping."""
        answers = {
            ITEM_SF36_VT_PEP: 0.0,     # clamped to 1 → recode 100
            ITEM_SF36_VT_ENERGY: 10.0,  # clamped to 6 → recode 0
        }
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 50.0, places=2)

    def test_metric_label(self):
        """original_metric_label should be 'SF-36 VT (0-100)'."""
        answers = {ITEM_SF36_VT_PEP: 3.0, ITEM_SF36_VT_ENERGY: 3.0}
        res = self._score(answers)
        assert res is not None
        self.assertEqual(res.original_metric_label, "SF-36 VT (0-100)")

    def test_items_required_is_4(self):
        """items_required should be 4."""
        answers = {iid: 3.0 for iid in SF36_VT_ITEMS}
        res = self._score(answers)
        assert res is not None
        self.assertEqual(res.items_required, 4)


class TestSF36VTTiers(unittest.TestCase):
    """Direct tier function boundary tests."""

    def test_severely_low_boundary(self):
        self.assertEqual(_sf36_vt_tier(0.0), "severely_low")
        self.assertEqual(_sf36_vt_tier(34.9), "severely_low")
        self.assertEqual(_sf36_vt_tier(35.0), "low")

    def test_low_boundary(self):
        self.assertEqual(_sf36_vt_tier(35.0), "low")
        self.assertEqual(_sf36_vt_tier(49.9), "low")
        self.assertEqual(_sf36_vt_tier(50.0), "average")

    def test_average_boundary(self):
        self.assertEqual(_sf36_vt_tier(50.0), "average")
        self.assertEqual(_sf36_vt_tier(65.0), "average")
        self.assertEqual(_sf36_vt_tier(65.1), "high")

    def test_high(self):
        self.assertEqual(_sf36_vt_tier(66.0), "high")
        self.assertEqual(_sf36_vt_tier(100.0), "high")


# ═══════════════════════════════════════════════════════════════════════════
# Subjective Vitality Scale (SVS)
# Uses generic 'mean' scoring — no custom scorer needed
# ═══════════════════════════════════════════════════════════════════════════

class TestSVSScoring(unittest.TestCase):
    """SVS: 6-item version (omitting reverse item 2), 1-7 Likert, mean scoring."""

    def _make_scale(self, min_items: int = 4) -> Scale:
        return _make_vitality_scale(
            "mean", SVS_ITEMS,
            min_items=min_items,
            response_type="svs_1_7",
            normalize_min=1.0,
            normalize_max=7.0,
            tags=("energy", "vitality"),
        )

    def _score(self, answers: Dict[str, float], min_items: int = 4) -> Optional[ScaleScoreResult]:
        return compute_scale_score(self._make_scale(min_items), answers)

    def test_maximum_vitality(self):
        """All items at 7 → raw mean=7, normalized=100."""
        answers = {iid: 7.0 for iid in SVS_ITEMS}
        res = self._score(answers)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 7.0, places=2)
        self.assertAlmostEqual(res.normalized_score, 100.0, places=2)

    def test_minimum_vitality(self):
        """All items at 1 → raw mean=1, normalized=0."""
        answers = {iid: 1.0 for iid in SVS_ITEMS}
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 1.0, places=2)
        self.assertAlmostEqual(res.normalized_score, 0.0, places=2)

    def test_midpoint(self):
        """All items at 4 → raw mean=4, normalized=50."""
        answers = {iid: 4.0 for iid in SVS_ITEMS}
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 4.0, places=2)
        self.assertAlmostEqual(res.normalized_score, 50.0, places=2)

    def test_mixed_scores(self):
        """Varied items → correct mean."""
        answers = {
            ITEM_SVS_1: 7.0,  # alive & vital
            ITEM_SVS_3: 5.0,  # energy burst
            ITEM_SVS_4: 6.0,  # spirit
            ITEM_SVS_5: 4.0,  # look forward
            ITEM_SVS_6: 3.0,  # alert
            ITEM_SVS_7: 5.0,  # energized
        }
        expected_mean = (7 + 5 + 6 + 4 + 3 + 5) / 6.0
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, expected_mean, places=4)
        self.assertEqual(res.answered_count, 6)

    def test_partial_answers_4_of_6(self):
        """4 of 6 items → mean of the 4 answered."""
        answers = {
            ITEM_SVS_1: 6.0,
            ITEM_SVS_3: 6.0,
            ITEM_SVS_4: 6.0,
            ITEM_SVS_5: 6.0,
        }
        res = self._score(answers, min_items=4)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 6.0, places=2)
        self.assertEqual(res.answered_count, 4)

    def test_insufficient_items_returns_none(self):
        """Fewer than min_items → None."""
        answers = {ITEM_SVS_1: 5.0, ITEM_SVS_3: 5.0}  # only 2
        res = self._score(answers, min_items=4)
        self.assertIsNone(res)

    def test_6_items_no_reverse(self):
        """Confirm the 6-item version has NO reverse-scored items."""
        scale = self._make_scale()
        for si in scale.items:
            self.assertFalse(si.reverse, f"Item {si.item_id} should not be reverse-scored")

    def test_items_required_is_6(self):
        """items_required should be 6."""
        answers = {iid: 4.0 for iid in SVS_ITEMS}
        res = self._score(answers)
        assert res is not None
        self.assertEqual(res.items_required, 6)

    def test_response_range_1_to_7(self):
        """Values clamped to 1-7 range by generic scorer."""
        answers = {iid: 0.0 for iid in SVS_ITEMS}  # 0 → clamped to 1.0
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 1.0, places=2)


# ═══════════════════════════════════════════════════════════════════════════
# PROMIS Fatigue Short Form 7a
# ═══════════════════════════════════════════════════════════════════════════

class TestPROMISFatigue7aScoring(unittest.TestCase):
    """PROMIS Fatigue 7a: 7 items (1-5), sum → T-score lookup, inverted normalization."""

    def _make_scale(self, min_items: int = 5) -> Scale:
        return _make_vitality_scale(
            VITALITY_METHOD_PROMIS_FATIGUE_7A, PF7_ITEMS,
            min_items=min_items,
            response_type="promis_freq_1_5",
            normalize_min=0.0,
            normalize_max=100.0,
        )

    def _answers(self, value: float) -> Dict[str, float]:
        """All 7 items at the same value."""
        return {iid: value for iid in PF7_ITEMS}

    def _score(self, answers: Dict[str, float], min_items: int = 5) -> Optional[ScaleScoreResult]:
        return compute_scale_score(self._make_scale(min_items), answers)

    def test_minimum_fatigue(self):
        """All items at 1 ('Never') → raw sum=7, T=33.7.

        Inverted normalization: T=33.7 maps to HIGH energy (low fatigue).
        normalized = (33.7 - 75.7) / (33.7 - 75.7) * 100 = 100.0
        """
        res = self._score(self._answers(1.0))
        self.assertIsNotNone(res)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 7.0, places=2)
        self.assertAlmostEqual(res.original_metric_value, 33.7, places=1)
        self.assertAlmostEqual(res.normalized_score, 100.0, places=1)
        self.assertEqual(res.risk_tier, "normal")

    def test_maximum_fatigue(self):
        """All items at 5 ('Always') → raw sum=35, T=75.7.

        Inverted normalization: T=75.7 → normalized ≈ 0.
        """
        res = self._score(self._answers(5.0))
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 35.0, places=2)
        self.assertAlmostEqual(res.original_metric_value, 75.7, places=1)
        self.assertAlmostEqual(res.normalized_score, 0.0, places=1)
        self.assertEqual(res.risk_tier, "severe")

    def test_population_mean_t50(self):
        """T=50 is the US general population mean.

        Raw sum=20 → T=49.3, raw sum=21 → T=50.1.
        """
        # raw=21 → T=50.1 ≈ population mean
        answers = {
            ITEM_PF7_TIRED: 3.0,       # 3
            ITEM_PF7_EXHAUSTION: 3.0,   # 3
            ITEM_PF7_RUN_OUT: 3.0,      # 3
            ITEM_PF7_LIMIT_WORK: 3.0,   # 3
            ITEM_PF7_THINK: 3.0,        # 3
            ITEM_PF7_BATH: 3.0,         # 3
            ITEM_PF7_EXERCISE: 3.0,     # 3 → sum=21, T=50.1
        }
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.original_metric_value, 50.1, places=1)
        self.assertEqual(res.risk_tier, "normal")

    def test_mild_fatigue_threshold_55(self):
        """T=55 is the mild fatigue threshold.

        raw=26 → T=54.9 (normal), raw=27 → T=56.0 (mild).
        """
        # raw=26 → T=54.9 → normal
        self.assertEqual(_promis_fatigue_tier(54.9), "normal")
        # raw=27 → T=56.0 → mild
        self.assertEqual(_promis_fatigue_tier(56.0), "mild")

    def test_moderate_fatigue_threshold_60(self):
        """T=60 is the moderate fatigue threshold.

        raw=30 → T=60.1.
        """
        self.assertEqual(_promis_fatigue_tier(59.9), "mild")
        self.assertEqual(_promis_fatigue_tier(60.1), "moderate")

    def test_severe_fatigue_threshold_70(self):
        """T=70 is the severe fatigue threshold.

        raw=34 → T=70.0.
        """
        self.assertEqual(_promis_fatigue_tier(69.9), "moderate")
        self.assertEqual(_promis_fatigue_tier(70.0), "severe")

    def test_t_score_lookup_table_completeness(self):
        """Lookup table should cover raw scores 7 through 35 (all possible sums)."""
        for raw in range(7, 36):
            self.assertIn(raw, _PROMIS_FATIGUE_7A_RAW_TO_T,
                          f"Missing T-score for raw={raw}")

    def test_t_score_monotonically_increasing(self):
        """T-scores should increase monotonically with raw scores (more fatigue)."""
        prev_t = 0.0
        for raw in range(7, 36):
            t = _PROMIS_FATIGUE_7A_RAW_TO_T[raw]
            self.assertGreater(t, prev_t,
                               f"T-score not monotonic at raw={raw}: T={t} <= prev={prev_t}")
            prev_t = t

    def test_t_score_population_calibration(self):
        """T-scores should be centered near 50 (US reference population).

        Raw=20 → T=49.3 (just below 50), raw=21 → T=50.1 (just above 50).
        """
        self.assertAlmostEqual(_PROMIS_FATIGUE_7A_RAW_TO_T[20], 49.3, places=1)
        self.assertAlmostEqual(_PROMIS_FATIGUE_7A_RAW_TO_T[21], 50.1, places=1)

    def test_inverted_normalization_direction(self):
        """Higher T-score (more fatigue) → LOWER normalized score (less energy).

        This inversion aligns with state_snapshots.py energy_vitality dimension
        where higher = better.
        """
        res_low_fatigue = self._score(self._answers(1.0))   # T=33.7
        res_high_fatigue = self._score(self._answers(5.0))   # T=75.7
        assert res_low_fatigue is not None
        assert res_high_fatigue is not None
        self.assertGreater(res_low_fatigue.normalized_score,
                           res_high_fatigue.normalized_score)

    def test_prorating_partial_answers(self):
        """With 5 of 7 items answered, pro-rate to 7-item equivalent."""
        # 5 items each at 3 → sum=15, pro-rated: 15*(7/5)=21, round=21
        answers = {
            ITEM_PF7_TIRED: 3.0,
            ITEM_PF7_EXHAUSTION: 3.0,
            ITEM_PF7_RUN_OUT: 3.0,
            ITEM_PF7_LIMIT_WORK: 3.0,
            ITEM_PF7_THINK: 3.0,
        }
        res = self._score(answers, min_items=5)
        self.assertIsNotNone(res)
        assert res is not None
        # Pro-rated raw: 15 * (7/5) = 21.0 → T=50.1
        self.assertAlmostEqual(res.original_metric_value, 50.1, places=1)
        self.assertEqual(res.answered_count, 5)

    def test_insufficient_items_returns_none(self):
        """Fewer than min_items → None."""
        answers = {ITEM_PF7_TIRED: 3.0, ITEM_PF7_EXHAUSTION: 3.0}  # only 2
        res = self._score(answers, min_items=5)
        self.assertIsNone(res)

    def test_values_clamped_1_5(self):
        """Values outside 1-5 should be clamped."""
        answers = {iid: 0.0 for iid in PF7_ITEMS}  # 0 → clamped to 1
        res = self._score(answers)
        assert res is not None
        self.assertAlmostEqual(res.raw_score, 7.0, places=2)

    def test_metric_label(self):
        """original_metric_label should be 'PROMIS Fatigue T-score'."""
        res = self._score(self._answers(3.0))
        assert res is not None
        self.assertEqual(res.original_metric_label, "PROMIS Fatigue T-score")

    def test_items_required_is_7(self):
        """items_required should be 7."""
        res = self._score(self._answers(3.0))
        assert res is not None
        self.assertEqual(res.items_required, 7)


class TestPROMISFatigueTiers(unittest.TestCase):
    """Direct tier function boundary tests."""

    def test_normal(self):
        self.assertEqual(_promis_fatigue_tier(33.7), "normal")
        self.assertEqual(_promis_fatigue_tier(50.0), "normal")
        self.assertEqual(_promis_fatigue_tier(54.9), "normal")

    def test_mild(self):
        self.assertEqual(_promis_fatigue_tier(55.0), "mild")
        self.assertEqual(_promis_fatigue_tier(57.0), "mild")
        self.assertEqual(_promis_fatigue_tier(59.9), "mild")

    def test_moderate(self):
        self.assertEqual(_promis_fatigue_tier(60.0), "moderate")
        self.assertEqual(_promis_fatigue_tier(65.0), "moderate")
        self.assertEqual(_promis_fatigue_tier(69.9), "moderate")

    def test_severe(self):
        self.assertEqual(_promis_fatigue_tier(70.0), "severe")
        self.assertEqual(_promis_fatigue_tier(75.7), "severe")


# ═══════════════════════════════════════════════════════════════════════════
# Cross-cutting: infer_risk_tier_from_scale dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestVitalityRiskTierDispatch(unittest.TestCase):
    """Verify infer_risk_tier_from_scale routes to vitality tiers correctly."""

    def _make_method_scale(self, method: str) -> Scale:
        return _make_vitality_scale(method, ["dummy_item"])

    def test_who5_dispatch(self):
        scale = self._make_method_scale(VITALITY_METHOD_WHO5)
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=20.0), "likely_depression")
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=60.0), "adequate_wellbeing")
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=80.0), "high_wellbeing")

    def test_sf36_vt_dispatch(self):
        scale = self._make_method_scale(VITALITY_METHOD_SF36_VT)
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=20.0), "severely_low")
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=55.0), "average")
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=80.0), "high")

    def test_promis_fatigue_dispatch(self):
        scale = self._make_method_scale(VITALITY_METHOD_PROMIS_FATIGUE_7A)
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=45.0), "normal")
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=57.0), "mild")
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=65.0), "moderate")
        self.assertEqual(infer_risk_tier_from_scale(scale, raw_score=72.0), "severe")

    def test_mean_method_returns_none(self):
        """SVS uses 'mean' method which doesn't have a custom tier → None."""
        scale = self._make_method_scale("mean")
        self.assertIsNone(infer_risk_tier_from_scale(scale, raw_score=4.5))


# ═══════════════════════════════════════════════════════════════════════════
# Cross-cutting: response type validation
# ═══════════════════════════════════════════════════════════════════════════

class TestVitalityResponseTypes(unittest.TestCase):
    """Verify the 4 vitality response types are properly registered."""

    def test_who5_response_type(self):
        from questions_agent_platform.pipeline.response_types import get_response_type
        rt = get_response_type("who5_0_5")
        self.assertEqual(rt.min_value, 0.0)
        self.assertEqual(rt.max_value, 5.0)
        self.assertEqual(len(rt.options), 6)  # 0-5

    def test_sf36_vt_response_type(self):
        from questions_agent_platform.pipeline.response_types import get_response_type
        rt = get_response_type("sf36_vt_1_6")
        self.assertEqual(rt.min_value, 1.0)
        self.assertEqual(rt.max_value, 6.0)
        self.assertEqual(len(rt.options), 6)  # 1-6

    def test_svs_response_type(self):
        from questions_agent_platform.pipeline.response_types import get_response_type
        rt = get_response_type("svs_1_7")
        self.assertEqual(rt.min_value, 1.0)
        self.assertEqual(rt.max_value, 7.0)
        self.assertEqual(len(rt.options), 7)  # 1-7

    def test_promis_response_type(self):
        from questions_agent_platform.pipeline.response_types import get_response_type
        rt = get_response_type("promis_freq_1_5")
        self.assertEqual(rt.min_value, 1.0)
        self.assertEqual(rt.max_value, 5.0)
        self.assertEqual(len(rt.options), 5)  # 1-5


# ═══════════════════════════════════════════════════════════════════════════
# Cross-cutting: state dimension wiring
# ═══════════════════════════════════════════════════════════════════════════

class TestVitalityStateDimensionWiring(unittest.TestCase):
    """Verify vitality tags map to correct state dimensions."""

    def test_energy_tag_maps_to_energy_vitality(self):
        from questions_agent_platform.pipeline.state_snapshots import infer_state_dimensions_from_tags
        dims = infer_state_dimensions_from_tags(["energy"])
        self.assertIn("energy_vitality", dims)

    def test_vitality_tag_maps_to_energy_vitality(self):
        from questions_agent_platform.pipeline.state_snapshots import infer_state_dimensions_from_tags
        dims = infer_state_dimensions_from_tags(["vitality"])
        self.assertIn("energy_vitality", dims)

    def test_fatigue_tag_maps_to_energy_vitality(self):
        from questions_agent_platform.pipeline.state_snapshots import infer_state_dimensions_from_tags
        dims = infer_state_dimensions_from_tags(["fatigue"])
        self.assertIn("energy_vitality", dims)

    def test_wellbeing_tag_maps_to_energy_and_mood(self):
        from questions_agent_platform.pipeline.state_snapshots import infer_state_dimensions_from_tags
        dims = infer_state_dimensions_from_tags(["wellbeing"])
        self.assertIn("energy_vitality", dims)
        self.assertIn("mood_affect", dims)

    def test_combined_vitality_tags(self):
        """A scale tagged ['energy', 'vitality', 'wellbeing'] maps correctly."""
        from questions_agent_platform.pipeline.state_snapshots import infer_state_dimensions_from_tags
        dims = infer_state_dimensions_from_tags(["energy", "vitality", "wellbeing"])
        self.assertIn("energy_vitality", dims)
        self.assertIn("mood_affect", dims)
        # Should NOT map to unrelated dimensions
        self.assertNotIn("gut_gi", dims)
        self.assertNotIn("cardiovascular_load", dims)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-cutting: allowed methods in registry
# ═══════════════════════════════════════════════════════════════════════════

class TestVitalityMethodsRegistered(unittest.TestCase):
    """Verify vitality scoring methods are in ALLOWED_SCORING_METHODS."""

    def test_who5_method_allowed(self):
        from questions_agent_platform.pipeline.registry import ALLOWED_SCORING_METHODS
        self.assertIn("instrument_who5", ALLOWED_SCORING_METHODS)

    def test_sf36_vt_method_allowed(self):
        from questions_agent_platform.pipeline.registry import ALLOWED_SCORING_METHODS
        self.assertIn("instrument_sf36_vt", ALLOWED_SCORING_METHODS)

    def test_promis_fatigue_method_allowed(self):
        from questions_agent_platform.pipeline.registry import ALLOWED_SCORING_METHODS
        self.assertIn("instrument_promis_fatigue_7a", ALLOWED_SCORING_METHODS)


# ═══════════════════════════════════════════════════════════════════════════
# Insufficient data: all vitality instruments
# ═══════════════════════════════════════════════════════════════════════════

class TestVitalityInsufficientData(unittest.TestCase):
    """All vitality instruments must return None when data is insufficient."""

    def test_who5_insufficient(self):
        scale = _make_vitality_scale(
            VITALITY_METHOD_WHO5, WHO5_ITEMS,
            min_items=3, response_type="who5_0_5",
        )
        self.assertIsNone(compute_scale_score(scale, {ITEM_WHO5_1: 3.0}))

    def test_sf36_vt_insufficient(self):
        scale = _make_vitality_scale(
            VITALITY_METHOD_SF36_VT, SF36_VT_ITEMS,
            min_items=3, response_type="sf36_vt_1_6",
        )
        self.assertIsNone(compute_scale_score(scale, {ITEM_SF36_VT_PEP: 3.0}))

    def test_promis_fatigue_insufficient(self):
        scale = _make_vitality_scale(
            VITALITY_METHOD_PROMIS_FATIGUE_7A, PF7_ITEMS,
            min_items=5, response_type="promis_freq_1_5",
        )
        self.assertIsNone(compute_scale_score(scale, {ITEM_PF7_TIRED: 3.0}))

    def test_svs_insufficient(self):
        scale = _make_vitality_scale(
            "mean", SVS_ITEMS,
            min_items=4, response_type="svs_1_7",
        )
        self.assertIsNone(compute_scale_score(scale, {ITEM_SVS_1: 5.0}))

    def test_empty_answers(self):
        """Empty answer dict → None for all instruments."""
        for method, items, rt in [
            (VITALITY_METHOD_WHO5, WHO5_ITEMS, "who5_0_5"),
            (VITALITY_METHOD_SF36_VT, SF36_VT_ITEMS, "sf36_vt_1_6"),
            (VITALITY_METHOD_PROMIS_FATIGUE_7A, PF7_ITEMS, "promis_freq_1_5"),
            ("mean", SVS_ITEMS, "svs_1_7"),
        ]:
            scale = _make_vitality_scale(method, items, min_items=2, response_type=rt)
            self.assertIsNone(
                compute_scale_score(scale, {}),
                f"Expected None for empty answers with method={method}",
            )


if __name__ == "__main__":
    unittest.main()
