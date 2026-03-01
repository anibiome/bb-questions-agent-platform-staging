"""Tests for the composite cardiometabolic risk index.

The CardioRiskIndex aggregates 6 validated instruments into a single
benchmark score for comparison with AniFold latent-space findings.
"""

import unittest
from typing import Dict

from questions_agent_platform.pipeline.scoring import ScaleScoreResult
from questions_agent_platform.pipeline.cardio_risk_index import (
    CardioRiskIndex,
    compute_cardio_risk_index,
    cardio_risk_to_dict,
    _composite_tier,
    _coverage_confidence,
    _INSTRUMENT_WEIGHTS,
    _INVERTED_INSTRUMENTS,
    ALL_CARDIO_METHODS,
    CARDIO_RISK_INDEX_VERSION,
)


def _mock_score(
    normalized_score: float,
    risk_tier: str = "low",
    raw_score: float = 0.0,
) -> ScaleScoreResult:
    """Build a minimal ScaleScoreResult for testing."""
    return ScaleScoreResult(
        raw_score=raw_score,
        normalized_score=normalized_score,
        answered_count=5,
        items_required=5,
        risk_tier=risk_tier,
    )


class TestCompositeComputation(unittest.TestCase):
    """Core composite risk computation."""

    def test_all_instruments_zero_risk(self):
        """All instruments at zero risk → composite ≈ 0."""
        scores = {
            "instrument_findrisc": _mock_score(0.0, "low"),
            "instrument_ez_cvd": _mock_score(0.0, "low"),
            "instrument_ipaq_sf": _mock_score(100.0, "high_activity"),  # inverted: 100→0
            "instrument_lee_nafld": _mock_score(0.0, "low_risk"),
            "instrument_scored": _mock_score(0.0, "negative_screen"),
            "instrument_audit_c": _mock_score(0.0, "no_use"),
        }
        result = compute_cardio_risk_index(scores)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.composite_risk, 0.0, places=4)
        self.assertEqual(result.risk_tier, "low")
        self.assertEqual(result.instruments_available, 6)
        self.assertAlmostEqual(result.coverage, 1.0, places=4)
        self.assertEqual(result.confidence, "high")

    def test_all_instruments_max_risk(self):
        """All instruments at maximum risk → composite ≈ 1.0."""
        scores = {
            "instrument_findrisc": _mock_score(100.0, "very_high"),
            "instrument_ez_cvd": _mock_score(100.0, "high"),
            "instrument_ipaq_sf": _mock_score(0.0, "low_activity"),  # inverted: 0→1.0
            "instrument_lee_nafld": _mock_score(100.0, "high_risk"),
            "instrument_scored": _mock_score(100.0, "positive_screen"),
            "instrument_audit_c": _mock_score(100.0, "high_risk"),
        }
        result = compute_cardio_risk_index(scores)
        assert result is not None
        self.assertAlmostEqual(result.composite_risk, 1.0, places=4)
        self.assertEqual(result.risk_tier, "high")

    def test_midpoint_risk(self):
        """All instruments at 50% risk → composite = 0.5."""
        scores = {
            "instrument_findrisc": _mock_score(50.0, "moderate"),
            "instrument_ez_cvd": _mock_score(50.0, "moderate"),
            "instrument_ipaq_sf": _mock_score(50.0, "moderate_activity"),  # inverted: 50→0.5
            "instrument_lee_nafld": _mock_score(50.0, "elevated_risk"),
            "instrument_scored": _mock_score(50.0, "positive_screen"),
            "instrument_audit_c": _mock_score(50.0, "positive_screen"),
        }
        result = compute_cardio_risk_index(scores)
        assert result is not None
        self.assertAlmostEqual(result.composite_risk, 0.5, places=4)

    def test_ipaq_inversion(self):
        """IPAQ-SF is inverted: high activity (norm=100) → low risk (0.0)."""
        scores_active = {
            "instrument_ipaq_sf": _mock_score(100.0, "high_activity"),
        }
        scores_sedentary = {
            "instrument_ipaq_sf": _mock_score(0.0, "low_activity"),
        }
        res_active = compute_cardio_risk_index(scores_active)
        res_sedentary = compute_cardio_risk_index(scores_sedentary)
        assert res_active is not None and res_sedentary is not None
        self.assertAlmostEqual(res_active.composite_risk, 0.0, places=4)
        self.assertAlmostEqual(res_sedentary.composite_risk, 1.0, places=4)

    def test_other_instruments_not_inverted(self):
        """FINDRISC, EZ-CVD, etc. are NOT inverted."""
        for method in ALL_CARDIO_METHODS - _INVERTED_INSTRUMENTS:
            scores = {method: _mock_score(100.0, "high")}
            result = compute_cardio_risk_index(scores)
            assert result is not None, f"None for {method}"
            self.assertAlmostEqual(
                result.composite_risk, 1.0, places=4,
                msg=f"{method} at norm=100 should give risk=1.0",
            )


class TestWeightRenormalization(unittest.TestCase):
    """Weights renormalize when instruments are missing."""

    def test_single_instrument(self):
        """One instrument → its weight becomes 1.0 after renormalization."""
        scores = {"instrument_findrisc": _mock_score(50.0, "moderate")}
        result = compute_cardio_risk_index(scores)
        assert result is not None
        self.assertEqual(len(result.components), 1)
        self.assertAlmostEqual(result.components[0].normalized_weight, 1.0, places=4)
        self.assertAlmostEqual(result.composite_risk, 0.5, places=4)

    def test_two_instruments_weights(self):
        """Two instruments → weights renormalized to sum to 1.0."""
        scores = {
            "instrument_findrisc": _mock_score(80.0, "high"),       # weight 0.25
            "instrument_ez_cvd": _mock_score(60.0, "moderate"),      # weight 0.20
        }
        result = compute_cardio_risk_index(scores)
        assert result is not None
        total_w = 0.25 + 0.20
        expected_composite = (0.25 * 0.8 + 0.20 * 0.6) / total_w
        self.assertAlmostEqual(result.composite_risk, expected_composite, places=4)
        # Normalized weights
        norm_w_sum = sum(c.normalized_weight for c in result.components)
        self.assertAlmostEqual(norm_w_sum, 1.0, places=4)

    def test_weights_sum_to_one(self):
        """With all instruments, normalized weights sum to 1.0."""
        scores = {m: _mock_score(50.0) for m in ALL_CARDIO_METHODS}
        result = compute_cardio_risk_index(scores)
        assert result is not None
        total = sum(c.normalized_weight for c in result.components)
        self.assertAlmostEqual(total, 1.0, places=4)


class TestPartialCoverage(unittest.TestCase):
    """Behavior with missing instruments."""

    def test_no_instruments_returns_none(self):
        """Empty dict → None."""
        self.assertIsNone(compute_cardio_risk_index({}))

    def test_unknown_method_ignored(self):
        """Non-cardiometabolic methods are silently ignored."""
        scores = {"instrument_who5": _mock_score(50.0)}
        self.assertIsNone(compute_cardio_risk_index(scores))

    def test_partial_coverage_ratio(self):
        """3 of 6 instruments → coverage = 0.5."""
        scores = {
            "instrument_findrisc": _mock_score(50.0),
            "instrument_ez_cvd": _mock_score(50.0),
            "instrument_scored": _mock_score(50.0),
        }
        result = compute_cardio_risk_index(scores)
        assert result is not None
        self.assertEqual(result.instruments_available, 3)
        self.assertEqual(result.instruments_total, 6)
        self.assertAlmostEqual(result.coverage, 0.5, places=4)
        self.assertEqual(result.confidence, "medium")

    def test_single_instrument_low_confidence(self):
        """1 instrument → coverage ≈ 0.17 → low confidence."""
        scores = {"instrument_audit_c": _mock_score(50.0)}
        result = compute_cardio_risk_index(scores)
        assert result is not None
        self.assertEqual(result.confidence, "low")


class TestCompositeTiers(unittest.TestCase):
    """Composite risk tier boundaries."""

    def test_low(self):
        self.assertEqual(_composite_tier(0.0), "low")
        self.assertEqual(_composite_tier(0.24), "low")

    def test_moderate(self):
        self.assertEqual(_composite_tier(0.25), "moderate")
        self.assertEqual(_composite_tier(0.44), "moderate")

    def test_elevated(self):
        self.assertEqual(_composite_tier(0.45), "elevated")
        self.assertEqual(_composite_tier(0.64), "elevated")

    def test_high(self):
        self.assertEqual(_composite_tier(0.65), "high")
        self.assertEqual(_composite_tier(1.0), "high")


class TestCoverageConfidence(unittest.TestCase):
    """Coverage → confidence mapping."""

    def test_high(self):
        self.assertEqual(_coverage_confidence(5, 6), "high")
        self.assertEqual(_coverage_confidence(6, 6), "high")

    def test_medium(self):
        self.assertEqual(_coverage_confidence(3, 6), "medium")
        self.assertEqual(_coverage_confidence(4, 6), "medium")

    def test_low(self):
        self.assertEqual(_coverage_confidence(1, 6), "low")
        self.assertEqual(_coverage_confidence(2, 6), "low")


class TestSerialization(unittest.TestCase):
    """JSON serialization round-trip."""

    def test_to_dict_structure(self):
        """Serialized dict has expected keys."""
        scores = {
            "instrument_findrisc": _mock_score(50.0, "moderate"),
            "instrument_ipaq_sf": _mock_score(75.0, "moderate_activity"),
        }
        result = compute_cardio_risk_index(scores)
        assert result is not None
        d = cardio_risk_to_dict(result)
        self.assertIn("composite_risk", d)
        self.assertIn("composite_risk_pct", d)
        self.assertIn("risk_tier", d)
        self.assertIn("components", d)
        self.assertIn("version", d)
        self.assertEqual(d["version"], CARDIO_RISK_INDEX_VERSION)
        self.assertEqual(len(d["components"]), 2)

    def test_component_fields(self):
        """Each component has required fields."""
        scores = {"instrument_findrisc": _mock_score(50.0, "moderate")}
        result = compute_cardio_risk_index(scores)
        assert result is not None
        d = cardio_risk_to_dict(result)
        comp = d["components"][0]
        for key in ("method", "risk_value", "weight", "normalized_weight",
                     "risk_tier", "normalized_score", "inverted"):
            self.assertIn(key, comp, f"Missing key: {key}")

    def test_pct_is_100x_composite(self):
        """composite_risk_pct = composite_risk × 100."""
        scores = {"instrument_findrisc": _mock_score(70.0)}
        result = compute_cardio_risk_index(scores)
        assert result is not None
        self.assertAlmostEqual(result.composite_risk_pct, result.composite_risk * 100, places=2)


if __name__ == "__main__":
    unittest.main()
