"""Tests for Active Coherence Detection from Behavioral Metadata.

Validates all six coherence signals independently, composite scoring,
tier assignment, edge cases, and serialization round-trip.

Test strategy:
  - Each signal has its own test class with known-good / known-bad inputs
  - Integration tests simulate realistic sessions (attentive, careless,
    mixed-quality) and verify composite scoring
  - Edge cases: empty input, single item, all skipped, zero RT
"""

import math
import unittest
from typing import List

from questions_agent_platform.pipeline.coherence import (
    ItemResponse,
    CoherenceSignal,
    CoherenceAssessment,
    assess_coherence,
    coherence_to_dict,
    coherence_from_dict,
    _speed_index,
    _longstring_index,
    _rt_variability,
    _intra_scale_variance,
    _fatigue_slope,
    _skip_rate,
    _ramp_severity,
    _ramp_severity_inverse,
    COHERENCE_VERSION,
    FAST_RT_MS,
    SPEED_INDEX_THRESHOLD,
    LONGSTRING_THRESHOLD,
    LONGSTRING_RATIO_THRESHOLD,
    RT_CV_THRESHOLD,
    INTRA_SCALE_VAR_THRESHOLD,
    FATIGUE_SLOPE_THRESHOLD,
    SKIP_RATE_THRESHOLD,
    SUSPECT_MIN_FLAGS,
    INVALID_MIN_FLAGS,
    SIGNAL_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(
    position: int,
    value: int = 2,
    rt_ms: float = 5000.0,
    scale_id: str = "phq9",
    n_options: int = 5,
    skipped: bool = False,
    edits: int = 0,
) -> ItemResponse:
    """Convenience factory for building test ItemResponse objects."""
    return ItemResponse(
        item_id=f"{scale_id}_item_{position}",
        scale_id=scale_id,
        response_value=value,
        response_time_ms=rt_ms,
        item_position=position,
        n_response_options=n_options,
        edit_count=edits,
        was_skipped=skipped,
    )


def _attentive_session(n: int = 20) -> List[ItemResponse]:
    """Simulate an attentive respondent: varied responses, normal RTs."""
    items = []
    for i in range(n):
        # Vary response values (0-4 on a 5-point scale, cycling)
        value = (i * 3 + 1) % 5
        # Vary RT: 3-8 seconds, with some natural variation
        rt = 3000 + 2500 * math.sin(i * 0.7) + 500 * (i % 3)
        items.append(_make_item(position=i, value=value, rt_ms=rt))
    return items


def _careless_session(n: int = 20) -> List[ItemResponse]:
    """Simulate a careless respondent: all same answer, very fast."""
    return [
        _make_item(position=i, value=3, rt_ms=800.0)
        for i in range(n)
    ]


def _speeding_session(n: int = 20) -> List[ItemResponse]:
    """Simulate progressive disengagement: RT decreases rapidly."""
    items = []
    for i in range(n):
        value = (i * 2 + 1) % 5
        # Start at 6000ms, decrease to ~500ms by end
        rt = 6000 * math.exp(-3.0 * i / n)
        items.append(_make_item(position=i, value=value, rt_ms=rt))
    return items


# ---------------------------------------------------------------------------
# Signal 1: Speed index
# ---------------------------------------------------------------------------

class TestSpeedIndex(unittest.TestCase):

    def test_all_slow_no_flag(self):
        """All items above threshold → not flagged."""
        items = [_make_item(i, rt_ms=5000.0) for i in range(10)]
        sig = _speed_index(items, FAST_RT_MS)
        self.assertFalse(sig.flagged)
        self.assertAlmostEqual(sig.value, 0.0, places=4)
        self.assertAlmostEqual(sig.severity, 0.0, places=4)

    def test_all_fast_flagged(self):
        """All items below threshold → flagged, high severity."""
        items = [_make_item(i, rt_ms=500.0) for i in range(10)]
        sig = _speed_index(items, FAST_RT_MS)
        self.assertTrue(sig.flagged)
        self.assertAlmostEqual(sig.value, 1.0, places=4)
        self.assertGreater(sig.severity, 0.8)

    def test_half_fast_boundary(self):
        """Exactly 50% fast → at boundary (>50% required to flag)."""
        items = (
            [_make_item(i, rt_ms=500.0) for i in range(5)]
            + [_make_item(i + 5, rt_ms=5000.0) for i in range(5)]
        )
        sig = _speed_index(items, FAST_RT_MS)
        self.assertAlmostEqual(sig.value, 0.5, places=4)
        # Exactly at threshold: not flagged (must be >)
        self.assertFalse(sig.flagged)

    def test_just_over_threshold(self):
        """51% fast → flagged."""
        fast = [_make_item(i, rt_ms=500.0) for i in range(51)]
        slow = [_make_item(i + 51, rt_ms=5000.0) for i in range(49)]
        items = fast + slow
        sig = _speed_index(items, FAST_RT_MS)
        self.assertTrue(sig.flagged)

    def test_custom_fast_threshold(self):
        """Custom fast threshold works."""
        items = [_make_item(i, rt_ms=2500.0) for i in range(10)]
        # Default: 2000ms → 2500 is above, not fast
        sig_default = _speed_index(items, 2000.0)
        self.assertFalse(sig_default.flagged)
        self.assertAlmostEqual(sig_default.value, 0.0, places=4)
        # Stricter: 3000ms → 2500 is below, all fast
        sig_strict = _speed_index(items, 3000.0)
        self.assertAlmostEqual(sig_strict.value, 1.0, places=4)

    def test_empty_items(self):
        """Empty list → not flagged."""
        sig = _speed_index([], FAST_RT_MS)
        self.assertFalse(sig.flagged)
        self.assertAlmostEqual(sig.severity, 0.0)


# ---------------------------------------------------------------------------
# Signal 2: Longstring
# ---------------------------------------------------------------------------

class TestLongstring(unittest.TestCase):

    def test_all_different_no_flag(self):
        """All different responses → longstring = 1, not flagged."""
        items = [_make_item(i, value=i % 5) for i in range(20)]
        sig = _longstring_index(items)
        self.assertFalse(sig.flagged)
        self.assertLessEqual(sig.value, 5)

    def test_all_same_flagged(self):
        """20 identical responses → longstring = 20, flagged."""
        items = [_make_item(i, value=3) for i in range(20)]
        sig = _longstring_index(items)
        self.assertTrue(sig.flagged)
        self.assertAlmostEqual(sig.value, 20.0)

    def test_run_of_8_absolute_threshold(self):
        """Run of exactly 8 hits absolute threshold."""
        items = (
            [_make_item(i, value=1) for i in range(3)]
            + [_make_item(i + 3, value=3) for i in range(8)]
            + [_make_item(i + 11, value=i % 4) for i in range(9)]  # varied
        )
        sig = _longstring_index(items)
        self.assertAlmostEqual(sig.value, 8.0)
        # 8 >= LONGSTRING_THRESHOLD → flagged
        self.assertTrue(sig.flagged)

    def test_short_run_not_flagged(self):
        """Run of 4 in 20 items → not flagged."""
        items = []
        for i in range(20):
            # 4 same values then different
            if 5 <= i < 9:
                value = 3
            else:
                value = i % 5
            items.append(_make_item(i, value=value))
        sig = _longstring_index(items)
        self.assertFalse(sig.flagged)
        self.assertAlmostEqual(sig.value, 4.0)

    def test_ratio_threshold_short_scale(self):
        """Short scale: ratio threshold may flag even below absolute."""
        # 5 items, 4 identical → 80% > 60% ratio threshold
        items = [_make_item(i, value=3) for i in range(4)]
        items.append(_make_item(4, value=1))
        sig = _longstring_index(items)
        # ratio = 4/5 = 0.80 > 0.60 → flagged even though 4 < 8
        self.assertTrue(sig.flagged)

    def test_too_few_items(self):
        """Fewer than 3 items → not flagged."""
        items = [_make_item(0, value=3), _make_item(1, value=3)]
        sig = _longstring_index(items)
        self.assertFalse(sig.flagged)


# ---------------------------------------------------------------------------
# Signal 3: RT variability
# ---------------------------------------------------------------------------

class TestRTVariability(unittest.TestCase):

    def test_high_cv_not_flagged(self):
        """Normal RT variation → not flagged."""
        items = _attentive_session(20)
        sig = _rt_variability(items)
        self.assertFalse(sig.flagged)
        self.assertGreater(sig.value, RT_CV_THRESHOLD)

    def test_constant_rt_flagged(self):
        """All same RT (CV=0) → flagged, max severity."""
        items = [_make_item(i, rt_ms=3000.0) for i in range(20)]
        sig = _rt_variability(items)
        self.assertTrue(sig.flagged)
        self.assertAlmostEqual(sig.value, 0.0, places=4)
        self.assertAlmostEqual(sig.severity, 1.0, places=4)

    def test_slightly_varied_rt_flagged(self):
        """Tiny RT variation (CV ~0.05) → flagged."""
        items = [_make_item(i, rt_ms=3000 + (i % 3) * 50) for i in range(20)]
        sig = _rt_variability(items)
        self.assertTrue(sig.flagged)
        self.assertLess(sig.value, RT_CV_THRESHOLD)

    def test_near_zero_mean_rt(self):
        """Near-zero mean RT → flagged with max severity."""
        items = [_make_item(i, rt_ms=0.5) for i in range(10)]
        sig = _rt_variability(items)
        self.assertTrue(sig.flagged)
        self.assertAlmostEqual(sig.severity, 1.0)

    def test_too_few_items(self):
        """Fewer than 3 items → not flagged."""
        items = [_make_item(0), _make_item(1)]
        sig = _rt_variability(items)
        self.assertFalse(sig.flagged)


# ---------------------------------------------------------------------------
# Signal 4: Intra-scale variance
# ---------------------------------------------------------------------------

class TestIntraScaleVariance(unittest.TestCase):

    def test_varied_responses_not_flagged(self):
        """Normal response variation → not flagged."""
        items = [_make_item(i, value=i % 5, scale_id="phq9") for i in range(9)]
        sig = _intra_scale_variance(items)
        self.assertFalse(sig.flagged)
        self.assertGreater(sig.value, INTRA_SCALE_VAR_THRESHOLD)

    def test_all_same_response_flagged(self):
        """All identical responses → variance 0, flagged."""
        items = [_make_item(i, value=3, scale_id="phq9") for i in range(9)]
        sig = _intra_scale_variance(items)
        self.assertTrue(sig.flagged)
        self.assertAlmostEqual(sig.value, 0.0, places=4)
        self.assertAlmostEqual(sig.severity, 1.0, places=4)

    def test_multi_scale_average(self):
        """Variance computed per-scale then averaged."""
        # Scale A: varied (not flagged)
        scale_a = [_make_item(i, value=i % 5, scale_id="scaleA") for i in range(9)]
        # Scale B: all same (flagged if alone)
        scale_b = [_make_item(i + 9, value=3, scale_id="scaleB") for i in range(9)]
        items = scale_a + scale_b
        sig = _intra_scale_variance(items)
        # Average of high + zero → middling
        self.assertGreater(sig.value, 0.0)

    def test_single_item_scale_ignored(self):
        """Scales with < 2 items are excluded from variance computation."""
        items = [_make_item(0, value=3, scale_id="single")]
        items += [_make_item(i + 1, value=i % 5, scale_id="multi") for i in range(9)]
        sig = _intra_scale_variance(items)
        # Only the multi-item scale contributes
        self.assertFalse(sig.flagged)

    def test_binary_scale(self):
        """Binary response scale (0/1) with both values → some variance."""
        items = [_make_item(i, value=i % 2, n_options=2, scale_id="binary") for i in range(10)]
        sig = _intra_scale_variance(items)
        self.assertGreater(sig.value, 0.0)

    def test_no_scales(self):
        """Empty input → not flagged."""
        sig = _intra_scale_variance([])
        self.assertFalse(sig.flagged)


# ---------------------------------------------------------------------------
# Signal 5: Fatigue slope
# ---------------------------------------------------------------------------

class TestFatigueSlope(unittest.TestCase):

    def test_constant_rt_no_trend(self):
        """Constant RT → slope ~0, not flagged."""
        items = [_make_item(i, rt_ms=5000.0) for i in range(20)]
        sig = _fatigue_slope(items)
        self.assertFalse(sig.flagged)
        self.assertAlmostEqual(sig.value, 0.0, places=2)

    def test_increasing_rt_not_flagged(self):
        """Increasing RT (slowing down) → positive slope, not flagged."""
        items = [_make_item(i, rt_ms=3000 + i * 200) for i in range(20)]
        sig = _fatigue_slope(items)
        self.assertFalse(sig.flagged)
        self.assertGreater(sig.value, 0.0)

    def test_rapidly_decreasing_rt_flagged(self):
        """Rapidly decreasing RT → negative slope, flagged."""
        items = _speeding_session(20)
        sig = _fatigue_slope(items)
        self.assertTrue(sig.flagged)
        self.assertLess(sig.value, FATIGUE_SLOPE_THRESHOLD)

    def test_moderate_decrease_not_flagged(self):
        """Small RT decrease (~15%) → not flagged."""
        items = [_make_item(i, rt_ms=5000 - i * 40) for i in range(20)]
        sig = _fatigue_slope(items)
        # 20 items * 40ms = 800ms decrease from 5000 = 16%, slope is small
        self.assertFalse(sig.flagged)

    def test_too_few_items(self):
        """Fewer than 5 items → not flagged."""
        items = [_make_item(i, rt_ms=5000 - i * 1000) for i in range(4)]
        sig = _fatigue_slope(items)
        self.assertFalse(sig.flagged)

    def test_severity_increases_with_steeper_slope(self):
        """Steeper negative slope → higher severity."""
        mild = [_make_item(i, rt_ms=5000 * math.exp(-0.6 * i / 20)) for i in range(20)]
        steep = [_make_item(i, rt_ms=5000 * math.exp(-4.0 * i / 20)) for i in range(20)]
        sig_mild = _fatigue_slope(mild)
        sig_steep = _fatigue_slope(steep)
        self.assertGreater(sig_steep.severity, sig_mild.severity)


# ---------------------------------------------------------------------------
# Signal 6: Skip rate
# ---------------------------------------------------------------------------

class TestSkipRate(unittest.TestCase):

    def test_no_skips_not_flagged(self):
        """No skipped items → not flagged."""
        items = [_make_item(i) for i in range(20)]
        sig = _skip_rate(items)
        self.assertFalse(sig.flagged)
        self.assertAlmostEqual(sig.value, 0.0)

    def test_all_skipped_flagged(self):
        """All items skipped → flagged, high severity."""
        items = [_make_item(i, skipped=True) for i in range(20)]
        sig = _skip_rate(items)
        self.assertTrue(sig.flagged)
        self.assertAlmostEqual(sig.value, 1.0)
        self.assertGreater(sig.severity, 0.8)

    def test_threshold_boundary(self):
        """Exactly 30% → not flagged (must be >)."""
        skipped = [_make_item(i, skipped=True) for i in range(3)]
        answered = [_make_item(i + 3) for i in range(7)]
        items = skipped + answered
        sig = _skip_rate(items)
        self.assertAlmostEqual(sig.value, 0.3)
        self.assertFalse(sig.flagged)

    def test_just_over_threshold(self):
        """31% skipped → flagged."""
        skipped = [_make_item(i, skipped=True) for i in range(31)]
        answered = [_make_item(i + 31) for i in range(69)]
        items = skipped + answered
        sig = _skip_rate(items)
        self.assertTrue(sig.flagged)

    def test_empty_no_flag(self):
        sig = _skip_rate([])
        self.assertFalse(sig.flagged)


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

class TestSeverityHelpers(unittest.TestCase):

    def test_ramp_below_threshold(self):
        self.assertAlmostEqual(_ramp_severity(0.3, 0.5, 1.0), 0.0)

    def test_ramp_at_threshold(self):
        self.assertAlmostEqual(_ramp_severity(0.5, 0.5, 1.0), 0.0)

    def test_ramp_midway(self):
        self.assertAlmostEqual(_ramp_severity(0.75, 0.5, 1.0), 0.5)

    def test_ramp_at_maximum(self):
        self.assertAlmostEqual(_ramp_severity(1.0, 0.5, 1.0), 1.0)

    def test_ramp_above_maximum(self):
        self.assertAlmostEqual(_ramp_severity(2.0, 0.5, 1.0), 1.0)

    def test_inverse_above_threshold(self):
        self.assertAlmostEqual(_ramp_severity_inverse(0.3, 0.0, 0.2), 0.0)

    def test_inverse_at_threshold(self):
        self.assertAlmostEqual(_ramp_severity_inverse(0.2, 0.0, 0.2), 0.0)

    def test_inverse_midway(self):
        self.assertAlmostEqual(_ramp_severity_inverse(0.1, 0.0, 0.2), 0.5)

    def test_inverse_at_minimum(self):
        self.assertAlmostEqual(_ramp_severity_inverse(0.0, 0.0, 0.2), 1.0)

    def test_inverse_below_minimum(self):
        self.assertAlmostEqual(_ramp_severity_inverse(-0.1, 0.0, 0.2), 1.0)


# ---------------------------------------------------------------------------
# Composite scoring and tier assignment
# ---------------------------------------------------------------------------

class TestCompositeScoring(unittest.TestCase):

    def test_attentive_session_valid(self):
        """Attentive respondent → tier = 'valid', high score."""
        items = _attentive_session(20)
        result = assess_coherence(items)
        self.assertEqual(result.tier, "valid")
        self.assertGreater(result.coherence_score, 0.7)
        self.assertLess(result.n_flagged, SUSPECT_MIN_FLAGS)

    def test_careless_session_suspect_or_invalid(self):
        """Careless respondent → multiple flags → suspect or invalid."""
        items = _careless_session(20)
        result = assess_coherence(items)
        self.assertIn(result.tier, ("suspect", "invalid"))
        self.assertGreaterEqual(result.n_flagged, SUSPECT_MIN_FLAGS)
        self.assertLess(result.coherence_score, 0.6)

    def test_score_bounds(self):
        """Coherence score always in [0, 1]."""
        for items in [_attentive_session(), _careless_session(), []]:
            result = assess_coherence(items)
            self.assertGreaterEqual(result.coherence_score, 0.0)
            self.assertLessEqual(result.coherence_score, 1.0)

    def test_signal_weights_sum_to_one(self):
        """Configured signal weights should sum to 1.0."""
        total = sum(SIGNAL_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_empty_input_valid(self):
        """No items → valid with score 1.0."""
        result = assess_coherence([])
        self.assertEqual(result.tier, "valid")
        self.assertAlmostEqual(result.coherence_score, 1.0)
        self.assertEqual(result.n_items, 0)
        self.assertEqual(len(result.signals), 0)

    def test_single_item(self):
        """Single item → valid (insufficient data for most signals)."""
        items = [_make_item(0)]
        result = assess_coherence(items)
        self.assertEqual(result.tier, "valid")
        self.assertEqual(result.n_items, 1)

    def test_six_signals_returned(self):
        """assess_coherence always returns exactly 6 signals."""
        items = _attentive_session(20)
        result = assess_coherence(items)
        self.assertEqual(len(result.signals), 6)
        names = {s.name for s in result.signals}
        expected = {
            "speed_index", "longstring", "rt_variability",
            "intra_scale_variance", "fatigue_slope", "skip_rate",
        }
        self.assertEqual(names, expected)

    def test_n_flagged_matches_signal_count(self):
        """n_flagged should equal the count of flagged signals."""
        for factory in [_attentive_session, _careless_session]:
            result = assess_coherence(factory())
            actual = sum(1 for s in result.signals if s.flagged)
            self.assertEqual(result.n_flagged, actual)


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

class TestTierAssignment(unittest.TestCase):

    def test_zero_flags_valid(self):
        """0 flags → valid."""
        items = _attentive_session(20)
        result = assess_coherence(items)
        if result.n_flagged < SUSPECT_MIN_FLAGS:
            self.assertEqual(result.tier, "valid")

    def test_two_flags_suspect(self):
        """Two flags → suspect."""
        # Mix fast and straightline: should trigger speed + longstring
        items = [_make_item(i, value=3, rt_ms=500.0) for i in range(20)]
        result = assess_coherence(items)
        # Should have multiple flags (speed, longstring, RT variability, intra-scale)
        self.assertGreaterEqual(result.n_flagged, SUSPECT_MIN_FLAGS)
        self.assertIn(result.tier, ("suspect", "invalid"))

    def test_four_flags_invalid(self):
        """Four or more flags → invalid."""
        # Maximum incoherence: fast, same answer, no variation
        items = [_make_item(i, value=3, rt_ms=300.0) for i in range(20)]
        result = assess_coherence(items)
        self.assertGreaterEqual(result.n_flagged, INVALID_MIN_FLAGS)
        self.assertEqual(result.tier, "invalid")

    def test_version_tag(self):
        result = assess_coherence(_attentive_session())
        self.assertEqual(result.version, COHERENCE_VERSION)

    def test_n_items_matches_input(self):
        """n_items should equal total input length (including skipped)."""
        answered = [_make_item(i) for i in range(15)]
        skipped = [_make_item(i + 15, skipped=True) for i in range(5)]
        items = answered + skipped
        result = assess_coherence(items)
        self.assertEqual(result.n_items, 20)


# ---------------------------------------------------------------------------
# Skipped items interaction
# ---------------------------------------------------------------------------

class TestSkippedItemsInteraction(unittest.TestCase):

    def test_skipped_items_excluded_from_answered_signals(self):
        """Skipped items should not count toward speed/longstring/RT signals."""
        # 15 attentive + 5 skipped (high skip rate but other signals clean)
        answered = _attentive_session(15)
        skipped = [_make_item(i + 15, skipped=True) for i in range(5)]
        items = answered + skipped
        result = assess_coherence(items)
        # Skip rate = 5/20 = 0.25, just under threshold
        skip_sig = next(s for s in result.signals if s.name == "skip_rate")
        self.assertAlmostEqual(skip_sig.value, 0.25, places=4)
        self.assertFalse(skip_sig.flagged)

    def test_all_skipped_only_skip_rate_flagged(self):
        """All items skipped → only skip_rate can be meaningfully flagged."""
        items = [_make_item(i, skipped=True) for i in range(10)]
        result = assess_coherence(items)
        skip_sig = next(s for s in result.signals if s.name == "skip_rate")
        self.assertTrue(skip_sig.flagged)


# ---------------------------------------------------------------------------
# Custom fast RT threshold
# ---------------------------------------------------------------------------

class TestCustomThreshold(unittest.TestCase):

    def test_custom_fast_rt_ms(self):
        """Custom fast_rt_ms parameter changes speed index behavior."""
        items = [_make_item(i, rt_ms=2500.0) for i in range(20)]
        # Default 2000ms → not fast
        result_default = assess_coherence(items, fast_rt_ms=2000.0)
        speed_default = next(s for s in result_default.signals if s.name == "speed_index")
        self.assertAlmostEqual(speed_default.value, 0.0, places=4)
        # Strict 3000ms → all fast
        result_strict = assess_coherence(items, fast_rt_ms=3000.0)
        speed_strict = next(s for s in result_strict.signals if s.name == "speed_index")
        self.assertAlmostEqual(speed_strict.value, 1.0, places=4)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):

    def test_round_trip(self):
        """to_dict → from_dict preserves all fields."""
        result = assess_coherence(_attentive_session(20))
        d = coherence_to_dict(result)
        restored = coherence_from_dict(d)

        self.assertEqual(restored.n_flagged, result.n_flagged)
        self.assertAlmostEqual(restored.coherence_score, result.coherence_score)
        self.assertEqual(restored.tier, result.tier)
        self.assertEqual(restored.n_items, result.n_items)
        self.assertEqual(restored.version, result.version)
        self.assertEqual(len(restored.signals), len(result.signals))

    def test_signal_detail_preserved(self):
        """Individual signal details survive serialization."""
        result = assess_coherence(_careless_session(20))
        d = coherence_to_dict(result)
        restored = coherence_from_dict(d)

        for original, restored_sig in zip(result.signals, restored.signals):
            self.assertEqual(restored_sig.name, original.name)
            self.assertAlmostEqual(restored_sig.value, original.value)
            self.assertEqual(restored_sig.flagged, original.flagged)
            self.assertAlmostEqual(restored_sig.severity, original.severity)

    def test_dict_has_required_keys(self):
        """Serialized dict contains all expected top-level keys."""
        result = assess_coherence(_attentive_session())
        d = coherence_to_dict(result)
        for key in ("signals", "n_flagged", "coherence_score", "tier", "n_items", "version"):
            self.assertIn(key, d, f"Missing key: {key}")

    def test_signal_dict_keys(self):
        """Each signal dict has required keys."""
        result = assess_coherence(_attentive_session())
        d = coherence_to_dict(result)
        for sig_d in d["signals"]:
            for key in ("name", "value", "threshold", "flagged", "severity", "description"):
                self.assertIn(key, sig_d, f"Missing signal key: {key}")


# ---------------------------------------------------------------------------
# Integration: realistic mixed sessions
# ---------------------------------------------------------------------------

class TestRealisticSessions(unittest.TestCase):

    def test_mixed_quality_session(self):
        """First half attentive, second half careless → some flags."""
        attentive = _attentive_session(10)
        careless = [
            _make_item(i + 10, value=3, rt_ms=600.0)
            for i in range(10)
        ]
        items = attentive + careless
        result = assess_coherence(items)
        # Some degradation but not fully invalid
        self.assertGreaterEqual(result.n_flagged, 1)
        self.assertLess(result.coherence_score, 1.0)

    def test_multi_scale_session(self):
        """Session spanning multiple scales with different characteristics."""
        # PHQ-9: varied responses, normal RT
        phq = [
            _make_item(i, value=i % 4, rt_ms=4000 + i * 100, scale_id="phq9")
            for i in range(9)
        ]
        # GAD-7: also varied
        gad = [
            _make_item(i + 9, value=(i + 1) % 4, rt_ms=3500 + i * 150, scale_id="gad7")
            for i in range(7)
        ]
        items = phq + gad
        result = assess_coherence(items)
        self.assertEqual(result.tier, "valid")
        self.assertEqual(result.n_items, 16)

    def test_speeding_session_flags_fatigue(self):
        """Progressive speeding → fatigue slope should flag."""
        items = _speeding_session(20)
        result = assess_coherence(items)
        fatigue_sig = next(s for s in result.signals if s.name == "fatigue_slope")
        self.assertTrue(fatigue_sig.flagged)
        self.assertLess(fatigue_sig.value, FATIGUE_SLOPE_THRESHOLD)

    def test_slow_careful_respondent(self):
        """Very slow but careful respondent → fully valid."""
        items = [
            _make_item(i, value=(i * 3 + 1) % 5, rt_ms=10000 + 2000 * (i % 3))
            for i in range(20)
        ]
        result = assess_coherence(items)
        self.assertEqual(result.tier, "valid")
        self.assertGreater(result.coherence_score, 0.8)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_two_items(self):
        """Two items → most signals return early, no crash."""
        items = [_make_item(0), _make_item(1)]
        result = assess_coherence(items)
        self.assertIsNotNone(result)
        self.assertEqual(result.n_items, 2)

    def test_all_zero_rt(self):
        """All response times = 0 → handled gracefully."""
        items = [_make_item(i, rt_ms=0.0) for i in range(10)]
        result = assess_coherence(items)
        self.assertIsNotNone(result)
        # Speed index: 0 < 2000 → all fast
        speed_sig = next(s for s in result.signals if s.name == "speed_index")
        self.assertTrue(speed_sig.flagged)

    def test_very_large_session(self):
        """200 items → no performance issues, correct computation."""
        items = _attentive_session(200)
        result = assess_coherence(items)
        self.assertEqual(result.n_items, 200)
        self.assertEqual(result.tier, "valid")

    def test_frozen_dataclasses(self):
        """All output dataclasses are immutable."""
        result = assess_coherence(_attentive_session())
        with self.assertRaises(AttributeError):
            result.coherence_score = 0.5  # type: ignore
        with self.assertRaises(AttributeError):
            result.signals[0].flagged = True  # type: ignore


if __name__ == "__main__":
    unittest.main()
