"""
Tests for Behavioural Metadata Uncertainty Modifiers (Patent Claim Family 5).

Validates:
  1. Fast confident responses reduce uncertainty
  2. Slow/edited responses increase uncertainty
  3. Skipped items get high uncertainty
  4. Declined items get maximum uncertainty
  5. Voice channel hesitation modifies uncertainty
  6. Circadian effects (late-night responses)
  7. Session profiles aggregate correctly
  8. Adjusted SE computation
  9. Engagement-based selection bonuses
  10. Edge cases and boundary conditions
"""

import unittest
import math
from typing import List

from questions_agent_platform.pipeline.response_metadata import (
    ResponseMetadata,
    UncertaintyModifier,
    SessionUncertaintyProfile,
    compute_uncertainty_modifier,
    compute_session_uncertainty_profile,
    adjust_se_with_metadata,
    engagement_selection_bonus,
    expected_latency_for_type,
)


class TestExpectedLatency(unittest.TestCase):
    """Test reference latency model."""

    def test_known_types(self):
        """Known response types should have expected latencies."""
        self.assertGreater(expected_latency_for_type("likert_0_4"), 0)
        self.assertGreater(expected_latency_for_type("bool_0_1"), 0)
        self.assertGreater(expected_latency_for_type("audit_binge_0_4"), 0)

    def test_unknown_type_uses_default(self):
        """Unknown response types should use default latency."""
        lat = expected_latency_for_type("unknown_type_xyz")
        self.assertEqual(lat, 4000.0)

    def test_sensitive_items_slower(self):
        """Sensitive items (AUDIT) should have higher expected latency than simple items."""
        audit = expected_latency_for_type("audit_binge_0_4")
        binary = expected_latency_for_type("bool_0_1")
        self.assertGreater(audit, binary)


class TestUncertaintyModifierComputation(unittest.TestCase):
    """Test core uncertainty modifier computation."""

    def test_fast_confident_reduces_uncertainty(self):
        """Fast response + no edits should produce multiplier < 1.0."""
        meta = ResponseMetadata(
            item_id="item_1",
            response_latency_ms=2000.0,  # 2s for a 4s expected item = 0.5 ratio
            edit_count=0,
        )
        mod = compute_uncertainty_modifier(meta, response_type="likert_0_4")
        self.assertLess(mod.multiplier, 1.0)
        # 0.85 * 0.95 = 0.8075 → "medium" (high requires ≤ 0.8)
        self.assertIn(mod.confidence_label, ("high", "medium"))
        self.assertIn("fast_response", mod.contributing_factors)
        self.assertIn("no_edits", mod.contributing_factors)

    def test_slow_edited_increases_uncertainty(self):
        """Slow response + multiple edits should produce multiplier > 1.0."""
        meta = ResponseMetadata(
            item_id="item_2",
            response_latency_ms=15000.0,  # 15s for a 4s expected item
            edit_count=3,
        )
        mod = compute_uncertainty_modifier(meta, response_type="likert_0_4")
        self.assertGreater(mod.multiplier, 1.0)
        self.assertIn("very_slow_response", mod.contributing_factors)
        self.assertIn("multiple_edits", mod.contributing_factors)

    def test_normal_response_near_unity(self):
        """Normal latency + no edits should produce multiplier near 1.0."""
        meta = ResponseMetadata(
            item_id="item_3",
            response_latency_ms=4000.0,  # exactly expected
            edit_count=0,
        )
        mod = compute_uncertainty_modifier(meta, response_type="likert_0_4")
        self.assertAlmostEqual(mod.multiplier, 0.95, delta=0.1)  # slight boost for no edits

    def test_suspiciously_fast_increases_uncertainty(self):
        """Very fast response (< 0.3 * expected) may indicate inattention."""
        meta = ResponseMetadata(
            item_id="item_4",
            response_latency_ms=500.0,  # 0.5s for a 4s expected item
            edit_count=0,
        )
        mod = compute_uncertainty_modifier(meta, response_type="likert_0_4")
        self.assertGreater(mod.multiplier, 1.0)
        self.assertIn("suspiciously_fast", mod.contributing_factors)

    def test_declined_item_max_uncertainty(self):
        """Declined items should get maximum uncertainty multiplier."""
        meta = ResponseMetadata(item_id="item_5", was_declined=True)
        mod = compute_uncertainty_modifier(meta)
        self.assertEqual(mod.multiplier, 3.0)
        self.assertEqual(mod.confidence_label, "declined")
        self.assertIn("declined", mod.contributing_factors)

    def test_skipped_then_returned(self):
        """Skip-then-return should increase uncertainty."""
        meta_normal = ResponseMetadata(
            item_id="item_6",
            response_latency_ms=4000.0,
            edit_count=0,
        )
        meta_skipped = ResponseMetadata(
            item_id="item_6",
            response_latency_ms=4000.0,
            edit_count=0,
            was_skipped=True,
        )
        mod_normal = compute_uncertainty_modifier(meta_normal)
        mod_skipped = compute_uncertainty_modifier(meta_skipped)
        self.assertGreater(mod_skipped.multiplier, mod_normal.multiplier)
        self.assertIn("skip_then_return", mod_skipped.contributing_factors)

    def test_single_edit_mild_increase(self):
        """One edit is a minor reconsideration, not a red flag."""
        meta = ResponseMetadata(
            item_id="item_7",
            response_latency_ms=4000.0,
            edit_count=1,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertGreater(mod.multiplier, 0.95)  # more than no-edit
        self.assertIn("single_edit", mod.contributing_factors)

    def test_excessive_edits(self):
        """4+ edits indicates high ambivalence."""
        meta = ResponseMetadata(
            item_id="item_8",
            response_latency_ms=4000.0,
            edit_count=5,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertGreater(mod.multiplier, 1.3)
        self.assertIn("excessive_edits", mod.contributing_factors)

    def test_multiplier_clamped_low(self):
        """Multiplier should never go below 0.5."""
        meta = ResponseMetadata(
            item_id="item_9",
            response_latency_ms=2000.0,
            edit_count=0,
            channel="voice",
            voice_hesitation_ms=100.0,
        )
        mod = compute_uncertainty_modifier(meta, response_type="likert_0_4")
        self.assertGreaterEqual(mod.multiplier, 0.5)

    def test_multiplier_clamped_high(self):
        """Multiplier should never exceed 3.0."""
        meta = ResponseMetadata(
            item_id="item_10",
            response_latency_ms=60000.0,
            edit_count=10,
            was_skipped=True,
            time_of_day_hour=3,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertLessEqual(mod.multiplier, 3.0)


class TestSensitiveItems(unittest.TestCase):
    """Test that sensitive items get appropriate handling."""

    def test_sensitive_item_slower_expected(self):
        """Sensitive items get 50% more expected time, so same latency = more confident."""
        # 9s response on a sensitive item (expected = 4s * 1.5 = 6s, ratio = 1.5) → normal
        # 9s response on non-sensitive item (expected = 4s, ratio = 2.25) → slow
        meta = ResponseMetadata(item_id="audit_1", response_latency_ms=9000.0, edit_count=0)
        mod_sensitive = compute_uncertainty_modifier(
            meta, response_type="likert_0_4", is_sensitive=True
        )
        mod_normal = compute_uncertainty_modifier(
            meta, response_type="likert_0_4", is_sensitive=False
        )
        # Sensitive should have lower multiplier (same latency is more expected)
        self.assertLess(mod_sensitive.multiplier, mod_normal.multiplier)


class TestVoiceChannel(unittest.TestCase):
    """Test voice-specific metadata handling."""

    def test_immediate_voice_response(self):
        """Immediate voice response = high confidence."""
        meta = ResponseMetadata(
            item_id="voice_1",
            response_latency_ms=3000.0,
            edit_count=0,
            channel="voice",
            voice_hesitation_ms=200.0,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertIn("voice_immediate", mod.contributing_factors)
        self.assertLess(mod.raw_components.get("voice_modifier", 1.0), 1.0)

    def test_voice_hesitation(self):
        """Long voice pause = lower confidence."""
        meta = ResponseMetadata(
            item_id="voice_2",
            response_latency_ms=6000.0,
            edit_count=0,
            channel="voice",
            voice_hesitation_ms=3000.0,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertIn("voice_hesitation", mod.contributing_factors)
        self.assertGreater(mod.raw_components.get("voice_modifier", 1.0), 1.0)

    def test_non_voice_no_voice_modifier(self):
        """Non-voice channels should not have voice modifier."""
        meta = ResponseMetadata(
            item_id="tap_1",
            response_latency_ms=4000.0,
            channel="tap",
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertNotIn("voice_modifier", mod.raw_components)


class TestCircadianEffects(unittest.TestCase):
    """Test time-of-day uncertainty effects."""

    def test_late_night_increases_uncertainty(self):
        """3 AM responses should increase uncertainty."""
        meta = ResponseMetadata(
            item_id="night_1",
            response_latency_ms=4000.0,
            edit_count=0,
            time_of_day_hour=3,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertIn("late_night_response", mod.contributing_factors)
        self.assertGreater(mod.raw_components.get("circadian_modifier", 1.0), 1.0)

    def test_daytime_neutral(self):
        """10 AM responses should have neutral circadian effect."""
        meta = ResponseMetadata(
            item_id="day_1",
            response_latency_ms=4000.0,
            edit_count=0,
            time_of_day_hour=10,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertNotIn("late_night_response", mod.contributing_factors)
        self.assertNotIn("night_response", mod.contributing_factors)

    def test_midnight_boundary(self):
        """23:00 should get mild night penalty."""
        meta = ResponseMetadata(
            item_id="midnight_1",
            response_latency_ms=4000.0,
            edit_count=0,
            time_of_day_hour=23,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertIn("night_response", mod.contributing_factors)


class TestSessionProfile(unittest.TestCase):
    """Test session-level aggregation."""

    def _make_session(self, multipliers: List[float]) -> SessionUncertaintyProfile:
        """Helper to create a session profile from a list of multipliers."""
        mods = [
            UncertaintyModifier(
                item_id=f"item_{i}",
                multiplier=m,
                confidence_label="medium",
                contributing_factors=(),
                raw_components={},
            )
            for i, m in enumerate(multipliers)
        ]
        metadata = [
            ResponseMetadata(
                item_id=f"item_{i}",
                response_latency_ms=4000.0,
                edit_count=0,
            )
            for i in range(len(multipliers))
        ]
        return compute_session_uncertainty_profile(
            mods, metadata, session_date="2026-01-15", user_id="u1"
        )

    def test_all_confident_session(self):
        """All fast/confident responses → focused session."""
        profile = self._make_session([0.8, 0.8, 0.85, 0.75, 0.8])
        self.assertLess(profile.session_multiplier, 1.0)
        self.assertEqual(profile.engagement_quality, "focused")

    def test_mixed_session(self):
        """Mix of confident and uncertain → normal session."""
        profile = self._make_session([0.9, 1.0, 1.1, 1.0, 0.95])
        self.assertAlmostEqual(profile.session_multiplier, 1.0, delta=0.15)
        self.assertEqual(profile.engagement_quality, "normal")

    def test_geometric_mean_computation(self):
        """Session multiplier should be geometric mean of item multipliers."""
        profile = self._make_session([0.8, 1.25])
        expected_geo = math.exp((math.log(0.8) + math.log(1.25)) / 2)
        self.assertAlmostEqual(profile.session_multiplier, expected_geo, delta=0.01)

    def test_empty_session(self):
        """Empty session should return neutral defaults."""
        profile = compute_session_uncertainty_profile(
            [], [], session_date="2026-01-15", user_id="u1"
        )
        self.assertEqual(profile.session_multiplier, 1.0)
        self.assertEqual(profile.engagement_quality, "normal")
        self.assertEqual(profile.total_edits, 0)

    def test_distracted_session(self):
        """High session multiplier → distracted."""
        mods = [
            UncertaintyModifier(
                item_id=f"item_{i}",
                multiplier=1.5,
                confidence_label="low",
                contributing_factors=("slow_response",),
                raw_components={},
            )
            for i in range(5)
        ]
        metadata = [
            ResponseMetadata(
                item_id=f"item_{i}",
                response_latency_ms=15000.0,
                edit_count=2,
                was_skipped=(i >= 3),
            )
            for i in range(5)
        ]
        profile = compute_session_uncertainty_profile(
            mods, metadata, session_date="2026-01-15", user_id="u1"
        )
        self.assertEqual(profile.engagement_quality, "distracted")
        self.assertEqual(profile.skip_count, 2)


class TestAdjustedSE(unittest.TestCase):
    """Test SE adjustment with metadata modifiers."""

    def test_confident_reduces_se(self):
        """Confident responses should reduce adjusted SE."""
        mods = [
            UncertaintyModifier(
                item_id="i1", multiplier=0.8, confidence_label="high",
                contributing_factors=(), raw_components={},
            ),
        ]
        se_raw = 0.5
        se_adj = adjust_se_with_metadata(se_raw, mods)
        self.assertLess(se_adj, se_raw)

    def test_uncertain_increases_se(self):
        """Uncertain responses should increase adjusted SE."""
        mods = [
            UncertaintyModifier(
                item_id="i1", multiplier=1.5, confidence_label="low",
                contributing_factors=(), raw_components={},
            ),
        ]
        se_raw = 0.5
        se_adj = adjust_se_with_metadata(se_raw, mods)
        self.assertGreater(se_adj, se_raw)

    def test_neutral_unchanged(self):
        """Neutral modifiers (1.0) should leave SE unchanged."""
        mods = [
            UncertaintyModifier(
                item_id="i1", multiplier=1.0, confidence_label="medium",
                contributing_factors=(), raw_components={},
            ),
        ]
        se_raw = 0.5
        se_adj = adjust_se_with_metadata(se_raw, mods)
        self.assertAlmostEqual(se_adj, se_raw, delta=0.01)

    def test_empty_modifiers_unchanged(self):
        """No modifiers should return original SE."""
        se_raw = 0.5
        se_adj = adjust_se_with_metadata(se_raw, [])
        self.assertEqual(se_adj, se_raw)

    def test_multiple_items_geometric_mean(self):
        """Multiple item modifiers should use geometric mean."""
        mods = [
            UncertaintyModifier(
                item_id="i1", multiplier=0.8, confidence_label="high",
                contributing_factors=(), raw_components={},
            ),
            UncertaintyModifier(
                item_id="i2", multiplier=1.25, confidence_label="low",
                contributing_factors=(), raw_components={},
            ),
        ]
        se_raw = 0.5
        se_adj = adjust_se_with_metadata(se_raw, mods)
        expected = se_raw * math.exp((math.log(0.8) + math.log(1.25)) / 2)
        self.assertAlmostEqual(se_adj, expected, delta=0.01)


class TestEngagementSelectionBonus(unittest.TestCase):
    """Test engagement-based item selection adjustments."""

    def test_focused_no_change(self):
        """Focused users should get no selection adjustment."""
        profile = SessionUncertaintyProfile(
            session_date="2026-01-15", user_id="u1",
            item_modifiers=(), session_multiplier=0.8,
            engagement_quality="focused",
            median_latency_ms=3000.0, total_edits=0,
            skip_count=0, decline_count=0,
        )
        bonus = engagement_selection_bonus(profile, "item_1")
        self.assertEqual(bonus, 0.0)

    def test_distracted_prefers_simple(self):
        """Distracted users should get bonus for simple items."""
        profile = SessionUncertaintyProfile(
            session_date="2026-01-15", user_id="u1",
            item_modifiers=(), session_multiplier=1.5,
            engagement_quality="distracted",
            median_latency_ms=12000.0, total_edits=5,
            skip_count=2, decline_count=0,
        )
        # Simple binary item → bonus
        bonus_simple = engagement_selection_bonus(
            profile, "item_1", response_type="bool_0_1"
        )
        self.assertGreater(bonus_simple, 0.0)

        # Complex item → penalty
        bonus_complex = engagement_selection_bonus(
            profile, "item_2", response_type="minutes_0_180"
        )
        self.assertLess(bonus_complex, 0.0)


class TestConfidenceLabels(unittest.TestCase):
    """Test that confidence labels map correctly to multiplier ranges."""

    def test_high_confidence_range(self):
        """Multiplier <= 0.8 should be 'high' confidence."""
        meta = ResponseMetadata(
            item_id="t1", response_latency_ms=2500.0, edit_count=0,
            channel="voice", voice_hesitation_ms=200.0,
        )
        mod = compute_uncertainty_modifier(meta, response_type="likert_0_4")
        if mod.multiplier <= 0.8:
            self.assertEqual(mod.confidence_label, "high")

    def test_all_labels_valid(self):
        """All produced labels should be in the valid set."""
        valid = {"high", "medium", "low", "very_low", "declined"}
        test_cases = [
            ResponseMetadata(item_id="a", response_latency_ms=2000.0, edit_count=0),
            ResponseMetadata(item_id="b", response_latency_ms=4000.0, edit_count=0),
            ResponseMetadata(item_id="c", response_latency_ms=15000.0, edit_count=4),
            ResponseMetadata(item_id="d", was_declined=True),
            ResponseMetadata(item_id="e", response_latency_ms=60000.0, edit_count=5, was_skipped=True),
        ]
        for meta in test_cases:
            mod = compute_uncertainty_modifier(meta)
            self.assertIn(mod.confidence_label, valid, f"Invalid label for {meta.item_id}")


class TestEdgeCases(unittest.TestCase):
    """Test boundary conditions and edge cases."""

    def test_zero_latency(self):
        """Zero latency should be handled without error."""
        meta = ResponseMetadata(item_id="e1", response_latency_ms=0.0, edit_count=0)
        mod = compute_uncertainty_modifier(meta)
        self.assertGreaterEqual(mod.multiplier, 0.5)
        self.assertLessEqual(mod.multiplier, 3.0)

    def test_negative_latency_handled(self):
        """Negative latency (clock error) should not crash."""
        meta = ResponseMetadata(item_id="e2", response_latency_ms=-100.0, edit_count=0)
        mod = compute_uncertainty_modifier(meta)
        self.assertGreaterEqual(mod.multiplier, 0.5)

    def test_no_metadata_at_all(self):
        """Minimal metadata (just item_id) should produce valid output."""
        meta = ResponseMetadata(item_id="e3")
        mod = compute_uncertainty_modifier(meta)
        self.assertAlmostEqual(mod.multiplier, 0.95, delta=0.1)

    def test_all_factors_combined(self):
        """Combining all negative factors should still clamp at 3.0."""
        meta = ResponseMetadata(
            item_id="e4",
            response_latency_ms=50000.0,
            edit_count=10,
            was_skipped=True,
            channel="voice",
            voice_hesitation_ms=10000.0,
            time_of_day_hour=3,
        )
        mod = compute_uncertainty_modifier(meta)
        self.assertEqual(mod.multiplier, 3.0)
        self.assertGreater(len(mod.contributing_factors), 3)

    def test_hour_wrapping(self):
        """Hour 24+ should wrap correctly."""
        meta = ResponseMetadata(item_id="e5", time_of_day_hour=27)
        mod = compute_uncertainty_modifier(meta)
        # 27 % 24 = 3, which is late_night
        self.assertIn("late_night_response", mod.contributing_factors)


if __name__ == "__main__":
    unittest.main()
