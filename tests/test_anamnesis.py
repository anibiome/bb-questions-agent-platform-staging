"""
Tests for Anamnesis Loop: Drift-Triggered Clinical Branching (Claim Family 4).

Validates:
  1. Drift trigger evaluation fires on correct conditions
  2. Episode creation and lifecycle management
  3. Episode resolution conditions (expired, confirmed, denied)
  4. Full anamnesis check integration
  5. Domain routing and item mapping
"""

import unittest
from datetime import date

from questions_agent_platform.pipeline.anamnesis import (
    evaluate_drift_triggers,
    create_anamnesis_episode,
    evaluate_episode_resolution,
    run_anamnesis_check,
    build_domain_to_items_map,
    build_domain_to_scales_map,
)
from questions_agent_platform.pipeline.baseline import BaselineState
from questions_agent_platform.pipeline.drift_routing import default_drift_routes


class TestDriftTriggers(unittest.TestCase):
    """Test drift trigger evaluation."""

    def test_ews_threshold_fires(self):
        """EWS score above threshold should trigger."""
        triggers = evaluate_drift_triggers(ews_score=0.7, ews_threshold=0.6)
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0][0], "ews_threshold")
        self.assertAlmostEqual(triggers[0][1], 0.7)

    def test_ews_below_threshold_no_trigger(self):
        """EWS score below threshold should not trigger."""
        triggers = evaluate_drift_triggers(ews_score=0.4, ews_threshold=0.6)
        self.assertEqual(len(triggers), 0)

    def test_radius_deviation_fires(self):
        """Large radius deviation from baseline should trigger."""
        baseline = BaselineState(mean=0.5, var=0.04, n=10)  # std = 0.2
        # current = 1.1, z = (1.1 - 0.5) / 0.2 = 3.0 > 2.0 threshold
        triggers = evaluate_drift_triggers(
            radius_current=1.1,
            radius_baseline=baseline,
        )
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0][0], "radius_deviation")

    def test_radius_insufficient_baseline(self):
        """Radius check with insufficient baseline (n<5) should not trigger."""
        baseline = BaselineState(mean=0.5, var=0.04, n=3)
        triggers = evaluate_drift_triggers(
            radius_current=1.1,
            radius_baseline=baseline,
        )
        self.assertEqual(len(triggers), 0)

    def test_velocity_trigger(self):
        """High velocity should trigger."""
        triggers = evaluate_drift_triggers(velocity=0.25, velocity_threshold=0.15)
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0][0], "velocity")

    def test_cross_modal_disagreement(self):
        """Cross-modal disagreement should trigger."""
        triggers = evaluate_drift_triggers(cross_modal_disagreement=True)
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0][0], "cross_modal")

    def test_multiple_triggers_simultaneous(self):
        """Multiple conditions can fire simultaneously."""
        baseline = BaselineState(mean=0.5, var=0.04, n=10)
        triggers = evaluate_drift_triggers(
            ews_score=0.8,
            radius_current=1.1,
            radius_baseline=baseline,
            velocity=0.25,
            ews_threshold=0.6,
        )
        self.assertGreater(len(triggers), 1)

    def test_no_triggers_when_all_normal(self):
        """Normal values should produce no triggers."""
        baseline = BaselineState(mean=0.5, var=0.04, n=10)
        triggers = evaluate_drift_triggers(
            ews_score=0.3,
            radius_current=0.55,  # z = 0.25, below threshold
            radius_baseline=baseline,
            velocity=0.05,
        )
        self.assertEqual(len(triggers), 0)


class TestEpisodeLifecycle(unittest.TestCase):
    """Test episode creation and resolution."""

    def test_create_episode(self):
        """Episode creation should set correct initial state."""
        ep = create_anamnesis_episode(
            user_id="u1",
            trigger_date=date(2026, 1, 15),
            trigger_type="ews_threshold",
            trigger_value=0.7,
            drift_domain="metabolic",
            triggered_scale_ids=["scale_findrisc"],
            follow_up_item_ids=["item_1", "item_2"],
        )
        self.assertEqual(ep.status, "active")
        self.assertEqual(ep.drift_domain, "metabolic")
        self.assertIn("scale_findrisc", ep.triggered_scale_ids)
        self.assertEqual(len(ep.follow_up_item_ids), 2)

    def test_episode_expires(self):
        """Episode should expire after max_days."""
        ep = create_anamnesis_episode(
            user_id="u1",
            trigger_date=date(2026, 1, 15),
            trigger_type="ews_threshold",
            trigger_value=0.7,
            drift_domain="metabolic",
            triggered_scale_ids=[],
            follow_up_item_ids=["item_1"],
            max_days=7,
        )
        resolved = evaluate_episode_resolution(
            ep, current_date=date(2026, 1, 23)  # 8 days later
        )
        self.assertEqual(resolved.status, "expired")
        self.assertEqual(resolved.resolution_reason, "max_days_exceeded")

    def test_episode_confirmed_on_scale_completion(self):
        """Episode should confirm when triggered scale is completed."""
        ep = create_anamnesis_episode(
            user_id="u1",
            trigger_date=date(2026, 1, 15),
            trigger_type="ews_threshold",
            trigger_value=0.7,
            drift_domain="metabolic",
            triggered_scale_ids=["scale_findrisc"],
            follow_up_item_ids=["item_1"],
        )
        resolved = evaluate_episode_resolution(
            ep, current_date=date(2026, 1, 17), scale_completed=True,
        )
        self.assertEqual(resolved.status, "confirmed")

    def test_episode_denied_when_ews_subsides(self):
        """Episode should be denied when EWS drops well below threshold."""
        ep = create_anamnesis_episode(
            user_id="u1",
            trigger_date=date(2026, 1, 15),
            trigger_type="ews_threshold",
            trigger_value=0.7,
            drift_domain="metabolic",
            triggered_scale_ids=[],
            follow_up_item_ids=["item_1"],
        )
        # 0.6 * 0.7 = 0.42, score of 0.3 < 0.42 → denied
        resolved = evaluate_episode_resolution(
            ep, current_date=date(2026, 1, 17), current_ews_score=0.3,
        )
        self.assertEqual(resolved.status, "denied")
        self.assertEqual(resolved.resolution_reason, "ews_subsided")

    def test_episode_stays_active(self):
        """Episode should stay active when no resolution condition met."""
        ep = create_anamnesis_episode(
            user_id="u1",
            trigger_date=date(2026, 1, 15),
            trigger_type="ews_threshold",
            trigger_value=0.7,
            drift_domain="metabolic",
            triggered_scale_ids=[],
            follow_up_item_ids=["item_1"],
        )
        resolved = evaluate_episode_resolution(
            ep, current_date=date(2026, 1, 17),
        )
        self.assertEqual(resolved.status, "active")


class TestAnamnesisCheck(unittest.TestCase):
    """Test full anamnesis check integration."""

    def test_new_episode_created(self):
        """New episode should be created when drift detected and no active episode."""
        actions = run_anamnesis_check(
            user_id="u1",
            current_date=date(2026, 1, 15),
            active_episodes=[],
            ews_score=0.75,
            drift_domain_to_scale_ids={"general": ["scale_findrisc"]},
            drift_domain_to_item_ids={"general": ["item_1", "item_2"]},
        )
        start_actions = [a for a in actions if a.action_type == "start_episode"]
        self.assertEqual(len(start_actions), 1)
        self.assertEqual(len(start_actions[0].follow_up_item_ids), 2)

    def test_no_duplicate_domain_episodes(self):
        """Should not create a new episode if domain already has active episode."""
        existing = create_anamnesis_episode(
            user_id="u1",
            trigger_date=date(2026, 1, 14),
            trigger_type="ews_threshold",
            trigger_value=0.65,
            drift_domain="general",
            triggered_scale_ids=["scale_findrisc"],
            follow_up_item_ids=["item_1"],
        )
        actions = run_anamnesis_check(
            user_id="u1",
            current_date=date(2026, 1, 15),
            active_episodes=[existing],
            ews_score=0.75,
            drift_domain_to_scale_ids={"general": ["scale_findrisc"]},
            drift_domain_to_item_ids={"general": ["item_1"]},
        )
        start_actions = [a for a in actions if a.action_type == "start_episode"]
        self.assertEqual(len(start_actions), 0)

    def test_continue_active_episode(self):
        """Active episodes should generate continue actions."""
        existing = create_anamnesis_episode(
            user_id="u1",
            trigger_date=date(2026, 1, 14),
            trigger_type="ews_threshold",
            trigger_value=0.65,
            drift_domain="metabolic",
            triggered_scale_ids=["scale_findrisc"],
            follow_up_item_ids=["item_1", "item_2"],
        )
        actions = run_anamnesis_check(
            user_id="u1",
            current_date=date(2026, 1, 15),
            active_episodes=[existing],
        )
        continue_actions = [a for a in actions if a.action_type == "continue_episode"]
        self.assertEqual(len(continue_actions), 1)
        self.assertGreater(continue_actions[0].priority_boost, 0)

    def test_max_concurrent_episodes(self):
        """Should respect max concurrent episodes limit."""
        ep1 = create_anamnesis_episode(
            user_id="u1", trigger_date=date(2026, 1, 14),
            trigger_type="ews_threshold", trigger_value=0.7,
            drift_domain="metabolic",
            triggered_scale_ids=[], follow_up_item_ids=["item_1"],
        )
        ep2 = create_anamnesis_episode(
            user_id="u1", trigger_date=date(2026, 1, 14),
            trigger_type="velocity", trigger_value=0.2,
            drift_domain="cardiovascular",
            triggered_scale_ids=[], follow_up_item_ids=["item_2"],
        )
        actions = run_anamnesis_check(
            user_id="u1",
            current_date=date(2026, 1, 15),
            active_episodes=[ep1, ep2],
            ews_score=0.8,
            max_concurrent_episodes=2,
            drift_domain_to_scale_ids={"general": ["s1"]},
            drift_domain_to_item_ids={"general": ["item_3"]},
        )
        start_actions = [a for a in actions if a.action_type == "start_episode"]
        self.assertEqual(len(start_actions), 0)  # at max capacity


class TestRoutingHelpers(unittest.TestCase):
    """Test routing table integration helpers."""

    def test_domain_to_scales_map(self):
        """Should map drift domains to scale IDs."""
        routes = default_drift_routes()
        mapping = build_domain_to_scales_map(routes)
        self.assertIn("metabolic", mapping)
        self.assertIn("scale_cm_findrisc", mapping["metabolic"])
        self.assertIn("cardiovascular", mapping)
        self.assertIn("scale_cm_ez_cvd", mapping["cardiovascular"])

    def test_domain_to_items_map(self):
        """Should expand scale IDs to item IDs."""
        routes = default_drift_routes()
        scale_items = {
            "scale_cm_findrisc": ["item_a", "item_b"],
            "scale_cm_ez_cvd": ["item_c", "item_d"],
        }
        mapping = build_domain_to_items_map(routes, scale_items)
        self.assertIn("metabolic", mapping)
        self.assertIn("item_a", mapping["metabolic"])


if __name__ == "__main__":
    unittest.main()
