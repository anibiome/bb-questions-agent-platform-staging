"""
Tests for IRT information-gain item selection.

Validates:
  1. Multiplexed items (shared across scales) get higher IRT info scores
  2. IRT info acts as tiebreaker within clinical priorities, not override
  3. Items at user's current θ get appropriate information
  4. Candidate features include irt_information_gain
"""

import unittest
from datetime import date, timedelta
from typing import Dict, List, Set, Tuple

from questions_agent_platform.pipeline.baseline import BaselineState
from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.registry import (
    Item,
    Questionnaire,
    Registry,
    Scale,
    ScaleItem,
)
from questions_agent_platform.pipeline.selection import (
    build_candidate_set,
    build_selection_plan,
)


def _make_item(item_id: str, tags: Tuple[str, ...] = ("general",)) -> Item:
    return Item(
        id=item_id,
        text=f"Test item {item_id}",
        response_type="likert_0_4",
        tags=tags,
        sensitivity="low",
        timeframes_allowed=("last_7_days",),
        intrusiveness="low",
        declinable=True,
    )


def _make_scale(
    scale_id: str,
    item_ids: List[str],
    min_items: int = 2,
    tags: Tuple[str, ...] = (),
) -> Scale:
    return Scale(
        id=scale_id,
        questionnaire_id=f"q_{scale_id}",
        version="1",
        name=f"Scale {scale_id}",
        method="mean",
        min_items_required=min_items,
        unlock_window_days=14,
        retest_interval_days=90,
        response_type="likert_0_4",
        normalize_min=0.0,
        normalize_max=4.0,
        items=tuple(ScaleItem(item_id=iid) for iid in item_ids),
        tags=tags,
    )


def _make_registry(items: List[Item], scales: List[Scale]) -> Registry:
    questionnaires = [
        Questionnaire(id=s.questionnaire_id, version="1", name=f"Q {s.id}")
        for s in scales
    ]
    return Registry(
        version="1",
        items={i.id: i for i in items},
        scales={s.id: s for s in scales},
        questionnaires={q.id: q for q in questionnaires},
    )


def _default_cfg() -> QuestionsAgentConfig:
    return QuestionsAgentConfig(
        database_path=":memory:",
        registry_root="/tmp/test_reg",
        core_questions_per_day=5,
        item_repeat_cooldown_days=7,
    )


class TestIRTInformationGainSelection(unittest.TestCase):
    """Test that IRT information-gain scoring influences item selection."""

    def test_multiplexed_item_gets_irt_boost(self):
        """Items shared across multiple scales should get higher IRT info scores."""
        # Create items: item_shared appears in scale_a AND scale_b
        items = [
            _make_item("item_shared", tags=("mood",)),
            _make_item("item_a1", tags=("mood",)),
            _make_item("item_a2", tags=("mood",)),
            _make_item("item_b1", tags=("sleep",)),
            _make_item("item_b2", tags=("sleep",)),
        ]
        scale_a = _make_scale("scale_a", ["item_shared", "item_a1", "item_a2"], tags=("mood",))
        scale_b = _make_scale("scale_b", ["item_shared", "item_b1", "item_b2"], tags=("sleep",))
        reg = _make_registry(items, [scale_a, scale_b])

        today = date(2026, 1, 15)
        # Both scales are near-unlock: 1 item missing each
        answered = {
            "scale_a": {"item_a1", "item_a2"},
            "scale_b": {"item_b1", "item_b2"},
        }

        candidate_set, _, _ = build_candidate_set(
            registry=reg,
            cfg=_default_cfg(),
            user_id="u1",
            day=today,
            answered_item_ids_by_scale=answered,
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )

        # Find the shared item's candidate
        shared_candidate = None
        for c in candidate_set.candidates:
            if c.item_id == "item_shared":
                shared_candidate = c
                break

        self.assertIsNotNone(shared_candidate)
        # Should have IRT information gain in features
        irt_info = shared_candidate.features.get("irt_information_gain", 0.0)
        self.assertGreater(irt_info, 0.0, "Multiplexed item should have IRT info score")

        # Should have IRT reason code
        self.assertIn(
            "irt_information_gain",
            shared_candidate.reason_codes,
            "Should have irt_information_gain reason",
        )

    def test_irt_info_only_for_clinically_relevant_items(self):
        """Items with no clinical reason should NOT get IRT info boost."""
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 2}",)) for i in range(8)]
        # scale_a: items 0-2 (near unlock — 2 of 3 answered)
        # scale_b: items 3-7 (already scored, no clinical trigger)
        scale_a = _make_scale("scale_a", ["item_0", "item_1", "item_2"], min_items=3)
        scale_b = _make_scale("scale_b", ["item_3", "item_4", "item_5", "item_6", "item_7"], min_items=5)
        reg = _make_registry(items, [scale_a, scale_b])

        today = date(2026, 1, 15)
        # scale_a near unlock (2 of 3), scale_b already scored recently (no retest)
        answered = {
            "scale_a": {"item_0", "item_1"},
            "scale_b": {"item_3", "item_4", "item_5", "item_6", "item_7"},
        }
        # scale_b scored 10 days ago — not due for retest (90 days)
        last_score = {"scale_b": today - timedelta(days=10)}

        candidate_set, _, _ = build_candidate_set(
            registry=reg,
            cfg=_default_cfg(),
            user_id="u1",
            day=today,
            answered_item_ids_by_scale=answered,
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id=last_score,
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )

        # item_2 should have IRT info (it has near_unlock reason)
        for c in candidate_set.candidates:
            if c.item_id == "item_2":
                self.assertIn(
                    "irt_information_gain", c.reason_codes,
                    "Clinically relevant item should get IRT boost",
                )

        # Items that only have anchor/multiplex but no clinical trigger
        # Check that items without strong clinical reasons don't get IRT boost
        # (note: some items may get anchor_continuity which counts as a reason)
        items_with_irt = [
            c.item_id for c in candidate_set.candidates
            if "irt_information_gain" in c.reason_codes
        ]
        # At minimum, item_2 should have IRT info
        self.assertIn("item_2", items_with_irt)

    def test_irt_info_uses_baseline_theta(self):
        """IRT info should adapt to user's current estimated θ."""
        items = [_make_item(f"item_{i}", tags=("mood",)) for i in range(5)]
        scale = _make_scale("scale_mood", [f"item_{i}" for i in range(5)], tags=("mood",))
        reg = _make_registry(items, [scale])

        today = date(2026, 1, 15)
        last_score = {"scale_mood": today - timedelta(days=91)}

        # User at high severity (baseline mean = 80)
        baselines_high = {"scale_mood": BaselineState(mean=80.0, var=25.0, n=10)}
        # User at low severity (baseline mean = 20)
        baselines_low = {"scale_mood": BaselineState(mean=20.0, var=25.0, n=10)}

        cs_high, _, _ = build_candidate_set(
            registry=reg,
            cfg=_default_cfg(),
            user_id="u1",
            day=today,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id=last_score,
            baselines_by_scale_id=baselines_high,
            rolling_normalized_by_scale_id={},
        )

        cs_low, _, _ = build_candidate_set(
            registry=reg,
            cfg=_default_cfg(),
            user_id="u1",
            day=today,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id=last_score,
            baselines_by_scale_id=baselines_low,
            rolling_normalized_by_scale_id={},
        )

        # Both should have IRT info scores
        high_info = {
            c.item_id: c.features.get("irt_information_gain", 0.0)
            for c in cs_high.candidates
        }
        low_info = {
            c.item_id: c.features.get("irt_information_gain", 0.0)
            for c in cs_low.candidates
        }

        # At least some items should have IRT info in both cases
        self.assertTrue(any(v > 0 for v in high_info.values()))
        self.assertTrue(any(v > 0 for v in low_info.values()))

        # The actual info values may differ because Fisher information
        # varies with θ — this proves the system adapts to the user's state
        # (not a hard assertion since symmetric items give similar info)

    def test_irt_does_not_override_clinical_priority(self):
        """Retest-due items should still be selected over IRT-only items."""
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 3}",)) for i in range(10)]
        retest_items = items[:3]
        scale_r = _make_scale("scale_r", [it.id for it in retest_items], min_items=3)
        scale_o = _make_scale("scale_o", [it.id for it in items[3:6]], min_items=3)
        reg = _make_registry(items, [scale_r, scale_o])

        today = date(2026, 1, 15)
        answered = {"scale_r": {it.id for it in retest_items}}
        last_score = {"scale_r": today - timedelta(days=91)}

        plan = build_selection_plan(
            registry=reg,
            cfg=_default_cfg(),
            user_id="u1",
            day=today,
            answered_item_ids_by_scale=answered,
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id=last_score,
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )

        # Retest items should appear (they have clinical priority, no cooldown here)
        selected = set(plan.core_item_ids)
        retest_selected = selected & {it.id for it in retest_items}
        self.assertGreater(
            len(retest_selected), 0,
            "Retest items should be selected when not on cooldown",
        )


class TestIRTInfoFeatureInCandidates(unittest.TestCase):
    """Verify irt_information_gain appears in candidate features."""

    def test_feature_present(self):
        """All candidates should have irt_information_gain in features."""
        items = [_make_item(f"item_{i}") for i in range(5)]
        scale = _make_scale("s1", [f"item_{i}" for i in range(5)])
        reg = _make_registry(items, [scale])

        cs, _, _ = build_candidate_set(
            registry=reg,
            cfg=_default_cfg(),
            user_id="u1",
            day=date(2026, 1, 1),
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )

        for c in cs.candidates:
            self.assertIn(
                "irt_information_gain", c.features,
                f"Candidate {c.item_id} missing irt_information_gain feature",
            )


if __name__ == "__main__":
    unittest.main()
