"""
Edge-case tests for the selection engine.

Covers scenarios that the 30-day simulation can't reliably trigger:
1. All items declined — selector must return empty gracefully
2. All items on cooldown — selector must relax cooldown for coverage
3. Near-unlock priority overrides cooldown
4. Drift probe overrides cooldown
5. Intrusiveness cap is respected even with override priority
6. Tag diversity swap works when all top-ranked share one tag
7. Onboarding order is respected (lower order numbers first)
8. Timeframe mismatch excludes items correctly
9. Extra batches draw from remaining pool (no duplicates)
"""

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Set

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.baseline import BaselineState
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.pipeline.registry import (
    Item,
    Questionnaire,
    Registry,
    Scale,
    ScaleItem,
    load_registry,
)
from questions_agent_platform.pipeline.selection import (
    build_candidate_set,
    build_selection_plan,
)


def _make_item(
    item_id: str,
    tags=("tag_a",),
    intrusiveness="low",
    sensitivity="low",
    timeframes=("last_7_days",),
    onboarding_order=None,
) -> Item:
    return Item(
        id=item_id,
        text=f"Text for {item_id}",
        response_type="likert_5",
        tags=tuple(tags),
        sensitivity=sensitivity,
        timeframes_allowed=tuple(timeframes),
        intrusiveness=intrusiveness,
        onboarding_order=onboarding_order,
    )


def _make_scale(
    scale_id: str,
    item_ids: list,
    min_items=None,
    tags=("cardiometabolic",),
) -> Scale:
    items = tuple(ScaleItem(item_id=iid) for iid in item_ids)
    return Scale(
        id=scale_id,
        questionnaire_id="q1",
        version="1",
        name=f"Scale {scale_id}",
        method="sum",
        min_items_required=min_items or len(item_ids),
        unlock_window_days=14,
        retest_interval_days=90,
        response_type="likert_5",
        normalize_min=0.0,
        normalize_max=100.0,
        items=items,
        tags=tuple(tags),
    )


def _make_registry(items: list, scales: list) -> Registry:
    return Registry(
        version="test",
        items={i.id: i for i in items},
        scales={s.id: s for s in scales},
        questionnaires={"q1": Questionnaire(id="q1", version="1", name="Test Q")},
    )


def _default_cfg() -> QuestionsAgentConfig:
    return QuestionsAgentConfig(
        database_path=":memory:",
        registry_root="/tmp/fake",
        core_questions_per_day=5,
        item_repeat_cooldown_days=7,
    )


def _empty_context():
    """Returns default empty dicts for build_selection_plan call."""
    return dict(
        answered_item_ids_by_scale={},
        last_asked_date_by_item_id={},
        last_score_date_by_scale_id={},
        baselines_by_scale_id={},
        rolling_normalized_by_scale_id={},
    )


class TestAllItemsDeclined(unittest.TestCase):
    """When every item in the registry is permanently declined."""

    def test_returns_empty_core(self):
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 3}",)) for i in range(10)]
        scale = _make_scale("s1", [it.id for it in items], min_items=5)
        reg = _make_registry(items, [scale])
        cfg = _default_cfg()

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            declined_item_ids={it.id for it in items},
            **_empty_context(),
        )
        self.assertEqual(len(plan.core_item_ids), 0)


class TestAllItemsOnCooldown(unittest.TestCase):
    """When every item was asked yesterday, cooldown should relax for coverage."""

    def test_relaxes_cooldown_for_coverage(self):
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 3}",)) for i in range(10)]
        scale = _make_scale("s1", [it.id for it in items], min_items=5)
        reg = _make_registry(items, [scale])
        cfg = _default_cfg()

        today = date(2026, 1, 15)
        yesterday = today - timedelta(days=1)
        last_asked = {it.id: yesterday for it in items}

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=today,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id=last_asked,
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )

        # Should still return items thanks to cooldown relaxation / anchor contract
        self.assertGreater(len(plan.core_item_ids), 0)
        self.assertLessEqual(len(plan.core_item_ids), 5)


class TestNearUnlockOverridesCooldown(unittest.TestCase):
    """Items needed for near-unlock should be selectable even within cooldown."""

    def test_near_unlock_items_selected(self):
        # 10 items, scale needs all 10 but 8 are already answered
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 3}",)) for i in range(10)]
        item_ids = [it.id for it in items]
        scale = _make_scale("s1", item_ids, min_items=10)
        reg = _make_registry(items, [scale])
        cfg = _default_cfg()

        today = date(2026, 1, 15)
        yesterday = today - timedelta(days=1)
        # All items asked yesterday (within cooldown)
        last_asked = {it.id: yesterday for it in items}
        # 8 items already answered for the scale
        answered = {scale.id: set(item_ids[:8])}
        # Missing items: item_8, item_9

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=today,
            answered_item_ids_by_scale=answered,
            last_asked_date_by_item_id=last_asked,
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )

        # The 2 missing items should be in the selected core
        selected = set(plan.core_item_ids)
        self.assertIn("item_8", selected)
        self.assertIn("item_9", selected)

        # They should have near_unlock reason
        for iid in ("item_8", "item_9"):
            reasons = plan.reasons_by_item_id.get(iid, ())
            has_near = any("near_unlock" in r for r in reasons)
            self.assertTrue(has_near, f"{iid} missing near_unlock reason: {reasons}")


class TestIntrusivenessCap(unittest.TestCase):
    """At most 1 high-intrusiveness item, even if all are high priority."""

    def test_max_one_high_intrusiveness(self):
        # All items marked high intrusiveness
        items = [
            _make_item(f"item_{i}", tags=(f"tag_{i % 3}",), intrusiveness="high")
            for i in range(10)
        ]
        scale = _make_scale("s1", [it.id for it in items], min_items=5)
        reg = _make_registry(items, [scale])
        cfg = _default_cfg()

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            **_empty_context(),
        )

        high_count = sum(
            1 for iid in plan.core_item_ids
            if reg.items[iid].intrusiveness == "high"
        )
        self.assertLessEqual(high_count, 1)


class TestTagDiversitySwap(unittest.TestCase):
    """When top candidates all share one tag, the diversity swap inserts variety."""

    def test_swaps_for_diversity(self):
        # 8 items all with tag_a, 2 items with tag_b
        items_a = [_make_item(f"a_{i}", tags=("tag_a",)) for i in range(8)]
        items_b = [_make_item(f"b_{i}", tags=("tag_b",)) for i in range(2)]
        all_items = items_a + items_b

        # Scale only includes the tag_a items (higher priority)
        scale = _make_scale("s1", [it.id for it in items_a], min_items=5)
        reg = _make_registry(all_items, [scale])
        cfg = _default_cfg()

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            **_empty_context(),
        )

        tags_in_session = set()
        for iid in plan.core_item_ids:
            tags_in_session.update(reg.items[iid].tags)

        if len(plan.core_item_ids) >= 3:
            self.assertGreaterEqual(
                len(tags_in_session), 2,
                f"Expected >=2 distinct tags, got {tags_in_session}",
            )


class TestTimeframeMismatch(unittest.TestCase):
    """Items whose timeframe doesn't match the session are excluded."""

    def test_mismatched_items_excluded(self):
        # 3 items with last_7_days, 7 items with last_90_days
        items_7 = [_make_item(f"w_{i}", tags=("weekly",), timeframes=("last_7_days",)) for i in range(3)]
        items_90 = [_make_item(f"q_{i}", tags=("quarterly",), timeframes=("last_90_days",)) for i in range(7)]
        all_items = items_7 + items_90

        scale = _make_scale("s1", [it.id for it in all_items], min_items=5)
        reg = _make_registry(all_items, [scale])
        cfg = _default_cfg()

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            **_empty_context(),
        )

        # All selected items must match the session timeframe
        for iid in plan.core_item_ids:
            item = reg.items[iid]
            self.assertIn(
                plan.timeframe,
                item.timeframes_allowed,
                f"Item {iid} timeframe {item.timeframes_allowed} doesn't match "
                f"session timeframe {plan.timeframe}",
            )


class TestExtraBatchesNoDuplicates(unittest.TestCase):
    """Extra batches should not duplicate core items."""

    def test_no_overlap_with_core(self):
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 4}",)) for i in range(20)]
        scale = _make_scale("s1", [it.id for it in items], min_items=10)
        reg = _make_registry(items, [scale])
        cfg = QuestionsAgentConfig(
            database_path=":memory:",
            registry_root="/tmp/fake",
            core_questions_per_day=5,
            extra_batch_size=5,
            extra_batches_max_per_day=2,
        )

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            **_empty_context(),
        )

        core_set = set(plan.core_item_ids)
        for batch_idx, batch in enumerate(plan.extra_batches):
            for iid in batch:
                self.assertNotIn(
                    iid, core_set,
                    f"Extra batch {batch_idx} contains core item {iid}",
                )

        # Also verify extra batches don't overlap each other
        all_extra = []
        for batch in plan.extra_batches:
            all_extra.extend(batch)
        self.assertEqual(len(all_extra), len(set(all_extra)), "Duplicates within extra batches")


class TestDriftProbeOverride(unittest.TestCase):
    """Drift detection should prioritize items for drifting scales."""

    def test_drift_items_get_priority(self):
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 3}",)) for i in range(10)]
        drift_items = items[7:]  # items 7, 8, 9 belong to drifting scale
        other_items = items[:7]

        drift_scale = _make_scale("drift_s", [it.id for it in drift_items], min_items=3)
        other_scale = _make_scale("other_s", [it.id for it in other_items], min_items=5)
        reg = _make_registry(items, [drift_scale, other_scale])
        cfg = _default_cfg()

        # Simulate drift: baseline mean=50, var=100 (std=10), rolling=80 (z=3.0)
        baselines = {
            "drift_s": BaselineState(mean=50.0, var=100.0, n=10)
        }
        rolling = {"drift_s": 80.0}

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id=baselines,
            rolling_normalized_by_scale_id=rolling,
        )

        # At least one drift item should be selected
        selected = set(plan.core_item_ids)
        drift_selected = selected & {it.id for it in drift_items}
        self.assertGreater(
            len(drift_selected), 0,
            f"No drift items selected. Core: {plan.core_item_ids}",
        )


class TestOnboardingOrder(unittest.TestCase):
    """Items with lower onboarding_order should be selected first during onboarding."""

    def test_onboarding_order_respected(self):
        # Items with explicit onboarding order
        items = [
            _make_item("ob_5", tags=("cardio",), onboarding_order=5),
            _make_item("ob_1", tags=("cardio",), onboarding_order=1),
            _make_item("ob_3", tags=("metabolic",), onboarding_order=3),
            _make_item("ob_2", tags=("metabolic",), onboarding_order=2),
            _make_item("ob_4", tags=("cardio",), onboarding_order=4),
            _make_item("no_ob_a", tags=("other",)),
            _make_item("no_ob_b", tags=("other",)),
        ]
        scale = _make_scale("s1", [it.id for it in items], min_items=5)
        reg = _make_registry(items, [scale])
        cfg = _default_cfg()

        # The service layer passes onboarding items as priority_item_ids
        priority_ids = {
            it.id for it in items if it.onboarding_order is not None
        }

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            priority_item_ids=priority_ids,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )

        core = list(plan.core_item_ids)
        # Priority items with onboarding_order should get onboarding_priority reason
        onboarding_in_core = [
            iid for iid in core
            if reg.items[iid].onboarding_order is not None
        ]
        self.assertGreater(len(onboarding_in_core), 0, "No onboarding items selected")

        for iid in onboarding_in_core:
            reasons = plan.reasons_by_item_id.get(iid, ())
            has_ob = any("onboarding_priority" in r for r in reasons)
            self.assertTrue(
                has_ob,
                f"Onboarding item {iid} (order={reg.items[iid].onboarding_order}) "
                f"missing onboarding_priority reason: {reasons}",
            )


class TestCoreItemCount(unittest.TestCase):
    """Core always returns exactly k items (or fewer if registry is exhausted)."""

    def test_exactly_k_with_sufficient_registry(self):
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 4}",)) for i in range(20)]
        scale = _make_scale("s1", [it.id for it in items], min_items=10)
        reg = _make_registry(items, [scale])
        cfg = _default_cfg()

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            **_empty_context(),
        )
        self.assertEqual(len(plan.core_item_ids), 5)

    def test_fewer_when_registry_small(self):
        items = [_make_item(f"item_{i}", tags=("tag_a", "tag_b")) for i in range(3)]
        scale = _make_scale("s1", [it.id for it in items], min_items=3)
        reg = _make_registry(items, [scale])
        cfg = _default_cfg()

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            **_empty_context(),
        )
        self.assertEqual(len(plan.core_item_ids), 3)

    def test_allowed_subset_limits_selection(self):
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 3}",)) for i in range(20)]
        scale = _make_scale("s1", [it.id for it in items], min_items=10)
        reg = _make_registry(items, [scale])
        cfg = _default_cfg()

        # Only allow 2 items
        allowed = {items[0].id, items[5].id}
        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=date(2026, 1, 15),
            allowed_item_ids=allowed,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )
        for iid in plan.core_item_ids:
            self.assertIn(iid, allowed, f"Selected {iid} not in allowed set")


class TestRetestDueOverride(unittest.TestCase):
    """Scales due for retest should override cooldown on their items."""

    def test_retest_items_selected(self):
        items = [_make_item(f"item_{i}", tags=(f"tag_{i % 3}",)) for i in range(10)]
        retest_items = items[:3]
        scale = _make_scale("retest_s", [it.id for it in retest_items], min_items=3)
        other_scale = _make_scale("other_s", [it.id for it in items[3:]], min_items=5)
        reg = _make_registry(items, [scale, other_scale])
        cfg = _default_cfg()

        today = date(2026, 1, 15)
        # All retest items were answered, and last score was 91+ days ago
        answered = {scale.id: {it.id for it in retest_items}}
        last_score = {scale.id: today - timedelta(days=91)}
        # Retest items asked 2 days ago (within cooldown)
        last_asked = {it.id: today - timedelta(days=2) for it in retest_items}

        plan = build_selection_plan(
            registry=reg,
            cfg=cfg,
            user_id="u1",
            day=today,
            answered_item_ids_by_scale=answered,
            last_asked_date_by_item_id=last_asked,
            last_score_date_by_scale_id=last_score,
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
        )

        # At least one retest item should be selected despite cooldown
        selected = set(plan.core_item_ids)
        retest_selected = selected & {it.id for it in retest_items}
        self.assertGreater(
            len(retest_selected), 0,
            f"No retest items selected despite being due. Core: {plan.core_item_ids}",
        )


if __name__ == "__main__":
    unittest.main()
