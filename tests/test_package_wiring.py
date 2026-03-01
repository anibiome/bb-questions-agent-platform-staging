"""Tests for measurement package ↔ selection engine wiring.

Verifies that the progressive measurement plan actually constrains
which items the selection engine prioritises, completing the loop:

    Progressive Plan  →  active_package_scale_ids  →  selection engine
         ↑                                                    │
         └───── package completion ← scored scales ← answers ←┘

Test coverage:
- Package scale extraction and priority maps
- Package completion evaluation
- resolve_current_package walks the plan correctly
- Selection engine applies package_focus boost
- Package_focus overrides cooldown (same as near_unlock)
- Items outside active package still selectable (soft boost, not filter)
- Package lifecycle state machine (pending → in_progress → completed)
- End-to-end: session creation → answers → package completion
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.baseline import BaselineState
from questions_agent_platform.pipeline.measurement_packages import (
    ALL_PACKAGES,
    AXIS_PACKAGES,
    CORE_SCREENER,
    DEEP_PACKAGES,
    DecoherenceMode,
    MeasurementPackage,
    PackageTier,
    ProgressivePlan,
    evaluate_package_completion,
    get_package_scale_ids,
    get_package_scale_priorities,
    get_progressive_plan,
    resolve_current_package,
)
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


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def _make_item(
    item_id: str,
    tags: Tuple[str, ...] = ("tag_a",),
    intrusiveness: str = "low",
    sensitivity: str = "low",
    timeframes: Tuple[str, ...] = ("last_7_days",),
    response_type: str = "likert_5",
) -> Item:
    return Item(
        id=item_id,
        text=f"Text for {item_id}",
        response_type=response_type,
        tags=tags,
        sensitivity=sensitivity,
        timeframes_allowed=timeframes,
        intrusiveness=intrusiveness,
    )


def _make_scale(
    scale_id: str,
    item_ids: List[str],
    min_items: Optional[int] = None,
    tags: Tuple[str, ...] = ("cardiometabolic",),
    response_type: str = "likert_5",
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
        response_type=response_type,
        normalize_min=0.0,
        normalize_max=100.0,
        items=items,
        tags=tags,
    )


def _make_questionnaire(qid: str = "q1") -> Questionnaire:
    return Questionnaire(id=qid, name="Test Questionnaire", version="1")


def _build_registry(
    items: Dict[str, Item],
    scales: Dict[str, Scale],
) -> Registry:
    return Registry(
        version="test_v1",
        items=items,
        scales=scales,
        questionnaires={"q1": _make_questionnaire()},
    )


def _default_cfg(**overrides: Any) -> QuestionsAgentConfig:
    return QuestionsAgentConfig(
        database_path=":memory:",
        registry_root="/tmp/reg",
        core_questions_per_day=overrides.get("core_questions_per_day", 5),
        extra_batch_size=overrides.get("extra_batch_size", 3),
        extra_batches_max_per_day=overrides.get("extra_batches_max_per_day", 1),
        item_repeat_cooldown_days=overrides.get("item_repeat_cooldown_days", 7),
        package_focus_boost=overrides.get("package_focus_boost", 5.0),
    )


# ---------------------------------------------------------------------------
# Package extraction tests
# ---------------------------------------------------------------------------
class TestGetPackageScaleIds:
    def test_core_screener_has_expected_scales(self) -> None:
        scale_ids = get_package_scale_ids("pkg_core_screener")
        assert isinstance(scale_ids, frozenset)
        assert len(scale_ids) >= 4  # at least 4 scales in core
        assert "scale_gad_7" in scale_ids
        assert "scale_pss" in scale_ids

    def test_metabolic_axis_has_expected_scales(self) -> None:
        scale_ids = get_package_scale_ids("pkg_axis_metabolic")
        assert "scale_findrisc" in scale_ids
        assert "scale_ipaq_sf" in scale_ids

    def test_unknown_package_returns_empty(self) -> None:
        assert get_package_scale_ids("pkg_nonexistent") == frozenset()

    def test_every_package_has_at_least_one_scale(self) -> None:
        for pkg_id in ALL_PACKAGES:
            scale_ids = get_package_scale_ids(pkg_id)
            assert len(scale_ids) >= 1, f"{pkg_id} has no scales"


class TestGetPackageScalePriorities:
    def test_priorities_are_floats_between_0_and_1(self) -> None:
        for pkg_id in ALL_PACKAGES:
            priorities = get_package_scale_priorities(pkg_id)
            for scale_id, p in priorities.items():
                assert 0.0 < p <= 1.0, f"{pkg_id}/{scale_id} priority {p} out of range"

    def test_core_screener_gad7_has_highest_priority(self) -> None:
        priorities = get_package_scale_priorities("pkg_core_screener")
        assert priorities["scale_gad_7"] == 1.0

    def test_unknown_package_returns_empty(self) -> None:
        assert get_package_scale_priorities("pkg_nonexistent") == {}


# ---------------------------------------------------------------------------
# Package completion evaluation
# ---------------------------------------------------------------------------
class TestEvaluatePackageCompletion:
    def test_all_scales_scored_completes_package(self) -> None:
        pkg_scales = get_package_scale_ids("pkg_core_screener")
        scored = frozenset(pkg_scales)  # all scored
        assert evaluate_package_completion("pkg_core_screener", scored) is True

    def test_missing_one_scale_does_not_complete(self) -> None:
        pkg_scales = set(get_package_scale_ids("pkg_core_screener"))
        # Remove one scale
        pkg_scales.pop()
        scored = frozenset(pkg_scales)
        assert evaluate_package_completion("pkg_core_screener", scored) is False

    def test_empty_scored_does_not_complete(self) -> None:
        assert evaluate_package_completion("pkg_core_screener", frozenset()) is False

    def test_extra_scored_scales_are_fine(self) -> None:
        pkg_scales = get_package_scale_ids("pkg_axis_metabolic")
        scored = frozenset(pkg_scales | {"scale_extra_one", "scale_extra_two"})
        assert evaluate_package_completion("pkg_axis_metabolic", scored) is True

    def test_unknown_package_returns_false(self) -> None:
        assert evaluate_package_completion("pkg_nonexistent", frozenset({"a"})) is False


# ---------------------------------------------------------------------------
# resolve_current_package
# ---------------------------------------------------------------------------
class TestResolveCurrentPackage:
    def test_fresh_plan_returns_first_package(self) -> None:
        plan = get_progressive_plan(user_id="u1")
        pkg = resolve_current_package(plan)
        assert pkg == "pkg_core_screener"

    def test_after_completing_core_returns_first_axis(self) -> None:
        plan = get_progressive_plan(
            user_id="u1",
            completed_packages=["pkg_core_screener"],
        )
        pkg = resolve_current_package(plan)
        assert pkg is not None
        assert pkg.startswith("pkg_axis_")

    def test_all_completed_returns_none(self) -> None:
        # Complete every possible package
        all_ids = list(ALL_PACKAGES.keys())
        plan = get_progressive_plan(
            user_id="u1",
            completed_sessions=10,
            completed_packages=all_ids,
        )
        assert resolve_current_package(plan) is None

    def test_drift_mode_changes_axis_order(self) -> None:
        plan_no_drift = get_progressive_plan(
            user_id="u1",
            completed_packages=["pkg_core_screener"],
        )
        plan_metabolic = get_progressive_plan(
            user_id="u1",
            completed_packages=["pkg_core_screener"],
            drift_mode=DecoherenceMode.METABOLIC,
        )
        first_no_drift = resolve_current_package(plan_no_drift)
        first_metabolic = resolve_current_package(plan_metabolic)
        assert first_metabolic == "pkg_axis_metabolic"
        # Without drift, default order starts with affective
        assert first_no_drift == "pkg_axis_affective"


# ---------------------------------------------------------------------------
# Selection engine: package_focus boost
# ---------------------------------------------------------------------------
class TestSelectionPackageFocusBoost:
    """Verify that items from the active package's scales score higher."""

    @staticmethod
    def _build_test_registry() -> Tuple[Registry, Dict[str, str]]:
        """Build a small registry with items in two groups:
        - 'package_items' belong to scale_gad_7 (in core screener)
        - 'non_package_items' belong to scale_other (NOT in any package)
        """
        package_items = [_make_item(f"pkg_i{i}", tags=("mood",)) for i in range(5)]
        non_package_items = [_make_item(f"other_i{i}", tags=("other",)) for i in range(5)]
        all_items = {it.id: it for it in package_items + non_package_items}

        scale_pkg = _make_scale(
            "scale_gad_7",
            [it.id for it in package_items],
            tags=("mood",),
        )
        scale_other = _make_scale(
            "scale_other",
            [it.id for it in non_package_items],
            tags=("other",),
        )
        registry = _build_registry(all_items, {"scale_gad_7": scale_pkg, "scale_other": scale_other})
        return registry, {}

    def test_package_items_score_higher(self) -> None:
        registry, _ = self._build_test_registry()
        cfg = _default_cfg(core_questions_per_day=5, package_focus_boost=5.0)
        day = date(2026, 2, 27)

        # Active package includes scale_gad_7
        active_scales = {"scale_gad_7"}

        cs, _, _ = build_candidate_set(
            registry=registry,
            cfg=cfg,
            user_id="u1",
            day=day,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
            active_package_scale_ids=active_scales,
        )

        # All package items should have "package_focus:" reason
        pkg_candidates = [
            c for c in cs.candidates
            if any(r.startswith("package_focus:") for r in c.reason_codes)
        ]
        assert len(pkg_candidates) == 5, f"Expected 5 package-focused items, got {len(pkg_candidates)}"

        # Package items should have higher scores than non-package items
        for c in cs.candidates:
            if c.item_id.startswith("pkg_i"):
                assert c.deterministic_score >= 5.0, (
                    f"Package item {c.item_id} score {c.deterministic_score} < 5.0"
                )
            elif c.item_id.startswith("other_i"):
                assert c.deterministic_score < 5.0, (
                    f"Non-package item {c.item_id} has unexpectedly high score {c.deterministic_score}"
                )

    def test_package_focus_in_ordered_candidates(self) -> None:
        """Package items should appear before non-package items in ordering."""
        registry, _ = self._build_test_registry()
        cfg = _default_cfg(core_questions_per_day=5, package_focus_boost=5.0)
        day = date(2026, 2, 27)

        cs, _, _ = build_candidate_set(
            registry=registry,
            cfg=cfg,
            user_id="u1",
            day=day,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
            active_package_scale_ids={"scale_gad_7"},
        )

        # First 5 ordered candidates should all be package items
        ordered = list(cs.ordered_candidate_item_ids)
        top_5 = set(ordered[:5])
        pkg_ids = {f"pkg_i{i}" for i in range(5)}
        assert pkg_ids.issubset(top_5), (
            f"Expected all package items in top 5, got {top_5}"
        )

    def test_no_package_no_boost(self) -> None:
        """When no active package, no items get package_focus reason."""
        registry, _ = self._build_test_registry()
        cfg = _default_cfg()
        day = date(2026, 2, 27)

        cs, _, _ = build_candidate_set(
            registry=registry,
            cfg=cfg,
            user_id="u1",
            day=day,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
            active_package_scale_ids=None,
        )

        for c in cs.candidates:
            assert not any(r.startswith("package_focus:") for r in c.reason_codes), (
                f"Item {c.item_id} has package_focus without active package"
            )

    def test_priority_scaling(self) -> None:
        """Higher-priority scales within a package get proportionally more boost."""
        items_high = [_make_item(f"hi_{i}", tags=("mood",)) for i in range(3)]
        items_low = [_make_item(f"lo_{i}", tags=("other",)) for i in range(3)]
        all_items = {it.id: it for it in items_high + items_low}

        scale_high = _make_scale("scale_high", [it.id for it in items_high])
        scale_low = _make_scale("scale_low", [it.id for it in items_low])
        registry = _build_registry(all_items, {"scale_high": scale_high, "scale_low": scale_low})

        cfg = _default_cfg(package_focus_boost=10.0)
        day = date(2026, 2, 27)

        cs, _, _ = build_candidate_set(
            registry=registry,
            cfg=cfg,
            user_id="u1",
            day=day,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
            active_package_scale_ids={"scale_high", "scale_low"},
            active_package_scale_priorities={"scale_high": 1.0, "scale_low": 0.5},
        )

        # Items from scale_high should have boost ~10.0, scale_low ~5.0
        for c in cs.candidates:
            if c.item_id.startswith("hi_"):
                assert c.deterministic_score >= 9.0, f"High-priority {c.item_id} score too low: {c.deterministic_score}"
            elif c.item_id.startswith("lo_"):
                assert 4.0 <= c.deterministic_score <= 6.0, f"Low-priority {c.item_id} score unexpected: {c.deterministic_score}"

    def test_package_focus_overrides_cooldown(self) -> None:
        """Items with package_focus can survive the cooldown exclusion."""
        items = [_make_item(f"item_{i}", tags=("mood",)) for i in range(5)]
        all_items = {it.id: it for it in items}
        scale = _make_scale("scale_pkg", [it.id for it in items])
        registry = _build_registry(all_items, {"scale_pkg": scale})

        cfg = _default_cfg(item_repeat_cooldown_days=7, core_questions_per_day=3)
        day = date(2026, 2, 27)
        # All items were asked 2 days ago (within cooldown)
        last_asked = {it.id: day - timedelta(days=2) for it in items}

        cs, _, _ = build_candidate_set(
            registry=registry,
            cfg=cfg,
            user_id="u1",
            day=day,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id=last_asked,
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
            active_package_scale_ids={"scale_pkg"},
        )

        # Items should NOT be excluded (package_focus overrides cooldown)
        excluded_set = set(cs.excluded_item_ids)
        for it in items:
            assert it.id not in excluded_set, (
                f"Item {it.id} was excluded despite package_focus"
            )

    def test_is_package_focus_feature_set(self) -> None:
        """Candidate features include is_package_focus flag for auditability."""
        registry, _ = self._build_test_registry()
        cfg = _default_cfg()
        day = date(2026, 2, 27)

        cs, _, _ = build_candidate_set(
            registry=registry,
            cfg=cfg,
            user_id="u1",
            day=day,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
            active_package_scale_ids={"scale_gad_7"},
        )

        for c in cs.candidates:
            assert "is_package_focus" in c.features
            if c.item_id.startswith("pkg_i"):
                assert c.features["is_package_focus"] is True
            elif c.item_id.startswith("other_i"):
                assert c.features["is_package_focus"] is False


# ---------------------------------------------------------------------------
# Selection plan integration: package_focus flows through to plan
# ---------------------------------------------------------------------------
class TestSelectionPlanPackageIntegration:
    def test_selection_plan_prefers_package_items(self) -> None:
        """build_selection_plan selects package items in core when boosted."""
        package_items = [_make_item(f"pkg_{i}", tags=("mood",)) for i in range(6)]
        other_items = [_make_item(f"oth_{i}", tags=("other",)) for i in range(6)]
        all_items = {it.id: it for it in package_items + other_items}

        scale_pkg = _make_scale("scale_gad_7", [it.id for it in package_items], tags=("mood",))
        scale_oth = _make_scale("scale_other", [it.id for it in other_items], tags=("other",))
        registry = _build_registry(all_items, {"scale_gad_7": scale_pkg, "scale_other": scale_oth})

        cfg = _default_cfg(core_questions_per_day=5, package_focus_boost=5.0)
        day = date(2026, 2, 27)

        plan = build_selection_plan(
            registry=registry,
            cfg=cfg,
            user_id="u1",
            day=day,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
            active_package_scale_ids={"scale_gad_7"},
        )

        core_ids = set(plan.core_item_ids)
        pkg_in_core = core_ids & {f"pkg_{i}" for i in range(6)}
        oth_in_core = core_ids & {f"oth_{i}" for i in range(6)}
        # Majority of core should be package items
        assert len(pkg_in_core) >= 3, f"Too few package items in core: {len(pkg_in_core)}"
        assert len(pkg_in_core) > len(oth_in_core), "Package items should outnumber non-package"


# ---------------------------------------------------------------------------
# Config knob
# ---------------------------------------------------------------------------
class TestConfigPackageFocusBoost:
    def test_default_value(self) -> None:
        cfg = QuestionsAgentConfig(database_path=":memory:", registry_root="/tmp")
        assert cfg.package_focus_boost == 5.0

    def test_custom_value(self) -> None:
        cfg = QuestionsAgentConfig(database_path=":memory:", registry_root="/tmp", package_focus_boost=8.0)
        assert cfg.package_focus_boost == 8.0

    def test_zero_boost_disables_package_focus(self) -> None:
        """Setting boost to 0.0 effectively disables package focusing."""
        items = [_make_item(f"item_{i}", tags=("mood",)) for i in range(5)]
        all_items = {it.id: it for it in items}
        scale = _make_scale("scale_pkg", [it.id for it in items])
        registry = _build_registry(all_items, {"scale_pkg": scale})

        cfg = _default_cfg(package_focus_boost=0.0)
        day = date(2026, 2, 27)

        cs, _, _ = build_candidate_set(
            registry=registry,
            cfg=cfg,
            user_id="u1",
            day=day,
            answered_item_ids_by_scale={},
            last_asked_date_by_item_id={},
            last_score_date_by_scale_id={},
            baselines_by_scale_id={},
            rolling_normalized_by_scale_id={},
            active_package_scale_ids={"scale_pkg"},
        )

        # With boost=0.0, package_focus reason still appears but adds 0 score.
        # Items may still have non-zero score from other signals (anchor, IRT).
        # What we verify is that the package_focus contribution itself is 0.
        for c in cs.candidates:
            pkg_reasons = [r for r in c.reason_codes if r.startswith("package_focus:")]
            non_pkg_reasons = [r for r in c.reason_codes if not r.startswith("package_focus:")]
            if pkg_reasons and not non_pkg_reasons:
                # Items whose ONLY reason is package_focus should have ~0 score
                assert c.deterministic_score < 0.01, (
                    f"Item {c.item_id} with only package_focus reason has score {c.deterministic_score}"
                )


# ---------------------------------------------------------------------------
# DB integration: package lifecycle
# ---------------------------------------------------------------------------
class TestPackageLifecycleDB:
    @pytest.fixture()
    def conn(self, tmp_path: Any) -> sqlite3.Connection:
        from questions_agent_platform.pipeline.db import init_db
        db_path = str(tmp_path / "test_pkg.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("INSERT INTO users(user_id, created_at) VALUES ('u1', '2026-01-01');")
        conn.commit()
        return conn

    def test_upsert_creates_package_row(self, conn: sqlite3.Connection) -> None:
        from questions_agent_platform.pipeline.service import _upsert_package_status
        _upsert_package_status(conn, user_id="u1", package_id="pkg_core_screener", status="in_progress")

        row = conn.execute(
            "SELECT * FROM user_measurement_packages WHERE user_id='u1';",
        ).fetchone()
        assert row is not None
        assert row["measurement_package_id"] == "pkg_core_screener"
        assert row["status"] == "in_progress"

    def test_upsert_upgrades_status(self, conn: sqlite3.Connection) -> None:
        from questions_agent_platform.pipeline.service import _upsert_package_status
        _upsert_package_status(conn, user_id="u1", package_id="pkg_core_screener", status="in_progress")
        _upsert_package_status(conn, user_id="u1", package_id="pkg_core_screener", status="completed")

        row = conn.execute(
            "SELECT * FROM user_measurement_packages WHERE user_id='u1' AND measurement_package_id='pkg_core_screener';",
        ).fetchone()
        assert row["status"] == "completed"
        assert row["completed_at"] is not None

    def test_upsert_does_not_downgrade(self, conn: sqlite3.Connection) -> None:
        from questions_agent_platform.pipeline.service import _upsert_package_status
        _upsert_package_status(conn, user_id="u1", package_id="pkg_core_screener", status="completed")
        _upsert_package_status(conn, user_id="u1", package_id="pkg_core_screener", status="in_progress")

        row = conn.execute(
            "SELECT * FROM user_measurement_packages WHERE user_id='u1' AND measurement_package_id='pkg_core_screener';",
        ).fetchone()
        assert row["status"] == "completed"  # not downgraded

    def test_get_scored_scale_ids(self, conn: sqlite3.Connection) -> None:
        from questions_agent_platform.pipeline.service import _get_scored_scale_ids
        # Insert some scale scores
        for sid in ["scale_gad_7", "scale_pss", "scale_brs"]:
            conn.execute(
                """INSERT INTO scale_scores(
                    score_id, user_id, scale_id, computed_at,
                    window_start, window_end,
                    raw_score, normalized_score, confidence_tier,
                    items_answered_count, items_required
                ) VALUES (?, 'u1', ?, '2026-02-27T00:00:00Z',
                    '2026-02-20', '2026-02-27',
                    10.0, 50.0, 'baseline', 5, 5);""",
                (str(uuid.uuid4()), sid),
            )
        conn.commit()

        scored = _get_scored_scale_ids(conn, user_id="u1")
        assert scored == frozenset({"scale_gad_7", "scale_pss", "scale_brs"})

    def test_evaluate_and_advance_marks_complete(self, conn: sqlite3.Connection) -> None:
        from questions_agent_platform.pipeline.service import (
            _evaluate_and_advance_packages,
            _upsert_package_status,
        )

        # Mark core screener as in_progress
        _upsert_package_status(conn, user_id="u1", package_id="pkg_core_screener", status="in_progress")

        # Score all scales required by core screener
        pkg_scales = get_package_scale_ids("pkg_core_screener")
        for sid in pkg_scales:
            conn.execute(
                """INSERT INTO scale_scores(
                    score_id, user_id, scale_id, computed_at,
                    window_start, window_end,
                    raw_score, normalized_score, confidence_tier,
                    items_answered_count, items_required
                ) VALUES (?, 'u1', ?, '2026-02-27T00:00:00Z',
                    '2026-02-20', '2026-02-27',
                    10.0, 50.0, 'baseline', 5, 5);""",
                (str(uuid.uuid4()), sid),
            )
        conn.commit()

        completed = _evaluate_and_advance_packages(conn, user_id="u1")
        assert completed == "pkg_core_screener"

        row = conn.execute(
            "SELECT status FROM user_measurement_packages WHERE user_id='u1' AND measurement_package_id='pkg_core_screener';",
        ).fetchone()
        assert row["status"] == "completed"

    def test_evaluate_does_not_complete_partially_scored(self, conn: sqlite3.Connection) -> None:
        from questions_agent_platform.pipeline.service import (
            _evaluate_and_advance_packages,
            _upsert_package_status,
        )

        _upsert_package_status(conn, user_id="u1", package_id="pkg_core_screener", status="in_progress")

        # Score only one of the core screener's scales
        conn.execute(
            """INSERT INTO scale_scores(
                score_id, user_id, scale_id, computed_at,
                window_start, window_end,
                raw_score, normalized_score, confidence_tier,
                items_answered_count, items_required
            ) VALUES (?, 'u1', 'scale_gad_7', '2026-02-27T00:00:00Z',
                '2026-02-20', '2026-02-27',
                10.0, 50.0, 'baseline', 5, 5);""",
            (str(uuid.uuid4()),),
        )
        conn.commit()

        completed = _evaluate_and_advance_packages(conn, user_id="u1")
        assert completed is None

        row = conn.execute(
            "SELECT status FROM user_measurement_packages WHERE user_id='u1' AND measurement_package_id='pkg_core_screener';",
        ).fetchone()
        assert row["status"] == "in_progress"  # still in progress
