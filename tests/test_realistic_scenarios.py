"""
Realistic stress-test scenarios — "take the car out and drive it."

Scenario 1: Flat-liner — user answers 3 to everything for 30 days.
  Tests: scale unlocks still happen, baseline doesn't crash,
         drift detection stays safe with zero variance,
         no div-by-zero anywhere.

Scenario 2: Aggressive decliner — user declines 2 items per day.
  Tests: graceful degradation when most items are declined,
         impossible-to-unlock scales are detectable,
         selector doesn't crash on exhausted registry.

Scenario 3: Mid-stream registry swap — 10 days on v1, then switch to v2.
  Tests: new items appear in sessions, old scores survive,
         no crash from orphaned session→registry references.
"""

import tempfile
import unittest
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.pipeline.demo_signals import demo_value_for_item
from questions_agent_platform.pipeline.registry import load_registry
from questions_agent_platform.pipeline.service import (
    compute_user_scale_progress,
    ensure_registry_active,
    get_or_create_daily_session,
    submit_answers,
)


class TestScenarioFlatLiner(unittest.TestCase):
    """
    Scenario 1: User answers 3 to every single question for 30 days.

    Real-world analog: a disengaged or confused user tapping the middle
    option on every Likert item. The system must not crash, must still
    unlock scales, and must not produce spurious drift alerts.
    """

    DAYS = 30
    FLAT_VALUE = 3  # middle of likert_5

    def test_flat_liner_30_days(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "flat.sqlite")
            reg_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=reg_root)
            init_db(db_path)
            seed_demo_registry(reg_root)

            unlocks = []
            total_questions = 0

            with connect(db_path) as conn:
                ensure_registry_active(conn, reg_root)

                for day_idx in range(self.DAYS):
                    day = date(2026, 1, 1) + timedelta(days=day_idx)

                    session = get_or_create_daily_session(
                        conn, cfg=cfg, registry_root=reg_root,
                        user_id="flat-user", day=day,
                    )

                    answers = [{
                        "client_event_id": f"flat::{day_idx}::{iid}",
                        "item_id": iid,
                        "value": self.FLAT_VALUE,
                        "answered_at": f"{day.isoformat()}T12:00:00Z",
                    } for iid in session.core_item_ids]

                    total_questions += len(answers)

                    result = submit_answers(
                        conn, registry_root=reg_root,
                        user_id="flat-user",
                        session_id=session.session_id,
                        answers=answers,
                    )

                    new_scores = result.get("new_scale_scores", [])
                    if new_scores:
                        unlocks.extend(new_scores)

                # --- Assertions ---

                # 1. No crashes over 30 days (we got here)
                self.assertGreater(total_questions, 0)

                # 2. At least some scales should unlock even with flat answers
                self.assertGreater(
                    len(unlocks), 0,
                    "Flat-liner: no scales unlocked in 30 days. "
                    "Scale unlock should not require answer variety."
                )

                # 3. Evidence rows should exist and show multiplexing
                evidence_count = conn.execute(
                    "SELECT COUNT(*) FROM scale_evidence WHERE user_id='flat-user'"
                ).fetchone()[0]
                self.assertGreater(evidence_count, total_questions)

                # 4. Baseline should be stable (variance near zero for a flat-liner)
                baselines = conn.execute(
                    "SELECT scale_id, mean, var FROM scale_baselines WHERE user_id='flat-user'"
                ).fetchall()
                for bl in baselines:
                    # Variance should be very low with constant input
                    self.assertLess(
                        float(bl["var"]), 50.0,
                        f"Scale {bl['scale_id']}: variance {bl['var']} is too high "
                        f"for a flat-liner (expected near-zero)"
                    )

                # 5. No drift follow-ups should exist (constant answers = no drift)
                drift_followups = conn.execute(
                    "SELECT COUNT(*) FROM follow_up_queue WHERE user_id='flat-user' AND status='pending'"
                ).fetchone()[0]
                self.assertEqual(
                    drift_followups, 0,
                    f"Flat-liner triggered {drift_followups} drift follow-ups — "
                    f"constant answers should never trigger drift"
                )


class TestScenarioAggressiveDecliner(unittest.TestCase):
    """
    Scenario 2: User permanently declines 2 items per session for 15 days,
    then continues answering normally for 15 more days.

    Real-world analog: a privacy-conscious user who doesn't want to answer
    certain medical questions. After enough declines, some scales become
    impossible to unlock (not enough remaining items).
    """

    def test_aggressive_decliner(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "decline.sqlite")
            reg_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=reg_root)
            init_db(db_path)
            seed_demo_registry(reg_root)

            declined_total: Set[str] = set()
            sessions_with_items = 0
            empty_sessions = 0
            exhaustion_warnings: List[Dict] = []

            with connect(db_path) as conn:
                reg_v = ensure_registry_active(conn, reg_root)
                registry = load_registry(reg_root, reg_v)
                total_items = len(registry.items)

                for day_idx in range(30):
                    day = date(2026, 1, 1) + timedelta(days=day_idx)

                    session = get_or_create_daily_session(
                        conn, cfg=cfg, registry_root=reg_root,
                        user_id="decliner", day=day,
                    )

                    # Capture pool exhaustion signals as they happen
                    if session.pool_exhaustion is not None:
                        exhaustion_warnings.append(session.pool_exhaustion)

                    core = list(session.core_item_ids)
                    if not core:
                        empty_sessions += 1
                        continue
                    sessions_with_items += 1

                    answers = []
                    for idx, iid in enumerate(core):
                        # First 15 days: decline the first 2 items each session
                        if day_idx < 15 and idx < 2 and iid not in declined_total:
                            answers.append({
                                "client_event_id": f"dec::{day_idx}::{iid}",
                                "item_id": iid,
                                "value": 0,
                                "raw": {"permanently_declined": True},
                                "answered_at": f"{day.isoformat()}T12:00:00Z",
                            })
                            declined_total.add(iid)
                        else:
                            answers.append({
                                "client_event_id": f"dec::{day_idx}::{iid}",
                                "item_id": iid,
                                "value": 3,
                                "answered_at": f"{day.isoformat()}T12:00:00Z",
                            })

                    submit_answers(
                        conn, registry_root=reg_root,
                        user_id="decliner",
                        session_id=session.session_id,
                        answers=answers,
                    )

                # --- Assertions ---

                # 1. Should never crash, even with heavy declining
                self.assertGreater(sessions_with_items, 0)

                # 2. Declined items should never reappear AFTER the day they were declined
                import json as _json
                cumulative_declined: Set[str] = set()
                for d_idx in range(30):
                    day = date(2026, 1, 1) + timedelta(days=d_idx)
                    row = conn.execute(
                        "SELECT core_questions_json FROM daily_sessions "
                        "WHERE user_id='decliner' AND date=?",
                        (day.isoformat(),),
                    ).fetchone()
                    if not row:
                        continue
                    core_ids = _json.loads(row["core_questions_json"])
                    for iid in core_ids:
                        if iid in cumulative_declined:
                            self.fail(
                                f"Day {d_idx}: declined item {iid} reappeared "
                                f"(was declined before day {d_idx})"
                            )
                    if d_idx < 15:
                        for iid in core_ids[:2]:
                            if iid not in cumulative_declined:
                                cumulative_declined.add(iid)

                # 3. Pool exhaustion must be surfaced when sessions are undersized.
                # The system MUST NOT silently return empty sessions.
                if empty_sessions > 0:
                    self.assertGreater(
                        len(exhaustion_warnings), 0,
                        "Sessions were empty but no pool_exhaustion warning was raised. "
                        "The API must tell clients WHY the pool is exhausted."
                    )
                    # Verify the exhaustion payload has required fields
                    last_warning = exhaustion_warnings[-1]
                    self.assertIn("declined_count", last_warning)
                    self.assertIn("impossible_scales", last_warning)
                    self.assertIn("warning", last_warning)
                    self.assertGreater(
                        last_warning["declined_count"], 0,
                        "Pool exhaustion should report declined items"
                    )

                # 4. Check which scales are now impossible to unlock
                impossible_scales = []
                for scale in registry.scales.values():
                    required_item_ids = {si.item_id for si in scale.items}
                    remaining = required_item_ids - declined_total
                    if len(remaining) < scale.min_items_required:
                        impossible_scales.append(scale.id)

                # 5. Progress endpoint should still work even with impossible scales
                progress = compute_user_scale_progress(
                    conn,
                    registry=registry,
                    user_id="decliner",
                    day=date(2026, 1, 30),
                )
                self.assertIsNotNone(progress)

                # 6. Verify exhaustion warnings match reality
                if exhaustion_warnings:
                    final = exhaustion_warnings[-1]
                    self.assertEqual(
                        len(final["impossible_scales"]),
                        len(impossible_scales),
                        "Exhaustion warning impossible_scales count doesn't match computed."
                    )


class TestScenarioRegistrySwap(unittest.TestCase):
    """
    Scenario 3: User has 10 days on registry v1, then registry changes to v2.

    Real-world analog: clinical team adds new questionnaires or modifies
    existing ones mid-study. Existing user data must survive, new items
    should appear in subsequent sessions.
    """

    def test_registry_swap_mid_stream(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "swap.sqlite")
            reg_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=reg_root)
            init_db(db_path)
            seed_demo_registry(reg_root)  # Seeds v1 and v2

            v1_items: Set[str] = set()
            v2_items: Set[str] = set()
            v2_new_items: Set[str] = set()

            with connect(db_path) as conn:
                # Phase 1: 10 days on v1
                reg_v1 = ensure_registry_active(conn, reg_root)
                registry_v1 = load_registry(reg_root, reg_v1)
                v1_items = set(registry_v1.items.keys())

                for day_idx in range(10):
                    day = date(2026, 1, 1) + timedelta(days=day_idx)
                    session = get_or_create_daily_session(
                        conn, cfg=cfg, registry_root=reg_root,
                        user_id="swap-user", day=day,
                    )
                    answers = [{
                        "client_event_id": f"v1::{day_idx}::{iid}",
                        "item_id": iid,
                        "value": demo_value_for_item(
                            tags=registry_v1.items[iid].tags if iid in registry_v1.items else (),
                            day_index=day_idx,
                        ),
                        "answered_at": f"{day.isoformat()}T12:00:00Z",
                    } for iid in session.core_item_ids]

                    submit_answers(
                        conn, registry_root=reg_root,
                        user_id="swap-user",
                        session_id=session.session_id,
                        answers=answers,
                    )

                # Count v1 scores
                v1_scores = conn.execute(
                    "SELECT COUNT(*) FROM scale_scores WHERE user_id='swap-user'"
                ).fetchone()[0]

                # --- Swap to v2 ---
                from questions_agent_platform.pipeline.registry import list_versions
                versions = list_versions(reg_root)
                v2_version = [v for v in versions if v != reg_v1]
                self.assertGreater(len(v2_version), 0, "No v2 version found in demo registry")
                v2_ver = v2_version[0]

                # Activate v2
                from questions_agent_platform.pipeline.db import set_registry_version_active, upsert_registry_version
                upsert_registry_version(conn, v2_ver)
                set_registry_version_active(conn, v2_ver)
                conn.commit()

                registry_v2 = load_registry(reg_root, v2_ver)
                v2_items = set(registry_v2.items.keys())
                v2_new_items = v2_items - v1_items

                # Phase 2: 20 more days on v2
                for day_idx in range(10, 30):
                    day = date(2026, 1, 1) + timedelta(days=day_idx)
                    session = get_or_create_daily_session(
                        conn, cfg=cfg, registry_root=reg_root,
                        user_id="swap-user", day=day,
                    )
                    answers = [{
                        "client_event_id": f"v2::{day_idx}::{iid}",
                        "item_id": iid,
                        "value": demo_value_for_item(
                            tags=registry_v2.items[iid].tags if iid in registry_v2.items else (),
                            day_index=day_idx,
                        ),
                        "answered_at": f"{day.isoformat()}T12:00:00Z",
                    } for iid in session.core_item_ids]

                    submit_answers(
                        conn, registry_root=reg_root,
                        user_id="swap-user",
                        session_id=session.session_id,
                        answers=answers,
                    )

                # --- Assertions ---

                # 1. V1 scores should survive the swap
                v1_scores_after = conn.execute(
                    "SELECT COUNT(*) FROM scale_scores WHERE user_id='swap-user'"
                ).fetchone()[0]
                self.assertGreaterEqual(
                    v1_scores_after, v1_scores,
                    "V1 scores were lost after registry swap"
                )

                # 2. V2 should have MORE scores (new scales)
                self.assertGreater(
                    v1_scores_after, v1_scores,
                    "No new scores computed after registry swap to v2"
                )

                # 3. If v2 has new items, at least some should appear in sessions
                if v2_new_items:
                    all_served = set()
                    for day_idx in range(10, 30):
                        day = date(2026, 1, 1) + timedelta(days=day_idx)
                        import json
                        row = conn.execute(
                            "SELECT core_questions_json FROM daily_sessions "
                            "WHERE user_id='swap-user' AND date=?",
                            (day.isoformat(),),
                        ).fetchone()
                        if row:
                            all_served.update(json.loads(row["core_questions_json"]))

                    new_items_served = all_served & v2_new_items
                    self.assertGreater(
                        len(new_items_served), 0,
                        f"No v2-new items were ever served after registry swap. "
                        f"New items: {sorted(v2_new_items)[:5]}..."
                    )

                # 4. Progress should reflect both v1 and v2 data
                progress = compute_user_scale_progress(
                    conn,
                    registry=registry_v2,
                    user_id="swap-user",
                    day=date(2026, 1, 30),
                )
                self.assertIsNotNone(progress)
                # V2 scales should show some progress (progress is flat: {scale_id: {...}})
                scales_with_progress = sum(
                    1 for sid, sp in progress.items()
                    if isinstance(sp, dict) and sp.get("answered_count", 0) > 0
                )
                self.assertGreater(scales_with_progress, 0, "No scales show progress after swap")


if __name__ == "__main__":
    unittest.main()
