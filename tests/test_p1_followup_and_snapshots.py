import json
import tempfile
import unittest
import uuid
from datetime import date, timedelta
from pathlib import Path

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.registry import (
    Item,
    Questionnaire,
    Registry,
    Scale,
    ScaleItem,
    save_registry,
)
from questions_agent_platform.pipeline.service import (
    get_circle_snapshots,
    get_or_create_daily_session,
    get_state_snapshots,
    submit_answers,
)
from questions_agent_platform.pipeline.time_utils import now_iso


def _seed_followup_snapshot_registry(registry_root: str) -> None:
    items = {
        "item_alpha": Item(
            id="item_alpha",
            text="Energy today",
            response_type="likert_0_4",
            tags=("energy",),
            sensitivity="low",
            timeframes_allowed=("last_7_days",),
            intrusiveness="low",
            declinable=True,
        ),
        "item_beta": Item(
            id="item_beta",
            text="Mood today",
            response_type="likert_0_4",
            tags=("mood",),
            sensitivity="low",
            timeframes_allowed=("last_7_days",),
            intrusiveness="low",
            declinable=True,
        ),
        "item_gamma": Item(
            id="item_gamma",
            text="Sleep quality",
            response_type="likert_0_4",
            tags=("sleep",),
            sensitivity="low",
            timeframes_allowed=("last_7_days",),
            intrusiveness="low",
            declinable=True,
        ),
    }
    registry = Registry(
        version="v_followup_snapshot",
        items=items,
        questionnaires={
            "q_core": Questionnaire(id="q_core", version="1", name="Core"),
        },
        scales={
            "scale_core": Scale(
                id="scale_core",
                questionnaire_id="q_core",
                version="1",
                name="Core scale",
                method="mean",
                min_items_required=1,
                unlock_window_days=1,
                retest_interval_days=1,
                response_type="likert_0_4",
                normalize_min=0.0,
                normalize_max=4.0,
                tags=("energy", "mood", "sleep"),
                items=(
                    ScaleItem(item_id="item_alpha", reverse=False, weight=1.0),
                    ScaleItem(item_id="item_beta", reverse=False, weight=1.0),
                    ScaleItem(item_id="item_gamma", reverse=False, weight=1.0),
                ),
            )
        },
    )
    save_registry(registry_root, registry)


class TestP1FollowUpAndSnapshots(unittest.TestCase):
    def test_followup_item_prioritized_and_resolved_on_answer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(
                database_path=db_path,
                registry_root=registry_root,
                core_questions_per_day=1,
                extra_batch_size=1,
                extra_batches_max_per_day=0,
            )
            init_db(db_path)
            _seed_followup_snapshot_registry(registry_root)

            day = date(2026, 2, 14)
            with connect(db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?);",
                    ("u_follow", now_iso()),
                )
                conn.execute(
                    """
                    INSERT INTO follow_up_queue(
                      queue_id, user_id, item_id, scale_id, reason_code, reason_detail_json,
                      priority, earliest_date, timeframe, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?);
                    """,
                    (
                        str(uuid.uuid4()),
                        "u_follow",
                        "item_gamma",
                        "scale_core",
                        "drift_followup",
                        json.dumps({"scale_id": "scale_core"}, ensure_ascii=False),
                        99.0,
                        day.isoformat(),
                        "last_7_days",
                        now_iso(),
                    ),
                )
                conn.commit()

                session = get_or_create_daily_session(
                    conn,
                    cfg=cfg,
                    registry_root=registry_root,
                    user_id="u_follow",
                    day=day,
                )
                self.assertEqual(tuple(session.core_item_ids), ("item_gamma",))
                self.assertIn("item_gamma", set(session.selection_explain.get("follow_up_items_due", [])))

                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u_follow",
                    session_id=session.session_id,
                    answers=[
                        {
                            "client_event_id": "u_follow::evt1",
                            "item_id": "item_gamma",
                            "value": 4,
                            "answered_at": f"{day.isoformat()}T09:00:00Z",
                        }
                    ],
                )

                row = conn.execute(
                    """
                    SELECT status, resolved_reason
                    FROM follow_up_queue
                    WHERE user_id=? AND item_id=?
                    ORDER BY created_at DESC
                    LIMIT 1;
                    """,
                    ("u_follow", "item_gamma"),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(str(row["status"]), "resolved")
                self.assertEqual(str(row["resolved_reason"]), "answered")

    def test_state_and_circle_snapshots_created_and_queryable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(
                database_path=db_path,
                registry_root=registry_root,
                core_questions_per_day=1,
                extra_batch_size=1,
                extra_batches_max_per_day=0,
            )
            init_db(db_path)
            _seed_followup_snapshot_registry(registry_root)

            day1 = date(2026, 2, 14)
            day2 = day1 + timedelta(days=1)

            with connect(db_path) as conn:
                s1 = get_or_create_daily_session(
                    conn,
                    cfg=cfg,
                    registry_root=registry_root,
                    user_id="u_snap",
                    day=day1,
                )
                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u_snap",
                    session_id=s1.session_id,
                    answers=[
                        {
                            "client_event_id": "u_snap::evt_day1",
                            "item_id": s1.core_item_ids[0],
                            "value": 0,
                            "answered_at": f"{day1.isoformat()}T10:00:00Z",
                        }
                    ],
                )

                s2 = get_or_create_daily_session(
                    conn,
                    cfg=cfg,
                    registry_root=registry_root,
                    user_id="u_snap",
                    day=day2,
                )
                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u_snap",
                    session_id=s2.session_id,
                    answers=[
                        {
                            "client_event_id": "u_snap::evt_day2",
                            "item_id": s2.core_item_ids[0],
                            "value": 4,
                            "answered_at": f"{day2.isoformat()}T10:00:00Z",
                        }
                    ],
                )

                state_rows = conn.execute(
                    "SELECT COUNT(*) AS c FROM state_snapshots WHERE user_id=?;",
                    ("u_snap",),
                ).fetchone()
                circle_rows = conn.execute(
                    "SELECT COUNT(*) AS c FROM circle_snapshots WHERE user_id=?;",
                    ("u_snap",),
                ).fetchone()
                self.assertEqual(int(state_rows["c"]), 2)
                self.assertEqual(int(circle_rows["c"]), 2)

                states = get_state_snapshots(conn, user_id="u_snap", day=day2, window_days=30)
                circles = get_circle_snapshots(conn, user_id="u_snap", day=day2, window_days=30)
                self.assertEqual(len(states), 2)
                self.assertEqual(len(circles), 2)
                self.assertEqual(states[0]["date"], day1.isoformat())
                self.assertEqual(states[1]["date"], day2.isoformat())
                self.assertEqual(circles[1]["projection_version"], "circle_projection_v1_fixed")
                self.assertIn("x_hat", states[1])
                self.assertIn("uncertainty", circles[1])
                self.assertGreaterEqual(float(circles[1]["velocity"]), 0.0)


if __name__ == "__main__":
    unittest.main()
