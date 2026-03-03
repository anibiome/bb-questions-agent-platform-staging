from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from questions_agent_platform.pipeline.db import connect, ensure_user, init_db
from questions_agent_platform.pipeline.follow_up_queue import (
    enqueue_follow_up_queue_item,
    get_due_follow_up_item_ids,
)


class TestFollowUpQueueHelpers(unittest.TestCase):
    def test_enqueue_updates_existing_pending_item(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            init_db(db_path)
            with connect(db_path) as conn:
                ensure_user(conn, "u1")
                enqueue_follow_up_queue_item(
                    conn,
                    user_id="u1",
                    item_id="item_001",
                    scale_id="scale_a",
                    reason_code="missing_answer",
                    reason_detail={"source": "score_gate"},
                    priority=0.2,
                    earliest_date=date(2026, 2, 16),
                    timeframe="last_7_days",
                )
                enqueue_follow_up_queue_item(
                    conn,
                    user_id="u1",
                    item_id="item_001",
                    scale_id="scale_b",
                    reason_code="anamnesis_drift_probe",
                    reason_detail={"source": "anamnesis"},
                    priority=0.9,
                    earliest_date=date(2026, 2, 14),
                    timeframe="last_30_days",
                )

                row = conn.execute(
                    """
                    SELECT scale_id, reason_code, priority, earliest_date, timeframe
                    FROM follow_up_queue
                    WHERE user_id='u1' AND item_id='item_001' AND status='pending'
                    """
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["scale_id"], "scale_b")
                self.assertEqual(row["reason_code"], "anamnesis_drift_probe")
                self.assertAlmostEqual(float(row["priority"]), 0.9, places=6)
                self.assertEqual(row["earliest_date"], "2026-02-14")
                self.assertEqual(row["timeframe"], "last_30_days")

                count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM follow_up_queue
                    WHERE user_id='u1' AND item_id='item_001' AND status='pending'
                    """
                ).fetchone()[0]
                self.assertEqual(int(count), 1)

    def test_get_due_follow_up_item_ids_respects_due_date(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            init_db(db_path)
            with connect(db_path) as conn:
                ensure_user(conn, "u2")
                enqueue_follow_up_queue_item(
                    conn,
                    user_id="u2",
                    item_id="item_due",
                    scale_id=None,
                    reason_code="anamnesis_drift_probe",
                    reason_detail=None,
                    priority=0.5,
                    earliest_date=date(2026, 2, 14),
                )
                enqueue_follow_up_queue_item(
                    conn,
                    user_id="u2",
                    item_id="item_future",
                    scale_id=None,
                    reason_code="anamnesis_drift_probe",
                    reason_detail=None,
                    priority=0.5,
                    earliest_date=date(2026, 2, 20),
                )
                due = get_due_follow_up_item_ids(conn, user_id="u2", day=date(2026, 2, 15))
                self.assertEqual(due, {"item_due"})


if __name__ == "__main__":
    unittest.main()
