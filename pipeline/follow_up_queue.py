from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date
from typing import Any, Optional, Set

from questions_agent_platform.pipeline.time_utils import now_iso, parse_date


def get_due_follow_up_item_ids(conn: sqlite3.Connection, *, user_id: str, day: date) -> Set[str]:
    rows = conn.execute(
        """
        SELECT item_id
        FROM follow_up_queue
        WHERE user_id=?
          AND status='pending'
          AND earliest_date<=?
        ORDER BY priority DESC, created_at ASC;
        """,
        (user_id, day.isoformat()),
    ).fetchall()
    return {str(r["item_id"]) for r in rows}


def enqueue_follow_up_queue_item(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    item_id: str,
    scale_id: Optional[str],
    reason_code: str,
    reason_detail: Optional[dict[str, Any]],
    priority: float,
    earliest_date: date,
    timeframe: Optional[str] = None,
) -> None:
    existing = conn.execute(
        """
        SELECT queue_id, priority, earliest_date
        FROM follow_up_queue
        WHERE user_id=? AND item_id=? AND status='pending'
        ORDER BY created_at ASC
        LIMIT 1;
        """,
        (user_id, item_id),
    ).fetchone()
    detail_json = json.dumps(reason_detail or {}, ensure_ascii=False)
    if existing:
        old_priority = float(existing["priority"] or 0.0)
        old_earliest = parse_date(str(existing["earliest_date"]))
        next_priority = max(old_priority, float(priority))
        next_earliest = min(old_earliest, earliest_date)
        conn.execute(
            """
            UPDATE follow_up_queue
            SET scale_id=?,
                reason_code=?,
                reason_detail_json=?,
                priority=?,
                earliest_date=?,
                timeframe=?
            WHERE queue_id=?;
            """,
            (
                scale_id,
                reason_code,
                detail_json,
                float(next_priority),
                next_earliest.isoformat(),
                timeframe,
                str(existing["queue_id"]),
            ),
        )
        return

    conn.execute(
        """
        INSERT INTO follow_up_queue(
          queue_id, user_id, item_id, scale_id, reason_code, reason_detail_json,
          priority, earliest_date, timeframe, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?);
        """,
        (
            str(uuid.uuid4()),
            user_id,
            item_id,
            scale_id,
            reason_code,
            detail_json,
            float(priority),
            earliest_date.isoformat(),
            timeframe,
            now_iso(),
        ),
    )
