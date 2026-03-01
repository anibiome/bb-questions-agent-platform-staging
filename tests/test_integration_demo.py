import tempfile
import unittest
from datetime import date
from pathlib import Path

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.pipeline.service import get_or_create_daily_session, submit_answers, take_next_extra_batch


class TestIntegrationDemo(unittest.TestCase):
    def test_demo_flow_unlocks_some_scales(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=registry_root)

            init_db(db_path)
            seed_demo_registry(registry_root)

            with connect(db_path) as conn:
                session = get_or_create_daily_session(
                    conn, cfg=cfg, registry_root=registry_root, user_id="u1", day=date(2026, 2, 2)
                )
                # Answer core
                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u1",
                    session_id=session.session_id,
                    answers=[
                        {
                            "client_event_id": f"core::{item_id}",
                            "item_id": item_id,
                            "value": 2,
                            "answered_at": "2026-02-02T12:00:00Z",
                        }
                        for item_id in session.core_item_ids
                    ],
                )
                # Take one extra batch, answer it
                batch = take_next_extra_batch(conn, session.session_id) or ()
                res = submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u1",
                    session_id=session.session_id,
                    answers=[
                        {
                            "client_event_id": f"extra::{item_id}",
                            "item_id": item_id,
                            "value": 2,
                            "answered_at": "2026-02-02T12:05:00Z",
                        }
                        for item_id in batch
                    ],
                )
                # Some demo scales are min_items_required=2, so we should unlock at least one.
                self.assertGreaterEqual(len(res.get("new_scale_scores") or []), 1)


if __name__ == "__main__":
    unittest.main()

