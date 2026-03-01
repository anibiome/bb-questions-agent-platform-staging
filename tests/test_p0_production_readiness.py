import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import connect, init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.pipeline.registry import Item, Questionnaire, Registry, Scale, ScaleItem, load_registry, save_registry
from questions_agent_platform.pipeline.service import (
    get_daily_session,
    get_or_create_daily_session,
    get_user_profile,
    submit_answers,
    take_next_extra_batch,
)


def _seed_cardio_onboarding_registry(registry_root: str) -> None:
    items = {}
    for idx in range(1, 7):
        item = Item(
            id=f"cardio_{idx}",
            text=f"Cardiometabolic onboarding item {idx}",
            response_type="likert_0_4",
            tags=("cardiometabolic", "metabolic"),
            sensitivity="low",
            timeframes_allowed=("last_7_days",),
            intrusiveness="low",
            declinable=True,
        )
        items[item.id] = item
    for idx in range(1, 7):
        item = Item(
            id=f"other_{idx}",
            text=f"Non-cardiometabolic item {idx}",
            response_type="likert_0_4",
            tags=("mood",),
            sensitivity="low",
            timeframes_allowed=("last_7_days",),
            intrusiveness="low",
            declinable=True,
        )
        items[item.id] = item

    registry = Registry(
        version="v_cardio_test",
        items=items,
        questionnaires={
            "q_cardio": Questionnaire(id="q_cardio", version="1", name="Cardiometabolic Core"),
            "q_other": Questionnaire(id="q_other", version="1", name="Other Domain"),
        },
        scales={
            "scale_cardio": Scale(
                id="scale_cardio",
                questionnaire_id="q_cardio",
                version="1",
                name="Cardio baseline",
                method="mean",
                min_items_required=6,
                unlock_window_days=14,
                retest_interval_days=90,
                response_type="likert_0_4",
                normalize_min=0.0,
                normalize_max=4.0,
                tags=("cardiometabolic",),
                items=tuple(ScaleItem(item_id=f"cardio_{idx}", reverse=False, weight=1.0) for idx in range(1, 7)),
            ),
            "scale_other": Scale(
                id="scale_other",
                questionnaire_id="q_other",
                version="1",
                name="Other scale",
                method="mean",
                min_items_required=3,
                unlock_window_days=14,
                retest_interval_days=90,
                response_type="likert_0_4",
                normalize_min=0.0,
                normalize_max=4.0,
                tags=("mood",),
                items=tuple(ScaleItem(item_id=f"other_{idx}", reverse=False, weight=1.0) for idx in range(1, 7)),
            ),
        },
    )
    save_registry(registry_root, registry)


class TestP0ProductionReadiness(unittest.TestCase):
    def test_onboarding_lock_keeps_cardiometabolic_only_and_five_items(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=registry_root)
            init_db(db_path)
            _seed_cardio_onboarding_registry(registry_root)

            day1 = date(2026, 2, 2)
            with connect(db_path) as conn:
                session1 = get_or_create_daily_session(
                    conn, cfg=cfg, registry_root=registry_root, user_id="u_cardio", day=day1
                )
                registry = load_registry(registry_root, session1.registry_version)
                cardio_items = {
                    item_id for item_id, item in registry.items.items() if "cardiometabolic" in set(item.tags)
                }
                self.assertEqual(len(session1.core_item_ids), 5)
                self.assertTrue(all(item_id in cardio_items for item_id in session1.core_item_ids))
                extras_day1 = {item_id for batch in session1.extra_batches for item_id in batch}
                self.assertTrue(all(item_id in cardio_items for item_id in extras_day1))

                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u_cardio",
                    session_id=session1.session_id,
                    answers=[
                        {
                            "client_event_id": f"u_cardio::{day1.isoformat()}::{item_id}",
                            "item_id": item_id,
                            "value": 2,
                            "answered_at": f"{day1.isoformat()}T12:00:00Z",
                        }
                        for item_id in session1.core_item_ids
                    ],
                )

                remaining = set(cardio_items) - set(session1.core_item_ids)
                self.assertEqual(len(remaining), 1)

                day2 = day1 + timedelta(days=1)
                session2 = get_or_create_daily_session(
                    conn, cfg=cfg, registry_root=registry_root, user_id="u_cardio", day=day2
                )
                self.assertEqual(len(session2.core_item_ids), 5)
                self.assertTrue(all(item_id in cardio_items for item_id in session2.core_item_ids))
                self.assertTrue(any(item_id in remaining for item_id in session2.core_item_ids))
                extras_day2 = {item_id for batch in session2.extra_batches for item_id in batch}
                self.assertTrue(all(item_id in cardio_items for item_id in extras_day2))

    def test_session_status_transitions_created_started_completed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=registry_root)

            init_db(db_path)
            seed_demo_registry(registry_root)

            day = date(2026, 2, 2)
            with connect(db_path) as conn:
                session = get_or_create_daily_session(conn, cfg=cfg, registry_root=registry_root, user_id="u1", day=day)
                current = get_daily_session(conn, "u1", day)
                self.assertIsNotNone(current)
                assert current is not None
                self.assertEqual(current.status, "created")

                first_item = session.core_item_ids[0]
                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u1",
                    session_id=session.session_id,
                    answers=[
                        {
                            "client_event_id": f"u1::{day.isoformat()}::first::{first_item}",
                            "item_id": first_item,
                            "value": 2,
                            "answered_at": f"{day.isoformat()}T12:00:00Z",
                        }
                    ],
                )
                current = get_daily_session(conn, "u1", day)
                assert current is not None
                self.assertEqual(current.status, "started")

                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u1",
                    session_id=session.session_id,
                    answers=[
                        {
                            "client_event_id": f"u1::{day.isoformat()}::rest::{item_id}",
                            "item_id": item_id,
                            "value": 2,
                            "answered_at": f"{day.isoformat()}T12:05:00Z",
                        }
                        for item_id in session.core_item_ids[1:]
                    ],
                )
                current = get_daily_session(conn, "u1", day)
                assert current is not None
                self.assertEqual(current.status, "completed")

    def test_scale_evidence_rows_written_for_multiplex_item(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=registry_root)

            init_db(db_path)
            seed_demo_registry(registry_root)

            day = date(2026, 2, 2)
            with connect(db_path) as conn:
                session = get_or_create_daily_session(conn, cfg=cfg, registry_root=registry_root, user_id="u1", day=day)
                item_id = "item_concentration_hard"
                client_event_id = f"u1::{day.isoformat()}::mux::{item_id}"
                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u1",
                    session_id=session.session_id,
                    answers=[
                        {
                            "client_event_id": client_event_id,
                            "item_id": item_id,
                            "value": 2,
                            "answered_at": f"{day.isoformat()}T12:00:00Z",
                        }
                    ],
                )

                row = conn.execute(
                    "SELECT event_id FROM answer_events WHERE user_id=? AND client_event_id=?;",
                    ("u1", client_event_id),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None

                evidence_rows = conn.execute(
                    "SELECT scale_id, timeframe, window_date FROM scale_evidence WHERE answer_event_id=?;",
                    (str(row["event_id"]),),
                ).fetchall()

                registry = load_registry(registry_root, session.registry_version)
                expected_scale_ids = {
                    scale.id
                    for scale in registry.scales.values()
                    if any(si.item_id == item_id for si in scale.items)
                }
                self.assertGreaterEqual(len(expected_scale_ids), 2)
                self.assertEqual({str(r["scale_id"]) for r in evidence_rows}, expected_scale_ids)
                self.assertTrue(all(str(r["timeframe"]) == session.timeframe for r in evidence_rows))
                self.assertTrue(all(str(r["window_date"]) == day.isoformat() for r in evidence_rows))

    def test_permanent_decline_excludes_future_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=registry_root)

            init_db(db_path)
            seed_demo_registry(registry_root)

            day1 = date(2026, 2, 2)
            with connect(db_path) as conn:
                session1 = get_or_create_daily_session(conn, cfg=cfg, registry_root=registry_root, user_id="u1", day=day1)
                declined_item = session1.core_item_ids[0]
                submit_answers(
                    conn,
                    registry_root=registry_root,
                    user_id="u1",
                    session_id=session1.session_id,
                    answers=[
                        {
                            "client_event_id": f"u1::{day1.isoformat()}::decline::{declined_item}",
                            "item_id": declined_item,
                            "raw": {"permanently_declined": True},
                        }
                    ],
                )

                profile = get_user_profile(conn, "u1")
                self.assertIn(declined_item, set(profile.get("permanently_declined_item_ids", [])))

                day2 = day1 + timedelta(days=1)
                session2 = get_or_create_daily_session(conn, cfg=cfg, registry_root=registry_root, user_id="u1", day=day2)
                self.assertNotIn(declined_item, set(session2.core_item_ids))
                extras_flat = {item_id for batch in session2.extra_batches for item_id in batch}
                self.assertNotIn(declined_item, extras_flat)

    def test_one_more_is_single_item_and_capped_at_three(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=registry_root)

            init_db(db_path)
            seed_demo_registry(registry_root)

            day = date(2026, 2, 2)
            with connect(db_path) as conn:
                session = get_or_create_daily_session(conn, cfg=cfg, registry_root=registry_root, user_id="u1", day=day)
                expected_extras = min(3, sum(len(batch) for batch in session.extra_batches))

                served = 0
                for _ in range(expected_extras + 5):
                    batch = take_next_extra_batch(conn, session.session_id)
                    if batch is None:
                        break
                    self.assertEqual(len(batch), 1)
                    served += 1

                self.assertEqual(served, expected_extras)
                self.assertIsNone(take_next_extra_batch(conn, session.session_id))
                row = conn.execute(
                    "SELECT extra_batches_used FROM daily_sessions WHERE session_id=?;",
                    (session.session_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(int(row["extra_batches_used"]), int(expected_extras))


if __name__ == "__main__":
    unittest.main()
