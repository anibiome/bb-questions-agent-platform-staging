import importlib
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

from questions_agent_platform.pipeline.api import _make_handler
from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry


def _http_json(method: str, url: str, payload=None):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, method=method, headers=headers, data=body)
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


class TestStateCircleApiContracts(unittest.TestCase):
    def test_reference_sqlite_state_circle_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            init_db(db_path)
            seed_demo_registry(registry_root)

            cfg = QuestionsAgentConfig(
                database_path=db_path,
                registry_root=registry_root,
                host="127.0.0.1",
                port=0,
            )
            server = ThreadingHTTPServer((cfg.host, 0), _make_handler(cfg))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                day = date(2026, 2, 14).isoformat()
                daily = _http_json("GET", f"{base}/v1/users/u_sqlite/daily-questions?date={day}")
                answers = [
                    {
                        "client_event_id": f"u_sqlite::{day}::{q['item_id']}",
                        "item_id": q["item_id"],
                        "value": 2,
                        "answered_at": f"{day}T12:00:00Z",
                    }
                    for q in daily["questions"]
                ]
                _http_json(
                    "POST",
                    f"{base}/v1/users/u_sqlite/answers",
                    {"session_id": daily["session_id"], "answers": answers},
                )
                state = _http_json("GET", f"{base}/v1/users/u_sqlite/state?date={day}&window=30d")
                circle = _http_json("GET", f"{base}/v1/users/u_sqlite/circle?date={day}&window=30d")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

            self.assertEqual(state["user_id"], "u_sqlite")
            self.assertEqual(circle["user_id"], "u_sqlite")
            self.assertEqual(int(state["window_days"]), 30)
            self.assertEqual(int(circle["window_days"]), 30)
            self.assertGreaterEqual(len(state["snapshots"]), 1)
            self.assertGreaterEqual(len(circle["snapshots"]), 1)
            self.assertIn("x_hat", state["snapshots"][-1])
            self.assertIn("x_uncertainty", state["snapshots"][-1])
            self.assertIn("z", circle["snapshots"][-1])
            self.assertIn("coherence", circle["snapshots"][-1])
            self.assertIn("score", circle["snapshots"][-1]["coherence"])
            self.assertIn("tier", circle["snapshots"][-1]["coherence"])
            self.assertIn("uncertainty", circle["snapshots"][-1])
            self.assertIn("projection_version", circle["snapshots"][-1])

    def test_prod_stack_state_circle_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "prod.sqlite")
            registry_root = str(Path(td) / "registry")
            seed_demo_registry(registry_root)
            database_url = f"sqlite+pysqlite:///{db_path}"

            env_backup = {k: os.environ.get(k) for k in (
                "DATABASE_URL",
                "REGISTRY_ROOT",
                "API_KEY",
                "DB_MIGRATION_MODE",
                "DB_MIGRATION_BASELINE_EXISTING",
            )}
            try:
                os.environ["DATABASE_URL"] = database_url
                os.environ["REGISTRY_ROOT"] = registry_root
                os.environ["API_KEY"] = "test-key"
                os.environ["DB_MIGRATION_MODE"] = "auto"
                os.environ["DB_MIGRATION_BASELINE_EXISTING"] = "false"

                mod_name = "questions_agent_platform.prod.app"
                if mod_name in sys.modules:
                    mod = importlib.reload(sys.modules[mod_name])
                else:
                    mod = importlib.import_module(mod_name)

                with TestClient(mod.app) as client:
                    headers = {"X-API-Key": "test-key"}
                    day = date(2026, 2, 14).isoformat()
                    daily = client.get(f"/v1/users/u_prod/daily-questions?date_param={day}", headers=headers)
                    self.assertEqual(daily.status_code, 200)
                    daily_payload = daily.json()
                    answers = [
                        {
                            "client_event_id": f"u_prod::{day}::{q['item_id']}",
                            "item_id": q["item_id"],
                            "value": 2,
                            "answered_at": f"{day}T12:00:00Z",
                        }
                        for q in daily_payload["questions"]
                    ]
                    posted = client.post(
                        "/v1/users/u_prod/answers",
                        headers=headers,
                        json={"session_id": daily_payload["session_id"], "answers": answers},
                    )
                    self.assertEqual(posted.status_code, 200)

                    state = client.get(f"/v1/users/u_prod/state?date_param={day}&window=30d", headers=headers)
                    circle = client.get(f"/v1/users/u_prod/circle?date_param={day}&window=30d", headers=headers)
                    self.assertEqual(state.status_code, 200)
                    self.assertEqual(circle.status_code, 200)
                    state_payload = state.json()
                    circle_payload = circle.json()
            finally:
                for key, old in env_backup.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old

            self.assertEqual(state_payload["user_id"], "u_prod")
            self.assertEqual(circle_payload["user_id"], "u_prod")
            self.assertEqual(int(state_payload["window_days"]), 30)
            self.assertEqual(int(circle_payload["window_days"]), 30)
            self.assertGreaterEqual(len(state_payload["snapshots"]), 1)
            self.assertGreaterEqual(len(circle_payload["snapshots"]), 1)
            self.assertIn("x_hat", state_payload["snapshots"][-1])
            self.assertIn("x_uncertainty", state_payload["snapshots"][-1])
            self.assertIn("z", circle_payload["snapshots"][-1])
            self.assertIn("coherence", circle_payload["snapshots"][-1])
            self.assertIn("score", circle_payload["snapshots"][-1]["coherence"])
            self.assertIn("tier", circle_payload["snapshots"][-1]["coherence"])
            self.assertIn("uncertainty", circle_payload["snapshots"][-1])
            self.assertIn("projection_version", circle_payload["snapshots"][-1])


if __name__ == "__main__":
    unittest.main()
