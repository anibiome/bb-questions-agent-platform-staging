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


def _answer_all_core(base: str, user_id: str, day: str):
    daily = _http_json("GET", f"{base}/v1/users/{user_id}/daily-questions?date={day}")
    answers = [
        {
            "client_event_id": f"{user_id}::{day}::{q['item_id']}",
            "item_id": q["item_id"],
            "value": 2,
            "answered_at": f"{day}T12:00:00Z",
        }
        for q in daily["questions"]
    ]
    _http_json(
        "POST",
        f"{base}/v1/users/{user_id}/answers",
        {"session_id": daily["session_id"], "answers": answers},
    )


class TestTrajectoryObservationApiContracts(unittest.TestCase):
    def test_reference_sqlite_trajectory_and_observations(self) -> None:
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
                user_id = "u_sqlite_obs"
                day = date(2026, 2, 14).isoformat()
                _answer_all_core(base, user_id, day)

                tier_contract = _http_json("GET", f"{base}/v1/coherence/tier-contract")
                self.assertEqual(tier_contract["version"], "coherence_tier_contract_v1")
                self.assertGreaterEqual(len(tier_contract["tiers"]), 5)

                ews = _http_json("GET", f"{base}/v1/users/{user_id}/ews?date={day}&window=30d")
                self.assertEqual(ews["user_id"], user_id)
                self.assertGreaterEqual(len(ews["features"]), 1)
                self.assertIn("ews_score", ews["features"][-1])

                drift = _http_json("GET", f"{base}/v1/users/{user_id}/drift-events?date={day}&window=30d")
                self.assertEqual(drift["user_id"], user_id)
                self.assertIn("events", drift)
                self.assertIsInstance(drift["events"], list)

                obs = _http_json(
                    "POST",
                    f"{base}/v1/users/{user_id}/observations",
                    {
                        "observations": [
                            {
                                "type": "wearable",
                                "observed_at": f"{day}T13:00:00Z",
                                "features": {"projection": [0.35, 0.55, 0.60]},
                                "confidence": "high",
                                "provenance": {"source": "test"},
                            }
                        ]
                    },
                )
                self.assertEqual(int(obs["inserted_observation_events"]), 1)
                self.assertEqual(len(obs.get("coupling_outputs", [])), 1)
                self.assertIn("agreement_label", obs["coupling_outputs"][0])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_prod_stack_trajectory_and_observations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "prod.sqlite")
            registry_root = str(Path(td) / "registry")
            seed_demo_registry(registry_root)
            database_url = f"sqlite+pysqlite:///{db_path}"

            env_backup = {
                k: os.environ.get(k)
                for k in (
                    "DATABASE_URL",
                    "REGISTRY_ROOT",
                    "API_KEY",
                    "DB_MIGRATION_MODE",
                    "DB_MIGRATION_BASELINE_EXISTING",
                )
            }
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
                    user_id = "u_prod_obs"
                    day = date(2026, 2, 14).isoformat()

                    daily = client.get(f"/v1/users/{user_id}/daily-questions?date_param={day}", headers=headers)
                    self.assertEqual(daily.status_code, 200)
                    daily_payload = daily.json()
                    answers = [
                        {
                            "client_event_id": f"{user_id}::{day}::{q['item_id']}",
                            "item_id": q["item_id"],
                            "value": 2,
                            "answered_at": f"{day}T12:00:00Z",
                        }
                        for q in daily_payload["questions"]
                    ]
                    posted = client.post(
                        f"/v1/users/{user_id}/answers",
                        headers=headers,
                        json={"session_id": daily_payload["session_id"], "answers": answers},
                    )
                    self.assertEqual(posted.status_code, 200)

                    tier_contract = client.get("/v1/coherence/tier-contract", headers=headers)
                    self.assertEqual(tier_contract.status_code, 200)
                    self.assertEqual(tier_contract.json()["version"], "coherence_tier_contract_v1")

                    ews = client.get(f"/v1/users/{user_id}/ews?date_param={day}&window=30d", headers=headers)
                    self.assertEqual(ews.status_code, 200)
                    self.assertGreaterEqual(len(ews.json()["features"]), 1)

                    drift = client.get(f"/v1/users/{user_id}/drift-events?date_param={day}&window=30d", headers=headers)
                    self.assertEqual(drift.status_code, 200)
                    self.assertIn("events", drift.json())
                    self.assertIsInstance(drift.json()["events"], list)

                    obs = client.post(
                        f"/v1/users/{user_id}/observations",
                        headers=headers,
                        json={
                            "observations": [
                                {
                                    "type": "wearable",
                                    "observed_at": f"{day}T13:00:00Z",
                                    "features": {"projection": [0.45, 0.50, 0.65]},
                                    "confidence": "high",
                                    "provenance": {"source": "test"},
                                }
                            ]
                        },
                    )
                    self.assertEqual(obs.status_code, 200)
                    payload = obs.json()
                    self.assertEqual(int(payload["inserted_observation_events"]), 1)
                    self.assertEqual(len(payload.get("coupling_outputs", [])), 1)
                    self.assertIn("agreement_label", payload["coupling_outputs"][0])
            finally:
                for key, old in env_backup.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old


if __name__ == "__main__":
    unittest.main()
