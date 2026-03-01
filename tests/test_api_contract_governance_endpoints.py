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


class TestGovernanceApiContracts(unittest.TestCase):
    def test_reference_sqlite_governance_and_experiment_endpoints(self) -> None:
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
                answers = []
                for idx, q in enumerate(daily["questions"]):
                    raw = {"safety_trigger": True, "safety_reason_code": "test_trigger"} if idx == 0 else {}
                    answers.append(
                        {
                            "client_event_id": f"u_sqlite::{day}::{q['item_id']}",
                            "item_id": q["item_id"],
                            "value": 2,
                            "answered_at": f"{day}T12:00:00Z",
                            "raw": raw,
                        }
                    )
                _http_json(
                    "POST",
                    f"{base}/v1/users/u_sqlite/answers",
                    {"session_id": daily["session_id"], "answers": answers},
                )

                profile = _http_json("GET", f"{base}/v1/users/u_sqlite/profile")
                self.assertEqual(profile["user_id"], "u_sqlite")
                self.assertIn("mode", profile["profile"])

                patched = _http_json(
                    "PATCH",
                    f"{base}/v1/users/u_sqlite/profile",
                    {"mode": "trial", "onboarding_complete": True},
                )
                self.assertEqual(patched["profile"]["mode"], "trial")
                self.assertTrue(bool(patched["profile"]["onboarding_complete"]))

                progress = _http_json(
                    "GET",
                    f"{base}/v1/users/u_sqlite/progress?date={day}&window=30d",
                )
                self.assertEqual(progress["user_id"], "u_sqlite")
                self.assertEqual(progress["window_days"], 30)
                self.assertIn("coverage", progress)
                self.assertIn("scales", progress)

                events_open = _http_json(
                    "GET",
                    f"{base}/v1/users/u_sqlite/safety-events?date={day}&days=30&status=open",
                )
                self.assertGreaterEqual(len(events_open["events"]), 1)
                event_id = str(events_open["events"][0]["event_id"])

                resolved = _http_json(
                    "POST",
                    f"{base}/v1/users/u_sqlite/safety-events/{event_id}/resolve",
                    {"resolved_by": "qa", "resolved_reason": "reviewed"},
                )
                self.assertEqual(resolved["status"], "resolved")

                deep_dive = _http_json("GET", f"{base}/v1/users/u_sqlite/emotion/deep_dive?date={day}")
                self.assertEqual(deep_dive["user_id"], "u_sqlite")
                self.assertIn("triggered", deep_dive)

                created = _http_json(
                    "POST",
                    f"{base}/v1/users/u_sqlite/experiments",
                    {
                        "name": "Trial arm A",
                        "status": "active",
                        "start_date": day,
                        "baseline_window_days": 14,
                        "eval_window_days": 14,
                        "target_metrics": {"primary": "coherence_r"},
                    },
                )
                exp_id = created["experiment_id"]
                listed = _http_json("GET", f"{base}/v1/users/u_sqlite/experiments")
                self.assertTrue(any(str(x.get("experiment_id")) == exp_id for x in listed["experiments"]))

                detail = _http_json("GET", f"{base}/v1/users/u_sqlite/experiments/{exp_id}")
                self.assertEqual(detail["experiment_id"], exp_id)

                results = _http_json("GET", f"{base}/v1/users/u_sqlite/experiments/{exp_id}/results?date={day}")
                self.assertEqual(results["experiment_id"], exp_id)
                self.assertIn("metrics", results)

                summary = _http_json("GET", f"{base}/v1/users/u_sqlite/ani/summary?date={day}")
                self.assertEqual(summary["user_id"], "u_sqlite")
                self.assertIn("guardrails", summary)

                readiness = _http_json("GET", f"{base}/v1/admin/domains/readiness")
                self.assertIn("domains", readiness)
                self.assertIn("ready_domains", readiness)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_prod_stack_governance_and_experiment_endpoints(self) -> None:
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
                    day = date(2026, 2, 14).isoformat()

                    daily = client.get(f"/v1/users/u_prod/daily-questions?date_param={day}", headers=headers)
                    self.assertEqual(daily.status_code, 200)
                    daily_payload = daily.json()
                    answers = []
                    for idx, q in enumerate(daily_payload["questions"]):
                        raw = {"safety_trigger": True, "safety_reason_code": "test_trigger"} if idx == 0 else {}
                        answers.append(
                            {
                                "client_event_id": f"u_prod::{day}::{q['item_id']}",
                                "item_id": q["item_id"],
                                "value": 2,
                                "answered_at": f"{day}T12:00:00Z",
                                "raw": raw,
                            }
                        )
                    posted = client.post(
                        "/v1/users/u_prod/answers",
                        headers=headers,
                        json={"session_id": daily_payload["session_id"], "answers": answers},
                    )
                    self.assertEqual(posted.status_code, 200)

                    profile = client.get("/v1/users/u_prod/profile", headers=headers)
                    self.assertEqual(profile.status_code, 200)
                    self.assertEqual(profile.json()["user_id"], "u_prod")

                    patched = client.patch(
                        "/v1/users/u_prod/profile",
                        headers=headers,
                        json={"mode": "trial", "onboarding_complete": True},
                    )
                    self.assertEqual(patched.status_code, 200)
                    self.assertEqual(patched.json()["profile"]["mode"], "trial")

                    progress = client.get(
                        f"/v1/users/u_prod/progress?date_param={day}&window=30d",
                        headers=headers,
                    )
                    self.assertEqual(progress.status_code, 200)
                    self.assertEqual(progress.json()["user_id"], "u_prod")
                    self.assertEqual(progress.json()["window_days"], 30)
                    self.assertIn("coverage", progress.json())
                    self.assertIn("scales", progress.json())

                    events_open = client.get(
                        f"/v1/users/u_prod/safety-events?date_param={day}&days=30&status=open",
                        headers=headers,
                    )
                    self.assertEqual(events_open.status_code, 200)
                    event_id = str(events_open.json()["events"][0]["event_id"])

                    resolved = client.post(
                        f"/v1/users/u_prod/safety-events/{event_id}/resolve",
                        headers=headers,
                        json={"resolved_by": "qa", "resolved_reason": "reviewed"},
                    )
                    self.assertEqual(resolved.status_code, 200)
                    self.assertEqual(resolved.json()["status"], "resolved")

                    deep_dive = client.get(f"/v1/users/u_prod/emotion/deep_dive?date_param={day}", headers=headers)
                    self.assertEqual(deep_dive.status_code, 200)
                    self.assertEqual(deep_dive.json()["user_id"], "u_prod")

                    created = client.post(
                        "/v1/users/u_prod/experiments",
                        headers=headers,
                        json={
                            "name": "Trial arm B",
                            "status": "active",
                            "start_date": day,
                            "baseline_window_days": 14,
                            "eval_window_days": 14,
                            "target_metrics": {"primary": "coherence_r"},
                        },
                    )
                    self.assertEqual(created.status_code, 200)
                    exp_id = created.json()["experiment_id"]

                    listed = client.get("/v1/users/u_prod/experiments", headers=headers)
                    self.assertEqual(listed.status_code, 200)
                    self.assertTrue(any(str(x.get("experiment_id")) == exp_id for x in listed.json()["experiments"]))

                    detail = client.get(f"/v1/users/u_prod/experiments/{exp_id}", headers=headers)
                    self.assertEqual(detail.status_code, 200)
                    self.assertEqual(detail.json()["experiment_id"], exp_id)

                    results = client.get(f"/v1/users/u_prod/experiments/{exp_id}/results?date_param={day}", headers=headers)
                    self.assertEqual(results.status_code, 200)
                    self.assertEqual(results.json()["experiment_id"], exp_id)

                    summary = client.get(f"/v1/users/u_prod/ani/summary?date_param={day}", headers=headers)
                    self.assertEqual(summary.status_code, 200)
                    self.assertEqual(summary.json()["user_id"], "u_prod")

                    readiness = client.get("/v1/admin/domains/readiness", headers=headers)
                    self.assertEqual(readiness.status_code, 200)
                    self.assertIn("domains", readiness.json())
            finally:
                for key, old in env_backup.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old


if __name__ == "__main__":
    unittest.main()
