import importlib
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from questions_agent_platform.pipeline.demo import seed_demo_registry


class TestP2OpsControls(unittest.TestCase):
    def test_slo_privacy_and_audit_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "prod.sqlite")
            registry_root = str(Path(td) / "registry")
            seed_demo_registry(registry_root)

            env_backup = {k: os.environ.get(k) for k in (
                "DATABASE_URL",
                "REGISTRY_ROOT",
                "API_KEY",
                "DB_MIGRATION_MODE",
                "DB_MIGRATION_BASELINE_EXISTING",
                "PHI_EXPORT_REDACT_USER_IDS",
                "PHI_EXPORT_REDACT_RAW_PAYLOADS",
                "PHI_EXPORT_USER_HASH_SALT",
            )}
            try:
                os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{db_path}"
                os.environ["REGISTRY_ROOT"] = registry_root
                os.environ["API_KEY"] = "test-key"
                os.environ["DB_MIGRATION_MODE"] = "auto"
                os.environ["DB_MIGRATION_BASELINE_EXISTING"] = "false"
                os.environ["PHI_EXPORT_REDACT_USER_IDS"] = "true"
                os.environ["PHI_EXPORT_REDACT_RAW_PAYLOADS"] = "true"
                os.environ["PHI_EXPORT_USER_HASH_SALT"] = "ops-test-salt"

                mod_name = "questions_agent_platform.prod.app"
                if mod_name in sys.modules:
                    mod = importlib.reload(sys.modules[mod_name])
                else:
                    mod = importlib.import_module(mod_name)

                with TestClient(mod.app) as client:
                    headers = {"X-API-Key": "test-key"}
                    day = date(2026, 2, 14).isoformat()
                    daily = client.get(f"/v1/users/u_ops/daily-questions?date_param={day}", headers=headers)
                    self.assertEqual(daily.status_code, 200)
                    payload = daily.json()
                    answers = [
                        {
                            "client_event_id": f"u_ops::{day}::{q['item_id']}",
                            "item_id": q["item_id"],
                            "value": 2,
                            "answered_at": f"{day}T12:00:00Z",
                            "raw": {"note": "sensitive payload"},
                        }
                        for q in payload["questions"]
                    ]
                    posted = client.post(
                        "/v1/users/u_ops/answers",
                        headers=headers,
                        json={"session_id": payload["session_id"], "answers": answers},
                    )
                    self.assertEqual(posted.status_code, 200)

                    slo = client.get("/v1/admin/slo?days=30", headers=headers)
                    self.assertEqual(slo.status_code, 200)
                    slo_payload = slo.json()
                    self.assertIn("metrics", slo_payload)
                    self.assertIn("alerts", slo_payload)
                    self.assertIn("answer_ingestion_latency_p95_ms", slo_payload["metrics"])

                    retention_dry = client.post("/v1/admin/privacy/retention/run?dry_run=true", headers=headers)
                    self.assertEqual(retention_dry.status_code, 200)
                    self.assertTrue(retention_dry.json().get("dry_run"))

                    audit = client.get(
                        f"/v1/admin/audit/export?start_date={day}&end_date={day}",
                        headers=headers,
                    )
                    self.assertEqual(audit.status_code, 200)
                    audit_payload = audit.json()
                    self.assertIn("redaction", audit_payload)
                    self.assertTrue(bool(audit_payload["redaction"].get("redact_user_ids")))
                    self.assertTrue(bool(audit_payload["redaction"].get("redact_raw_payloads")))
                    if audit_payload.get("answers"):
                        self.assertIsNone(audit_payload["answers"][0].get("raw"))
                        self.assertNotEqual(str(audit_payload["answers"][0].get("user_id")), "u_ops")
            finally:
                for key, old in env_backup.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old


if __name__ == "__main__":
    unittest.main()

