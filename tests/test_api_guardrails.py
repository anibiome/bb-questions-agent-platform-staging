from __future__ import annotations

import importlib
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from questions_agent_platform.pipeline.demo import seed_demo_registry


class ApiGuardrailsTests(unittest.TestCase):
    def test_prod_stack_invalid_date_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "prod.sqlite")
            registry_root = str(Path(td) / "registry")
            seed_demo_registry(registry_root)
            database_url = f"sqlite+pysqlite:///{db_path}"

            env_keys = (
                "DATABASE_URL",
                "REGISTRY_ROOT",
                "API_KEY",
                "DB_MIGRATION_MODE",
                "DB_MIGRATION_BASELINE_EXISTING",
            )
            env_backup = {k: os.environ.get(k) for k in env_keys}
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
                    resp = client.get(
                        "/v1/users/u_prod/scales?date_param=not-a-date",
                        headers=headers,
                    )
                    self.assertEqual(resp.status_code, 400)
                    self.assertIn("invalid date", str(resp.text).lower())
            finally:
                for key, old in env_backup.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old

    def test_prod_stack_request_more_context_blocks_cross_user_session(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "prod.sqlite")
            registry_root = str(Path(td) / "registry")
            seed_demo_registry(registry_root)
            database_url = f"sqlite+pysqlite:///{db_path}"

            env_keys = (
                "DATABASE_URL",
                "REGISTRY_ROOT",
                "API_KEY",
                "DB_MIGRATION_MODE",
                "DB_MIGRATION_BASELINE_EXISTING",
            )
            env_backup = {k: os.environ.get(k) for k in env_keys}
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
                    daily = client.get("/v1/users/alice/daily-questions", headers=headers)
                    self.assertEqual(daily.status_code, 200)
                    session_id = str(daily.json()["session_id"])

                    # Force one extra item so the ownership check is exercised on a valid extra pool.
                    conn = sqlite3.connect(db_path)
                    try:
                        conn.execute(
                            "UPDATE daily_sessions SET extra_batches_json=?, extra_batches_used=0 WHERE session_id=?;",
                            ('[["item_concentration_hard"]]', session_id),
                        )
                        conn.commit()
                    finally:
                        conn.close()

                    cross = client.post(
                        "/v1/users/bob/request-more-context",
                        headers=headers,
                        json={"session_id": session_id},
                    )
                    self.assertEqual(cross.status_code, 403)
                    self.assertIn("does not belong", str(cross.text).lower())
            finally:
                for key, old in env_backup.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old

    def test_prod_stack_uses_configured_extra_batch_cap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "prod.sqlite")
            registry_root = str(Path(td) / "registry")
            seed_demo_registry(registry_root)
            database_url = f"sqlite+pysqlite:///{db_path}"

            env_keys = (
                "DATABASE_URL",
                "REGISTRY_ROOT",
                "API_KEY",
                "DB_MIGRATION_MODE",
                "DB_MIGRATION_BASELINE_EXISTING",
                "EXTRA_BATCHES_MAX_PER_DAY",
            )
            env_backup = {k: os.environ.get(k) for k in env_keys}
            try:
                os.environ["DATABASE_URL"] = database_url
                os.environ["REGISTRY_ROOT"] = registry_root
                os.environ["API_KEY"] = "test-key"
                os.environ["DB_MIGRATION_MODE"] = "auto"
                os.environ["DB_MIGRATION_BASELINE_EXISTING"] = "false"
                os.environ["EXTRA_BATCHES_MAX_PER_DAY"] = "1"

                mod_name = "questions_agent_platform.prod.app"
                if mod_name in sys.modules:
                    mod = importlib.reload(sys.modules[mod_name])
                else:
                    mod = importlib.import_module(mod_name)

                with TestClient(mod.app) as client:
                    headers = {"X-API-Key": "test-key"}
                    daily = client.get("/v1/users/u_cap/daily-questions", headers=headers)
                    self.assertEqual(daily.status_code, 200)
                    payload = daily.json()
                    self.assertEqual(int(payload["extra"]["max_batches"]), 1)
            finally:
                for key, old in env_backup.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old


if __name__ == "__main__":
    unittest.main()
