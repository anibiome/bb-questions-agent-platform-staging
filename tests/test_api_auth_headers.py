from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from questions_agent_platform.prod.auth import require_api_key


class TestAPIAuthHeaders(TestCase):
    def test_require_api_key_accepts_x_api_key(self) -> None:
        require_api_key("secret", "secret")

    def test_require_api_key_accepts_bearer_fallback(self) -> None:
        require_api_key(None, "secret", authorization="Bearer secret")

    def test_require_api_key_rejects_mismatch(self) -> None:
        with self.assertRaises(HTTPException) as exc:
            require_api_key(None, "secret", authorization="Bearer wrong")
        self.assertEqual(exc.exception.status_code, 401)

    def test_health_echoes_request_id_header(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env_patch = {
            "DATABASE_URL": "sqlite:///:memory:",
            "REGISTRY_ROOT": str(repo_root / "data" / "registry_demo"),
            "POLICY_ROOT": str(repo_root / "data" / "policy_demo"),
            "API_KEY": "test-key",
            "DB_MIGRATION_MODE": "off",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            import questions_agent_platform.prod.app as app_module

            app_module = importlib.reload(app_module)
            with TestClient(app_module.app) as client:
                response = client.get("/health", headers={"x-request-id": "req_qa_test"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers.get("x-request-id"), "req_qa_test")
