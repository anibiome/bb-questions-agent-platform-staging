import importlib
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
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


class TestDriftRoutingContractApi(unittest.TestCase):
    def test_reference_sqlite_drift_routing_contract_default_and_custom(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            init_db(db_path)
            seed_demo_registry(registry_root)

            custom_table_path = Path(td) / "drift_routes.json"
            custom_table_path.write_text(
                json.dumps(
                    {
                        "version": "drift_routes_test_v1",
                        "routes": [
                            {
                                "route_id": "route_custom_general",
                                "drift_domain": "general",
                                "action": "full_27_rescan",
                                "reason_code": "route_custom_general",
                                "priority": 1,
                                "domain_ids": ["general"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            cfg_default = QuestionsAgentConfig(
                database_path=db_path,
                registry_root=registry_root,
                host="127.0.0.1",
                port=0,
            )
            server_default = ThreadingHTTPServer((cfg_default.host, 0), _make_handler(cfg_default))
            thread_default = threading.Thread(target=server_default.serve_forever, daemon=True)
            thread_default.start()
            try:
                base = f"http://127.0.0.1:{server_default.server_port}"
                payload = _http_json("GET", f"{base}/v1/drift-routing-contract")
            finally:
                server_default.shutdown()
                server_default.server_close()
                thread_default.join(timeout=3)

            self.assertEqual(payload["version"], "drift_routing_v1")
            self.assertGreaterEqual(len(payload.get("routes", [])), 1)
            self.assertIn("route_general_rescan", {r.get("route_id") for r in payload.get("routes", [])})

            cfg_custom = QuestionsAgentConfig(
                database_path=db_path,
                registry_root=registry_root,
                host="127.0.0.1",
                port=0,
                drift_routing_table_path=str(custom_table_path),
            )
            server_custom = ThreadingHTTPServer((cfg_custom.host, 0), _make_handler(cfg_custom))
            thread_custom = threading.Thread(target=server_custom.serve_forever, daemon=True)
            thread_custom.start()
            try:
                base = f"http://127.0.0.1:{server_custom.server_port}"
                payload = _http_json("GET", f"{base}/v1/drift-routing-contract")
            finally:
                server_custom.shutdown()
                server_custom.server_close()
                thread_custom.join(timeout=3)

            self.assertEqual(payload["version"], "drift_routes_test_v1")
            self.assertEqual(len(payload.get("routes", [])), 1)
            self.assertEqual(payload["routes"][0]["route_id"], "route_custom_general")

    def test_prod_stack_drift_routing_contract_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "prod.sqlite")
            registry_root = str(Path(td) / "registry")
            seed_demo_registry(registry_root)
            database_url = f"sqlite+pysqlite:///{db_path}"

            custom_table_path = Path(td) / "drift_routes_prod.json"
            custom_table_path.write_text(
                json.dumps(
                    {
                        "version": "drift_routes_prod_v1",
                        "routes": [
                            {
                                "route_id": "route_prod_general",
                                "drift_domain": "general",
                                "action": "full_27_rescan",
                                "reason_code": "route_prod_general",
                                "priority": 1,
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            env_backup = {
                k: os.environ.get(k)
                for k in (
                    "DATABASE_URL",
                    "REGISTRY_ROOT",
                    "API_KEY",
                    "DB_MIGRATION_MODE",
                    "DB_MIGRATION_BASELINE_EXISTING",
                    "DRIFT_ROUTING_TABLE_PATH",
                )
            }
            try:
                os.environ["DATABASE_URL"] = database_url
                os.environ["REGISTRY_ROOT"] = registry_root
                os.environ["API_KEY"] = "test-key"
                os.environ["DB_MIGRATION_MODE"] = "auto"
                os.environ["DB_MIGRATION_BASELINE_EXISTING"] = "false"
                os.environ["DRIFT_ROUTING_TABLE_PATH"] = str(custom_table_path)

                mod_name = "questions_agent_platform.prod.app"
                if mod_name in sys.modules:
                    mod = importlib.reload(sys.modules[mod_name])
                else:
                    mod = importlib.import_module(mod_name)

                with TestClient(mod.app) as client:
                    headers = {"X-API-Key": "test-key"}
                    response = client.get("/v1/drift-routing-contract", headers=headers)
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
            finally:
                for key, old in env_backup.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old

            self.assertEqual(payload["version"], "drift_routes_prod_v1")
            self.assertEqual(len(payload.get("routes", [])), 1)
            self.assertEqual(payload["routes"][0]["route_id"], "route_prod_general")


if __name__ == "__main__":
    unittest.main()
