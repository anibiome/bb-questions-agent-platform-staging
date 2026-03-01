import json
import tempfile
import threading
import unittest
import urllib.request
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

from questions_agent_platform.pipeline.api import _make_handler
from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.tools.run_anifold_roundtrip_smoke import main as roundtrip_main
from questions_agent_platform.tools.run_mock_anifold_server import build_mock_server


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


class TestAnifoldRoundtripSmokeTool(unittest.TestCase):
    def test_roundtrip_smoke_runs_end_to_end(self) -> None:
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
            qa_server = ThreadingHTTPServer((cfg.host, 0), _make_handler(cfg))
            qa_thread = threading.Thread(target=qa_server.serve_forever, daemon=True)
            qa_thread.start()

            anifold_server = build_mock_server(host="127.0.0.1", port=0)
            anifold_thread = threading.Thread(target=anifold_server.serve_forever, daemon=True)
            anifold_thread.start()
            try:
                qa_base = f"http://127.0.0.1:{qa_server.server_port}"
                anifold_base = f"http://127.0.0.1:{anifold_server.server_port}"
                rc = roundtrip_main(
                    [
                        "--questions-base-url",
                        qa_base,
                        "--anifold-base-url",
                        anifold_base,
                        "--user-id",
                        "u_roundtrip_smoke",
                        "--date",
                        date(2026, 2, 16).isoformat(),
                        "--questions-date-param-name",
                        "date",
                    ]
                )
                self.assertEqual(rc, 0)

                logs = _http_json("GET", f"{anifold_base}/v1/logs")
                self.assertEqual(int(logs["evidence_count"]), 1)
            finally:
                qa_server.shutdown()
                qa_server.server_close()
                qa_thread.join(timeout=3)
                anifold_server.shutdown()
                anifold_server.server_close()
                anifold_thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
