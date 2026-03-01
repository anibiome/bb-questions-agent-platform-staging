import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from questions_agent_platform.pipeline.api import _make_handler
from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.pipeline.db import init_db
from questions_agent_platform.pipeline.demo import seed_demo_registry
from questions_agent_platform.tools.load_test_questions_agent import main as load_test_main


class TestP2LoadHarness(unittest.TestCase):
    def test_load_harness_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            report_path = str(Path(td) / "load_report.json")
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
                exit_code = load_test_main(
                    [
                        "--base-url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--api-key",
                        "unused-for-reference-stack",
                        "--users",
                        "2",
                        "--days",
                        "2",
                        "--concurrency",
                        "2",
                        "--date-param-name",
                        "date",
                        "--out",
                        report_path,
                        "--fail-on-errors",
                    ]
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

            self.assertEqual(exit_code, 0)
            payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
            self.assertEqual(str(payload.get("status")), "ok")
            self.assertGreater(int(payload.get("requests_scenarios") or 0), 0)
            self.assertGreater(int(payload["latency"]["daily_select_ms"]["count"]), 0)
            self.assertGreater(int(payload["latency"]["answer_ingest_ms"]["count"]), 0)
            self.assertGreater(int(payload["latency"]["state_read_ms"]["count"]), 0)
            self.assertGreater(int(payload["latency"]["circle_read_ms"]["count"]), 0)


if __name__ == "__main__":
    unittest.main()

