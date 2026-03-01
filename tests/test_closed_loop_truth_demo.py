import json
import tempfile
import unittest
from pathlib import Path

from questions_agent_platform.pipeline.config import QuestionsAgentConfig
from questions_agent_platform.tools.generate_closed_loop_truth_demo import generate_closed_loop_truth_demo


class TestClosedLoopTruthDemo(unittest.TestCase):
    def test_closed_loop_truth_demo_generates_full_chain_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "qa.sqlite")
            registry_root = str(Path(td) / "registry")
            out_dir = str(Path(td) / "out")
            cfg = QuestionsAgentConfig(database_path=db_path, registry_root=registry_root)

            result = generate_closed_loop_truth_demo(
                cfg=cfg,
                out_dir=out_dir,
                days=14,
                user_id="u_closed_loop",
                seed=123,
                epsilon=0.25,
                reset_db=True,
            )

            self.assertTrue(result.json_path.exists())
            self.assertTrue(result.html_path.exists())

            payload = json.loads(result.json_path.read_text(encoding="utf-8"))
            gates = payload.get("quality_gates") if isinstance(payload.get("quality_gates"), dict) else {}
            coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
            daily = payload.get("daily") if isinstance(payload.get("daily"), list) else []

            self.assertEqual(str(payload.get("meta", {}).get("verdict")), "PASS")
            self.assertTrue(all(bool(v) for v in gates.values()))
            self.assertEqual(int(coverage.get("policy_decisions") or 0), 14)
            self.assertEqual(int(coverage.get("decision_with_projection_anchor_count") or 0), 14)
            self.assertEqual(int(coverage.get("decision_with_outcome_count") or 0), 14)
            self.assertEqual(int(coverage.get("decision_with_full_chain_count") or 0), 14)
            self.assertEqual(len(daily), 14)
            self.assertTrue(all(bool(row.get("projection_anchor_ok")) for row in daily if isinstance(row, dict)))
            self.assertTrue(all(bool(row.get("outcome_ok")) for row in daily if isinstance(row, dict)))


if __name__ == "__main__":
    unittest.main()
