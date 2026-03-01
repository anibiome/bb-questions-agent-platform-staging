import unittest
from unittest.mock import patch
from urllib.parse import unquote

from questions_agent_platform.tools.policy_release_gate import build_policy_release_report
from questions_agent_platform.tools.policy_rollback_drill import run_rollback_drill


class TestP2PolicyReleaseTools(unittest.TestCase):
    def test_release_gate_report_ready(self) -> None:
        report = build_policy_release_report(
            candidate_version="v2",
            gate_report={
                "quality_gates": {
                    "safety_no_violations": True,
                    "offline_eval_sample_ge_threshold": True,
                }
            },
            policy_versions={"active_version": "v1", "versions": ["v1", "v2"]},
            policy_metrics={"rollback_guard": {"rollback": False}},
            slo_report={"status": "ok", "alerts": []},
        )
        self.assertTrue(bool(report.get("ready_for_promotion")))
        self.assertEqual(report.get("blocked_reasons"), [])

    def test_release_gate_report_blocked(self) -> None:
        report = build_policy_release_report(
            candidate_version="v2",
            gate_report={"quality_gates": {"safety_no_violations": False}},
            policy_versions={"active_version": "v1", "versions": ["v1", "v2"]},
            policy_metrics={"rollback_guard": {"rollback": True}},
            slo_report={"status": "degraded", "alerts": ["slo_alert"]},
        )
        self.assertFalse(bool(report.get("ready_for_promotion")))
        self.assertIn("quality_gates_pass", report.get("blocked_reasons", []))
        self.assertIn("slo_status_ok", report.get("blocked_reasons", []))
        self.assertIn("rollback_guard_clear", report.get("blocked_reasons", []))

    def test_rollback_drill_switch_and_restore(self) -> None:
        state = {"active": "v2", "versions": ["v1", "v2"]}

        def fake_http(method, url, *, api_key, payload=None):
            if url.endswith("/v1/admin/policy/versions"):
                return {"active_version": state["active"], "versions": list(state["versions"])}
            if "/v1/admin/policy/activate/" in url:
                target = unquote(url.split("/v1/admin/policy/activate/", 1)[1])
                if target not in state["versions"]:
                    raise RuntimeError("unknown version")
                state["active"] = target
                return {"active_version": target}
            raise RuntimeError(f"unexpected url: {url}")

        with patch("questions_agent_platform.tools.policy_rollback_drill._http_json", side_effect=fake_http):
            report = run_rollback_drill(
                base_url="http://test.local",
                api_key="test-key",
                fallback_version="v1",
                restore_original=True,
            )
        self.assertEqual(str(report.get("status")), "pass")
        self.assertEqual(str(report.get("active_before")), "v2")
        self.assertEqual(str(report.get("fallback_version")), "v1")
        self.assertEqual(str(report.get("active_final")), "v2")


if __name__ == "__main__":
    unittest.main()
