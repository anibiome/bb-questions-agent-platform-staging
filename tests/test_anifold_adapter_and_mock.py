import json
import threading
import unittest
import urllib.request

from questions_agent_platform.contracts import CONTRACT_FUSION_TO_QUESTIONS_CONTEXT
from questions_agent_platform.pipeline.anifold_adapter import (
    AnifoldAdapterClient,
    AnifoldAdapterConfig,
    normalize_context_payload,
    normalize_outcome_payload,
)
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


def _sample_projection_payload(user_id: str, day: str, session_id: str, decision_id: str):
    vector = [round(float(i) / 1000.0, 6) for i in range(128)]
    return {
        "contract": {
            "name": "questions_to_fusion_evidence",
            "schema_version": "1.0",
            "compatibility": ["1.0"],
            "producer": "test-suite",
            "join_keys": {
                "subject_id": user_id,
                "date": day,
                "session_id": session_id,
                "decision_id": decision_id,
            },
        },
        "subject_id": user_id,
        "timestamp": f"{day}T12:00:00Z",
        "modality_projections": {
            "questionnaires": {
                "projection": vector,
                "uncertainty": {"diag": [0.2 for _ in range(128)]},
                "velocity": [0.0 for _ in range(128)],
                "attractor_candidate": [0.1 for _ in range(128)],
            }
        },
    }


class TestAnifoldAdapterAndMock(unittest.TestCase):
    def test_fetch_context_and_build_select_payload(self) -> None:
        server = build_mock_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            client = AnifoldAdapterClient(AnifoldAdapterConfig(base_url=base_url))
            context = client.fetch_questions_context(
                user_id="u_adapter_1",
                day="2026-02-16",
                identity_mask_id="identity_mask_default",
            )
            self.assertEqual(context["contract_name"], CONTRACT_FUSION_TO_QUESTIONS_CONTEXT)
            self.assertEqual(context["schema_version"], "1.0")
            self.assertEqual(len(context["anifold_z"]), 3)
            self.assertEqual(len(context["z_uncertainty_diag"]), 3)
            self.assertEqual(len(context["z_velocity"]), 3)

            req = client.build_daily_select_request(
                day="2026-02-16",
                context_payload=context,
                selection_mode="policy_shadow",
                include_explanations=True,
                allow_context_batches=True,
                k_core=5,
                identity_mask_id="identity_mask_default",
            )
            self.assertEqual(req["date"], "2026-02-16")
            self.assertEqual(req["selection_mode"], "policy_shadow")
            self.assertEqual(req["k_core"], 5)
            self.assertIn("context", req)
            self.assertEqual(req["context"]["contract_name"], CONTRACT_FUSION_TO_QUESTIONS_CONTEXT)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_submit_evidence_and_outcome_to_mock(self) -> None:
        server = build_mock_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            client = AnifoldAdapterClient(AnifoldAdapterConfig(base_url=base_url))

            evidence_payload = _sample_projection_payload(
                user_id="u_adapter_2",
                day="2026-02-16",
                session_id="sess-1",
                decision_id="dec-1",
            )
            evidence_result = client.submit_questions_evidence(
                user_id="u_adapter_2",
                day="2026-02-16",
                evidence_payload=evidence_payload,
                session_id="sess-1",
                decision_id="dec-1",
            )
            self.assertEqual(evidence_result["status"], "accepted")
            self.assertEqual(evidence_result["contract_name"], "questions_to_fusion_evidence")

            outcome_result = client.submit_questions_outcome(
                user_id="u_adapter_2",
                outcome_payload={
                    "decision_id": "dec-1",
                    "z_before": [0.1, 0.2, 0.3],
                    "z_after": [0.2, 0.25, 0.35],
                    "uncertainty_before_diag": [0.30, 0.31, 0.32],
                    "uncertainty_after_diag": [0.26, 0.27, 0.28],
                },
            )
            self.assertEqual(outcome_result["status"], "accepted")
            self.assertEqual(outcome_result["contract_name"], "fusion_to_questions_outcome")

            logs = _http_json("GET", f"{base_url}/v1/logs")
            self.assertEqual(int(logs["evidence_count"]), 1)
            self.assertEqual(int(logs["outcome_count"]), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_contract_normalization_guards(self) -> None:
        with self.assertRaises(ValueError):
            normalize_context_payload(
                {
                    "contract_name": "fusion_to_questions_context",
                    "anifold_z": [0.1, 0.2],
                    "z_uncertainty_diag": [0.3],
                }
            )

        with self.assertRaises(ValueError):
            normalize_outcome_payload(
                {
                    "decision_id": "d1",
                    "z_before": [0.1, 0.2],
                    "z_after": [0.2, 0.3],
                    "uncertainty_before_diag": [0.2],
                    "uncertainty_after_diag": [0.1, 0.2],
                }
            )

    def test_context_normalization_and_request_builder_harden_string_flags(self) -> None:
        context = normalize_context_payload(
            {
                "contract_name": CONTRACT_FUSION_TO_QUESTIONS_CONTEXT,
                "schema_version": "1.0",
                "safety_trigger_active": "false",
                "allow_context_batches": "false",
            }
        )

        client = AnifoldAdapterClient(AnifoldAdapterConfig(base_url="http://127.0.0.1:9999"))
        request_payload = client.build_daily_select_request(
            day="2026-02-16",
            context_payload=context,
            include_explanations="false",
            allow_context_batches="false",
        )

        self.assertFalse(context["safety_trigger_active"])
        self.assertFalse(context["allow_context_batches"])
        self.assertFalse(request_payload["allow_context_batches"])
        self.assertFalse(request_payload["include_explanations"])


if __name__ == "__main__":
    unittest.main()
