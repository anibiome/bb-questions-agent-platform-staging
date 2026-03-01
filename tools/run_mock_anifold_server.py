from __future__ import annotations

import argparse
import json
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from questions_agent_platform.contracts import (
    CONTRACT_FUSION_TO_QUESTIONS_CONTEXT,
    CONTRACT_FUSION_TO_QUESTIONS_OUTCOME,
    CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE,
    canonical_contract_version,
    validate_projection_payload_contract,
)
from questions_agent_platform.pipeline.anifold_adapter import normalize_outcome_payload


def build_mock_server(host: str = "127.0.0.1", port: int = 8091) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, int(port)), _make_handler())
    server.evidence_events = []  # type: ignore[attr-defined]
    server.outcome_events = []  # type: ignore[attr-defined]
    return server


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mock AniFold integration server for Questions Agent contract testing.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args(argv)

    server = build_mock_server(host=args.host, port=args.port)
    host, port = server.server_address
    print(f"Mock AniFold server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            try:
                if path == "/health":
                    self._json(HTTPStatus.OK, {"ok": True, "service": "mock_anifold"})
                    return
                if path == "/v1/context/questions":
                    user_id = str((qs.get("user_id") or [""])[0]).strip() or "unknown-user"
                    day = str((qs.get("date") or [date.today().isoformat()])[0]).strip() or date.today().isoformat()
                    identity_mask_id = str((qs.get("identity_mask_id") or ["default"])[0]).strip() or "default"
                    self._json(HTTPStatus.OK, {"context": _build_context(user_id=user_id, day=day, identity_mask_id=identity_mask_id)})
                    return
                if path == "/v1/logs":
                    evidence = getattr(self.server, "evidence_events", [])
                    outcomes = getattr(self.server, "outcome_events", [])
                    self._json(
                        HTTPStatus.OK,
                        {
                            "evidence_count": len(evidence),
                            "outcome_count": len(outcomes),
                            "evidence_events": evidence,
                            "outcome_events": outcomes,
                        },
                    )
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._json_body()
                if path == "/v1/evidence/questions":
                    self._handle_evidence(body)
                    return
                if path == "/v1/outcomes/questions":
                    self._handle_outcome(body)
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def _handle_evidence(self, body: Dict[str, Any]) -> None:
            user_id = str(body.get("user_id") or "").strip()
            evidence = body.get("evidence")
            if not user_id:
                raise ValueError("user_id is required")
            if not isinstance(evidence, dict):
                raise ValueError("evidence object is required")
            validate_projection_payload_contract(evidence, vector_dim=128)
            contract = evidence.get("contract") or {}
            if str(contract.get("name") or "") != CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE:
                raise ValueError("invalid evidence contract name")
            event = {
                "user_id": user_id,
                "date": str(body.get("date") or ""),
                "session_id": body.get("session_id"),
                "decision_id": body.get("decision_id"),
                "schema_version": canonical_contract_version(contract.get("schema_version")),
            }
            getattr(self.server, "evidence_events").append(event)
            self._json(
                HTTPStatus.OK,
                {
                    "status": "accepted",
                    "contract_name": CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE,
                    "evidence_event_id": f"evidence-{len(getattr(self.server, 'evidence_events'))}",
                },
            )

        def _handle_outcome(self, body: Dict[str, Any]) -> None:
            user_id = str(body.get("user_id") or "").strip()
            outcome = body.get("outcome")
            if not user_id:
                raise ValueError("user_id is required")
            if not isinstance(outcome, dict):
                raise ValueError("outcome object is required")
            normalized = normalize_outcome_payload(outcome)
            event = {
                "user_id": user_id,
                "decision_id": normalized["decision_id"],
                "schema_version": normalized["schema_version"],
                "contract_name": normalized["contract_name"],
            }
            if str(normalized["contract_name"]) != CONTRACT_FUSION_TO_QUESTIONS_OUTCOME:
                raise ValueError("invalid outcome contract name")
            getattr(self.server, "outcome_events").append(event)
            self._json(
                HTTPStatus.OK,
                {
                    "status": "accepted",
                    "contract_name": CONTRACT_FUSION_TO_QUESTIONS_OUTCOME,
                    "outcome_event_id": f"outcome-{len(getattr(self.server, 'outcome_events'))}",
                },
            )

        def _json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            if not raw.strip():
                return {}
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            return payload

        def _json(self, code: HTTPStatus, payload: Dict[str, Any]) -> None:
            blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(code))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, format: str, *args):  # noqa: A003
            return

    return Handler


def _build_context(*, user_id: str, day: str, identity_mask_id: str) -> Dict[str, Any]:
    seed = float((sum(ord(ch) for ch in user_id) + len(identity_mask_id) * 17) % 97) / 100.0
    dscale = float((sum(ord(ch) for ch in day) % 31) + 1) / 100.0
    z = [round(seed - 0.20, 6), round(0.10 + dscale, 6), round(-0.05 + (seed / 2.0), 6)]
    uncertainty = [0.35, 0.30, 0.28]
    velocity = [round((z[0] - 0.01), 6), round((z[1] - 0.02), 6), round((z[2] + 0.01), 6)]
    return {
        "contract_name": CONTRACT_FUSION_TO_QUESTIONS_CONTEXT,
        "schema_version": canonical_contract_version("1.0"),
        "identity_mask_id": str(identity_mask_id),
        "anifold_z": z,
        "z_uncertainty_diag": uncertainty,
        "z_velocity": velocity,
        "z_distance_to_attractor": round(abs(z[0]) + abs(z[1]) + abs(z[2]), 6),
        "completion_rate_7d": 0.71,
        "completion_rate_14d": 0.74,
        "completion_rate_30d": 0.79,
        "burden_ms_median_14d": 2100.0,
        "safety_trigger_active": False,
        "allow_context_batches": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
