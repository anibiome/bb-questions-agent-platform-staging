from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PUBLIC_PREFIXES = ("/v1/",)
SNAPSHOT_PATH = Path(__file__).resolve().parent / "contracts" / "prod_public_api_contract_snapshot.json"
KNOWN_RESPONSE_SCHEMA_GAPS = {
    ("get", "/v1/admin/audit/export"),
    ("get", "/v1/admin/domains/readiness"),
    ("post", "/v1/admin/policy/activate/{version}"),
    ("get", "/v1/admin/policy/metrics"),
    ("get", "/v1/admin/policy/versions"),
    ("post", "/v1/admin/privacy/retention/run"),
    ("post", "/v1/admin/registry/activate/{version}"),
    ("post", "/v1/admin/registry/upload"),
    ("get", "/v1/admin/registry/versions"),
    ("get", "/v1/admin/slo"),
    ("get", "/v1/coherence/tier-contract"),
    ("get", "/v1/drift-routing-contract"),
    ("post", "/v1/users/{user_id}/anamnesis/episodes/{episode_id}/resolve"),
    ("get", "/v1/users/{user_id}/ani/summary"),
    ("post", "/v1/users/{user_id}/calibrations/{scale_id}"),
    ("post", "/v1/users/{user_id}/cardio-risk/compute"),
    ("get", "/v1/users/{user_id}/circle"),
    ("get", "/v1/users/{user_id}/drift-events"),
    ("get", "/v1/users/{user_id}/emotion/deep_dive"),
    ("get", "/v1/users/{user_id}/ews"),
    ("get", "/v1/users/{user_id}/experiments"),
    ("post", "/v1/users/{user_id}/experiments"),
    ("get", "/v1/users/{user_id}/experiments/{experiment_id}"),
    ("get", "/v1/users/{user_id}/experiments/{experiment_id}/results"),
    ("get", "/v1/users/{user_id}/follow-up-queue"),
    ("post", "/v1/users/{user_id}/policy/outcomes"),
    ("get", "/v1/users/{user_id}/profile"),
    ("patch", "/v1/users/{user_id}/profile"),
    ("get", "/v1/users/{user_id}/progress"),
    ("get", "/v1/users/{user_id}/projection/questions"),
    ("post", "/v1/users/{user_id}/request-more-context"),
    ("get", "/v1/users/{user_id}/safety-events"),
    ("post", "/v1/users/{user_id}/safety-events/{event_id}/resolve"),
    ("get", "/v1/users/{user_id}/scales"),
    ("get", "/v1/users/{user_id}/scales/{scale_id}/history"),
    ("get", "/v1/users/{user_id}/sessions/{session_id}/coherence"),
    ("get", "/v1/users/{user_id}/state"),
}


def _load_prod_app():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    os.environ.setdefault("REGISTRY_ROOT", str(root / "data" / "registry_demo"))
    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("POLICY_ROOT", str(root / "data" / "policy_demo"))

    from questions_agent_platform.prod.app import app

    return app


def _is_public_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


def _schema_ref_or_inline(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {}
    if "$ref" in schema:
        return {"$ref": str(schema["$ref"])}
    keep_keys = {"type", "properties", "required", "items", "oneOf", "anyOf", "allOf", "enum", "format", "nullable"}
    return {k: v for k, v in schema.items() if k in keep_keys}


def _build_snapshot() -> dict[str, Any]:
    app = _load_prod_app()
    openapi = app.openapi()
    out: dict[str, Any] = {}

    for path in sorted(openapi.get("paths", {})):
        if not _is_public_path(path):
            continue
        methods = openapi["paths"][path]
        method_map: dict[str, Any] = {}
        for method in sorted(methods):
            op = methods[method]
            request_schema = (
                op.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            response_schema = (
                op.get("responses", {})
                .get("200", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            method_map[method] = {
                "operation_id": op.get("operationId"),
                "request_schema": _schema_ref_or_inline(request_schema),
                "response_schema": _schema_ref_or_inline(response_schema),
            }
        out[path] = method_map

    return {
        "title": openapi.get("info", {}).get("title"),
        "version": openapi.get("info", {}).get("version"),
        "public_paths": out,
    }


def test_public_endpoints_have_request_response_schema_contracts() -> None:
    app = _load_prod_app()
    openapi = app.openapi()
    snapshot = _build_snapshot()
    actual_response_gaps: set[tuple[str, str]] = set()

    for path, methods in snapshot["public_paths"].items():
        for method, contract in methods.items():
            if contract["response_schema"] == {}:
                actual_response_gaps.add((method, path))

            operation = openapi["paths"][path][method]
            has_request_body = bool(
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json")
            )
            if has_request_body:
                assert contract["request_schema"] != {}, f"{method.upper()} {path} missing request body schema"

    unexpected_gaps = actual_response_gaps - KNOWN_RESPONSE_SCHEMA_GAPS
    assert not unexpected_gaps, f"New endpoints missing response schema contract: {sorted(unexpected_gaps)}"


def test_public_api_snapshot_backwards_compatibility() -> None:
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    current = _build_snapshot()
    assert current == expected, (
        "Public API contract snapshot changed. "
        "If intentional, run: python tools/update_prod_api_contract_snapshot.py"
    )
