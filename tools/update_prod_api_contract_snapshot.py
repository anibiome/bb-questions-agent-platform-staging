#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PUBLIC_PREFIXES = ("/v1/",)
SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "prod_public_api_contract_snapshot.json"


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
        method_map: dict[str, Any] = {}
        for method in sorted(openapi["paths"][path]):
            op = openapi["paths"][path][method]
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


def main() -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(_build_snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
