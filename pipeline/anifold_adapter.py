from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from questions_agent_platform.contracts import (
    CONTRACT_FUSION_TO_QUESTIONS_CONTEXT,
    CONTRACT_FUSION_TO_QUESTIONS_OUTCOME,
    canonical_contract_version,
    validate_projection_payload_contract,
)


DEFAULT_CONTEXT_ENDPOINT = "/v1/context/questions"
DEFAULT_EVIDENCE_ENDPOINT = "/v1/evidence/questions"
DEFAULT_OUTCOME_ENDPOINT = "/v1/outcomes/questions"


DateLike = Union[date, str]


def _boolish(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"false", "0", "no", "n", "off", "none", "null"}:
        return False
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    return default


@dataclass(frozen=True)
class AnifoldAdapterConfig:
    base_url: str
    api_key: Optional[str] = None
    timeout_seconds: float = 15.0
    context_endpoint: str = DEFAULT_CONTEXT_ENDPOINT
    evidence_endpoint: str = DEFAULT_EVIDENCE_ENDPOINT
    outcome_endpoint: str = DEFAULT_OUTCOME_ENDPOINT


class AnifoldAdapterClient:
    def __init__(self, config: AnifoldAdapterConfig):
        self._cfg = config
        self._base_url = str(config.base_url).rstrip("/")

    def fetch_questions_context(
        self,
        *,
        user_id: str,
        day: DateLike,
        identity_mask_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {"user_id": str(user_id), "date": _day_iso(day)}
        if identity_mask_id:
            params["identity_mask_id"] = str(identity_mask_id)
        payload = self._request_json("GET", self._cfg.context_endpoint, query=params)
        return normalize_context_payload(payload)

    def build_daily_select_request(
        self,
        *,
        day: DateLike,
        context_payload: Mapping[str, Any],
        selection_mode: str = "policy_live",
        include_explanations: bool = True,
        allow_context_batches: bool = True,
        k_core: int = 5,
        identity_mask_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = normalize_context_payload(dict(context_payload))
        req: Dict[str, Any] = {
            "schema_version": canonical_contract_version(context.get("schema_version")),
            "date": _day_iso(day),
            "selection_mode": str(selection_mode),
            "k_core": int(k_core),
            "allow_context_batches": _boolish(allow_context_batches, default=True),
            "include_explanations": _boolish(include_explanations, default=True),
            "context": context,
        }
        if identity_mask_id:
            req["identity_mask_id"] = str(identity_mask_id)
        return req

    def submit_questions_evidence(
        self,
        *,
        user_id: str,
        day: DateLike,
        evidence_payload: Mapping[str, Any],
        session_id: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = dict(evidence_payload)
        validate_projection_payload_contract(payload, vector_dim=128)
        body = {
            "user_id": str(user_id),
            "date": _day_iso(day),
            "session_id": str(session_id) if session_id else None,
            "decision_id": str(decision_id) if decision_id else None,
            "evidence": payload,
        }
        return self._request_json("POST", self._cfg.evidence_endpoint, payload=body)

    def submit_questions_outcome(
        self,
        *,
        user_id: str,
        outcome_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        normalized = normalize_outcome_payload(dict(outcome_payload))
        body = {"user_id": str(user_id), "outcome": normalized}
        return self._request_json("POST", self._cfg.outcome_endpoint, payload=body)

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        query: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        path = str(endpoint).strip() or "/"
        if not path.startswith("/"):
            path = "/" + path
        url = self._base_url + path
        if query:
            items = [(k, str(v)) for k, v in query.items() if v is not None]
            url = f"{url}?{urllib.parse.urlencode(items)}"

        headers: Dict[str, str] = {}
        if self._cfg.api_key:
            headers["X-API-Key"] = str(self._cfg.api_key)

        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")

        request = urllib.request.Request(url=url, method=str(method).upper(), headers=headers, data=body)
        try:
            with urllib.request.urlopen(request, timeout=float(self._cfg.timeout_seconds)) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail: str
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"Anifold adapter HTTP {exc.code}: {detail}") from exc


def normalize_context_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    obj = dict(payload)
    if isinstance(obj.get("context"), dict):
        obj = dict(obj["context"])

    contract_name = str(obj.get("contract_name") or CONTRACT_FUSION_TO_QUESTIONS_CONTEXT).strip()
    if contract_name != CONTRACT_FUSION_TO_QUESTIONS_CONTEXT:
        raise ValueError(f"context.contract_name must be {CONTRACT_FUSION_TO_QUESTIONS_CONTEXT}")

    schema_version = canonical_contract_version(obj.get("schema_version"))
    z = _vector_or_none(obj.get("anifold_z"), field="anifold_z")
    u = _vector_or_none(obj.get("z_uncertainty_diag"), field="z_uncertainty_diag")
    v = _vector_or_none(obj.get("z_velocity"), field="z_velocity")
    if z is not None:
        if u is not None and len(u) != len(z):
            raise ValueError("z_uncertainty_diag length must match anifold_z length")
        if v is not None and len(v) != len(z):
            raise ValueError("z_velocity length must match anifold_z length")

    out: Dict[str, Any] = {
        "contract_name": contract_name,
        "schema_version": schema_version,
        "anifold_z": z,
        "z_uncertainty_diag": u,
        "z_velocity": v,
        "z_distance_to_attractor": _float_or_none(obj.get("z_distance_to_attractor")),
        "completion_rate_7d": _float_or_none(obj.get("completion_rate_7d")),
        "completion_rate_14d": _float_or_none(obj.get("completion_rate_14d")),
        "completion_rate_30d": _float_or_none(obj.get("completion_rate_30d")),
        "burden_ms_median_14d": _float_or_none(obj.get("burden_ms_median_14d")),
        "safety_trigger_active": _boolish(obj.get("safety_trigger_active", False), default=False),
        "allow_context_batches": _boolish(obj.get("allow_context_batches", True), default=True),
    }
    return out


def normalize_outcome_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    obj = dict(payload)
    if isinstance(obj.get("outcome"), dict):
        obj = dict(obj["outcome"])

    contract_name = str(obj.get("contract_name") or CONTRACT_FUSION_TO_QUESTIONS_OUTCOME).strip()
    if contract_name != CONTRACT_FUSION_TO_QUESTIONS_OUTCOME:
        raise ValueError(f"outcome.contract_name must be {CONTRACT_FUSION_TO_QUESTIONS_OUTCOME}")

    decision_id = str(obj.get("decision_id") or "").strip()
    if not decision_id:
        raise ValueError("outcome.decision_id is required")

    schema_version = canonical_contract_version(obj.get("schema_version"))
    z_before = _vector_or_none(obj.get("z_before"), field="z_before")
    z_after = _vector_or_none(obj.get("z_after"), field="z_after")
    u_before = _vector_or_none(obj.get("uncertainty_before_diag"), field="uncertainty_before_diag")
    u_after = _vector_or_none(obj.get("uncertainty_after_diag"), field="uncertainty_after_diag")

    if z_before is not None and z_after is not None and len(z_before) != len(z_after):
        raise ValueError("z_before and z_after must have equal length")
    if u_before is not None and u_after is not None and len(u_before) != len(u_after):
        raise ValueError("uncertainty_before_diag and uncertainty_after_diag must have equal length")
    if z_before is not None and u_before is not None and len(z_before) != len(u_before):
        raise ValueError("uncertainty_before_diag length must match z_before length")
    if z_after is not None and u_after is not None and len(z_after) != len(u_after):
        raise ValueError("uncertainty_after_diag length must match z_after length")

    return {
        "contract_name": contract_name,
        "schema_version": schema_version,
        "source_event_id": str(obj.get("source_event_id")) if obj.get("source_event_id") else None,
        "decision_id": decision_id,
        "z_before": z_before,
        "z_after": z_after,
        "uncertainty_before_diag": u_before,
        "uncertainty_after_diag": u_after,
        "reward_overrides": dict(obj.get("reward_overrides")) if isinstance(obj.get("reward_overrides"), dict) else None,
    }


def _day_iso(v: DateLike) -> str:
    if isinstance(v, date):
        return v.isoformat()
    s = str(v).strip()
    if not s:
        raise ValueError("date is required")
    return s


def _vector_or_none(value: Any, *, field: str) -> Optional[list[float]]:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a numeric vector")
    out: list[float] = []
    for idx, x in enumerate(value):
        try:
            out.append(float(x))
        except Exception as exc:
            raise ValueError(f"{field}[{idx}] is not numeric") from exc
    return out


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
