from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Sequence


CANONICAL_CONTRACT_VERSION = "1.0"
CONTRACT_VERSION_ALIASES = {
    "1": CANONICAL_CONTRACT_VERSION,
    "1.0": CANONICAL_CONTRACT_VERSION,
    "1.0.0": CANONICAL_CONTRACT_VERSION,
    "v1": CANONICAL_CONTRACT_VERSION,
    "v1.0": CANONICAL_CONTRACT_VERSION,
}

CONTRACT_FUSION_TO_QUESTIONS_CONTEXT = "fusion_to_questions_context"
CONTRACT_FUSION_TO_QUESTIONS_OUTCOME = "fusion_to_questions_outcome"
CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE = "questions_to_fusion_evidence"


def canonical_contract_version(raw: Optional[str]) -> str:
    if raw is None:
        return CANONICAL_CONTRACT_VERSION
    s = str(raw).strip().lower()
    if not s:
        return CANONICAL_CONTRACT_VERSION
    if s in CONTRACT_VERSION_ALIASES:
        return CONTRACT_VERSION_ALIASES[s]
    raise ValueError(f"Unsupported schema_version: {raw}")


def build_contract_header(*, name: str, schema_version: Optional[str] = None, producer: str) -> Dict[str, Any]:
    v = canonical_contract_version(schema_version)
    return {
        "name": str(name),
        "schema_version": v,
        "compatibility": [CANONICAL_CONTRACT_VERSION],
        "producer": str(producer),
    }


def build_integration_anchor(
    *,
    user_id: str,
    day: date,
    session_id: Optional[str],
    decision_id: Optional[str],
) -> Dict[str, Any]:
    return {
        "subject_id": str(user_id),
        "date": day.isoformat(),
        "session_id": str(session_id) if session_id else None,
        "decision_id": str(decision_id) if decision_id else None,
    }


def validate_projection_payload_contract(payload: Dict[str, Any], *, vector_dim: int) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Projection payload must be an object")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Projection payload is missing contract metadata")
    name = str(contract.get("name") or "")
    if name != CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE:
        raise ValueError(f"Unexpected contract name: {name}")
    canonical_contract_version(contract.get("schema_version"))

    mp = payload.get("modality_projections")
    if not isinstance(mp, dict):
        raise ValueError("Projection payload missing modality_projections")
    q = mp.get("questionnaires")
    if not isinstance(q, dict):
        raise ValueError("Projection payload missing questionnaires projection")

    projection = q.get("projection")
    uncertainty_diag = ((q.get("uncertainty") or {}).get("diag")) if isinstance(q.get("uncertainty"), dict) else None
    velocity = q.get("velocity")
    attractor = q.get("attractor_candidate")
    _assert_vector_len(projection, vector_dim, "projection")
    _assert_vector_len(uncertainty_diag, vector_dim, "uncertainty.diag")
    _assert_vector_len(velocity, vector_dim, "velocity")
    _assert_vector_len(attractor, vector_dim, "attractor_candidate")


def _assert_vector_len(v: Any, expected: int, field: str) -> None:
    if not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
        raise ValueError(f"{field} must be a numeric vector")
    if len(v) != int(expected):
        raise ValueError(f"{field} length mismatch: expected {expected}, got {len(v)}")
    for idx, x in enumerate(v):
        try:
            float(x)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"{field}[{idx}] is not numeric") from exc
