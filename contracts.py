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
QUESTION_EVIDENCE_CLASS = (
    "behavioral_or_derived_evidence_not_raw_biological_or_clinical_truth"
)
QUESTION_CLAIM_BOUNDARY = (
    "behavioral_question_runtime_not_biological_or_clinical_source_truth"
)
QUESTION_PROMOTION_GATES = (
    "questionnaire_registry_identity",
    "response_provenance",
    "runtime_bridge_identity",
    "backend_runtime_identity",
    "calibration_or_validation_trace",
    "evidence_class_on_exported_packets",
    "privacy_security_review",
    "clinical_or_protocol_review_for_action_claims",
    "explicit_promotion_approval",
)


def canonical_contract_version(raw: Optional[str]) -> str:
    if raw is None:
        return CANONICAL_CONTRACT_VERSION
    s = str(raw).strip().lower()
    if not s:
        return CANONICAL_CONTRACT_VERSION
    if s in CONTRACT_VERSION_ALIASES:
        return CONTRACT_VERSION_ALIASES[s]
    raise ValueError(f"Unsupported schema_version: {raw}")


def build_contract_header(
    *, name: str, schema_version: Optional[str] = None, producer: str
) -> Dict[str, Any]:
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


def build_behavioral_evidence_boundary() -> Dict[str, Any]:
    return {
        "claim_boundary": QUESTION_CLAIM_BOUNDARY,
        "evidence_class": QUESTION_EVIDENCE_CLASS,
        "allowed_use": [
            "behavioral_question_runtime",
            "adaptive_question_selection",
            "derived_questionnaire_evidence",
            "downstream_state_reconstruction_input",
        ],
        "blocked_claims": {
            "participant_source_truth": False,
            "raw_biological_truth": False,
            "raw_omics_truth": False,
            "clinical_diagnosis": False,
            "treatment_recommendation": False,
            "treatment_efficacy": False,
            "public_clinical_claim": False,
            "public_science_claim": False,
            "collaborator_proof": False,
            "investor_proof": False,
            "patent_ready_proof": False,
            "regulatory_clearance": False,
            "regulated_software": False,
        },
        "promotion_required_gates": list(QUESTION_PROMOTION_GATES),
    }


def validate_projection_payload_contract(
    payload: Dict[str, Any], *, vector_dim: int
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Projection payload must be an object")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Projection payload is missing contract metadata")
    name = str(contract.get("name") or "")
    if name != CONTRACT_QUESTIONS_TO_FUSION_EVIDENCE:
        raise ValueError(f"Unexpected contract name: {name}")
    canonical_contract_version(contract.get("schema_version"))

    if payload.get("evidence_class") != QUESTION_EVIDENCE_CLASS:
        raise ValueError("Projection payload missing behavioral evidence class")
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, dict):
        raise ValueError("Projection payload missing claim boundary")
    if claim_boundary.get("claim_boundary") != QUESTION_CLAIM_BOUNDARY:
        raise ValueError("Projection payload has unexpected claim boundary")
    if claim_boundary.get("evidence_class") != QUESTION_EVIDENCE_CLASS:
        raise ValueError(
            "Projection payload claim boundary has unexpected evidence class"
        )
    allowed_use = claim_boundary.get("allowed_use")
    if (
        not isinstance(allowed_use, list)
        or "behavioral_question_runtime" not in allowed_use
    ):
        raise ValueError(
            "Projection payload claim boundary missing allowed behavioral runtime use"
        )
    blocked_claims = claim_boundary.get("blocked_claims")
    if not isinstance(blocked_claims, dict):
        raise ValueError("Projection payload claim boundary missing blocked claims")
    for blocked_claim in (
        "raw_biological_truth",
        "raw_omics_truth",
        "clinical_diagnosis",
        "treatment_recommendation",
        "treatment_efficacy",
        "public_clinical_claim",
        "public_science_claim",
        "collaborator_proof",
        "investor_proof",
        "patent_ready_proof",
        "regulatory_clearance",
        "regulated_software",
    ):
        if blocked_claims.get(blocked_claim) is not False:
            raise ValueError(
                f"Projection payload missing blocked claim gate: {blocked_claim}"
            )
    promotion_gates = claim_boundary.get("promotion_required_gates")
    if not isinstance(promotion_gates, list) or not set(
        QUESTION_PROMOTION_GATES
    ).issubset(set(promotion_gates)):
        raise ValueError("Projection payload claim boundary missing promotion gates")

    mp = payload.get("modality_projections")
    if not isinstance(mp, dict):
        raise ValueError("Projection payload missing modality_projections")
    q = mp.get("questionnaires")
    if not isinstance(q, dict):
        raise ValueError("Projection payload missing questionnaires projection")

    projection = q.get("projection")
    uncertainty_diag = (
        ((q.get("uncertainty") or {}).get("diag"))
        if isinstance(q.get("uncertainty"), dict)
        else None
    )
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
