from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date
from typing import Any, Dict, Optional, Sequence

from questions_agent_platform.policy.types import CandidateSet, PolicyContext, PolicyDecision


def serialize_context(context: PolicyContext, *, quantize: int = 3) -> Dict[str, Any]:
    """
    Context snapshot for audit / reproducible training.

    Privacy:
    - Quantizes vectors by default
    - Does not store raw answers
    """
    q = int(max(0, quantize))

    def qv(v: Optional[Sequence[float]]) -> Optional[list]:
        if v is None:
            return None
        if q <= 0:
            return [float(x) for x in v]
        return [round(float(x), q) for x in v]

    return {
        "anifold_z": qv(context.anifold_z),
        "z_uncertainty_diag": qv(context.z_uncertainty_diag),
        "z_velocity": qv(context.z_velocity),
        "z_distance_to_attractor": round(float(context.z_distance_to_attractor or 0.0), q) if context.z_distance_to_attractor is not None else None,
        "completion_rate_7d": round(float(context.completion_rate_7d or 0.0), 4) if context.completion_rate_7d is not None else None,
        "completion_rate_14d": round(float(context.completion_rate_14d or 0.0), 4) if context.completion_rate_14d is not None else None,
        "completion_rate_30d": round(float(context.completion_rate_30d or 0.0), 4) if context.completion_rate_30d is not None else None,
        "burden_ms_median_14d": round(float(context.burden_ms_median_14d or 0.0), 2) if context.burden_ms_median_14d is not None else None,
        "response_validity_score": round(float(context.response_validity_score or 0.0), 4) if context.response_validity_score is not None else None,
        "response_validity_tier": str(context.response_validity_tier) if context.response_validity_tier else None,
        "biological_coherence_score": round(float(context.biological_coherence_score or 0.0), 4) if context.biological_coherence_score is not None else None,
        "biological_decoherence_radius": round(float(context.biological_decoherence_radius or 0.0), 4) if context.biological_decoherence_radius is not None else None,
        "day_of_week": int(context.day_of_week) if context.day_of_week is not None else None,
        "safety_trigger_active": bool(context.safety_trigger_active),
        "allow_context_batches": bool(context.allow_context_batches),
        "identity_mask_id": str(context.identity_mask_id) if context.identity_mask_id else None,
    }


def serialize_candidate_set(candidate_set: CandidateSet) -> Dict[str, Any]:
    return {
        "user_id": candidate_set.user_id,
        "day": candidate_set.day.isoformat(),
        "k_core": int(candidate_set.k_core),
        "mandatory_item_ids": list(candidate_set.mandatory_item_ids),
        "deterministic_baseline_selected": list(candidate_set.deterministic_baseline_selected),
        "candidates": [
            {
                "item_id": c.item_id,
                "item_type": c.item_type,
                "scale_ids": list(c.scale_ids),
                "deterministic_score": float(c.deterministic_score),
                "constraint_tags": list(c.constraint_tags),
                "reason_codes": list(c.reason_codes),
                "features": dict(c.features),
            }
            for c in candidate_set.candidates
        ],
    }


def decision_row(
    *,
    decision: PolicyDecision,
    selection_mode: str,
    identity_mask_id: Optional[str],
    context: Optional[PolicyContext] = None,
    candidate_set: Optional[CandidateSet] = None,
    world_model: Optional[Dict[str, Any]] = None,
    include_context: bool = False,
    include_candidate_set: bool = False,
) -> Dict[str, Any]:
    """
    Row for policy_decisions table (DB-agnostic dict).
    """
    row: Dict[str, Any] = {
        "decision_id": decision.decision_id,
        "user_id": decision.user_id,
        "date": decision.day.isoformat(),
        "identity_mask_id": identity_mask_id,
        "selection_mode": str(selection_mode),
        "mode": decision.mode,
        "policy_version": decision.policy_version,
        "feature_version": decision.feature_version,
        "context_hash": decision.context_hash,
        "candidate_set_hash": decision.candidate_set_hash,
        "selected_item_ids_json": json.dumps(list(decision.selected_item_ids), ensure_ascii=False),
        "deterministic_baseline_selected_json": json.dumps(list(decision.deterministic_baseline_selected), ensure_ascii=False),
        "propensities_json": json.dumps(decision.propensities, ensure_ascii=False) if decision.propensities is not None else None,
        "explanations_json": json.dumps([_as_json(e) for e in decision.explanations], ensure_ascii=False),
        "counterfactuals_json": json.dumps([_as_json(c) for c in decision.counterfactuals], ensure_ascii=False),
    }
    if include_context and context is not None:
        context_payload = serialize_context(context)
        if world_model:
            context_payload["world_model"] = world_model
        row["context_json"] = json.dumps(context_payload, ensure_ascii=False)
    else:
        row["context_json"] = None
    if include_candidate_set and candidate_set is not None:
        row["candidate_set_json"] = json.dumps(serialize_candidate_set(candidate_set), ensure_ascii=False)
    else:
        row["candidate_set_json"] = None
    return row


def _as_json(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj
