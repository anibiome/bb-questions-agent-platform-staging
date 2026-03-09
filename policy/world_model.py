from __future__ import annotations

from typing import Any, Dict

from questions_agent_platform.policy.types import CandidateItem, CandidateSet, PolicyContext, PolicyDecision


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif value in (None, ""):
        raw_items = []
    else:
        raw_items = [value]
    normalized: list[str] = []
    for item in raw_items:
        token = str(item).strip()
        if token:
            normalized.append(token)
    return normalized


def response_validity_snapshot(context: PolicyContext) -> Dict[str, Any]:
    raw_score = context.response_validity_score
    score = _safe_float(raw_score, default=None) if raw_score not in (None, "") else None
    tier = str(context.response_validity_tier or "").strip().lower()
    if score is not None and not tier:
        if score >= 0.85:
            tier = "high"
        elif score >= 0.6:
            tier = "medium"
        else:
            tier = "low"
    return {
        "score": round(float(score), 4) if score is not None else None,
        "tier": tier or None,
    }


def build_measurement_actions(
    *,
    decision: PolicyDecision,
    candidate_set: CandidateSet,
    context: PolicyContext,
) -> list[Dict[str, Any]]:
    candidate_lookup = {candidate.item_id: candidate for candidate in candidate_set.candidates}
    explanation_lookup = {explanation.item_id: explanation for explanation in decision.explanations}
    max_policy_score = max((_safe_float(explanation.policy_score) for explanation in decision.explanations), default=1.0)
    max_det_score = max((_safe_float(candidate.deterministic_score) for candidate in candidate_set.candidates), default=1.0)
    response_validity = response_validity_snapshot(context)

    actions: list[Dict[str, Any]] = []
    for rank, item_id in enumerate(decision.selected_item_ids, start=1):
        candidate = candidate_lookup.get(item_id)
        explanation = explanation_lookup.get(item_id)
        if candidate is None:
            continue
        actions.append(
            {
                "action_id": f"qa_action:{decision.decision_id}:{item_id}",
                "action_type": "measurement",
                "status": "selected",
                "source_system": "questions-agent-platform",
                "label": f"Ask item {item_id}",
                "description": _describe_candidate(candidate),
                "intent": _infer_intent(candidate),
                "target_signal_id": f"question_item:{item_id}",
                "targeted_modalities": ["psychometric"],
                "expected_information_gain": _estimate_information_gain(
                    candidate=candidate,
                    explanation=explanation,
                    max_policy_score=max_policy_score,
                    max_det_score=max_det_score,
                ),
                "expected_biological_coherence_delta": None,
                "expected_risk_deltas": {},
                "parameters": {
                    "decision_id": decision.decision_id,
                    "selection_rank": rank,
                    "item_id": item_id,
                    "item_type": candidate.item_type,
                    "scale_ids": _normalized_string_list(candidate.scale_ids),
                    "constraint_tags": _normalized_string_list(candidate.constraint_tags),
                    "reason_codes": _normalized_string_list(candidate.reason_codes),
                    "expected_burden": candidate.features.get("expected_burden"),
                    "sensitivity": candidate.features.get("sensitivity"),
                    "multiplex_count": candidate.features.get("multiplex_count"),
                },
                "provenance": {
                    "policy_version": decision.policy_version,
                    "feature_version": decision.feature_version,
                    "mode": decision.mode,
                    "identity_mask_id": context.identity_mask_id,
                    "response_validity": response_validity,
                    "biological_state_context": {
                        "biological_coherence_score": context.biological_coherence_score,
                        "biological_decoherence_radius": context.biological_decoherence_radius,
                    },
                },
            }
        )
    return actions


def _infer_intent(candidate: CandidateItem) -> str:
    tags = {str(tag).strip().lower() for tag in candidate.constraint_tags}
    reasons = {str(reason).strip().lower() for reason in candidate.reason_codes}
    if "mandatory" in tags:
        return "stabilize"
    if any(reason.startswith("near_unlock:") for reason in reasons):
        return "disambiguate"
    if any(reason.startswith("drift") or "follow" in reason for reason in reasons):
        return "repair"
    return "observe"


def _describe_candidate(candidate: CandidateItem) -> str:
    scale_ids = _normalized_string_list(candidate.scale_ids)
    scale_part = ", ".join(scale_ids) if scale_ids else "unscoped"
    return f"{candidate.item_type} targeting {scale_part}"


def _estimate_information_gain(
    *,
    candidate: CandidateItem,
    explanation: Any,
    max_policy_score: float,
    max_det_score: float,
) -> float:
    det_component = _safe_float(candidate.deterministic_score) / max(max_det_score, 1.0)
    policy_component = (
        _safe_float(getattr(explanation, "policy_score", 0.0)) / max(max_policy_score, 1.0)
        if explanation is not None
        else det_component
    )
    burden = _safe_float(candidate.features.get("expected_burden"), 1.0)
    multiplex = _safe_float(candidate.features.get("multiplex_count"), 1.0)
    novelty_days = _safe_float(candidate.features.get("novelty_days"), 0.0)
    burden_penalty = min(0.35, max(0.0, (burden - 1.0) * 0.08))
    multiplex_bonus = min(0.2, max(0.0, (multiplex - 1.0) * 0.05))
    novelty_bonus = min(0.15, max(0.0, novelty_days / 60.0))
    score = (0.55 * det_component) + (0.35 * policy_component) + multiplex_bonus + novelty_bonus - burden_penalty
    return round(max(0.0, min(1.0, score)), 4)
