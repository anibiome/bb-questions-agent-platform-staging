from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from questions_agent_platform.policy.adapter import adapt_candidate_set
from questions_agent_platform.policy.bandit import PolicyRuntimeConfig, make_policy_decision
from questions_agent_platform.policy.features import build_feature_mapping_v1
from questions_agent_platform.policy.logging import decision_row
from questions_agent_platform.policy.registry import get_active_policy_version, load_policy_params, make_default_policy_params
from questions_agent_platform.policy.types import CandidateSet, PolicyContext, PolicyDecision


@dataclass(frozen=True)
class PolicyExecutionResult:
    decision: PolicyDecision
    decision_row_payload: Dict[str, Any]
    summary: Dict[str, Any]
    served_core_item_ids: Tuple[str, ...]
    candidate_set: CandidateSet


def build_policy_context(*, day: date, context_obj: Mapping[str, Any], identity_mask_id: Optional[str]) -> PolicyContext:
    return PolicyContext(
        anifold_z=context_obj.get("anifold_z"),
        z_uncertainty_diag=context_obj.get("z_uncertainty_diag"),
        z_velocity=context_obj.get("z_velocity"),
        z_distance_to_attractor=context_obj.get("z_distance_to_attractor"),
        completion_rate_7d=context_obj.get("completion_rate_7d"),
        completion_rate_14d=context_obj.get("completion_rate_14d"),
        completion_rate_30d=context_obj.get("completion_rate_30d"),
        burden_ms_median_14d=context_obj.get("burden_ms_median_14d"),
        day_of_week=int(day.weekday()),
        safety_trigger_active=bool(context_obj.get("safety_trigger_active", False)),
        allow_context_batches=bool(context_obj.get("allow_context_batches", True)),
        identity_mask_id=str(identity_mask_id) if identity_mask_id else None,
    )


def execute_policy_selection(
    *,
    pipeline_candidate_set: Any,
    user_id: str,
    day: date,
    deterministic_baseline_selected: Sequence[str],
    selection_mode: str,
    runtime_mode: str,
    identity_mask_id: Optional[str],
    context_obj: Mapping[str, Any],
    policy_root: str,
    policy_default_version: str,
    epsilon_explore: float,
    include_context_snapshot: bool,
    include_candidate_set_snapshot: bool,
) -> PolicyExecutionResult:
    policy_candidate_set = adapt_candidate_set(
        pipeline_candidate_set=pipeline_candidate_set,
        user_id=user_id,
        day=day,
        deterministic_baseline_selected=tuple(str(i) for i in deterministic_baseline_selected),
    )

    mapping = build_feature_mapping_v1()
    policy_root_str = str(policy_root or "").strip()
    if policy_root_str:
        policy_version = get_active_policy_version(policy_root_str) or str(policy_default_version)
        try:
            params = load_policy_params(policy_root_str, policy_version)
        except Exception:
            params = make_default_policy_params(policy_version=str(policy_version), mapping=mapping, lambda_reg=1.0)
    else:
        policy_version = str(policy_default_version)
        params = make_default_policy_params(policy_version=str(policy_version), mapping=mapping, lambda_reg=1.0)

    ctx = build_policy_context(day=day, context_obj=context_obj, identity_mask_id=identity_mask_id)
    decision = make_policy_decision(
        candidate_set=policy_candidate_set,
        context=ctx,
        params=params,
        mapping=mapping,
        mode=str(runtime_mode),
        runtime=PolicyRuntimeConfig(
            epsilon_explore=float(epsilon_explore),
            max_counterfactuals=10,
        ),
    )

    row_payload = decision_row(
        decision=decision,
        selection_mode=selection_mode,
        identity_mask_id=str(identity_mask_id) if identity_mask_id else None,
        context=ctx,
        candidate_set=policy_candidate_set,
        include_context=bool(include_context_snapshot),
        include_candidate_set=bool(include_candidate_set_snapshot),
    )

    summary = {
        "decision_id": decision.decision_id,
        "policy_version": decision.policy_version,
        "feature_version": decision.feature_version,
        "mode": decision.mode,
        "selected_item_ids": list(decision.selected_item_ids),
        "propensities": decision.propensities,
        "counterfactuals": [
            {
                "item_id": str(c.item_id),
                "policy_score": float(c.policy_score),
                "reason_codes": list(c.reason_codes),
            }
            for c in decision.counterfactuals
        ],
    }

    served_core_item_ids = (
        tuple(decision.selected_item_ids)
        if selection_mode == "policy_live"
        else tuple(str(i) for i in deterministic_baseline_selected)
    )

    return PolicyExecutionResult(
        decision=decision,
        decision_row_payload=row_payload,
        summary=summary,
        served_core_item_ids=served_core_item_ids,
        candidate_set=policy_candidate_set,
    )
