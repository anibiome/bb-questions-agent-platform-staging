from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from questions_agent_platform.pipeline.governance import (
    CARDIOMETABOLIC_DOMAIN,
    item_matches_domains,
    registry_item_domains,
)
from questions_agent_platform.pipeline.registry import Registry


SELECTION_MODE_DETERMINISTIC = "deterministic"
SELECTION_MODE_POLICY_LIVE = "policy_live"
SELECTION_MODE_POLICY_SHADOW = "policy_shadow"
SELECTION_MODE_POLICY_OFFLINE_REPLAY = "policy_offline_replay"

VALID_SELECTION_MODES = {
    SELECTION_MODE_DETERMINISTIC,
    SELECTION_MODE_POLICY_LIVE,
    SELECTION_MODE_POLICY_SHADOW,
    SELECTION_MODE_POLICY_OFFLINE_REPLAY,
}

POLICY_SELECTION_MODES = {
    SELECTION_MODE_POLICY_LIVE,
    SELECTION_MODE_POLICY_SHADOW,
    SELECTION_MODE_POLICY_OFFLINE_REPLAY,
}


@dataclass(frozen=True)
class OnboardingCardiometabolicGate:
    """Computed onboarding gating decision for the current daily selection run."""

    allowed_item_ids: Optional[Set[str]]
    priority_item_ids: Optional[Set[str]]
    mark_onboarding_complete: bool


def normalize_selection_mode(selection_mode: Optional[str], *, default: str = SELECTION_MODE_DETERMINISTIC) -> str:
    """Normalize and validate API selection mode."""

    mode = str(selection_mode or default).strip().lower()
    if mode not in VALID_SELECTION_MODES:
        allowed = ", ".join(sorted(VALID_SELECTION_MODES))
        raise ValueError(f"Unknown selection_mode: {mode}. Allowed: {allowed}")
    return mode


def policy_runtime_mode(selection_mode: str) -> str:
    """Map API selection mode to persisted policy runtime mode."""

    if selection_mode == SELECTION_MODE_POLICY_LIVE:
        return "live"
    if selection_mode == SELECTION_MODE_POLICY_SHADOW:
        return "shadow"
    if selection_mode == SELECTION_MODE_POLICY_OFFLINE_REPLAY:
        return "replay"
    raise ValueError(f"selection_mode is not policy-enabled: {selection_mode}")


def filter_extra_batches(
    *,
    extra_batches: Sequence[Sequence[str]],
    core_item_ids: Sequence[str],
    declined_item_ids: Set[str],
    allowed_item_ids: Optional[Set[str]],
) -> Tuple[Tuple[str, ...], ...]:
    """Filter extra batches against core, declined, and domain/onboarding constraints."""

    already_core = set(str(i) for i in core_item_ids)
    filtered: list[Tuple[str, ...]] = []
    for batch in extra_batches:
        batch_items: list[str] = []
        for item_id in batch:
            item_id_s = str(item_id)
            if item_id_s in already_core:
                continue
            if item_id_s in declined_item_ids:
                continue
            if allowed_item_ids is not None and item_id_s not in allowed_item_ids:
                continue
            batch_items.append(item_id_s)
        if batch_items:
            filtered.append(tuple(batch_items))
    return tuple(filtered)


def evaluate_onboarding_cardiometabolic_gate(
    *,
    onboarding_complete: bool,
    registry: Registry,
    declined_item_ids: Set[str],
    answered_item_ids: Set[str],
) -> OnboardingCardiometabolicGate:
    """Evaluate cardiometabolic-first onboarding gating without persistence side effects."""

    if onboarding_complete:
        return OnboardingCardiometabolicGate(
            allowed_item_ids=None,
            priority_item_ids=None,
            mark_onboarding_complete=False,
        )

    cardiometabolic_item_ids = {
        item.id for item in registry.items.values() if CARDIOMETABOLIC_DOMAIN in set(item.tags)
    }
    if not cardiometabolic_item_ids:
        return OnboardingCardiometabolicGate(
            allowed_item_ids=None,
            priority_item_ids=None,
            mark_onboarding_complete=True,
        )

    pending = {
        item_id
        for item_id in cardiometabolic_item_ids
        if item_id not in answered_item_ids and item_id not in declined_item_ids
    }
    if pending:
        return OnboardingCardiometabolicGate(
            allowed_item_ids={item_id for item_id in cardiometabolic_item_ids if item_id not in declined_item_ids},
            priority_item_ids=set(pending),
            mark_onboarding_complete=False,
        )

    return OnboardingCardiometabolicGate(
        allowed_item_ids=None,
        priority_item_ids=None,
        mark_onboarding_complete=True,
    )


def allowed_item_ids_for_domains(*, registry: Registry, allowed_domains: Set[str]) -> Set[str]:
    """Return registry item ids allowed under current domain-governance constraints."""

    if not allowed_domains:
        return set(registry.items.keys())
    item_domains = registry_item_domains(registry)
    return {
        item_id
        for item_id in registry.items.keys()
        if item_matches_domains(item_domains, item_id, allowed_domains)
    }


def merge_allowed_item_ids(
    *,
    current_allowed_item_ids: Optional[Set[str]],
    domain_allowed_item_ids: Set[str],
) -> Set[str]:
    """Merge onboarding allow-list with domain allow-list using set intersection semantics."""

    if current_allowed_item_ids is None:
        return set(domain_allowed_item_ids)
    return set(current_allowed_item_ids) & set(domain_allowed_item_ids)


def build_selection_explain_payload(
    *,
    reasons_by_item_id: Mapping[str, Sequence[str]],
    primary_scale_by_item_id: Mapping[str, str],
    near_unlock_scales: Sequence[str],
    due_retest_scales: Sequence[str],
    drift_scales: Sequence[str],
    follow_up_item_ids: Set[str],
    profile: Mapping[str, Any],
    allowed_domains: Set[str],
    requested_selection_mode: str,
    selection_mode: str,
    policy_summary: Optional[Dict[str, Any]],
    deterministic_baseline_core_item_ids: Sequence[str],
    served_core_item_ids: Sequence[str],
    rollback_guard: Optional[Dict[str, Any]],
    safety_mode_active: bool,
    open_safety_events: Sequence[Dict[str, Any]],
    timeframe: str,
) -> Dict[str, Any]:
    """Build a stable explainability payload shape for both SQLite and Postgres runtimes."""

    return {
        "reasons_by_item_id": {k: list(v) for k, v in reasons_by_item_id.items()},
        "primary_scale_by_item_id": dict(primary_scale_by_item_id),
        "near_unlock_scales": list(near_unlock_scales),
        "due_retest_scales": list(due_retest_scales),
        "drift_scales": list(drift_scales),
        "follow_up_items_due": sorted(follow_up_item_ids),
        "profile_mode": str(profile.get("mode") or "consumer"),
        "active_domains": list(profile.get("active_domains") or []),
        "queued_domains": list(profile.get("queued_domains") or []),
        "allowed_domains": sorted(list(allowed_domains)),
        "requested_selection_mode": requested_selection_mode,
        "selection_mode": selection_mode,
        "policy": policy_summary,
        "deterministic_baseline_core_item_ids": list(deterministic_baseline_core_item_ids),
        "served_core_item_ids": list(served_core_item_ids),
        "rollback_guard": rollback_guard,
        "safety_mode_active": bool(safety_mode_active),
        "safety_events_open": list(open_safety_events[:3]),
        "gamification_paused": bool(safety_mode_active),
        "timeframe": str(timeframe),
    }
