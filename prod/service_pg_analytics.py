"""Analytics/reporting domain exports for PG service."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_clinical_audit_export",
    "build_domain_promotion_readiness",
    "build_slo_report",
    "build_tta_summary",
    "compute_experiment_results",
    "compute_user_scale_progress",
    "evaluate_policy_live_rollback_guard_pg",
    "format_score_for_user",
    "get_anamnesis_episode",
    "get_anamnesis_episodes",
    "get_circle_snapshots",
    "get_coherence_history",
    "get_coherence_tier_contract",
    "get_drift_events",
    "get_drift_routing_contract",
    "get_ews_features",
    "get_experiment",
    "get_state_snapshots",
    "list_experiments",
    "list_safety_events",
    "record_operational_metric",
    "resolve_anamnesis_episode",
    "resolve_safety_event",
    "resolve_site_config",
    "run_phi_retention",
    "suggest_emotion_deep_dive",
    "upsert_experiment",
    "upsert_policy_outcome_update",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import service_pg as _facade

    return getattr(_facade, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
