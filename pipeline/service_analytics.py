"""Analytics/reporting domain exports for pipeline service."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_domain_promotion_readiness",
    "build_tta_summary",
    "compute_experiment_results",
    "compute_user_fold",
    "compute_user_scale_progress",
    "get_circle_snapshots",
    "get_coherence_tier_contract",
    "get_drift_events",
    "get_drift_routing_contract",
    "get_ews_features",
    "get_experiment",
    "get_state_snapshots",
    "list_experiments",
    "suggest_emotion_deep_dive",
    "upsert_experiment",
    "upsert_policy_outcome_update",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import service as _facade

    return getattr(_facade, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
