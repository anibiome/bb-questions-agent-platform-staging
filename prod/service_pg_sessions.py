"""Session/question flow domain exports for PG service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .service_pg import (
        SnapshotWriteError,
        compute_and_store_cardio_risk,
        get_cardio_risk_history,
        get_follow_up_queue,
        get_or_create_daily_session,
        get_scale_history,
        get_session_coherence,
        get_session_engagement_profile,
        get_user_calibrations,
        list_site_configs,
        list_users,
        submit_answers,
        submit_observations,
        take_next_extra_batch,
        upsert_calibration,
    )

__all__ = [
    "SnapshotWriteError",
    "compute_and_store_cardio_risk",
    "get_cardio_risk_history",
    "get_follow_up_queue",
    "get_or_create_daily_session",
    "get_scale_history",
    "get_session_coherence",
    "get_session_engagement_profile",
    "get_user_calibrations",
    "list_site_configs",
    "list_users",
    "submit_answers",
    "submit_observations",
    "take_next_extra_batch",
    "upsert_calibration",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import service_pg as _facade

    return getattr(_facade, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
