"""Session/question flow domain exports for pipeline service."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DailySession",
    "get_or_create_daily_session",
    "get_scale_history",
    "list_users",
    "submit_answers",
    "submit_observations",
    "take_next_extra_batch",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import service as _facade

    return getattr(_facade, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
