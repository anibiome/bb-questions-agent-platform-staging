"""Utility/helper exports for threaded API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import (
        _build_questions_payload,
        _extract_poe_from_fold,
        _get_latest_session,
        _get_query_date,
        _get_query_float,
        _get_query_int,
        _get_query_window_days,
        _item_to_question,
        _list_registry_versions,
        _progress_hints_by_item,
        _scale_names_by_item,
    )

__all__ = [
    "_build_questions_payload",
    "_extract_poe_from_fold",
    "_get_latest_session",
    "_get_query_date",
    "_get_query_float",
    "_get_query_int",
    "_get_query_window_days",
    "_item_to_question",
    "_list_registry_versions",
    "_progress_hints_by_item",
    "_scale_names_by_item",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import api as _facade

    return getattr(_facade, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
