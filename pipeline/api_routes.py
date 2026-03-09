"""HTTP route handler exports for threaded API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .api import _make_handler, run_server

__all__ = ["_make_handler", "run_server"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import api as _facade

    return getattr(_facade, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
