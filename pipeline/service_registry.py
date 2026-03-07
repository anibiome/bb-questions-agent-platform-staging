"""Registry/profile domain exports for pipeline service."""

from __future__ import annotations

from typing import Any

__all__ = [
    "activate_registry_version",
    "ensure_registry_active",
    "ensure_user_profile",
    "get_user_profile",
    "upload_registry_bundle",
    "upsert_user_profile",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import service as _facade

    return getattr(_facade, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
