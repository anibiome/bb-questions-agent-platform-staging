"""Facade for pipeline service modules.

Keeps stable import path while exposing bounded domain modules.
"""

from . import service_monolith as _legacy
from . import service_analytics as _analytics
from . import service_registry as _registry
from . import service_sessions as _sessions

for _module in (_legacy,):
    for _name in dir(_module):
        if _name.startswith("__"):
            continue
        globals().setdefault(_name, getattr(_module, _name))

for _module in (_registry, _sessions, _analytics):
    for _name in dir(_module):
        if _name.startswith("__"):
            continue
        globals().setdefault(_name, getattr(_module, _name))

del _module, _name

__all__ = [name for name in globals() if not name.startswith("__")]
