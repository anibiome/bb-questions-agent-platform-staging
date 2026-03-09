"""Facade for threaded HTTP API modules.

Keeps stable import path while exposing route/helper modules.
"""

from . import api_monolith as _legacy
from . import api_helpers as _helpers
from . import api_routes as _routes

for _module in (_legacy,):
    for _name in dir(_module):
        if _name.startswith("__"):
            continue
        globals().setdefault(_name, getattr(_module, _name))

for _module in (_routes, _helpers):
    for _name in dir(_module):
        if _name.startswith("__"):
            continue
        globals().setdefault(_name, getattr(_module, _name))

del _module, _name

__all__ = [name for name in globals() if not name.startswith("__")]
