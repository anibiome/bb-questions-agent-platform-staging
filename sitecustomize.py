"""Import compatibility for the archived repository layout.

The code packages live at the repository root (`pipeline`, `policy`, `prod`),
while tests and integrations import them through `questions_agent_platform`.
Pytest used `conftest.py` for that alias, but the CI unittest runner imports
tests before pytest hooks exist. Python loads this module automatically when the
repo root is on `PYTHONPATH`, giving unittest the same namespace.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_repo_root = Path(__file__).resolve().parent

if (_repo_root / "pipeline").is_dir() and "questions_agent_platform" not in sys.modules:
    package = types.ModuleType("questions_agent_platform")
    package.__file__ = str(_repo_root / "__init__.py")
    package.__path__ = [str(_repo_root)]
    sys.modules["questions_agent_platform"] = package

