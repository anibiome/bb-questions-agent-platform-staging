"""Root conftest -- ensure questions_agent_platform is importable."""

import os
import sys
import types

# Add the parent directory to sys.path so that
# `from questions_agent_platform.pipeline.xxx import ...` works
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

_repo_root = os.path.dirname(os.path.abspath(__file__))
if "questions_agent_platform" not in sys.modules:
    package = types.ModuleType("questions_agent_platform")
    package.__path__ = [_repo_root]
    sys.modules["questions_agent_platform"] = package
