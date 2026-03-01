"""Root conftest — ensure questions_agent_platform is importable."""

import os
import sys

# Add the parent directory to sys.path so that
# `from questions_agent_platform.pipeline.xxx import ...` works
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
