"""Install the historical package alias for raw checkout execution.

The repository checkout is named `bb-questions-agent-platform`, while the
package import contract is `questions_agent_platform`. Pytest imports this via
`conftest.py`; plain `python -m unittest` and scripts need the same alias before
test modules import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _install_repo_package_alias() -> None:
    package_name = "questions_agent_platform"
    if package_name in sys.modules:
        return

    repo_root = Path(__file__).resolve().parent
    init_file = repo_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


_install_repo_package_alias()
