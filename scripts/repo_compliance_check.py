#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    runpy.run_path(str(repo_root / "tools" / "repo_compliance_check.py"), run_name="__main__")
