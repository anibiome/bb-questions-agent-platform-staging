#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "SECURITY.md",
    "CHANGE_CONTROL_PLAN.md",
    "FDA_TECHNICAL_FILE_INDEX.md",
    "SOFTWARE_REQUIREMENTS.md",
    "SOFTWARE_ARCHITECTURE.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    warnings: list[str] = []

    for rel_path in REQUIRED_FILES:
        if not (repo_root / rel_path).exists():
            failures.append(f"missing required file: {rel_path}")

    workflow_dir = repo_root / ".github" / "workflows"
    workflow_files = sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml"))
    if not workflow_files:
        failures.append("missing GitHub Actions workflow")

    env_example = repo_root / "prod" / ".env.example"
    if env_example.exists():
        text = env_example.read_text(encoding="utf-8")
        if "API_KEY=change_me" in text:
            warnings.append("prod/.env.example contains placeholder API key and must not be promoted unchanged")

    if (repo_root / "output").exists():
        warnings.append("review committed `output/` artifacts before release packaging")

    result = {
        "status": "pass" if not failures else "fail",
        "repo": str(repo_root),
        "failures": failures,
        "warnings": warnings,
        "workflow_files": [str(path.relative_to(repo_root)) for path in workflow_files],
    }
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
