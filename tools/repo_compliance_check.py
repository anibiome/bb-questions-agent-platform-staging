#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "CODE_ROOM_README.md",
    "ani-manifest.yaml",
    "anifesto.yaml",
    "SECURITY.md",
    "CHANGE_CONTROL_PLAN.md",
    "FDA_TECHNICAL_FILE_INDEX.md",
    "SOFTWARE_REQUIREMENTS.md",
    "SOFTWARE_ARCHITECTURE.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
]

REQUIRED_TOKENS = {
    "README.md": [
        "## Evidence Boundary",
        "behavioral-question runtime",
        "not raw biological source truth",
        "treatment efficacy",
        "patent-ready proof",
    ],
    "CODE_ROOM_README.md": [
        "participant source truth",
        "raw biological truth",
        "treatment efficacy",
        "patent-ready proof",
        "explicit_promotion_approval",
    ],
    "ani-manifest.yaml": [
        "source_of_truth: false",
        "behavioral_question_runtime_and_runtime_bridge_contracts_only",
        "behavioral_or_derived_evidence_not_raw_biological_or_clinical_truth",
        "participant_source_truth: false",
        "treatment_efficacy: false",
        "patent_ready_proof: false",
        "explicit_promotion_approval",
    ],
    "anifesto.yaml": [
        "behavioral_question_runtime_and_bridge_contract_surface",
        "participant_source_truth",
        "raw_biological_truth",
        "treatment_efficacy",
        "patent_ready_proof",
        "explicit_promotion_approval",
    ],
    ".github/workflows/code-room-readiness.yml": [
        "tests.test_manifest_controls",
    ],
    ".github/workflows/regulated-review.yml": [
        "tests.test_manifest_controls",
    ],
}


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    warnings: list[str] = []

    for rel_path in REQUIRED_FILES:
        if not (repo_root / rel_path).exists():
            failures.append(f"missing required file: {rel_path}")

    code_room = repo_root / "CODE_ROOM_README.md"
    if code_room.exists():
        text = code_room.read_text(encoding="utf-8")
        for section in ("## Role In ANI.AI", "## Current Audit Posture", "## Claim Boundary"):
            if section not in text:
                failures.append(f"CODE_ROOM_README.md missing section: {section}")

    for rel_path, tokens in REQUIRED_TOKENS.items():
        path = repo_root / rel_path
        if not path.exists():
            failures.append(f"missing required claim-control file: {rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                failures.append(f"missing claim-control token in {rel_path}: {token}")

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
