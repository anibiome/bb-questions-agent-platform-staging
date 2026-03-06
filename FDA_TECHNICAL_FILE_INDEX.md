# FDA Technical File Index

This index maps the main software-facing review artifacts for `questions-agent-platform`.

## Core Description

- `README.md`: repository overview and operating commands
- `SOFTWARE_REQUIREMENTS.md`: software requirements baseline
- `SOFTWARE_ARCHITECTURE.md`: component and data-flow description
- `SECURITY.md`: vulnerability intake and response targets
- `CHANGE_CONTROL_PLAN.md`: review and release controls

## Source and Runtime

- `questions_agent_platform/`: packaged runtime code
- `pipeline/`: question flow, scoring, snapshots, and API logic
- `policy/`: policy runtime, offline evaluation, and registry logic
- `prod/`: production API and deployment settings
- `SCHEMA_CONTRACTS_V1.md`: contract notes
- `POLICY_ROLLOUT_PLAYBOOK_V1.md`: rollout governance notes
- `PRODUCTION_READINESS_CHECKLIST_V2.md`: readiness checklist

## Verification

- `tests/`: automated regression coverage
- `.github/workflows/ci.yml`: repository CI
- `.github/workflows/control-plane-gate.yml`: cross-repo gate
- `.github/workflows/regulated-review.yml`: documentation and governance baseline check
- `tools/repo_compliance_check.py`: repository compliance gate

## Governance

- `.github/CODEOWNERS`: review accountability
- `.github/dependabot.yml`: dependency update monitoring
- `docs/release-process.md`: release process notes
- `docs/observability-baseline.md`: observability notes

## External Records

The following artifacts are expected outside this repository for formal review packages:

- rollout and policy-approval records,
- deployment configuration approvals,
- evaluation reports backing policy changes,
- incident, CAPA, and rollback records.
