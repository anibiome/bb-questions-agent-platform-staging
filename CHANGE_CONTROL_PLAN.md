# Change Control Plan

This repository contains the ANI questions agent reference and governed runtime surfaces, including policy logic, state tracking, and API behavior.

## Scope

- `questions_agent_platform/`, `pipeline/`, `policy/`, and `prod/`
- Contract and rollout documentation
- Test suites, offline evaluation, and tooling
- Production environment examples and operational scripts

## Change Classes

### Standard

- Documentation-only edits
- Non-behavioral tooling
- Test additions that do not change runtime behavior

### Controlled

- Item selection or policy logic changes
- State snapshot, trajectory, or projection logic changes
- Scale scoring or safety-event logic changes
- API contract changes
- Production settings changes

### Release-Critical

- Changes affecting user-facing question selection
- Changes affecting safety routing or escalation outputs
- Changes affecting production authentication or persistence
- Changes affecting rollout-gate or offline-evaluation decisions

## Required Controls

1. Every change requires pull request review by the code owner in `.github/CODEOWNERS`.
2. Controlled and release-critical changes require documented review of user-impact, policy, and safety implications.
3. Breaking contract changes must be versioned or explicitly rolled out; in-place silent changes are not permitted.
4. Demo data, fixtures, and committed outputs must remain non-production and de-identified.
5. Production environment placeholders must not be promoted unchanged into deployed environments.

## Release Evidence

Before release or promotion:

- CI must pass.
- `python3 tools/repo_compliance_check.py` must pass.
- Relevant tests or rollout gates must pass for the changed area.
- Policy or scoring changes must be traceable to an evaluation or rollout note outside this repo.
- Commit SHA, environment, and configuration set must be recorded in the release log.

## Emergency Changes

Emergency fixes may ship for security, integrity, or blocking runtime defects, but they still require post-release review, regression confirmation, and retrospective documentation.
