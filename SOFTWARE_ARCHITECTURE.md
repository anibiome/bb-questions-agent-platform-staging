# Software Architecture

## Purpose

`questions-agent-platform` provides the questionnaire-selection, scoring, state-tracking, and policy-evaluation layer for ANI question workflows.

## Major Components

- `questions_agent_platform/`: packaged entrypoints and configuration
- `pipeline/`: session lifecycle, scoring, snapshots, trajectories, and serving logic
- `policy/`: policy runtime, registry, logging, and evaluation
- `prod/`: production API settings and deployment-specific code
- `tests/`: regression coverage
- `tools/`: quality gates and evaluation utilities

## Runtime Boundary

- User or study context enters through API and pipeline entrypoints.
- Pipeline logic selects questions, applies scoring, persists state, and emits projection or evidence outputs.
- Policy modules evaluate or constrain runtime selection behavior.
- Production behavior depends on externally supplied environment configuration and deployment controls.

## Security Boundary

- The main software risks are integrity and safety: incorrect item selection, incorrect scoring, or incorrect policy decisions.
- Example environment files are templates only and must not become production source-of-truth configuration.
- This repository contains code, fixtures, and governance artifacts rather than approved clinical record custody.

## Verification Boundary

- CI and automated tests provide repository-level verification.
- Rollout decisions and policy authorization remain external governance steps.
