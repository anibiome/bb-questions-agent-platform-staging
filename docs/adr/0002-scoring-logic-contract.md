# ADR 0002: Scoring Logic Contract and Stability Window

- Status: Accepted
- Date: 2026-03-03
- Owners: Questions Agent Platform Team

## Context

Scoring logic drives user-facing trajectory and risk outputs. Silent scoring drift creates correctness and trust regressions.

## Decision

Treat scoring as versioned contract behavior:

1. Keep deterministic score computation in pipeline modules with explicit confidence/risk tiers.
2. Snapshot critical scoring payload schemas and public endpoint contracts in tests.
3. Require regression tests for scoring/routing behavior whenever scoring logic changes.
4. Use semver with MINOR/PATCH restrictions: breaking scoring semantics require MAJOR.

## Consequences

- Positive: predictable score evolution and explicit compatibility posture.
- Negative: additional test maintenance and snapshot updates.
- Follow-up actions: monitor score distribution drift in observability dashboards.

## Alternatives Considered

1. Rapid iteration without contract snapshots.
2. Runtime-only guards without CI-level schema checks.
3. Fully dynamic scoring configuration per deployment.
