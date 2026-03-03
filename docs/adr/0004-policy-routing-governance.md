# ADR 0004: Policy Routing and Safety Governance

- Status: Accepted
- Date: 2026-03-03
- Owners: Questions Agent Platform Team

## Context

Policy routing influences question selection, operational safety, and rollback behavior. Unclear governance can cause silent degradation.

## Decision

Define policy routing as governed runtime surface:

1. Keep policy modes explicit (`deterministic`, `policy_shadow`, `policy_live`, `policy_offline_replay`).
2. Enforce rollback guard checks and operational metrics in CI/tested code paths.
3. Maintain public contracts for coherence tier and drift routing endpoints.
4. Require changelog + migration notes for policy schema/behavior changes.

## Consequences

- Positive: safer live policy rollout and reproducible release decisions.
- Negative: more process overhead for policy changes.
- Follow-up actions: expand alerting on rollback-trigger metrics.

## Alternatives Considered

1. Free-form policy experimentation in production.
2. Static deterministic policy only.
3. Live policy without rollback automation.
