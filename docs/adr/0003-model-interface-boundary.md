# ADR 0003: Model/Service Interface Boundary

- Status: Accepted
- Date: 2026-03-03
- Owners: Questions Agent Platform Team

## Context

Large service modules mixed domain orchestration, persistence, and transport concerns, reducing maintainability and increasing merge risk.

## Decision

Adopt bounded module facades with stable import paths:

1. Keep import paths stable (`pipeline.service`, `prod.service_pg`, `pipeline.api`) via facade modules.
2. Segment domain surfaces into registry/profile, sessions, and analytics modules.
3. Continue extracting behavior from legacy monolith backends incrementally behind these facades.
4. Hold a hard target that top-level core entry files stay below 800 LOC.

## Consequences

- Positive: clearer ownership boundaries and safer incremental refactors.
- Negative: transitional dual-layer modules until full extraction is complete.
- Follow-up actions: retire monolith backends as domain modules absorb implementations.

## Alternatives Considered

1. Keep single monolith files.
2. Perform one-shot rewrite (high risk).
3. Split by infrastructure layer only (DB/API) without domain boundaries.
