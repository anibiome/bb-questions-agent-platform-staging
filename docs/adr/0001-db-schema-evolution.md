# ADR 0001: Database Schema Evolution Policy

- Status: Accepted
- Date: 2026-03-03
- Owners: Questions Agent Platform Team

## Context

The production API now spans session state, scale scoring, calibration, cardio-risk, and anamnesis entities. Schema drift across environments has caused startup/runtime mismatch risk.

## Decision

Use migration-first schema evolution with these rules:

1. All schema changes must be represented as Alembic migrations.
2. Backward-compatible additive changes are default (new nullable columns/tables first).
3. Destructive changes require one full release cycle deprecation notice and migration notes.
4. Runtime startup checks schema head (`DB_MIGRATION_MODE=auto|require|off`) and fails safe in `require` mode.

## Consequences

- Positive: deterministic deploys, safer rollbacks, explicit schema lineage.
- Negative: slower iteration for ad-hoc DB changes.
- Follow-up actions: enforce migration notes in release checklist and changelog.

## Alternatives Considered

1. Auto-create schema from ORM models only.
2. Manual SQL scripts without revision tracking.
3. Hybrid with no startup verification.
