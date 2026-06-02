# Questions Agent v2.0 Production Readiness Checklist

Date: 2026-02-14  
Scope: `questions_agent_platform`  
Source of truth: your “QUESTIONS AGENT The Complete Specification v2.0”

## Goal
Bridge the current implementation to a release-reviewable v2 baseline with strict clinical/engineering constraints:
- 5 questions/day (except safety override)
- one timeframe per session
- item-level multiplexing with auditable evidence rows
- hard onboarding ordering (cardiometabolic-first)
- constrained policy with full auditability
- safe fallback and safety overrides

---

## P0 (must be done first)

### 1) Data model + migrations
- [x] Add user profile state:
  - `mode` (`trial|consumer`)
  - `permanently_declined_items_json`
  - `onboarding_complete`
  - `state_schema_version`
- [x] Add session state:
  - `status` (`created|started|completed|abandoned`)
  - `timeframe`
- [x] Add `scale_evidence` table:
  - one row per answer-to-scale mapping
  - fields: `answer_event_id`, `item_id`, `scale_id`, `weight`, `contribution_value`, `timeframe`, `window_date`
- [x] Add schema indexes for high-read paths.

Files:
- `pipeline/db.py`
- `prod/sql/001_init.sql`
- `prod/models.py`

### 2) Selector and session invariants
- [x] Enforce one timeframe per session via item-allowed timeframes.
- [x] Enforce onboarding lock:
  - cardiometabolic sequence first
  - no non-cardiometabolic questions until completed.
- [x] Enforce diversity and intrusiveness:
  - minimum 2 distinct concept tags/session (unless safety mode)
  - maximum 1 high-intrusiveness item/day.
- [x] Mark stale incomplete previous sessions as `abandoned` when a new day starts.

Files:
- `pipeline/registry.py`
- `pipeline/selection.py`
- `pipeline/service.py`
- `prod/service_pg.py`

### 3) Answer ingestion + evidence trail
- [x] On each answer write:
  - `AnswerEvent`
  - N rows in `scale_evidence` for every mapped scale item
- [x] Add support for permanent decline events:
  - persist item as declined
  - exclude from future candidate sets.

Files:
- `pipeline/service.py`
- `prod/service_pg.py`

### 4) API behavior alignment
- [x] Expose session `timeframe` and `status`.
- [x] Align “Ask Ani more” to one-item behavior:
  - return one extra question/call
  - cap at 3 extras/day.
- [x] Expose dedicated progress contract:
  - `GET /v1/users/{user_id}/progress?date=YYYY-MM-DD&window=30d`
  - coverage + completion + scale progress in one payload.
- [x] Keep strict compatibility for existing endpoints.

Files:
- `pipeline/api.py`
- `prod/app.py`
- `prod/schemas.py`

### 5) Registry support for timeframe and flags
- [x] Extend item schema with:
  - `timeframes_allowed`
  - `intrusiveness`
  - `declinable`
  - optional onboarding and domain tags.
- [x] Keep backward compatibility (defaults if fields absent).

Files:
- `pipeline/registry.py`
- `data/registry_demo/*` (for demo parity)

---

## P1 (after P0)

### 6) Cardiometabolic V1 content pack
- [x] Add canonical 27-item deduplicated cardiometabolic battery.
- [x] Add six scoring modules (FINDRISC, EZ-CVD, SCORED, Lee NAFLD, IPAQ-SF, AUDIT-C).
- [x] Add strict onboarding progression and unlock messaging.

Files:
- `data/registry_demo/versions/<new_version>/...`
- `pipeline/scoring.py`
- `pipeline/service.py`
- `prod/service_pg.py`

### 7) Anamnesis runtime object
- [x] Add persistent follow-up queue with priorities/reasons/earliest-date.
- [x] Pull queue items first in daily selection, then normal pool.
- [x] Auto-resolve branch when confidence stabilizes.

Files:
- `pipeline/db.py`, `prod/sql/001_init.sql`, `prod/models.py`
- `pipeline/selection.py`, `pipeline/service.py`, `prod/service_pg.py`

### 8) State + circle snapshots
- [x] Add state snapshot object:
  - `x_hat`, `x_uncertainty`, coverage, model version
- [x] Add circle snapshot object:
  - `z`, `z_star`, `r`, `theta`, `v`, `a`, uncertainties
  - immutable `projection_version` and schema version.
- [x] Expose snapshot retrieval endpoints (`/state`, `/circle`) in reference + prod APIs.

Files:
- `pipeline/db.py`, `prod/sql/001_init.sql`, `prod/models.py`
- `pipeline/state_snapshots.py`, `pipeline/service.py`, `pipeline/api.py`
- `prod/service_pg.py`, `prod/app.py`

---

## P2 (hardening / ops)

### 9) Production operations
- [x] Add formal migration tool (Alembic).
- [x] Add API/load tests and SLO dashboards.
- [x] Add backup/restore drill scripts.
- [x] Add audit export jobs for clinical review.

### 10) Governance and release
- [x] Add policy/version promotion gates and checklists.
- [x] Add explicit rollout playbook and rollback drills.
- [x] Add data minimization + retention policy config.

### 11) Final production cut blockers (closed)
- [x] Replace `Base.metadata.create_all` startup bootstrap with strict DB migration workflow (Alembic + immutable migration history).
- [x] Add contract-level API tests for `/v1/users/{user_id}/state` and `/v1/users/{user_id}/circle` (both SQLite and Postgres stacks).
- [x] Add load/performance tests for daily selection + answer ingestion + snapshot writes under realistic concurrency.
- [x] Add alerting/SLOs for safety fallback rate, answer ingestion latency, and snapshot write failures.
- [x] Add PHI handling controls for retention windows and export redaction policies.
- [x] Remove FastAPI startup deprecation (`on_event`) by migrating to lifespan handlers.
- [x] Remove Pydantic v1 validator/config deprecations (`validator/root_validator/Config`) by migrating to v2 APIs.

### 12) Trajectory + scientific evaluation closure
- [x] Persist first-class `ews_features` and `drift_events` entities (SQLite + Postgres + Alembic migration).
- [x] Persist and expose coherence score/tier contract (`GET /v1/coherence/tier-contract`).
- [x] Persist and expose cross-modal uncertainty coupling payload on `POST /v1/users/{user_id}/observations`.
- [x] Add consolidated offline scientific evaluation tool with ablations + calibration:
  - `python3 -m questions_agent_platform.tools.run_scientific_offline_eval ...`

---

## Current pass summary
1. P0/P1 implementation closure ✅  
2. Alembic migration hardening ✅  
3. Contract + load + ops control tests ✅  
4. SLO + PHI controls ✅  
5. Backup/restore drills ✅  
6. Policy promotion checklist + rollback drills ✅
7. Trajectory persistence + scientific eval consolidation ✅

---

## Latest validation snapshot
- Full test suite: `52 passed` (`python3 -m pytest -q questions_agent_platform/tests --maxfail=1`)
- Warning summary: 1 non-project warning (`python_multipart` pending deprecation from Starlette import path).
- New production-readiness tests:
  - onboarding cardiometabolic lock + 5-item continuity
  - onboarding extra-batch remains cardiometabolic-only while lock is active
  - session status transitions (`created → started → completed`)
  - multiplex `scale_evidence` writes
  - permanent decline exclusion
  - one-more single-item + 3/day cap
  - follow-up queue prioritization + answer-based resolution
  - state/circle snapshot persistence + retrieval
  - coherence payload contract (`score` + `tier`) in circle snapshots
  - coherence tier contract endpoint + EWS/drift endpoint parity (SQLite + Postgres)
  - drift routing contract endpoint parity (SQLite + Postgres)
  - observation coupling payload contract on observation ingestion (SQLite + Postgres)
  - AniFold adapter + mock server contract loop tests (context fetch, evidence submit, outcome submit)
  - full roundtrip smoke harness test (context → select → answers → projection → evidence submit → outcome callback)
  - scientific offline evaluation consolidation output (full + ablations + calibration)
  - governance/API parity for profile, safety events, emotion deep dive, experiments, ANI summary, and domain readiness (SQLite + Postgres)
  - progress endpoint parity (`/v1/users/{user_id}/progress`) across SQLite + Postgres stacks
  - backup/restore drill parity (`baseline_row_counts == restored_row_counts`)
  - policy promotion checklist gate logic (`ready_for_promotion` vs blocked reasons)
  - policy rollback drill activation/restore verification
