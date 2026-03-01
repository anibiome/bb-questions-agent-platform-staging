# Questions Agent v2.0 — Architecture Map (Implementation Reality)

Date: 2026-02-16  
Scope: `questions_agent_platform`  
Goal: one clear map from your v2 spec to concrete modules your team will edit.

## 1) System boundary (strict)

- **AniFold/Fusion owns Z inference**.
- **Questions Agent reads context, never writes Z directly**.
- **Questions Agent writes evidence + outcomes** for AniFold to consume.

Current contract surfaces:
- context in: `/v1/users/{user_id}/daily-questions/select` (`context` block)
- evidence out: `/v1/users/{user_id}/projection/questions`
- delayed outcomes in: `/v1/users/{user_id}/policy/outcomes`

## 2) Runtime layers

1. **Registry layer** (`pipeline/registry.py`, `data/registry_demo/*`)
   - canonical item bank, scale structures, mappings, repeat/intrusiveness/timeframe metadata.
2. **Selection layer** (`pipeline/selection.py`)
   - deterministic constrained selector and candidate-set generation.
3. **Orchestration layer** (`pipeline/service.py`, `prod/service_pg.py`, `pipeline/session_helpers.py`)
   - daily session lifecycle, onboarding lock, follow-up queue, safety mode, scoring updates.
4. **Policy layer** (`policy/*`, `policy/runtime.py`)
   - constrained contextual bandit rerank over candidate set `C`, with audit/counterfactuals.
5. **State/trajectory layer** (`pipeline/state_snapshots.py`, `pipeline/trajectory_metrics.py`)
   - state snapshots, circle geometry, EWS, drift events.
6. **API layer** (`pipeline/api.py`, `prod/app.py`)
   - contract-safe endpoints for app, ops, and integration.

## 3) Database ownership by concern

- **Session + answers**: `daily_sessions`, `answer_events`
- **Multiplex evidence**: `scale_evidence`
- **Scoring**: `scale_scores`, `scale_baselines`
- **State geometry**: `state_snapshots`, `circle_snapshots`
- **Branching**: `follow_up_queue`
- **Safety**: `safety_events`
- **Policy audit + training**: `policy_decisions`, `policy_outcomes`
- **Trajectory analytics**: `ews_features`, `drift_events`
- **Multimodal ingress**: `observation_events`
- **N-of-1 layer**: `experiments`

## 4) Where your devs should make changes

### Add or modify questionnaires
- edit registry bundle (`items.json`, `scales.json`, mappings)
- validate/upload/activate via admin registry endpoints
- do **not** hardcode questionnaire logic in service code

### Change selection behavior
- deterministic constraints/objective: `pipeline/selection.py`
- policy rerank behavior: `policy/bandit.py`, `policy/features.py`
- rollout and gates: `POLICY_ROLLOUT_PLAYBOOK_V1.md`, ops tools

### Integrate with AniFold
- adapter + contracts: `pipeline/anifold_adapter.py`, `contracts.py`, `SCHEMA_CONTRACTS_V1.md`
- roundtrip smoke: `tools/run_anifold_roundtrip_smoke.py`

### Harden production ops
- migrations: `prod/alembic/*`
- SLO + alerting: `/v1/admin/slo`, `tools/check_slo_alerts.py`
- PHI retention/redaction: `tools/run_phi_retention.py`, audit export controls

## 5) Quality rules that must stay true

1. Exactly 5 core/day (except safety override).
2. One timeframe per session.
3. Item is the interaction unit; scales are derived projections.
4. One answer can feed many scales (`scale_evidence` rows are mandatory).
5. Policy can only pick from candidate set `C`; no item invention.
6. Safety mode bypasses exploration/rerank.
7. Z is recomputed by AniFold, never by Questions Agent.

## 6) Practical implementation status

- This repo is a **production-ready baseline** for the Questions Agent side.
- Remaining product integration work is mostly:
  - loading your full registry content,
  - wiring real AniFold endpoints,
  - app UX rendering and messaging,
  - deployment environment hardening in your infra.
