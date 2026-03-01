# Questions Agent: Current State + Vision Integration Plan

Date: 2026-02-06  
Scope: `questions_agent_platform` (questionnaire intelligence layer)

## 1) What we actually have now

### A) Running backend (reference + production)
- Reference engine: Python + SQLite flow for selection, scoring, projection payloads, policy logs.
- Production engine: FastAPI + Postgres (`prod/`) with API key auth, dashboard, Docker deployment path.
- Registry version management (upload/activate versions of questionnaires/scales).

### B) Daily measurement system
- Exactly 5 core questions/day, plus optional extra context batches.
- Deterministic candidate generation with hard constraints (mandatory anchors, blocked/cooldown handling).
- Multiplexing support (one item can feed multiple scales).
- Scale progress/unlock/retest logic and history endpoints.

### C) Within-person scoring + trajectory primitives
- Per-scale baseline + drift-aware updates in the question/scoring layer.
- Historical scale tracking and confidence-style scoring metadata.
- Question-derived projection payload endpoint for fusion consumption.

### D) Constrained policy layer (deep-tech part already implemented)
- Contextual-bandit reranking with strict action-space constraint: only from candidate set `C`.
- Policy modes: `deterministic`, `policy_live`, `policy_shadow`, `policy_offline_replay`.
- Full decision logging (selected items, propensities, explanations, counterfactuals).
- Outcome logging and reward attachment path.
- Offline evaluation (IPS/DR) including segment views.
- Rollout gates and policy brief generation.
- Runtime auto-rollback guard from `policy_live` to deterministic on safety/performance degradation.
- Weekly policy report generator (HTML + JSON).

## 2) What is partially implemented (exists but not complete for your full vision)

- Fusion/Anifold interface is present as contract points, but not the full multimodal Anifold model itself.
- `identity_mask_id` is plumbed through APIs/context, but full identity-mask reasoning runtime is not complete.
- Clinical/investor-facing reporting exists technically, but not yet the final polished narrative product layer.
- Segment gates work, but current data is sparse in some segments (not enough eval power yet).

## 3) What is not yet in this repo (must be integrated from other systems)

- True multimodal fusion stack (imaging/voice/wearables/omics joint inference).
- Anifold model training/retraining and production inference service.
- End-to-end ANI conversational reasoning layer bound to identity masks as primary interface.
- Cohort analytics, intervention optimization, and polytherapy recommendation engine under governance.

## 4) Vision-fit assessment (honest)

- Questionnaire intelligence backend: **strong** (foundationally solid).
- Constrained adaptive policy + auditability: **strong**.
- Closed loop with Anifold: **partially wired** (interfaces exist; full loop needs external fusion runtime).
- Full ANI “interface for biology” product: **not complete in this repo alone**.

## 5) Integration plan to match your vision (decision-complete)

## Phase 1: Contract hardening (1 week)
Goal: make cross-system integration non-ambiguous.
- Freeze canonical schemas for:
  - `fusion -> questions` context payload,
  - `questions -> fusion` evidence payload,
  - `policy decisions/outcomes` join keys.
- Add strict validation + schema versioning in API boundary.
- Add compatibility tests for old/new schema versions.

Acceptance:
- Any integration mismatch fails fast with explicit error codes.
- One canonical `decision_id` chain links selection -> outcomes -> fusion deltas.

## Phase 2: Closed-loop activation (1–2 weeks)
Goal: operational loop `fusion/anifold context -> questions policy -> evidence -> fusion`.
- Wire scheduled daily context ingestion from fusion.
- Enforce required context fields (with fallback behavior if missing).
- Persist loop-latency and loop-completeness metrics.

Acceptance:
- Daily loop success rate >= 99% in staging.
- Zero silent drops of context/evidence events.

## Phase 3: Policy live hardening (1–2 weeks)
Goal: safe live policy rollout with guardrails.
- Increase policy-live sample for segment power.
- Keep shadow/live dual logging for comparison.
- Tune reward weights per segment if DR under baseline.

Acceptance:
- Segment min rows thresholds satisfied.
- Segment DR not worse than deterministic baseline.
- Safety violations = 0.

## Phase 4: Identity mask runtime (2 weeks)
Goal: implement your identity-mask semantics as first-class runtime.
- Replace placeholder mask usage with explicit mask state objects.
- Add mask-aware trajectory summaries and reasoning controls.
- Add mask-level explainability in API outputs.

Acceptance:
- Mask changes alter interpretation and query behavior deterministically.
- No effect on validated scale scoring rules.

## Phase 5: Clinical + investor product layer (1–2 weeks)
Goal: make system legible to non-technical stakeholders.
- Produce final narrative artifacts from live data:
  - weekly PDF,
  - clinician summary,
  - investor system map tied to real metrics.
- Add “what changed / why / confidence / next measurement intent” summaries.

Acceptance:
- Stakeholder outputs generated automatically each week.
- Narrative is traceable back to logs and numeric evidence.

## 6) Immediate next execution block (recommended)

1. Run 2–4 weeks of higher-volume policy logging (multi-user, mixed adherence profiles).  
2. Close segment gate failures by data volume + targeted reward tuning.  
3. Lock identity-mask schema and runtime behavior definition.  
4. Finish fusion loop SLO dashboard (success rate, latency, missingness).

## 7) “Done” definition for your vision (questionnaire layer)

The questionnaire subsystem is “vision-complete” when:
- It selects only validated approved items under constraints.
- It adapts daily to true state/trajectory context from fusion.
- It emits structured evidence that materially improves downstream state inference.
- It is auditable (why chosen, what not chosen, what changed).
- It is safe (auto-rollback + zero constraint violations in live operation).
- It is explainable to clinician, user, and investor in one coherent narrative.
