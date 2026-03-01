# Questions Agent: Current State + Deep-Tech Integration Roadmap

Date: 2026-02-07  
Scope: `questions_agent_platform` only (questionnaire intelligence subsystem)

## 1) Executive status (what exists now)

The questionnaire subsystem is no longer a concept. It is implemented as:

- A runnable reference backend (`pipeline/`) with daily selection, scoring, projections, and policy logs.
- A production API backend (`prod/`) with FastAPI + Postgres + auth + Docker.
- A constrained adaptive policy layer (`policy/`) with explainability, counterfactuals, offline eval (IPS/DR), and rollback guardrails.
- A strict cross-system schema contract layer (`contracts.py`, `SCHEMA_CONTRACTS_V1.md`) for fusion/anifold integration.

This means the "Questions Agent core" is real and testable.  
What is still missing is the full external fusion/anifold runtime and full identity-mask conversational runtime.

## 2) Concrete implementation map (repo-grounded)

### A) Daily question engine
- Core logic: `questions_agent_platform/pipeline/selection.py`
- API orchestration: `questions_agent_platform/pipeline/api.py`
- Scoring + trajectories: `questions_agent_platform/pipeline/scoring.py`
- Supports:
  - 5 core questions/day
  - optional extra context batches
  - multiplexing (one item feeding multiple scales)
  - unlock/retest/progress tracking

### B) Production backend
- Service: `questions_agent_platform/prod/app.py`
- Business logic: `questions_agent_platform/prod/service_pg.py`
- Schemas: `questions_agent_platform/prod/schemas.py`
- SQLAlchemy models: `questions_agent_platform/prod/models.py`
- Deployment: `questions_agent_platform/prod/docker-compose.yml`, `questions_agent_platform/prod/Dockerfile`

### C) Policy layer (constrained learning)
- Features: `questions_agent_platform/policy/features.py`
- Bandit/reranker: `questions_agent_platform/policy/bandit.py`
- Logging + propensities + counterfactuals: `questions_agent_platform/policy/logging.py`
- Reward model hooks: `questions_agent_platform/policy/rewards.py`
- Offline evaluation (IPS/DR): `questions_agent_platform/policy/offline_eval.py`
- Versioning/activation: `questions_agent_platform/policy/registry.py`
- CLIs:
  - training/eval: `questions_agent_platform/policy/cli.py`
  - operational simulation/gates/reports: `questions_agent_platform/pipeline/cli.py`

### D) Contract hardening for fusion/anifold
- Contract helpers: `questions_agent_platform/contracts.py`
- Exact schema doc: `questions_agent_platform/SCHEMA_CONTRACTS_V1.md`
- Projection payload integration:
  - reference path: `questions_agent_platform/pipeline/projection.py`
  - prod path: `questions_agent_platform/prod/projection_pg.py`
- Enforced:
  - contract names
  - schema version compatibility aliases
  - vector shape checks (including 128-d projection contract)
  - join-key chain (`subject_id`, `date`, `session_id`, `decision_id`)

## 3) Vision fit assessment (honest)

### Strong fit (implemented)
- Validated-scale daily adaptive selection.
- "Selector only" policy discipline (no invented item IDs).
- Hard constraints and safe fallback behavior.
- Auditability (reason codes, scores, counterfactual alternatives).
- Offline policy evaluation and quality gate mechanics.
- Production-ready service skeleton.

### Partial fit (interfaces exist, runtime not complete here)
- Closed-loop fusion/anifold operation: contracts are ready, full external service is not in this repo.
- Identity masks as first-class runtime semantics: `identity_mask_id` is plumbed, full stateful mask engine is not complete.
- Rich clinical narrative layer over trajectory/interventions: basic reporting exists, final productized narrative layer is partial.

### Not in this subsystem (must live in adjacent systems)
- True multimodal fusion model training/inference.
- Anifold state-space model training/retraining pipeline.
- Intervention optimization/politherapy recommender under clinical governance.

## 4) Quality status now

- Unit/integration tests passing for this subsystem:
  - `questions_agent_platform.tests.test_contracts`
  - `questions_agent_platform.tests.test_policy`
  - `questions_agent_platform.tests.test_integration_demo`
  - `questions_agent_platform.tests.test_scoring`
- Operational outputs already generated:
  - policy briefs (`questions_agent_platform/output/policy_brief_*.html|json`)
  - weekly report (`questions_agent_platform/output/weekly_policy_report_*.html|json`)

## 5) Next step change (the single highest-impact move)

## Step Change #1: "Closed-Loop Truth Demo" (2 weeks)

Build one deterministic, replayable loop proving end-to-end behavior:

1. Ingest fusion/anifold context into `daily-questions/select`.
2. Run constrained policy selection with explanation + counterfactuals.
3. Emit `questions_to_fusion_evidence` payload with join-keys.
4. Attach delayed outcomes and show measurable uncertainty reduction.
5. Render one evidence-linked narrative output (clinician + investor lens).

Why this is the step change:
- It collapses "we have pieces" into a single proof of loop integrity.
- It validates your core claim: questions are not random UX; they are adaptive information acquisition for state inference.
- It gives a strong due-diligence artifact for technical investors.

Acceptance criteria:
- 0 contract mismatches over a 30-day replay.
- 0 constraint violations in policy mode.
- full traceability from `decision_id` -> outcomes -> evidence payload.
- reproducible report from one command.

## 6) Deep-tech roadmap to full vision

### Phase 1 (now complete): schema and policy foundation
- Contract/version hardening
- constrained policy + logging + eval + rollback

### Phase 2: closed-loop runtime integrity
- automated daily context ingestion
- strict event-chain observability and SLOs
- replay harness with failure injection

### Phase 3: identity-mask runtime semantics
- formal mask-state object and transitions
- mask-aware trajectory interpretation
- deterministic test vectors for mask behavior

### Phase 4: intervention response intelligence
- robust delayed reward attribution
- response confidence and de-escalation "review-only" flags
- cohort vs N-of-1 comparison views

### Phase 5: stakeholder-facing outputs
- clinician report from same evidence chain
- investor architecture narrative with live metrics
- user-facing "what changed / why / next best measurement" layer

## 7) What to improve next for a killer raise pitch

For the questionnaire subsystem specifically:

1. Show one full loop artifact, not fragments:
   - context in -> selected questions -> answers -> evidence out -> measurable update.
2. Put hard numbers on reliability:
   - loop completion rate, constraint-violation count, fallback rate, uncertainty reduction trend.
3. Show governance:
   - "no LLM item generation", strict approved-ID policy, audit logs, rollback behavior.
4. Show product consequence:
   - fewer random questions, faster scale completion, lower burden, better signal capture.
5. Tie each slide claim to one metric and one artifact path.

This converts the story from "interesting architecture" to "de-risked execution."
