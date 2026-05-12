# Questions Agent — Developer Implementation Guide

**For:** New developer joining the project
**Date:** 2026-02-27
**Test suite:** run `./tools/quality_gate.sh` before quoting current counts

---

## 1. What This System Does (30-second version)

The Questions Agent asks a person ~5 smart health questions per day. Over days/weeks, it builds a complete health profile across 9 dimensions (energy, sleep, gut, metabolic risk, cardiovascular, mood, cognition, agency, social). It uses Item Response Theory (IRT) to pick the most informative questions, and one question can feed multiple clinical instruments simultaneously.

## 2. Project Structure

```
questions_agent_platform/
  pipeline/               # Core engine (pure Python, no external deps except stdlib)
    irt.py                # IRT scoring — Graded Response Model (Samejima 1969)
    scoring.py            # Scale scoring — maps IRT scores to clinical instruments
    selection.py          # Item selection — cross-instrument Fisher information optimiser
    service.py            # Orchestration — session lifecycle, answer processing, state updates
    registry.py           # Item bank — scales, items, metadata, versioning
    measurement_packages.py  # Progressive plan — which instruments to prioritise when
    state_snapshots.py    # 9D latent state + Coherence Circle geometry
    questions_fold.py     # QuestionsFold — Identity Mask fragment for AniFold fusion
    trajectory_metrics.py # EWS (Early Warning System), drift detection, coherence tiers
    n_of_1.py             # Person-specific calibration — Kalman filter + Reliable Change
    coherence.py          # Careless responding detection — 6-signal multiple-hurdle
    cardio_risk_index.py  # Composite cardiometabolic risk score
    response_metadata.py  # Behavioural uncertainty modifiers (Claim Family 5)
    anamnesis.py          # Drift-triggered clinical branching (Claim Family 4)
    concordance.py        # Validation framework — ICC, Bland-Altman, Cohen's kappa
    drift_routing.py      # Drift-to-instrument routing table
    site_config.py        # Multi-version deployment configs (wellness/CDS/SaMD)
    anifold_adapter.py    # HTTP bridge to AniFold (mock-ready)
    projection.py         # Evidence payload builder for AniFold fusion
    baseline.py           # EWMA baseline tracking
    config.py             # QuestionsAgentConfig — all tuneable parameters
    governance.py         # Domain governance — onboarding, promotion, cardiometabolic gate
    dashboard.py          # HTML dashboard renderer
    session_helpers.py    # Selection mode helpers, onboarding gate, domain filtering
    api.py                # Lightweight HTTP API (stdlib, SQLite — for dev/demo)
    db.py                 # SQLite schema + helpers (for dev/demo)
    cli.py                # CLI entry point
    demo.py               # Demo scenario generator
    time_utils.py         # Date/time helpers
    response_types.py     # Likert scale definitions (0-4, 0-5, 0-6, etc.)
  policy/                 # Contextual bandit reranking layer
    bandit.py             # UCB-based contextual bandit
    features.py           # Feature engineering for policy decisions
    runtime.py            # Policy execution runtime
    rewards.py            # Reward computation (completion, burden, uncertainty reduction)
    hashing.py            # Stable hashing for quantised Z vectors
    offline_eval.py       # Offline policy evaluation (IPS, DR estimators)
    types.py              # Policy data structures
    registry.py           # Policy version management
    adapter.py            # Policy adapter
    logging.py            # Policy audit logging
    cli.py                # Policy CLI tools
  prod/                   # Governed service path (FastAPI + PostgreSQL + Alembic)
    app.py                # FastAPI application — promoted endpoints require release review
    service_pg.py         # PostgreSQL service layer (mirrors pipeline/service.py)
    models.py             # SQLAlchemy ORM models (26 tables)
    schemas.py            # Pydantic request/response schemas
    settings.py           # Environment-based settings
    auth.py               # API key authentication
    db.py                 # SQLAlchemy engine + session factory
    migrations.py         # Alembic migration runner
    projection_pg.py      # PostgreSQL projection builder
    manage.py             # Management CLI
    Dockerfile            # Container image
    docker-compose.yml    # Local dev stack (Postgres + API)
    alembic/              # Database migrations
  tests/                  # Test suite; run the current gate before quoting counts
  tools/                  # Operational scripts (backup, audit, load test, policy tools)
  data/                   # Registry bundles (items, scales, questionnaires)
  contracts.py            # Integration contracts with AniFold
```

## 3. Key Architectural Rules

### Rule 1: AniFold Owns Z
The Questions Agent NEVER writes the fused latent vector Z. It writes *evidence* (scale scores, state snapshots, projections) for AniFold to consume. AniFold computes Z via cross-attention fusion.

### Rule 2: Pure-Functional Pipeline
All core pipeline modules use frozen dataclasses, no side effects, no global state. Every function takes inputs and returns outputs. The service layer (`service.py`) is the only place with database I/O.

### Rule 3: Two Service Layers
- `pipeline/service.py` + `pipeline/db.py` — SQLite (for testing, demos, local dev)
- `prod/service_pg.py` + `prod/models.py` — PostgreSQL path for promoted deployments
They implement the same logic. The SQLite layer is the reference implementation; the Postgres layer mirrors it for promoted deployments.

### Rule 4: SiteConfig Enforces Regulatory Boundaries
Config A (consumer) must NEVER show clinical thresholds. Config B (CDS) shows transparent scoring. Config C (SaMD) enables AniFold. The `anifold_enabled` flag is a hard regulatory gate. Do not bypass this.

## 4. How a Request Flows

### GET /v1/users/{user_id}/daily-questions

1. `get_or_create_daily_session()` in service.py
2. Resolves registry version, loads item bank
3. Computes answered history, baselines, domain governance
4. Evaluates progressive measurement plan (which package is active)
5. Calls `build_selection_plan()` — the cross-instrument Fisher info optimiser
6. If policy mode is active, passes candidate set to contextual bandit for reranking
7. Creates session record, returns selected questions with progress hints

### POST /v1/users/{user_id}/answers

1. `submit_answers()` in service.py
2. Validates items against registry, stores answer events (idempotent)
3. Captures behavioural metadata (response time, edits, skips) → uncertainty modifiers
4. Inserts scale evidence rows (one answer → multiple scale contributions)
5. Computes scale scores with IRT (GRM) + uncertainty adjustment from metadata
6. Updates baselines (EWMA), state snapshots (9D), Coherence Circle (2D projection)
7. Runs EWS features → drift detection → anamnesis check → follow-up queue
8. Evaluates package completion → advances progressive plan
9. Returns new scores, progress, completion status

## 5. The Database (26 Tables)

### Core Tables
- `users` — user registry
- `user_profiles` — mode, domains, onboarding status, declined items
- `daily_sessions` — one per user per day (lifecycle: created→started→completed)
- `answer_events` — every answer, immutable, idempotent on client_event_id
- `scale_evidence` — per-answer contributions to each scale (the multiplexing)
- `scale_scores` — computed scale scores with IRT theta, SE, reliability
- `scale_baselines` — EWMA baselines for drift detection

### State/Geometry Tables
- `state_snapshots` — 9D latent state (x_hat, x_uncertainty, coverage)
- `circle_snapshots` — 2D Coherence Circle (z, z*, r, theta, velocity)
- `minifold_snapshots` — 4 sub-circles (metabolic, mind, body, social)
- `ews_features` — Early Warning System features (variance, autocorrelation, trend)
- `drift_events` — detected drift episodes
- `follow_up_queue` — anamnesis-triggered follow-up items

### Safety/Governance Tables
- `safety_events` — flagged high-severity responses
- `registry_versions` — item bank versioning
- `user_measurement_packages` — progressive plan tracking

### Policy/Audit Tables
- `policy_decisions` — every bandit decision with context + counterfactuals
- `policy_outcomes` — observed outcomes (completion, burden, uncertainty reduction)
- `observation_events` — non-questionnaire signals (wearables, voice, imaging)
- `operational_metrics` — SLO tracking (latency, error rates)
- `response_metadata` — behavioural metadata per answer
- `session_uncertainty_profiles` — session-level engagement quality
- `experiments` — N-of-1 experiment definitions

### Calibration/Coherence/Risk Tables (v15)
- `personal_calibrations` — N-of-1 Kalman calibration state per (user, scale)
- `session_coherence` — careless responding assessment per session
- `cardio_risk_snapshots` — composite cardiometabolic risk per (user, date)

## 6. Running Locally

```bash
# Install dependencies
pip install -r prod/requirements.txt

# Run tests (should see 721 passed)
python -m pytest tests/ -q

# Start dev API (SQLite)
python -m questions_agent_platform.pipeline.cli serve

# Start production API (PostgreSQL)
docker-compose -f prod/docker-compose.yml up --build

# The API will be at http://localhost:8080
# Health check: GET /health
```

## 7. Key Configuration (QuestionsAgentConfig / Settings)

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `core_questions_per_day` | 5 | Questions per daily session |
| `extra_batch_size` | 1 | Extra questions when user requests more |
| `item_repeat_cooldown_days` | 7 | Min days before re-asking same item |
| `max_active_new_domains` | 2 | Max new domain activations at once |
| `safety_event_lookback_days` | 30 | Window for active safety events |
| `policy_default_mode` | deterministic | Selection mode (deterministic/policy_shadow/policy_live) |

## 8. Adding a New Questionnaire

1. Add items to `data/registry_production/v3/items.json` (item_id, text, response_type, tags)
2. Add scale to `data/registry_production/v3/scales.json` (scale_id, items with weights, scoring_method)
3. If items are shared with existing scales, they automatically get multiplexed
4. Add scoring method in `pipeline/scoring.py` if it's a new instrument type
5. Add IRT parameters in `pipeline/irt.py` (or use defaults which work well)
6. Upload via API: `POST /v1/admin/registry/upload` then `POST /v1/admin/registry/activate/{version}`

## 9. Recently Wired Modules (v15)

These pipeline modules are now fully wired into the production API:

| Module | Tests | What it does | Production |
|--------|-------|-------------|------------|
| `n_of_1.py` | 59 | Person-specific Kalman calibration | `GET/POST /v1/users/{uid}/calibrations` |
| `coherence.py` | 72 | Careless responding detection | `GET /v1/users/{uid}/sessions/{sid}/coherence` |
| `cardio_risk_index.py` | 22 | Composite cardiometabolic score | `POST /v1/users/{uid}/cardio-risk/compute` |

Production wiring includes:
- SQLAlchemy models (`prod/models.py`): `PersonalCalibrationRow`, `SessionCoherence`, `CardioRiskSnapshot`
- Pydantic schemas (`prod/schemas.py`): `CalibrationOut/ListOut`, `CoherenceOut/HistoryOut`, `CardioRiskOut/HistoryOut`
- Service functions (`prod/service_pg.py`): 8 new functions
- Alembic migration `20260227_0005`: 3 new tables + indexes
- FastAPI endpoints (`prod/app.py`): 7 new endpoints

## 10. Step 7 (Multi-Ontology) — DEFERRED

Multi-ontology interpretation layer was designed but deferred. When Bruno says to do it, this maps ICD-10, SNOMED-CT, and LOINC codes to the internal scale/item ontology for clinical interoperability.

---

## Quick Reference: API Endpoints (Production)

### Core Flow
- `GET  /v1/users/{uid}/daily-questions` — Get today's questions
- `POST /v1/users/{uid}/daily-questions/select` — Get questions with policy context
- `POST /v1/users/{uid}/answers` — Submit answers
- `POST /v1/users/{uid}/request-more-context` — Request extra batch

### State & Trajectory
- `GET  /v1/users/{uid}/state` — 9D state snapshots
- `GET  /v1/users/{uid}/circle` — Coherence Circle snapshots
- `GET  /v1/users/{uid}/ews` — Early Warning System features
- `GET  /v1/users/{uid}/drift-events` — Detected drift episodes
- `GET  /v1/users/{uid}/scales` — Scale progress
- `GET  /v1/users/{uid}/scales/{sid}/history` — Scale score history

### N-of-1 Calibration
- `GET  /v1/users/{uid}/calibrations` — List all personal calibrations (with RCI + sufficiency)
- `POST /v1/users/{uid}/calibrations/{scale_id}` — Update calibration with new observation

### Coherence Detection
- `GET  /v1/users/{uid}/sessions/{sid}/coherence` — Get coherence assessment for a session
- `GET  /v1/users/{uid}/coherence/history` — Coherence assessments over time window

### Cardiometabolic Risk
- `POST /v1/users/{uid}/cardio-risk/compute` — Compute composite risk from latest scores
- `GET  /v1/users/{uid}/cardio-risk/history` — Risk snapshots over time window

### Profile & Safety
- `GET/PATCH /v1/users/{uid}/profile` — User profile management
- `GET  /v1/users/{uid}/safety-events` — Safety event listing
- `POST /v1/users/{uid}/safety-events/{eid}/resolve` — Resolve safety event

### Admin
- `GET  /v1/admin/registry/versions` — Registry management
- `POST /v1/admin/registry/upload` — Upload new item bank
- `GET  /v1/admin/slo` — SLO dashboard
- `GET  /v1/admin/audit/export` — Clinical audit export
- `GET  /v1/admin/policy/metrics` — Policy performance
- `POST /v1/admin/privacy/retention/run` — PHI retention cleanup

### Integration
- `GET  /v1/users/{uid}/projection/questions` — Evidence payload for AniFold
- `POST /v1/users/{uid}/observations` — Ingest non-questionnaire signals
- `POST /v1/users/{uid}/policy/outcomes` — AniFold delayed reward feedback
- `GET  /v1/users/{uid}/ani/summary` — Talk-to-Ani summary
- `GET  /v1/users/{uid}/emotion/deep_dive` — Emotion deep dive suggestions

### Contracts
- `GET  /v1/coherence/tier-contract` — Coherence tier definitions
- `GET  /v1/drift-routing-contract` — Drift routing table
