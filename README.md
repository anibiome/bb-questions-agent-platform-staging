# Questions Agent Platform (V1) — Reference Implementation

This folder is a runnable blueprint for a **Questions Agent** that:
- serves **5 daily questions** (plus optional one-by-one “Ask Ani more” probes, up to 3/day),
- scores **validated scale structures** with **within-person baselines**,
- supports **item multiplexing** (one item can feed multiple scales),
- outputs **structured evidence for Anifold fusion** (projection + uncertainty + attractor + velocity),
- includes a minimal **dashboard** for sanity-checking unlocks/scores over time.

P0 production-readiness invariants implemented:
- one timeframe per session (`timeframe` persisted on `daily_sessions`)
- session lifecycle (`created|started|completed|abandoned`)
- onboarding lock (cardiometabolic-tagged items first, no domain mixing until completion)
- permanent item decline support (persisted and excluded from future selection)
- auditable multiplex trail (`scale_evidence`: one row per answer→scale contribution)

P1 cardiometabolic package implemented (registry `v2`):
- canonical 27-item deduplicated cardiometabolic battery and six instrument scales,
- canonical scoring engines for FINDRISC, EZ-CVD, SCORED, Lee NAFLD, IPAQ-SF, and AUDIT-C,
- persisted `risk_tier` on scored outputs (SQLite + Postgres paths),
- strict onboarding enforcement across both core and optional extra batches (cardiometabolic only until completion),
- unlock UX metadata (`progress_hint`, `unlock_hint`, `unlock_message`, latest scale status).

P1 runtime state loop implemented:
- persistent anamnesis follow-up queue (`follow_up_queue`) with due-date + priority routing,
- follow-up-aware candidate selection (`anamnesis_followup` reason code, `follow_up` item type),
- auto-resolution of queued follow-ups on answer, decline, or stabilization,
- daily `state_snapshots` (`x_hat`, uncertainty, coverage) and `circle_snapshots` (`z`, `z*`, `r`, `theta`, `velocity`, `acceleration`) with immutable versions,
- first-class trajectory persistence (`ews_features`, `drift_events`) with API retrieval endpoints,
- explicit coherence tier contract (`/v1/coherence/tier-contract`) and cross-modal observation coupling payloads,
- state/circle retrieval endpoints in both reference and production APIs.

The reference engine uses **SQLite + a minimal Python dependency set**. Current runtime requirements are pinned in `prod/requirements.txt` (including `numpy` for the item-efficiency analyzer) to keep behavior reproducible across environments.

## Quick start (demo)

1) Initialize DB and seed demo registry:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli --config questions_agent_platform/config.example.json init-db

PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli --config questions_agent_platform/config.example.json seed-demo
```

2) Run a 30‑day demo simulation (writes an HTML report next to the DB):

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli --config questions_agent_platform/config.example.json simulate
```

Open the generated file:
- `questions_agent_platform/output/questions_agent_demo.report.html`

3) Run unit/integration tests:

```bash
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m unittest -q
```

4) Generate a policy readiness brief (HTML + JSON):

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli \
  --config questions_agent_platform/config.example.json \
  policy-brief
```

This writes files into `questions_agent_platform/output/` and summarizes:
- decision/outcome coverage,
- safety constraint checks,
- explainability payload completeness,
- mode-level outcome metrics,
- rollout recommendation gates.

5) Collect policy evaluation data quickly (for IPS/DR):

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli \
  --config questions_agent_platform/config.example.json \
  simulate-policy \
  --days 120 \
  --users 1 \
  --selection-mode policy_live \
  --epsilon 0.35 \
  --reset-db
```

6) Run rollout gate (exits non-zero if gates fail):

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli \
  --config questions_agent_platform/config.example.json \
  policy-gate \
  --min-eval-rows 100
```

7) Generate weekly policy report (HTML + JSON, print/PDF-ready):

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli \
  --config questions_agent_platform/config.example.json \
  weekly-policy-report \
  --days 7 \
  --min-eval-rows 100 \
  --min-segment-rows 30
```

8) Run the closed-loop truth demo (deterministic end-to-end replay + chain checks):

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli \
  --config questions_agent_platform/config.example.json \
  closed-loop-truth-demo \
  --days 30 \
  --user-id closed-loop-user-1 \
  --epsilon 0.25 \
  --reset-db
```

This writes JSON + HTML into `questions_agent_platform/output/` and verifies:
- policy decisions remain inside candidate set C with constraints enforced,
- candidate/context snapshots are present for auditable replay,
- projection payload anchors link decision/session/date,
- policy outcomes are attached to `decision_id`,
- full-chain coverage and gate verdict (`PASS`/`FAIL`).

## Run the HTTP API + dashboard

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.pipeline.cli --config questions_agent_platform/config.example.json serve
```

Endpoints:
- Health: `GET /health`
- Dashboard: `GET /dashboard`
- Coherence tier contract: `GET /v1/coherence/tier-contract`
- Drift routing contract: `GET /v1/drift-routing-contract`
- Daily questions: `GET /v1/users/{user_id}/daily-questions?date=YYYY-MM-DD`
- Daily questions (policy-aware): `POST /v1/users/{user_id}/daily-questions/select`
- Submit answers: `POST /v1/users/{user_id}/answers`
- More context (one item/call, max 3/day): `POST /v1/users/{user_id}/request-more-context`
- User profile: `GET /v1/users/{user_id}/profile`
- User profile patch: `PATCH /v1/users/{user_id}/profile`
- Progress summary: `GET /v1/users/{user_id}/progress?date=YYYY-MM-DD&window=30d`
- Safety events: `GET /v1/users/{user_id}/safety-events?date=YYYY-MM-DD&days=30&status=open|resolved`
- Resolve safety event: `POST /v1/users/{user_id}/safety-events/{event_id}/resolve`
- Emotion deep dive: `GET /v1/users/{user_id}/emotion/deep_dive?date=YYYY-MM-DD`
- Experiments upsert: `POST /v1/users/{user_id}/experiments`
- Experiments list: `GET /v1/users/{user_id}/experiments?status=active|paused|completed|cancelled`
- Experiment detail: `GET /v1/users/{user_id}/experiments/{experiment_id}`
- Experiment results: `GET /v1/users/{user_id}/experiments/{experiment_id}/results?date=YYYY-MM-DD`
- Scales summary: `GET /v1/users/{user_id}/scales`
- Scale history: `GET /v1/users/{user_id}/scales/{scale_id}/history`
- Questions evidence payload (for Anifold fusion): `GET /v1/users/{user_id}/projection/questions?date=YYYY-MM-DD`
- State snapshots: `GET /v1/users/{user_id}/state?date=YYYY-MM-DD&window=30d`
- Circle snapshots: `GET /v1/users/{user_id}/circle?date=YYYY-MM-DD&window=30d`
- EWS trajectory features: `GET /v1/users/{user_id}/ews?date=YYYY-MM-DD&window=30d`
- Drift events: `GET /v1/users/{user_id}/drift-events?date=YYYY-MM-DD&window=30d`
- ANI narrative summary: `GET /v1/users/{user_id}/ani/summary?date=YYYY-MM-DD`
- Observation events (wearables/voice/imaging/omics): `POST /v1/users/{user_id}/observations`
- Policy outcomes (attach Z/uncertainty): `POST /v1/users/{user_id}/policy/outcomes`

Contract spec:
- `questions_agent_platform/SCHEMA_CONTRACTS_V1.md`
- Includes canonical versioning, compatibility aliases, strict validation rules, and join-key chain.
- Integration execution order:
  - `questions_agent_platform/DEV_EXECUTION_ORDER_INTEGRATION.md`
- Dev handoff quickstart:
  - `questions_agent_platform/DEV_HANDOFF_QUICKSTART.md`
- Architecture map:
  - `questions_agent_platform/ARCHITECTURE_MAP_V2.md`

Admin:
- List versions: `GET /v1/admin/registry/versions`
- Upload registry bundle: `POST /v1/admin/registry/upload`
- Activate version: `POST /v1/admin/registry/activate/{version}`
- Domain promotion readiness: `GET /v1/admin/domains/readiness`

---

## Production deployment (FastAPI + Postgres)

We also shipped a **production‑grade FastAPI service** in:
- `questions_agent_platform/prod/`

It includes:
- FastAPI + Pydantic validation
- SQLAlchemy models for Postgres
- Alembic migrations (strict schema control; no `create_all` bootstrap)
- Dockerfile + docker‑compose
- API key auth
- Same selection/scoring logic as the reference engine

### Local production run (Docker)

```bash
cd questions_agent_platform/prod
cp .env.example .env
docker compose up --build
```

Open:
- API: `http://localhost:8080/health`
- Dashboard: `http://localhost:8080/dashboard` (requires `X-API-Key` header)
- Snapshot APIs (same contract as reference engine):
  - `GET /v1/coherence/tier-contract`
  - `GET /v1/drift-routing-contract`
  - `GET /v1/users/{user_id}/state`
  - `GET /v1/users/{user_id}/circle`
  - `GET /v1/users/{user_id}/ews`
  - `GET /v1/users/{user_id}/drift-events`

### AniFold adapter + local mock integration

Adapter module:
- `questions_agent_platform/pipeline/anifold_adapter.py`

Mock AniFold server:
```bash
python3 -m questions_agent_platform.tools.run_mock_anifold_server --host 127.0.0.1 --port 8091
```

What this gives you:
- contract-safe fetch of selection context (`fusion_to_questions_context`)
- contract-safe submit of evidence (`questions_to_fusion_evidence`)
- contract-safe submit of delayed outcomes (`fusion_to_questions_outcome`)

### Production env vars

See `questions_agent_platform/prod/.env.example:1`

Required:
- `DATABASE_URL`
- `REGISTRY_ROOT`
- `API_KEY`

### Auth

All endpoints require the header:
- `X-API-Key: <your_api_key>`

### Registry volume

In the compose example, the demo registry is mounted:
- `questions_agent_platform/data/registry_demo/v1` → `/app/registry/versions/v1`

In production, mount your own registry root and manage versions with:
- `POST /v1/admin/registry/upload`
- `POST /v1/admin/registry/activate/{version}`

### DB migrations (Alembic)

Migration files:
- `questions_agent_platform/prod/alembic.ini`
- `questions_agent_platform/prod/alembic/env.py`
- `questions_agent_platform/prod/alembic/versions/20260214_0001_initial_schema.py`
- `questions_agent_platform/prod/alembic/versions/20260214_0002_operational_metrics.py`

Runtime behavior is controlled by env:
- `DB_MIGRATION_MODE=auto|require|off`
- `DB_MIGRATION_BASELINE_EXISTING=true|false`

Recommended:
- staging/prod: `DB_MIGRATION_MODE=require`
- local/dev: `DB_MIGRATION_MODE=auto`

Manual commands:
```bash
python3 -m questions_agent_platform.prod.manage upgrade
python3 -m questions_agent_platform.prod.manage require
```

### SLO + alerting endpoints

- `GET /v1/admin/slo?days=14`
  - reports:
    - safe fallback rate
    - answer ingestion p95 latency
    - snapshot write failure rate
  - returns `status=ok|degraded` plus explicit `alerts[]`.

Operational checker:
```bash
python3 -m questions_agent_platform.tools.check_slo_alerts --fail-on-degraded
```

### PHI retention + audit export controls

Retention/redaction endpoint:
- `POST /v1/admin/privacy/retention/run?dry_run=true|false`

Audit export endpoint:
- `GET /v1/admin/audit/export?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
- redaction knobs:
  - `redact_user_ids` (default from env)
  - `redact_raw_payloads` (default from env)

CLI jobs:
```bash
python3 -m questions_agent_platform.tools.run_phi_retention --dry-run
python3 -m questions_agent_platform.tools.export_clinical_audit --out /tmp/qa_audit.json
```

Key PHI env vars:
- `PHI_RETENTION_ANSWER_RAW_DAYS`
- `PHI_RETENTION_OBSERVATION_RAW_DAYS`
- `PHI_RETENTION_POLICY_CONTEXT_DAYS`
- `PHI_EXPORT_REDACT_USER_IDS`
- `PHI_EXPORT_REDACT_RAW_PAYLOADS`
- `PHI_EXPORT_USER_HASH_SALT`

### Load/performance harness

Run concurrency load test against a running API:
```bash
python3 -m questions_agent_platform.tools.load_test_questions_agent \
  --base-url http://localhost:8080 \
  --api-key change_me \
  --users 25 \
  --days 7 \
  --concurrency 10 \
  --out /tmp/qa_load_report.json
```

### AniFold roundtrip smoke harness

Run the full contract loop:
`Anifold context -> daily select -> answers -> projection evidence -> evidence submit -> outcome callback`

```bash
python3 -m questions_agent_platform.tools.run_anifold_roundtrip_smoke \
  --questions-base-url http://localhost:8080 \
  --questions-api-key change_me \
  --anifold-base-url http://localhost:8091 \
  --user-id smoke-user-1 \
  --date 2026-02-16 \
  --selection-mode policy_shadow \
  --questions-date-param-name date_param \
  --out /tmp/qa_roundtrip_smoke.json
```

For the reference SQLite API, use:
- `--questions-date-param-name date`

### Backup/restore drill scripts

Create backup artifact:
```bash
python3 -m questions_agent_platform.tools.db_backup \
  --database-url sqlite+pysqlite:///questions_agent_platform/output/questions_agent_demo.sqlite \
  --out-dir questions_agent_platform/output/backups
```

Restore from backup:
```bash
python3 -m questions_agent_platform.tools.db_restore \
  --backup questions_agent_platform/output/backups/qa_backup_YYYYMMDD_HHMMSS.sqlite3 \
  --database-url sqlite+pysqlite:///questions_agent_platform/output/questions_agent_demo.sqlite
```

Run full backup/restore drill:
```bash
python3 -m questions_agent_platform.tools.run_backup_restore_drill \
  --database-url sqlite+pysqlite:///questions_agent_platform/output/questions_agent_demo.sqlite \
  --fail-on-mismatch
```

For Postgres, the same tools use `pg_dump`/`pg_restore` automatically.  
To verify restore parity against a separate target DB, pass `--restore-database-url`.

### Policy promotion checklist + rollback drill

Promotion gate/checklist:
```bash
python3 -m questions_agent_platform.tools.policy_release_gate \
  --candidate-version v2 \
  --base-url http://localhost:8080 \
  --api-key change_me \
  --db questions_agent_platform/output/questions_agent_demo.sqlite \
  --policy-root questions_agent_platform/data/policy_demo \
  --fail-on-block
```

Rollback drill:
```bash
python3 -m questions_agent_platform.tools.policy_rollback_drill \
  --base-url http://localhost:8080 \
  --api-key change_me \
  --fail-on-error
```

Rollout playbook:
- `questions_agent_platform/POLICY_ROLLOUT_PLAYBOOK_V1.md`

---

## Policy layer (constrained contextual bandit, v1)

This repo now includes an **optional policy layer** that learns to re-rank the daily questions **under strict constraints**:
- The policy can **only choose from the candidate set C** emitted by the deterministic selector (no new item IDs).
- Mandatory anchors are always included.
- Blocked/cooldown items cannot be selected unless explicitly marked `override_allowed`.
- If a safety trigger is active, the system runs a minimal safe protocol and disables exploration.
- The policy **never changes validated content** (question wording, scoring rules, or scale composition). It only changes *timing/selection* within C.

**No LLM is allowed to choose or invent questions.** (An LLM may only rephrase already selected questions if you add such a post-processing step in your own stack.)

Important distinction:
- The Questions Agent produces **evidence**. **Anifold recomputes Z from evidence** (the Questions Agent does not output Z).

### Policy context `x` (unambiguous)

Policy selection uses a context object `x` (per user-day), provided by fusion/Anifold and/or derived from app behavior:
- `Z` (optional) + `Z` uncertainty diag (optional)
- `Z` velocity (optional) + distance-to-attractor (optional)
- adherence stats (completion rates)
- burden stats (median response time proxy)
- protocol/safety flags (`safety_trigger_active`, `allow_context_batches`)
- `identity_mask_id` (optional)

### Production API

New endpoint (supports policy modes + context from fusion/Anifold):
- `POST /v1/users/{user_id}/daily-questions/select`

Body (example):
```json
{
  "date": "2026-02-02",
  "selection_mode": "policy_shadow",
  "identity_mask_id": "work",
  "include_explanations": true,
  "context": {
    "anifold_z": [0.0, 0.1, -0.2],
    "z_uncertainty_diag": [0.3, 0.2, 0.4],
    "z_velocity": [0.01, 0.0, -0.02],
    "z_distance_to_attractor": 0.42
  }
}
```

Delayed reward attachment (fusion → Questions Agent):
- `POST /v1/users/{user_id}/policy/outcomes`

Explanation payload fields (minimum interpretability contract):
- `reason_codes`, `policy_score`, `deterministic_score`
- `top_feature_contributions`
- `counterfactuals` (top alternatives + scores)

Safety trigger mode:
- If safety triggers exist, selection bypasses ML and uses the predefined safe protocol (no exploration).

Online monitoring + rollback:
- Monitor constraint violations (should be 0), adherence, burden, and uncertainty reduction.
- Use `GET /v1/admin/policy/metrics` (includes current rollback guard status).
- Runtime auto-rollback can switch `policy_live` requests to deterministic when safety/performance thresholds degrade.
- Manual rollback remains available by switching active policy version (`POST /v1/admin/policy/activate/{version}`) or pinning via env.

### Policy artifacts (versioned + rollback)

Artifacts are file-based and mounted into the service:
- `POLICY_ROOT=/app/policy`
- versions live in: `{POLICY_ROOT}/versions/<version>/policy_params.json`
- active version file: `{POLICY_ROOT}/active_policy_version.json`

Demo policy artifact included:
- `questions_agent_platform/data/policy_demo/`

### Train + offline eval (SQLite reference engine)

Policy CLI:
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.policy.cli --help
```

Examples:
```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.policy.cli init-demo --policy-root questions_agent_platform/data/policy_demo

PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.policy.cli train-sqlite \
  --db questions_agent_platform/output/questions_agent_demo.sqlite \
  --policy-root questions_agent_platform/output/policy \
  --out-version v2

PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.policy.cli eval-sqlite \
  --db questions_agent_platform/output/questions_agent_demo.sqlite \
  --policy-root questions_agent_platform/output/policy

PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m questions_agent_platform.tools.run_scientific_offline_eval \
  --db questions_agent_platform/output/questions_agent_demo.sqlite \
  --policy-root questions_agent_platform/output/policy \
  --out-json questions_agent_platform/output/scientific_offline_eval.json
```

Note: training/eval requires decision snapshots (`context_json`, `candidate_set_json`) to be stored. Enable via:
- `policy_log_context_snapshot=true`
- `policy_log_candidate_set_snapshot=true`
- For valid IPS/DR estimates, prefer `policy_live` during collection so logged actions equal served actions.
- Scientific eval tool produces one consolidated artifact with:
  - full IPS/DR replay metrics,
  - ablations (`without_z_features`, `without_uncertainty_features`, `without_z_and_uncertainty`),
  - reward calibration bins and ECE summary.

Policy outcomes (what is stored):
- completion + burden proxies
- Anifold-provided `Z` before/after or privacy-safe summaries (quantized `ΔZ` + norms)
- uncertainty before/after summaries (mean/max) enabling uncertainty reduction metrics

Schema hardening status:
- Contract-bound request payloads (`daily-questions/select` context and `policy/outcomes`) are strict (`extra=forbid`).
- Legacy clients without `schema_version` are still accepted (default canonical `1.0`).
- Alias versions (`1`, `v1`, etc.) normalize to `1.0`; unknown versions are rejected.

## Registry format (add/remove questionnaires)

Registries are **versioned** under:
- `{registry_root}/versions/{version}/items.json`
- `{registry_root}/versions/{version}/scales.json`
- `{registry_root}/versions/{version}/questionnaires.json`

Demo registry ships in:
- `questions_agent_platform/data/registry_demo/v1/`

To upload a new registry via API, POST a JSON body like:

```json
{
  "version": "v2",
  "items": [ { "id": "...", "text": "...", "response_type": "likert_0_4", "tags": ["mood"] } ],
  "questionnaires": [ { "id": "...", "version": "1", "name": "...", "domains": ["mood"] } ],
  "scales": [
    {
      "id": "scale_x",
      "questionnaire_id": "...",
      "version": "1",
      "name": "...",
      "response_type": "likert_0_4",
      "min_items_required": 4,
      "unlock_window_days": 14,
      "retest_interval_days": 90,
      "scoring": { "method": "mean", "normalize_min": 0, "normalize_max": 4 },
      "items": [ { "item_id": "...", "reverse": false, "weight": 1.0 } ]
    }
  ]
}
```

## Notes on questionnaire text / licensing

The demo registry uses **synthetic question text**. In production, your team should upload real questionnaire content only if you have the appropriate permissions/licenses.

## Anifold integration (evidence payload)

`GET /v1/users/{user_id}/projection/questions` returns an object shaped like:
- projection vector (128d),
- uncertainty (diag),
- attractor_candidate (baseline vector),
- velocity (diff vs previous),
- distance_from_attractor,
- plus derived scale score summaries.

This follows the separation described in your *Multi‑Omics Projection Specification*:
- **Questions Agent produces geometry‑ready evidence**
- **Anifold produces scores / circle / coherence**
