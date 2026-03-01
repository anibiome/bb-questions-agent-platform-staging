# Questions Agent × AniFold Integration — Dev Execution Order

Date: 2026-02-16  
Scope: practical handoff order for engineering teams integrating Questions Agent with AniFold/fusion.

## 0) Outcomes this order guarantees
- Questions Agent can run standalone with deterministic + policy selection.
- AniFold/fusion can provide context (`Z`, uncertainty, velocity) to selection.
- Questions Agent can return structured evidence and delayed outcomes back to AniFold.
- Drift routing is explicit and auditable via a configurable routing table.

---

## 1) Bring up Questions Agent first (no external dependency)
1. Configure and run either stack:
   - Reference: `python3 -m questions_agent_platform.pipeline.cli --config questions_agent_platform/config.example.json serve`
   - Production: `cd questions_agent_platform/prod && docker compose up --build`
2. Verify baseline endpoints:
   - `GET /health`
   - `GET /v1/coherence/tier-contract`
   - `GET /v1/drift-routing-contract`

Exit criteria:
- Service starts cleanly.
- Daily session and answer ingestion work end-to-end.

---

## 2) Lock registry + drift routing contracts
1. Add/activate your real questionnaire registry version.
2. Optional: provide explicit drift routing config JSON and set:
   - reference config: `drift_routing_table_path`
   - prod env: `DRIFT_ROUTING_TABLE_PATH`
3. Validate with:
   - `GET /v1/drift-routing-contract`

Exit criteria:
- Returned contract matches your intended route IDs/domains/instruments.
- Drift events include routing metadata in `details_json.routing`.

---

## 3) Validate contract boundary with mock AniFold
1. Start mock server:
   - `python3 -m questions_agent_platform.tools.run_mock_anifold_server --host 127.0.0.1 --port 8091`
2. Use adapter client (`pipeline/anifold_adapter.py`) to:
   - fetch context from `/v1/context/questions`
   - build `POST /v1/users/{user_id}/daily-questions/select` request
   - submit evidence to `/v1/evidence/questions`
   - submit outcomes to `/v1/outcomes/questions`
3. Inspect mock receipts:
   - `GET /v1/logs`

Exit criteria:
- Context contract validates.
- Evidence and outcome contracts validate and are accepted.

---

## 4) Wire real AniFold service using same contract surface
AniFold side should expose equivalents of:
- `GET /v1/context/questions?user_id=...&date=...`
- `POST /v1/evidence/questions`
- `POST /v1/outcomes/questions`

Questions Agent side already supports:
- context ingestion: `POST /v1/users/{user_id}/daily-questions/select`
- evidence export: `GET /v1/users/{user_id}/projection/questions`
- delayed outcome ingest: `POST /v1/users/{user_id}/policy/outcomes`

Exit criteria:
- No contract mismatches for schema version/name/vector lengths.
- Join keys (`subject_id`, `date`, `session_id`, `decision_id`) remain consistent across systems.

---

## 5) Production hardening sequence
1. Run migrations in controlled mode (`DB_MIGRATION_MODE=require` in prod).
2. Run load harness (`tools/load_test_questions_agent.py`).
3. Enable SLO checks (`GET /v1/admin/slo`, `tools/check_slo_alerts.py`).
4. Enable PHI retention + redaction jobs (`tools/run_phi_retention.py`, audit export controls).
5. Run policy promotion + rollback drills.

Exit criteria:
- Load/SLO/rollback/retention checks pass.
- Safety fallback rate and snapshot-write failure rate stay within thresholds.

---

## 6) Suggested day-one deploy order
1. Deploy Questions Agent API (deterministic mode on).
2. Enable policy in shadow mode.
3. Integrate AniFold context feed for selection.
4. Start sending projection evidence to AniFold.
5. Enable delayed policy outcome updates.
6. Promote policy live gradually with rollback guard enabled.

This order minimizes coupling risk and keeps every boundary observable.
