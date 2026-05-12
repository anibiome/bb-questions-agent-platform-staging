# Questions Agent — Dev Handoff Quickstart (Respectful + Practical)

Date: 2026-03-01 (v17)
Audience: engineering team integrating Questions Agent with app + AniFold/fusion

## 1) What this package already gives you

This repository is not a starter skeleton; it is a working backend baseline with:
- deterministic + constrained policy daily selection (5/day core contract),
- one-timeframe-per-session enforcement,
- multiplexed evidence trail (`answer_events` + `scale_evidence`),
- cardiometabolic onboarding lock + unlock flow,
- follow-up queue (anamnesis) + state/circle snapshots,
- **behavioural metadata uncertainty modifiers** (Claim Family 5) — per-response latency, edit count, skip/decline signals adjust SE(theta),
- **drift-triggered anamnesis episodes** (Claim Family 4) — automatic clinical probing when EWS detects drift,
- **session engagement profiling** — focused/normal/distracted/fatigued classification feeds back into selection,
- **N-of-1 calibration**, **coherence detection**, **cardiometabolic risk index**,
- API endpoints across user, admin, and operational domains; run the current contract gate before quoting endpoint counts,
- **SiteConfig regulatory separation** — consumer/cds_default/trial_trial/samd_full with feature gates,
- staged FastAPI/Postgres reference stack with Alembic migrations and ops tools,
- current test count must come from the latest `./tools/quality_gate.sh` run.

You can treat it as an integration reference implementation and contract surface.

## 2) What your team still needs to plug in

1. **App-layer UX wiring**
   - Connect daily session, unlock messaging, “ask one more,” safety cards, engagement feedback in your mobile/web client.
   - Wire the response metadata payload: when displaying questions, capture `response_latency_ms`, `edit_count`, `channel` (tap/voice/keyboard/swipe), and send in the `metadata` field of each answer.
2. **Real AniFold endpoints**
   - Wire your production context/evidence/outcome endpoints to the existing adapter contracts.
3. **Ops environment values**
   - Set production keys, DB URLs, retention policy env vars, and SLO thresholds for your infra.
4. **Production registry activation**
   - Registry v3 (941 items, 65 scales, 44 response types) is included and mounted in Docker. Activate it via `POST /v1/admin/registry/activate/v3`.

## 3) Integration order (recommended)

Use this exact sequence:
1. Bring up Questions Agent standalone and pass health/smoke.
2. Lock registry version + drift routing contract.
3. Validate boundary against mock AniFold.
4. Swap mock to real AniFold using same contracts.
5. Run load + SLO + rollback + retention drills.
6. Roll out policy in shadow first, then gradual live traffic.

Detailed runbook:
- `DEV_EXECUTION_ORDER_INTEGRATION.md`

## 4) “Mini holes” reviewers may point out (expected seams)

These are normal integration seams, not fundamental architecture breaks:

1. **Field-name mismatches across systems**
   - Example: date parameter naming or optional context keys.
   - Mitigation: keep contract tests and roundtrip smoke in CI.
2. **Registry quality issues**
   - Missing mappings/tags can weaken selection quality.
   - Mitigation: enforce registry lint + version review before activation.
3. **Environment-specific performance variance**
   - SQLite/local runs are not production latency profiles.
   - Mitigation: use Postgres + load harness before go-live.
4. **Policy data sparsity during early rollout**
   - IPS/DR quality depends on enough logged coverage.
   - Mitigation: shadow mode first; promote only after gate passes.
5. **PHI export/retention interpretation differences**
   - Legal/security teams may request stricter defaults.
   - Mitigation: tune retention/redaction envs and verify audit exports.

## 5) 30-minute verification commands (first-day handoff)

From workspace root:

```bash
python3 -m questions_agent_platform.pipeline.cli --config questions_agent_platform/config.example.json init-db
python3 -m questions_agent_platform.pipeline.cli --config questions_agent_platform/config.example.json seed-demo
python3 -m pytest -q questions_agent_platform/tests --maxfail=1
```

Roundtrip contract smoke (mock AniFold + Questions Agent):

```bash
python3 -m questions_agent_platform.tools.run_anifold_roundtrip_smoke \
  --questions-base-url http://localhost:8080 \
  --questions-api-key change_me \
  --anifold-base-url http://localhost:8091 \
  --user-id smoke-user-1 \
  --date 2026-02-16 \
  --selection-mode policy_shadow
```

## 6) “Done” definition for your dev team

You are promotion-ready for this scope when all are true:
- migrations are controlled by Alembic in your deployment path,
- roundtrip contracts pass against real AniFold,
- load + SLO + rollback + retention drills pass in staging,
- policy stays in-constraint with zero safety-constraint violations,
- your app renders daily flow, unlocks, progress, and safety UX correctly.

## 7) New in v17: Response Metadata Payload Format

When submitting answers, the client can include behavioural metadata per answer:

```json
{
  "session_id": "...",
  "answers": [
    {
      "client_event_id": "evt_1",
      "item_id": "item_mood_1",
      "value": 3.0,
      "answered_at": "2026-03-01T08:30:00Z",
      "metadata": {
        "response_latency_ms": 3200,
        "edit_count": 0,
        "was_skipped": false,
        "was_declined": false,
        "channel": "tap",
        "time_of_day_hour": 8
      }
    }
  ]
}
```

The backend computes uncertainty modifiers and adjusts SE(theta) automatically. The response includes `session_engagement` with engagement quality classification.

## 8) New in v17: Anamnesis Episode Lifecycle

When drift is detected (EWS >= 0.65, radius deviation > 2 sigma, or velocity spike):
1. An anamnesis episode is created for the affected clinical domain
2. Follow-up items are enqueued in the follow-up queue with elevated priority
3. The episode stays active until: scale completed, EWS subsides, or max_days exceeded
4. Resolution states: `confirmed` (scale completed), `denied` (EWS subsided), `expired` (timeout), `resolved` (manual)

Endpoints:
- `GET /v1/users/{user_id}/anamnesis/episodes` — list episodes (filter by `?status=active`)
- `GET /v1/users/{user_id}/anamnesis/episodes/{episode_id}` — single episode
- `POST /v1/users/{user_id}/anamnesis/episodes/{episode_id}/resolve` — manual resolution
- `GET /v1/users/{user_id}/sessions/{session_id}/engagement` — engagement profile
- `GET /v1/users/{user_id}/follow-up-queue` — pending follow-up items

## 9) Feature Gates (SiteConfig)

All new behaviour is gated behind SiteConfig flags:

| Flag | Default | Effect |
|------|---------|--------|
| `behavioural_metadata_enabled` | `true` (consumer) | Captures response metadata, adjusts SE(theta) |
| `anamnesis_enabled` | `true` (consumer) | Creates anamnesis episodes on drift |
| `concordance_mode` | `false` | Enables concordance validation (offline) |

SiteConfig types: `consumer`, `cds_default`, `trial_trial`, `samd_full`

## 10) Docker Quick Start

```bash
cd prod/
docker-compose up --build
# Wait for health check: curl http://localhost:8080/health
# Activate production registry:
curl -X POST http://localhost:8080/v1/admin/registry/activate/v3 \
  -H "x-api-key: change_me"
```

---

If needed for stakeholder review, pair this with:
- `PRODUCTION_READINESS_CHECKLIST_V2.md` (full closure checklist)
- `SPEC_V2_23_SECTION_COMPLIANCE_MATRIX_2026-02-14.md` (section-by-section spec mapping)
