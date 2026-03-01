# Questions Agent v2.0 — 23-Section Compliance Matrix

Date: 2026-02-14  
Scope: `questions_agent_platform`  
Reference: "QUESTIONS AGENT — The Complete Specification (v2.0, 23 sections)"

Status legend:
- `✅` Implemented in code
- `🟨` Implemented with scoped gaps
- `❌` Not implemented

| # | Spec section | Status | Current implementation evidence | Gap to literal 100% |
|---|---|---|---|---|
| 1 | What the Questions Agent is | ✅ | Item-level selector, 5/day contract, multiplexing, no diagnosis language, constrained item bank (`pipeline/selection.py`, `pipeline/service.py`, `prod/service_pg.py`) | None for v2 backend scope |
| 2 | Versioning model: progressive promotion | 🟨 | Active/queued/promoted domains, readiness gate endpoint (`build_domain_promotion_readiness`) and domain governance (`pipeline/governance.py`, `pipeline/service.py`, `prod/service_pg.py`, `/v1/admin/domains/readiness`) | External AniFold retraining/promotion orchestration remains outside this repo |
| 3 | Cardiometabolic core (6 instruments) | ✅ | Canonical scoring for FINDRISC, EZ-CVD, SCORED, Lee NAFLD, IPAQ-SF, AUDIT-C (`pipeline/scoring.py`) + demo registry content | None for backend scope |
| 4 | Initial scan onboarding | ✅ | Hard cardiometabolic-first lock until completion, enforced in reference + prod (`pipeline/service.py`, `prod/service_pg.py`) | None |
| 5 | Multiplexing: one answer many scales | ✅ | `answer_events` + per-scale `scale_evidence` writes with auditable mappings (`pipeline/service.py`, `prod/service_pg.py`, `pipeline/db.py`, `prod/sql/001_init.sql`) | None |
| 6 | Selector: dual objective and constraints | ✅ | Timeframe coherence, diversity, intrusiveness, repeat logic, unlock bias, policy modes, safe fallback (`pipeline/selection.py`, `pipeline/service.py`, `prod/service_pg.py`) | None for implemented policy generation |
| 7 | Gamification and unlock cadence | ✅ | Progress hints, unlock hints/messages, scale completion metadata in APIs (`pipeline/service.py`, `pipeline/api.py`, `prod/app.py`) | None |
| 8 | Anamnesis branching | ✅ | Persistent follow-up queue, priority pull-first behavior, resolution on evidence stabilization (`follow_up_queue` + service logic) | None |
| 9 | AniFold → Questions feedback loop | ✅ | Policy context accepts Z/uncertainty/velocity and supports drift/safety-triggered targeted probing; drift routing is now an explicit configurable protocol map (`pipeline/drift_routing.py`, `pipeline/service.py`, `prod/service_pg.py`, `/v1/drift-routing-contract`) | None for backend scope |
| 10 | Repetition strategy | ✅ | `trial|consumer` mode, repeat policy, onboarding repeats, extras cap (`user_profiles.mode`, selector constraints, services) | None |
| 11 | Questions latent space construction | ✅ | Daily `state_snapshots` and `circle_snapshots` with immutable schema/projection version fields + first-class `ews_features` persistence (`pipeline/state_snapshots.py`, `pipeline/db.py`, `prod/models.py`) | None for backend scope |
| 12 | Branching beyond cardiometabolic | ✅ | Max concurrent new domains, active/queued/promoted governance, promotion readiness criteria including paired ratio + variance (`pipeline/governance.py`, `build_domain_promotion_readiness`) | None for backend control plane |
| 13 | Behavioral analytics | ✅ | Latency, edit count, channel, skip flags captured per answer/observation metadata | None |
| 14 | Wearable + imaging integration | ✅ | Observation ingestion APIs + modality-specific uncertainty coupling payload persisted in `observation_events.coupling_json` and returned by API (`pipeline/service.py`, `prod/service_pg.py`, `/v1/users/{user_id}/observations`) | None for current backend scope |
| 15 | Coherence scale (5-tier UX layer) | ✅ | Explicit coherence tier contract object via `GET /v1/coherence/tier-contract` + persisted per-snapshot score/tier (`circle_snapshots.coherence_score`, `coherence_tier_json`) | None |
| 16 | Safety and clinical guardrails | ✅ | Safety events, safety override selection, escalation/resolve endpoints, guardrail language in payloads | None for current backend scope |
| 17 | Data model | ✅ | Core objects implemented including first-class `drift_events` + `ews_features` entities (SQLite + Postgres + Alembic) (`pipeline/db.py`, `prod/models.py`, `prod/alembic/versions/20260215_0004_ews_drift_events.py`) | None |
| 18 | API surface | ✅ | Spec endpoints covered, including dedicated progress contract (`GET /v1/users/{user_id}/progress`) and full experiment/deep-dive/policy routes | None for backend scope |
| 19 | Delivery phases | 🟨 | Phase 1 + core Phase 2 delivered; Phase 3 foundations present (observations/experiments) | Full Phase 3 multimodal fusion + affective deep-dive orchestration remains incremental |
| 20 | Success metrics | ✅ | Product metrics, policy metrics, SLOs, rollback guard, load/ops checks plus consolidated offline scientific eval pipeline (`tools/run_scientific_offline_eval.py`) with ablations and calibration | None for backend scope |
| 21 | What not to build (anti-goals) | ✅ | Item-unit interaction, no diagnosis claims, no mixed timeframe sessions, constrained question bank enforced | None |
| 22 | Source documents archived | ✅ | Consolidated operational docs/checklists/readme in repo (`README.md`, `PRODUCTION_READINESS_CHECKLIST_V2.md`) | None for code-delivery scope |
| 23 | "Remember 3 things" operating rule | ✅ | Enforced in logic: one timeframe, multiplexed evidence, next-best-question selection (`selection.py`, `service.py`) | None |

## Net result
- Fully implemented (`✅`): 21 / 23
- Implemented with scoped gaps (`🟨`): 2 / 23
- Not implemented (`❌`): 0 / 23

## Highest-impact remaining closures for literal 100%
1. Externalize AniFold retraining/promotion orchestration into a dedicated control-plane service (currently represented via readiness gates and flags only).
2. Extend Phase 3 multimodal fusion and affective deep-dive orchestration from foundational objects to full production workflows.
