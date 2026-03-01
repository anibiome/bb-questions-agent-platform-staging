# Questions Agent Policy Rollout Playbook (v1)

Date: 2026-02-14  
Scope: Production policy promotion, staged rollout, and rollback drills.

## 1) Promotion Preconditions (must pass)

Run the release checklist:

```bash
python3 -m questions_agent_platform.tools.policy_release_gate \
  --candidate-version v2 \
  --base-url http://localhost:8080 \
  --api-key change_me \
  --db questions_agent_platform/output/questions_agent_demo.sqlite \
  --policy-root questions_agent_platform/data/policy_demo \
  --fail-on-block
```

Required pass conditions:
- candidate policy version exists and is not already active
- policy quality gates from `policy_brief` are all `PASS`
- `/v1/admin/slo` returns `status=ok`
- `/v1/admin/policy/metrics` rollback guard is clear (`rollback=false`)

If any condition fails, rollout is blocked.

## 2) Staged Rollout (manual traffic split)

Use staged production traffic exposure:
- Phase A: 1% `policy_live` for 24h
- Phase B: 5% `policy_live` for 48h
- Phase C: 20% `policy_live` for 72h
- Phase D: 50% `policy_live` after sustained stability

At each phase boundary:
- check `GET /v1/admin/slo?days=14`
- check `GET /v1/admin/policy/metrics?days=30`
- verify no safety constraint violations
- verify burden/adherence are not degraded beyond rollback thresholds

If any guard fails, immediately rollback to deterministic or previous policy version.

## 3) Rollback Drill (required before/after release)

Execute rollback activation drill:

```bash
python3 -m questions_agent_platform.tools.policy_rollback_drill \
  --base-url http://localhost:8080 \
  --api-key change_me \
  --fail-on-error
```

Expected drill behavior:
- switch from active policy -> fallback policy
- verify fallback is active
- restore original policy
- verify original is active again

Persist both generated artifacts:
- `policy_rollback_drill_*.json`
- `policy_rollback_drill_*.md`

## 4) Backup/Restore Drill (required before release window)

Run database backup/restore drill:

```bash
python3 -m questions_agent_platform.tools.run_backup_restore_drill \
  --database-url sqlite+pysqlite:///questions_agent_platform/output/questions_agent_demo.sqlite \
  --fail-on-mismatch
```

Persist artifact:
- `qa_backup_*.sqlite3` (or `.dump` for Postgres)
- drill JSON report for release evidence

## 5) Release Evidence Pack

Attach these artifacts to each policy promotion record:
- policy release checklist (`policy_release_checklist_*.json` + `.md`)
- rollback drill report (`policy_rollback_drill_*.json` + `.md`)
- backup/restore drill report
- latest SLO report (`/v1/admin/slo`)
- latest policy metrics report (`/v1/admin/policy/metrics`)

Without this pack, release is not complete.
