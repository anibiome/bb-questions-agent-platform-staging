# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `pipeline/follow_up_queue.py` to centralize follow-up queue enqueue and due-item logic.
- Regression coverage for follow-up queue behavior in `tests/test_follow_up_queue.py`.
- `.python-version` and `mypy.ini` for reproducible runtime + static type checking baseline.
- Engineering process baselines:
  - `CONTRIBUTING.md`
  - `docs/release-process.md`
  - `docs/observability-baseline.md`
  - `docs/adr/0000-template.md`
- `tools/quality_gate.sh` for compile, lint, type-check, unit tests, and clean-tree enforcement.

### Changed
- `.github/workflows/ci.yml` now runs a stricter CI baseline around the quality gate and typed test dependencies.
- `pipeline/service.py` now imports queue helpers from `pipeline/follow_up_queue.py` and removes duplicate internal implementations.
- `prod/service_pg.py` includes a missing `SessionUncertaintyProfile` import fix.
- `requirements-test.txt` now uses valid, modern tool pins for test/lint/type gates.
- Extra-batch retrieval in both SQLite and Postgres services now supports explicit `user_id` ownership checks and configurable per-day caps.
