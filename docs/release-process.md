# Release Process

## Versioning

Use semantic versioning (`MAJOR.MINOR.PATCH`).

- `MAJOR`: breaking API/schema changes.
- `MINOR`: backward-compatible features.
- `PATCH`: bug fixes and non-breaking ops/docs updates.

## Release Checklist

1. Run `./tools/quality_gate.sh`.
2. Confirm DB and API contract compatibility notes are documented.
3. Update `CHANGELOG.md` with user-visible changes.
4. Tag release (`vX.Y.Z`) from `main`.
5. Publish release notes with:
   - schema/API deltas,
   - migration steps,
   - rollback plan.
6. Deploy only from signed release tags; no direct production deploy from arbitrary commits.

## Hotfix Policy

- Branch from the latest release tag.
- Include regression test for the bug.
- Merge back to `main` after patch release.
