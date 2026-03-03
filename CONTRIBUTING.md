# Contributing

## Quality Bar

Before opening a PR, run:

```bash
./tools/quality_gate.sh
```

This gate enforces:
- compile checks,
- lint checks for syntax/runtime errors,
- strict type checks on extracted critical helpers,
- full automated test suite,
- clean git tree after tests.

## Pull Request Rules

1. Keep PRs scoped to one concern (feature, refactor, bug fix, or infra).
2. Include tests for behavior changes.
3. Do not commit generated runtime/demo output.
4. Keep `main` releasable: no partial migrations or half-wired APIs.

## Commit Format

Use concise conventional prefixes:
- `feat:`
- `fix:`
- `refactor:`
- `test:`
- `ci:`
- `docs:`
- `chore:`
