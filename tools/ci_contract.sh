#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUFF_BIN="${RUFF_BIN:-ruff}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache}"

run_lint() {
  "$PYTHON_BIN" -m compileall -q pipeline policy prod tools scripts tests sitecustomize.py
  "$RUFF_BIN" check pipeline policy prod tools scripts tests sitecustomize.py --exclude .git,.venv,.audit_venv,.audit_venv312,node_modules,output,data
}

run_type_check() {
  local targets=(
    pipeline/service.py
    pipeline/follow_up_queue.py
    pipeline/api.py
    prod/service_pg.py
  )
  if [[ "${STRICT_TYPECHECK_FULL:-0}" == "1" ]]; then
    local extra_targets=(
      pipeline/service_monolith.py
      pipeline/api_monolith.py
      prod/service_pg_monolith.py
      prod/app.py
    )
    local extra
    for extra in "${extra_targets[@]}"; do
      if [[ -f "$extra" ]]; then
        targets+=("$extra")
      fi
    done
  fi
  PYTHONPATH=. "$PYTHON_BIN" -m mypy --config-file mypy.ini --explicit-package-bases "${targets[@]}"
}

run_unit() {
  PYTHONPATH=. "$PYTHON_BIN" -m unittest -q
}

run_architecture_contract() {
  PYTHONPATH=. "$PYTHON_BIN" -m unittest -q tests.test_architecture_boundaries
}

run_integration_smoke() {
  PYTHONPATH=. "$PYTHON_BIN" -m unittest -q \
    tests.test_integration_demo \
    tests.test_p2_anifold_roundtrip_smoke
  PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    "$PYTHON_BIN" -m pytest -q tests/test_prod_public_api_contract_snapshot.py --maxfail=1 -c /dev/null --rootdir .
}

run_clean_tree() {
  if [[ "${ALLOW_DIRTY_WORKTREE:-0}" == "1" ]]; then
    echo "ALLOW_DIRTY_WORKTREE=1; skipping clean-tree gate."
    return 0
  fi

  if [[ -n "$(git status --porcelain)" ]]; then
    git status --short
    echo "clean-tree gate failed: repository contains uncommitted changes."
    exit 1
  fi
}

run_generated_guard() {
  ./tools/check_generated_outputs.sh
}

case "$TARGET" in
  lint)
    run_lint
    ;;
  type-check)
    run_type_check
    ;;
  unit)
    run_unit
    ;;
  architecture)
    run_architecture_contract
    ;;
  integration-smoke)
    run_integration_smoke
    ;;
  clean-tree)
    run_clean_tree
    ;;
  generated-guard)
    run_generated_guard
    ;;
  all)
    run_lint
    run_type_check
    run_architecture_contract
    run_unit
    run_integration_smoke
    run_generated_guard
    run_clean_tree
    ;;
  *)
    echo "Unknown gate '$TARGET'. Expected: lint|type-check|architecture|unit|integration-smoke|generated-guard|clean-tree|all"
    exit 2
    ;;
esac
