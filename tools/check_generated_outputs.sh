#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${CI_BASE_REF:-}"
if [[ -z "$BASE_REF" && -n "${GITHUB_BASE_REF:-}" ]]; then
  git fetch --no-tags --depth=1 origin "${GITHUB_BASE_REF}" >/dev/null 2>&1 || true
  BASE_REF="origin/${GITHUB_BASE_REF}"
fi
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git rev-parse HEAD~1 2>/dev/null || true)"
fi

if [[ -z "$BASE_REF" ]]; then
  echo "No base ref available; skipping generated-output diff check."
  exit 0
fi

changed_files="$(git diff --name-only "${BASE_REF}"...HEAD)"
if [[ -z "${changed_files}" ]]; then
  exit 0
fi

blocked_prefixes=(
  "data/policy_demo/"
  "data/registry_demo/"
)

violations=()
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  for prefix in "${blocked_prefixes[@]}"; do
    if [[ "$path" == "$prefix"* ]]; then
      violations+=("$path")
      break
    fi
  done
done <<EOF2
${changed_files}
EOF2

if [[ ${#violations[@]} -gt 0 ]]; then
  echo "Generated/demo outputs must not be modified in PRs:"
  printf ' - %s\n' "${violations[@]}"
  exit 1
fi
