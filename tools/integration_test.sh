#!/usr/bin/env bash
# ============================================================================
# Questions Agent — Integration Test Kit (curl)
#
# Runs the core daily workflow against a live API instance:
#   1. Health check
#   2. Get daily questions for a user
#   3. Submit answers
#   4. Check progress
#   5. Request extra batch
#
# Usage:
#   export QA_BASE_URL=http://localhost:8088   # or your deployment URL
#   export QA_API_KEY=your-api-key             # if auth is enabled
#   bash tools/integration_test.sh
#
# Prerequisites:
#   - jq (brew install jq / apt install jq)
#   - Running API instance
# ============================================================================

set -euo pipefail

BASE="${QA_BASE_URL:-http://localhost:8088}"
AUTH="${QA_API_KEY:-}"
USER_ID="${QA_TEST_USER:-integration-test-user-1}"
DATE="${QA_TEST_DATE:-$(date +%Y-%m-%d)}"

# Auth header
AUTH_HEADER=""
if [ -n "$AUTH" ]; then
  AUTH_HEADER="-H \"Authorization: Bearer $AUTH\""
fi

PASS=0
FAIL=0

run_test() {
  local name="$1"
  local cmd="$2"
  local check="$3"

  echo ""
  echo "--- TEST: $name ---"
  RESPONSE=$(eval "$cmd" 2>/dev/null) || { echo "  FAIL: curl error"; FAIL=$((FAIL+1)); return; }

  if echo "$RESPONSE" | eval "$check" > /dev/null 2>&1; then
    echo "  PASS"
    PASS=$((PASS+1))
  else
    echo "  FAIL"
    echo "  Response: $(echo "$RESPONSE" | head -c 500)"
    FAIL=$((FAIL+1))
  fi
}

echo "============================================================"
echo "Questions Agent Integration Test"
echo "============================================================"
echo "Base URL:  $BASE"
echo "User ID:   $USER_ID"
echo "Date:      $DATE"
echo "============================================================"

# --- 1. Health check ---
run_test "Health check" \
  "curl -s $BASE/health" \
  "jq -e '.status == \"ok\"'"

# --- 2. Get daily questions ---
run_test "GET daily questions" \
  "curl -s $AUTH_HEADER $BASE/v1/users/$USER_ID/daily-questions?date_param=$DATE" \
  "jq -e '.session_id and (.questions | length) > 0'"

# Capture session_id and item_ids for next step
SESSION_RESPONSE=$(curl -s $AUTH_HEADER "$BASE/v1/users/$USER_ID/daily-questions?date_param=$DATE" 2>/dev/null || echo "{}")
SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id // empty')
ITEM_IDS=$(echo "$SESSION_RESPONSE" | jq -r '[.questions[].item_id] | join(" ")' 2>/dev/null || echo "")

if [ -z "$SESSION_ID" ]; then
  echo ""
  echo "FATAL: Could not get session_id. Is the server running at $BASE?"
  echo "Total: $PASS passed, $FAIL failed"
  exit 1
fi

echo ""
echo "  Session ID: $SESSION_ID"
echo "  Items: $ITEM_IDS"

# --- 3. Submit answers ---
ANSWERS="["
FIRST=true
for ITEM_ID in $ITEM_IDS; do
  if [ "$FIRST" = true ]; then FIRST=false; else ANSWERS+=","; fi
  ANSWERS+="{\"client_event_id\":\"test-${USER_ID}-${DATE}-${ITEM_ID}\","
  ANSWERS+="\"item_id\":\"${ITEM_ID}\","
  ANSWERS+="\"value\":3,"
  ANSWERS+="\"answered_at\":\"${DATE}T12:00:00Z\"}"
done
ANSWERS+="]"

SUBMIT_BODY="{\"session_id\":\"${SESSION_ID}\",\"answers\":${ANSWERS}}"

run_test "POST answers" \
  "curl -s -X POST $AUTH_HEADER -H 'Content-Type: application/json' -d '$SUBMIT_BODY' $BASE/v1/users/$USER_ID/answers" \
  "jq -e '.inserted_answer_events >= 0'"

# --- 4. Check progress ---
run_test "GET user progress" \
  "curl -s $AUTH_HEADER $BASE/v1/users/$USER_ID/progress" \
  "jq -e '.scales'"

# --- 5. Get user profile ---
run_test "GET user profile" \
  "curl -s $AUTH_HEADER $BASE/v1/users/$USER_ID/profile" \
  "jq -e '.user_id'"

# --- 6. Get user scales ---
run_test "GET user scales" \
  "curl -s $AUTH_HEADER $BASE/v1/users/$USER_ID/scales" \
  "jq -e '. | type == \"array\" or type == \"object\"'"

# --- 7. Request more context (extra batch) ---
MORE_BODY="{\"session_id\":\"${SESSION_ID}\"}"
run_test "POST request-more-context" \
  "curl -s -X POST $AUTH_HEADER -H 'Content-Type: application/json' -d '$MORE_BODY' $BASE/v1/users/$USER_ID/request-more-context" \
  "jq -e '. | type == \"object\"'"

# --- 8. Idempotency check (re-submit same answers) ---
run_test "POST answers (idempotent re-submit)" \
  "curl -s -X POST $AUTH_HEADER -H 'Content-Type: application/json' -d '$SUBMIT_BODY' $BASE/v1/users/$USER_ID/answers" \
  "jq -e '.inserted_answer_events == 0'"

# --- 9. Contract endpoints ---
run_test "GET tier contract" \
  "curl -s $BASE/v1/coherence/tier-contract" \
  "jq -e '. | type == \"object\"'"

run_test "GET drift routing contract" \
  "curl -s $BASE/v1/drift-routing-contract" \
  "jq -e '. | type == \"object\"'"

# --- 10. Admin: registry versions ---
run_test "GET registry versions" \
  "curl -s $AUTH_HEADER $BASE/v1/admin/registry/versions" \
  "jq -e '.versions | length > 0'"

# --- Summary ---
echo ""
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed (total: $((PASS+FAIL)))"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
