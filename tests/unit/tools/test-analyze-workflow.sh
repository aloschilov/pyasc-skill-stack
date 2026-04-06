#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: Session analysis tools against fixture data
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Session Analysis Tools ==="

FIXTURE="$TESTS_DIR/fixtures/sample-session.json"
ERRORS=0

if [ ! -f "$FIXTURE" ]; then
    echo "[FAIL] Fixture not found: $FIXTURE"
    exit 1
fi

# --- analyze-session.sh ---
echo ""
echo "--- analyze-session.sh ---"

SESSION_TOOL="$TOOLS_DIR/analyze-session.sh"
if [ -f "$SESSION_TOOL" ]; then
    output=$(bash "$SESSION_TOOL" "$FIXTURE" --brief 2>&1) || true
    if echo "$output" | grep -qi "session"; then
        print_pass "analyze-session.sh --brief runs"
    else
        print_fail "analyze-session.sh --brief produced no session output"
        ERRORS=$((ERRORS + 1))
    fi

    output_json=$(bash "$SESSION_TOOL" "$FIXTURE" --json 2>&1) || true
    if echo "$output_json" | grep -q '"file"'; then
        print_pass "analyze-session.sh --json produces JSON"
    else
        print_fail "analyze-session.sh --json did not produce JSON"
        ERRORS=$((ERRORS + 1))
    fi
else
    print_skip "analyze-session.sh not found"
fi

# --- analyze-workflow.sh ---
echo ""
echo "--- analyze-workflow.sh ---"

WORKFLOW_TOOL="$TOOLS_DIR/analyze-workflow.sh"
if [ -f "$WORKFLOW_TOOL" ]; then
    output=$(bash "$WORKFLOW_TOOL" "$FIXTURE" 2>&1) || true

    if echo "$output" | grep -qi "workflow"; then
        print_pass "analyze-workflow.sh runs"
    else
        print_fail "analyze-workflow.sh produced no workflow output"
        ERRORS=$((ERRORS + 1))
    fi

    if echo "$output" | grep -qi "phase"; then
        print_pass "Phase progression detected"
    else
        print_fail "No phase progression found"
        ERRORS=$((ERRORS + 1))
    fi

    if echo "$output" | grep -qi "pyasc-codegen-workflow"; then
        print_pass "Workflow skill reference found"
    else
        print_warn "Workflow skill not detected in output"
    fi

    if echo "$output" | grep -qi "OK\|PASS\|no premature"; then
        print_pass "No premature actions detected"
    else
        print_warn "Premature action check unclear"
    fi
else
    print_skip "analyze-workflow.sh not found"
fi

# --- analyze-tokens.sh ---
echo ""
echo "--- analyze-tokens.sh ---"

TOKEN_TOOL="$TOOLS_DIR/analyze-tokens.sh"
if [ -f "$TOKEN_TOOL" ]; then
    output=$(bash "$TOKEN_TOOL" "$FIXTURE" 2>&1) || true
    if echo "$output" | grep -qiE "token|usage"; then
        print_pass "analyze-tokens.sh runs"
    else
        print_fail "analyze-tokens.sh produced no token output"
        ERRORS=$((ERRORS + 1))
    fi
else
    print_skip "analyze-tokens.sh not found"
fi

# --- Summary ---
echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo "[FAIL] $ERRORS error(s) in session analysis tools"
    exit 1
else
    echo "[PASS] All session analysis tool checks passed"
fi
