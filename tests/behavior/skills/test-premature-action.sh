#!/usr/bin/env bash
# =============================================================================
# L2 Behavior Test: Premature Action Prevention (Agent-in-the-Loop)
# =============================================================================
# Runs OpenCode with a kernel development prompt, then exports the session
# and analyzes it to verify the agent loaded the workflow skill before
# writing any code files.
#
# Requires: opencode CLI on PATH
# Estimated time: 2-5 minutes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L2 Behavior: Premature Action Prevention${NC}"
echo "========================================"
echo ""
echo "Tests that the agent loads skills before writing code."
echo "Requires: opencode CLI"
echo ""

if ! check_opencode; then
    echo -e "${RED}[ERROR]${NC} opencode CLI not found on PATH"
    exit 1
fi

FAILED=0

# ============================================
# Scenario 1: Kernel development prompt
# ============================================

print_section_header "Scenario 1: Kernel Dev Prompt"

PROMPT="Implement a pyasc vector add kernel for two float32 tensors of size 1024.
Use @asc.jit and manual sync with set_flag/wait_flag.
Follow the pyasc-codegen-workflow phases."

TEST_PROJECT=$(create_test_project "premature-test")
trap "cleanup_test_project '$TEST_PROJECT'" EXIT

OUTPUT_FILE="$TEST_PROJECT/output.json"
SESSION_FILE="$TEST_PROJECT/session.json"

echo "Running OpenCode with kernel dev prompt..."
echo "  Working dir: $TEST_PROJECT"
echo "  Timeout: 120s"
echo ""

if timeout 120 opencode run "$PROMPT" \
    --format json \
    --dir "$TEST_PROJECT" > "$OUTPUT_FILE" 2>&1; then
    echo "  Execution completed."
else
    ec=$?
    if [ "$ec" -eq 124 ]; then
        print_warn "OpenCode timed out after 120s"
    else
        print_info "OpenCode exited with code $ec"
    fi
fi

echo ""

# ============================================
# Session export and analysis
# ============================================

print_section_header "Session Analysis"

SESSION_ID=$(find_recent_session)

if [ -n "$SESSION_ID" ]; then
    print_info "Found session: $SESSION_ID"
    export_session "$SESSION_ID" "$SESSION_FILE"

    if [ -s "$SESSION_FILE" ]; then
        echo ""
        echo "  Analyzing premature actions..."
        if ! analyze_premature_actions "$SESSION_FILE" "skill" "pyasc-codegen-workflow"; then
            FAILED=$((FAILED + 1))
        fi

        echo ""
        echo "  Tool chain:"
        analyze_tool_chain "$SESSION_FILE"
    else
        print_warn "Session export was empty"
    fi
else
    print_skip "No recent session found (opencode session list returned nothing)"
    print_info "Falling back to output-text analysis..."

    if [ -f "$OUTPUT_FILE" ]; then
        output=$(cat "$OUTPUT_FILE")
        if echo "$output" | grep -qiE "workflow|phase|codegen-workflow|skill"; then
            print_pass "Output references workflow/skill concepts"
        else
            print_warn "Output does not mention workflow/skill"
        fi
        if echo "$output" | grep -qiE "design.*before.*implement|phase 0.*phase 1|environment.*design"; then
            print_pass "Output indicates phased approach"
        else
            print_info "Could not confirm phased approach from output alone"
        fi
    fi
fi

# ============================================
# Summary
# ============================================

echo ""
echo "========================================"
echo -e " ${BOLD}Premature Action Results${NC}"
echo "========================================"
echo ""

if [ "$FAILED" -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
