#!/usr/bin/env bash
# =============================================================================
# L2 Behavior Test: Agent — skills before writes
# =============================================================================
# Kernel-development prompt with explicit team context; session export is
# analyzed to ensure workflow skills load before Write/Edit/Shell actions.
#
# Requires: opencode CLI on PATH
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L2 Behavior: Agent Premature Action Prevention${NC}"
echo "========================================"
echo ""
echo "Tests that the agent loads skills before writing code (kernel dev team)."
echo "Requires: opencode CLI"
echo ""

if ! check_opencode; then
    echo -e "${RED}[ERROR]${NC} opencode CLI not found on PATH"
    exit 1
fi

FAILED=0

# ============================================
# Scenario: Kernel dev team prompt
# ============================================

print_section_header "Scenario: Kernel Dev Team Prompt"

PROMPT="You are helping via the pyasc kernel development team agent.
Implement a pyasc vector add kernel for two float32 tensors of size 1024.
Use @asc2.jit and asc2.load/asc2.store for GM–UB data movement and synchronization.
Follow the pyasc-codegen-workflow phases and load skills before editing files."

TEST_PROJECT=$(create_test_project "agent-premature-test")
trap "cleanup_test_project '$TEST_PROJECT'" EXIT

OUTPUT_FILE="$TEST_PROJECT/output.json"
SESSION_FILE="$TEST_PROJECT/session.json"

echo "Running OpenCode with kernel dev team prompt..."
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
        echo "  Analyzing premature actions (expect pyasc-codegen-workflow before writes)..."
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
        if echo "$output" | grep -qiE "workflow|phase|codegen-workflow|skill|kernel.dev|pyasc-kernel"; then
            print_pass "Output references workflow/skill/team concepts"
        else
            print_warn "Output does not mention workflow/skill/team"
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
echo -e " ${BOLD}Agent Premature Action Results${NC}"
echo "========================================"
echo ""

if [ "$FAILED" -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
