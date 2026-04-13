#!/usr/bin/env bash
# =============================================================================
# L3 Integration Test: Full Workflow Execution (Agent-in-the-Loop)
# =============================================================================
# Runs OpenCode through a complete pyasc-codegen-workflow: environment check,
# design, implementation with review, and verification.  Validates that the
# agent follows the phased workflow via both artifact inspection and session
# analysis.
#
# Requires: opencode CLI, python3.10
# Estimated time: 5-15 minutes
# WARNING: This is a long-running integration test.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L3 Integration: Workflow Execution${NC}"
echo "========================================"
echo ""
echo "Tests full phased workflow via OpenCode."
echo "Estimated time: 5-15 minutes"
echo "Requires: opencode CLI"
echo ""

if ! check_opencode; then
    echo -e "${RED}[ERROR]${NC} opencode CLI not found on PATH"
    exit 1
fi

# ============================================
# Setup test project
# ============================================

TEST_PROJECT=$(create_test_project "workflow-exec")

# Run init_kernel_project.sh to set up the kernel directory
INIT_SCRIPT="$SKILLS_DIR/skills/pyasc-codegen-workflow/scripts/init_kernel_project.sh"
if [ -x "$INIT_SCRIPT" ]; then
    (cd "$TEST_PROJECT" && bash "$INIT_SCRIPT" test_vadd 2>/dev/null) || true
fi

trap "cleanup_test_project '$TEST_PROJECT'" EXIT

echo "Test project: $TEST_PROJECT"
echo ""

# ============================================
# Agent execution with full workflow prompt
# ============================================

print_section_header "Phase: Agent Workflow Execution"

PROMPT="You are a pyasc kernel development engineer. Your task is to develop
a vector add kernel following the pyasc-codegen-workflow phases strictly:

Phase 0 - Environment: Check that pyasc is available.
Phase 1 - Design: Create a design document at kernels/test_vadd/docs/design.md
  describing the API selection, buffer strategy, sync strategy, and verification plan.
Phase 2 - Implementation: Write the kernel at kernels/test_vadd/kernel.py using
  @asc2.jit, asc2.tensor, asc2.load, asc2.store, asc2.range, and kernel[cores](...) launch.
  Run a self-review against the pyasc code review checklist.
Phase 3 - Verification: Verify the kernel produces z = x + y using torch.allclose.

Follow all phases in order. Do not skip phases.
Use only supported pyasc syntax."

OUTPUT_FILE="$TEST_PROJECT/workflow-output.txt"
SESSION_FILE="$TEST_PROJECT/session.json"

echo "Running OpenCode (long timeout: 600s)..."
echo ""

if timeout 600 opencode run "$PROMPT" \
    --format json \
    --dir "$TEST_PROJECT" > "$OUTPUT_FILE" 2>&1; then
    echo "  Agent completed."
else
    ec=$?
    if [ "$ec" -eq 124 ]; then
        print_warn "Agent timed out after 600s"
    else
        print_info "Agent exited with code $ec"
    fi
fi

echo ""

# ============================================
# Checkpoint verification
# ============================================

print_section_header "Checkpoint Verification"

FAILED=0

# CP-0: Environment (init script ran, or agent mentions environment)
echo "CP-0: Environment"
if [ -f "$TEST_PROJECT/kernels/test_vadd/docs/environment.json" ] ||
   grep -qiE "environment|python.*version|pyasc.*import|Phase 0" "$OUTPUT_FILE" 2>/dev/null; then
    print_pass "CP-0: Environment phase evidence found"
else
    print_warn "CP-0: No clear environment phase evidence"
fi

echo ""

# CP-1: Design artifact
echo "CP-1: Design"
DESIGN_MD=""
for candidate in \
    "$TEST_PROJECT/kernels/test_vadd/docs/design.md" \
    "$TEST_PROJECT/design.md" \
    "$TEST_PROJECT/docs/design.md"; do
    if [ -f "$candidate" ]; then
        DESIGN_MD="$candidate"
        break
    fi
done

if [ -n "$DESIGN_MD" ]; then
    print_pass "CP-1: Design document created: $DESIGN_MD"
    if grep -qiE "api|buffer|sync|verification" "$DESIGN_MD" 2>/dev/null; then
        print_pass "CP-1: Design covers key sections"
    else
        print_warn "CP-1: Design may be incomplete"
    fi
else
    print_warn "CP-1: No design document found"
fi

echo ""

# CP-2: Implementation artifact
echo "CP-2: Implementation"
KERNEL_PY=""
for candidate in \
    "$TEST_PROJECT/kernels/test_vadd/kernel.py" \
    "$TEST_PROJECT/kernel.py"; do
    if [ -f "$candidate" ]; then
        KERNEL_PY="$candidate"
        break
    fi
done

if [ -n "$KERNEL_PY" ]; then
    print_pass "CP-2: Kernel file created: $KERNEL_PY"

    echo ""
    echo "  Static verification:"
    if $PYTHON "$TOOLS_DIR/verify_kernel.py" "$KERNEL_PY" 2>&1; then
        print_pass "CP-2: Static verification passed"
    else
        print_fail "CP-2: Static verification failed"
        FAILED=$((FAILED + 1))
    fi

    echo ""
    echo "  Automated scoring:"
    SCORE_JSON=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$KERNEL_PY" --json 2>&1) || true
    score_val=$(echo "$SCORE_JSON" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")
    echo "  Score: $score_val / 10"
    score_int=${score_val%%.*}
    if [ "${score_int:-0}" -ge 8 ]; then
        print_pass "CP-2: Acceptance score >= 8"
    else
        print_fail "CP-2: Acceptance score too low"
        FAILED=$((FAILED + 1))
    fi
else
    print_fail "CP-2: No kernel file found"
    FAILED=$((FAILED + 1))
fi

echo ""

# CP-3: Verification evidence
echo "CP-3: Verification"
if grep -qiE "allclose|PASS|verified|verification.*pass|Phase 3" "$OUTPUT_FILE" 2>/dev/null; then
    print_pass "CP-3: Verification evidence in output"
else
    print_warn "CP-3: No clear verification evidence in output"
fi

if [ -n "$KERNEL_PY" ]; then
    echo ""
    echo "  Runtime verification (optional):"
    $PYTHON "$TOOLS_DIR/run_and_verify.py" "$KERNEL_PY" 2>&1 || true
fi

echo ""

# ============================================
# Session analysis (if available)
# ============================================

print_section_header "Session Analysis"

SESSION_ID=$(find_recent_session)

if [ -n "$SESSION_ID" ]; then
    export_session "$SESSION_ID" "$SESSION_FILE"

    if [ -s "$SESSION_FILE" ]; then
        echo "  Tool chain:"
        analyze_tool_chain "$SESSION_FILE"
        echo ""

        echo "  Tool counts:"
        for tool in Read Write Edit Shell Skill Grep Glob; do
            count=$(count_tool_invocations "$SESSION_FILE" "$tool")
            [ "$count" -gt 0 ] && print_info "$tool: $count invocations"
        done
        echo ""

        analyze_premature_actions "$SESSION_FILE" "skill" "pyasc-codegen-workflow" || true
    else
        print_info "Session export was empty"
    fi
else
    print_info "No session available for analysis"
fi

# ============================================
# Workflow order in output text
# ============================================

print_section_header "Workflow Order (from output text)"

output_text=$(cat "$OUTPUT_FILE" 2>/dev/null || true)

if [ -n "$output_text" ]; then
    assert_order "$output_text" "Phase 0\|environment\|setup" "Phase 1\|design\|Design" "Environment before Design" || true
    assert_order "$output_text" "Phase 1\|design\|Design" "Phase 2\|implement\|kernel\.py" "Design before Implementation" || true
    assert_order "$output_text" "Phase 2\|implement\|kernel" "Phase 3\|verif\|allclose" "Implementation before Verification" || true
else
    print_skip "No output text for order analysis"
fi

# ============================================
# Summary
# ============================================

echo ""
echo "========================================"
echo -e " ${BOLD}Workflow Execution Results${NC}"
echo "========================================"
echo ""
echo "  Test project: $TEST_PROJECT"
echo "  Kernel:       ${KERNEL_PY:-not found}"
echo "  Design:       ${DESIGN_MD:-not found}"
echo "  Score:        ${score_val:-n/a} / 10"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "  Critical failures: $FAILED"
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
