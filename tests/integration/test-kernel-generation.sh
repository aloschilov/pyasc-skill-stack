#!/usr/bin/env bash
# =============================================================================
# L3 Integration Test: Full Kernel Generation (Agent-in-the-Loop)
# =============================================================================
# End-to-end test: OpenCode generates a pyasc vector add kernel, then the
# test harness verifies the output with static AST checks, automated scoring,
# and optional runtime verification.
#
# Requires: opencode CLI, python3.10 with pyasc importable
# Estimated time: 5-15 minutes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L3 Integration: Kernel Generation E2E${NC}"
echo "========================================"
echo ""
echo "Tests full agent-driven pyasc kernel generation."
echo "Estimated time: 5-15 minutes"
echo "Requires: opencode CLI, python3.10"
echo ""

if ! check_opencode; then
    echo -e "${RED}[ERROR]${NC} opencode CLI not found on PATH"
    exit 1
fi

# ============================================
# Setup
# ============================================

TEST_PROJECT=$(create_test_project "kernel-gen")
KERNEL_DIR="$TEST_PROJECT/kernels/add"
mkdir -p "$KERNEL_DIR/docs"

trap "cleanup_test_project '$TEST_PROJECT'" EXIT

echo "Test project: $TEST_PROJECT"
echo ""

# ============================================
# Agent execution
# ============================================

print_section_header "Phase: Agent Execution"

PROMPT="You are a pyasc kernel development engineer.

Implement a simple vector add kernel for two float32 tensors of size 1024.
Follow the pyasc-codegen-workflow phases:
1. Check environment
2. Create a design document in docs/design.md
3. Implement the kernel in kernel.py using @asc.jit with manual sync
4. Include verification using torch.allclose

Requirements:
- Use asc.GlobalTensor, asc.LocalTensor, asc.data_copy
- Use asc.set_flag / asc.wait_flag for pipeline sync
- Use only supported pyasc syntax (no print/break/continue/lambda inside @asc.jit)
- Create files under kernels/add/

Use only supported pyasc syntax and documented APIs."

OUTPUT_FILE="$TEST_PROJECT/agent-output.txt"

echo "Running OpenCode..."
echo "  Timeout: 300s"
echo ""

if timeout 300 opencode run "$PROMPT" \
    --format json \
    --dir "$TEST_PROJECT" > "$OUTPUT_FILE" 2>&1; then
    echo "  Agent completed successfully."
else
    ec=$?
    if [ "$ec" -eq 124 ]; then
        print_warn "Agent timed out after 300s"
    else
        print_info "Agent exited with code $ec"
    fi
fi

echo ""

# ============================================
# Artifact verification
# ============================================

print_section_header "Phase: Artifact Verification"

FAILED=0

# Look for kernel.py in expected and fallback locations
KERNEL_PY=""
for candidate in \
    "$KERNEL_DIR/kernel.py" \
    "$TEST_PROJECT/kernel.py" \
    "$TEST_PROJECT/kernels/add/kernel.py"; do
    if [ -f "$candidate" ]; then
        KERNEL_PY="$candidate"
        break
    fi
done

if [ -n "$KERNEL_PY" ]; then
    print_pass "kernel.py found: $KERNEL_PY"
else
    print_fail "kernel.py not found in any expected location"
    FAILED=$((FAILED + 1))
fi

DESIGN_MD=""
for candidate in \
    "$KERNEL_DIR/docs/design.md" \
    "$TEST_PROJECT/docs/design.md" \
    "$TEST_PROJECT/kernels/add/docs/design.md"; do
    if [ -f "$candidate" ]; then
        DESIGN_MD="$candidate"
        break
    fi
done

if [ -n "$DESIGN_MD" ]; then
    print_pass "design.md found: $DESIGN_MD"
else
    print_warn "design.md not found (agent may have skipped design phase)"
fi

echo ""

# ============================================
# Static verification
# ============================================

print_section_header "Phase: Static Verification"

if [ -n "$KERNEL_PY" ]; then
    echo "Running verify_kernel.py..."
    if $PYTHON "$TOOLS_DIR/verify_kernel.py" "$KERNEL_PY" 2>&1; then
        print_pass "Static verification passed"
    else
        print_fail "Static verification failed"
        FAILED=$((FAILED + 1))
    fi
else
    print_skip "No kernel.py to verify"
    FAILED=$((FAILED + 1))
fi

echo ""

# ============================================
# Automated scoring
# ============================================

print_section_header "Phase: Automated Scoring"

if [ -n "$KERNEL_PY" ]; then
    echo "Running score_kernel.py..."
    SCORE_OUTPUT=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$KERNEL_PY" --json 2>&1) || true
    echo "$SCORE_OUTPUT" | $PYTHON -m json.tool 2>/dev/null || echo "$SCORE_OUTPUT"

    score_val=$(echo "$SCORE_OUTPUT" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")
    accepted=$(echo "$SCORE_OUTPUT" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('accepted',False))" 2>/dev/null || echo "False")

    echo ""
    echo "  Score: $score_val / 10"
    if [ "$accepted" = "True" ]; then
        print_pass "Score meets acceptance threshold (>= 8.5)"
    else
        print_fail "Score below acceptance threshold"
        FAILED=$((FAILED + 1))
    fi
else
    print_skip "No kernel.py to score"
    FAILED=$((FAILED + 1))
fi

echo ""

# ============================================
# Runtime verification (optional)
# ============================================

print_section_header "Phase: Runtime Verification"

if [ -n "$KERNEL_PY" ]; then
    echo "Running run_and_verify.py..."
    $PYTHON "$TOOLS_DIR/run_and_verify.py" "$KERNEL_PY" --json 2>&1 || true
    runtime_exit=$?

    if [ "$runtime_exit" -eq 0 ]; then
        print_pass "Runtime verification passed"
    elif [ "$runtime_exit" -eq 2 ]; then
        print_skip "Runtime unavailable (CANN simulator not present)"
    else
        print_warn "Runtime verification failed (non-blocking for this test)"
    fi
else
    print_skip "No kernel.py for runtime verification"
fi

echo ""

# ============================================
# Summary
# ============================================

echo "========================================"
echo -e " ${BOLD}Kernel Generation E2E Results${NC}"
echo "========================================"
echo ""
echo "  Test project: $TEST_PROJECT"
echo "  Kernel file:  ${KERNEL_PY:-not found}"
echo "  Design doc:   ${DESIGN_MD:-not found}"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "  Critical failures: $FAILED"
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
