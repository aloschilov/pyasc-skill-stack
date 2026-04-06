#!/usr/bin/env bash
# =============================================================================
# L3 Integration Test: Abs Kernel Generation (Agent-in-the-Loop)
# =============================================================================
# End-to-end test: OpenCode generates a pyasc abs kernel for float16,
# then the harness verifies artifacts, runs static checks, JIT verification,
# and compares structural similarity to the golden abs kernel.
#
# Requires: opencode CLI, python3.10 with pyasc
# Estimated time: 5-15 minutes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L3 Integration: Abs Kernel E2E${NC}"
echo "========================================"
echo ""
echo "Tests agent-driven abs(float16) kernel generation."
echo "Requires: opencode CLI, python3.10"
echo ""

if ! check_opencode; then
    echo -e "${RED}[ERROR]${NC} opencode CLI not found on PATH"
    exit 1
fi

GOLDEN_ABS="$SKILLS_DIR/golden/kernels/abs_f16.py"

# ============================================
# Setup
# ============================================

TEST_PROJECT=$(create_test_project "abs-kernel")
trap "cleanup_test_project '$TEST_PROJECT'" EXIT

echo "Test project: $TEST_PROJECT"
echo ""

# ============================================
# Agent execution
# ============================================

print_section_header "Phase: Agent Execution"

PROMPT="Help me develop an abs operator that supports float16 data type. The shape is mainly [1,128], [4,2048], [32,4096].

Follow the pyasc-codegen-workflow phases strictly.
Use @asc.jit, asc.GlobalTensor, asc.LocalTensor, asc.data_copy, set_flag/wait_flag.
Include torch.allclose verification for all three shapes.
Write the kernel to kernels/abs_f16/kernel.py."

OUTPUT_FILE="$TEST_PROJECT/agent-output.txt"

echo "Running OpenCode..."
if timeout 300 opencode run "$PROMPT" \
    --format json \
    --dir "$TEST_PROJECT" > "$OUTPUT_FILE" 2>&1; then
    echo "  Agent completed."
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

# Find kernel.py
KERNEL_PY=""
for candidate in \
    "$TEST_PROJECT/kernels/abs_f16/kernel.py" \
    "$TEST_PROJECT/kernel.py" \
    "$TEST_PROJECT/abs_f16/kernel.py"; do
    if [ -f "$candidate" ]; then
        KERNEL_PY="$candidate"
        break
    fi
done

if [ -z "$KERNEL_PY" ]; then
    KERNEL_PY=$(find "$TEST_PROJECT" -name "kernel.py" -path "*abs*" 2>/dev/null | head -1)
    [ -z "$KERNEL_PY" ] && KERNEL_PY=$(find "$TEST_PROJECT" -name "*.py" ! -name "__*" 2>/dev/null | head -1)
fi

if [ -n "$KERNEL_PY" ] && [ -f "$KERNEL_PY" ]; then
    print_pass "kernel.py found: $KERNEL_PY"
else
    print_fail "kernel.py not found"
    FAILED=$((FAILED + 1))
fi

# Check Phase artifacts
for artifact in "design.md" "self_review.md" "acceptance_review.md" "verification.md"; do
    found=false
    for dir in "$TEST_PROJECT/kernels/abs_f16/docs" "$TEST_PROJECT/docs"; do
        if [ -f "$dir/$artifact" ]; then
            found=true
            print_pass "$artifact found"
            break
        fi
    done
    if ! $found; then
        print_warn "$artifact not found"
    fi
done

echo ""

# ============================================
# Static verification
# ============================================

print_section_header "Phase: Static Verification"

if [ -n "$KERNEL_PY" ] && [ -f "$KERNEL_PY" ]; then
    echo "Running verify_kernel.py..."
    if $PYTHON "$TOOLS_DIR/verify_kernel.py" "$KERNEL_PY" 2>&1; then
        print_pass "Static verification passed"
    else
        print_fail "Static verification failed"
        FAILED=$((FAILED + 1))
    fi

    echo ""
    echo "Running score_kernel.py..."
    SCORE_OUTPUT=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$KERNEL_PY" --json 2>&1) || true
    echo "$SCORE_OUTPUT" | $PYTHON -m json.tool 2>/dev/null || echo "$SCORE_OUTPUT"

    score_val=$(echo "$SCORE_OUTPUT" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")
    echo ""
    echo "  Score: $score_val / 10"
    score_int=${score_val%%.*}
    if [ "${score_int:-0}" -ge 8 ]; then
        print_pass "Score meets threshold (>= 8)"
    else
        print_fail "Score below threshold ($score_val < 8)"
        FAILED=$((FAILED + 1))
    fi
else
    print_skip "No kernel.py to verify"
    FAILED=$((FAILED + 1))
fi

echo ""

# ============================================
# JIT verification
# ============================================

print_section_header "Phase: JIT Verification"

if [ -n "$KERNEL_PY" ] && [ -f "$KERNEL_PY" ]; then
    JIT_TOOL="$TOOLS_DIR/pytest_verify_kernel.py"
    if [ -f "$JIT_TOOL" ]; then
        echo "Running pytest_verify_kernel.py..."
        jit_exit=0
        $PYTHON "$JIT_TOOL" "$KERNEL_PY" 2>&1 || jit_exit=$?
        if [ "$jit_exit" -eq 0 ]; then
            print_pass "JIT verification passed"
        elif [ "$jit_exit" -eq 2 ]; then
            print_skip "pyasc not available for JIT check"
        else
            print_warn "JIT verification failed (non-blocking)"
        fi
    else
        print_skip "pytest_verify_kernel.py not found"
    fi
else
    print_skip "No kernel.py for JIT verification"
fi

echo ""

# ============================================
# Golden comparison
# ============================================

print_section_header "Phase: Golden Comparison"

if [ -n "$KERNEL_PY" ] && [ -f "$KERNEL_PY" ] && [ -f "$GOLDEN_ABS" ]; then
    gen_src=$(cat "$KERNEL_PY")

    for pattern in "data_copy" "set_flag" "wait_flag" "allclose" "abs" "float16"; do
        if echo "$gen_src" | grep -q "$pattern"; then
            print_pass "Generated has '$pattern'"
        else
            print_fail "Generated missing '$pattern'"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""

    for event in "MTE2_V" "V_MTE3" "MTE3_MTE2"; do
        if echo "$gen_src" | grep -q "$event"; then
            print_pass "Sync event '$event' present"
        else
            print_warn "Sync event '$event' missing"
        fi
    done

    echo ""

    golden_score=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$GOLDEN_ABS" --json 2>&1 | \
        $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")
    gen_score=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$KERNEL_PY" --json 2>&1 | \
        $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")
    echo "  Golden score:    $golden_score / 10"
    echo "  Generated score: $gen_score / 10"
elif [ ! -f "$GOLDEN_ABS" ]; then
    print_skip "Golden abs kernel not found at $GOLDEN_ABS"
else
    print_skip "No kernel.py for golden comparison"
fi

# ============================================
# Summary
# ============================================

echo ""
echo "========================================"
echo -e " ${BOLD}Abs Kernel E2E Results${NC}"
echo "========================================"
echo ""
echo "  Kernel: ${KERNEL_PY:-not found}"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "  Critical failures: $FAILED"
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
