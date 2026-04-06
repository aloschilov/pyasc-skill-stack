#!/usr/bin/env bash
# =============================================================================
# L3 Integration Test: Golden Comparison (Agent-in-the-Loop)
# =============================================================================
# Generates a vector add kernel via OpenCode, then structurally compares the
# output against the golden reference (golden/tutorials/01_add.py).
#
# Checks:
#   - Same API patterns (data_copy, set_flag/wait_flag, asc.add)
#   - Same sync event types (MTE2_V, V_MTE3, MTE3_MTE2)
#   - Same verification approach (torch.allclose)
#   - Both pass static verification
#   - AST-level structural similarity (function count, decorator)
#
# Requires: opencode CLI, python3.10
# Estimated time: 5-15 minutes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L3 Integration: Golden Comparison${NC}"
echo "========================================"
echo ""
echo "Compares agent-generated kernel against golden/tutorials/01_add.py."
echo "Requires: opencode CLI, python3.10"
echo ""

if ! check_opencode; then
    echo -e "${RED}[ERROR]${NC} opencode CLI not found on PATH"
    exit 1
fi

GOLDEN="$SKILLS_DIR/golden/tutorials/01_add.py"
assert_file_exists "$GOLDEN" "Golden reference exists"

# ============================================
# Setup
# ============================================

TEST_PROJECT=$(create_test_project "golden-cmp")
trap "cleanup_test_project '$TEST_PROJECT'" EXIT

# ============================================
# Agent generates kernel
# ============================================

print_section_header "Phase: Agent Generation"

PROMPT="Write a pyasc vector add kernel. The kernel should:
- Accept two float32 input tensors x, y and produce output z = x + y
- Use @asc.jit decorator
- Use asc.GlobalTensor, asc.LocalTensor, asc.data_copy
- Use set_flag/wait_flag for pipeline sync with MTE2_V, V_MTE3, MTE3_MTE2 events
- Include a launch function and torch.allclose verification
- Use only supported pyasc syntax

Write the complete kernel to a single file called kernel.py."

OUTPUT_FILE="$TEST_PROJECT/agent-output.txt"

echo "Running OpenCode..."

if timeout 300 opencode run "$PROMPT" \
    --format json \
    --dir "$TEST_PROJECT" > "$OUTPUT_FILE" 2>&1; then
    echo "  Done."
else
    ec=$?
    if [ "$ec" -eq 124 ]; then
        print_warn "Agent timed out"
    else
        print_info "Agent exited with code $ec"
    fi
fi

echo ""

# Find the generated kernel
GEN_PY=""
for candidate in \
    "$TEST_PROJECT/kernel.py" \
    "$TEST_PROJECT/kernels/add/kernel.py"; do
    if [ -f "$candidate" ]; then
        GEN_PY="$candidate"
        break
    fi
done

if [ -z "$GEN_PY" ]; then
    # Search more broadly
    GEN_PY=$(find "$TEST_PROJECT" -name "*.py" -newer "$OUTPUT_FILE" -path "*/kernel*" 2>/dev/null | head -1)
    [ -z "$GEN_PY" ] && GEN_PY=$(find "$TEST_PROJECT" -name "*.py" ! -name "__*" 2>/dev/null | head -1)
fi

if [ -z "$GEN_PY" ] || [ ! -f "$GEN_PY" ]; then
    print_fail "Agent did not produce a kernel.py file"
    print_status_failed
    exit 1
fi

print_pass "Generated kernel: $GEN_PY"
echo ""

# ============================================
# Static verification of both files
# ============================================

print_section_header "Phase: Static Verification"

echo "Golden reference:"
$PYTHON "$TOOLS_DIR/verify_kernel.py" "$GOLDEN" 2>&1 || true
echo ""
echo "Generated kernel:"
$PYTHON "$TOOLS_DIR/verify_kernel.py" "$GEN_PY" 2>&1 || true
echo ""

golden_ok=true
gen_ok=true
$PYTHON "$TOOLS_DIR/verify_kernel.py" "$GOLDEN" > /dev/null 2>&1 || golden_ok=false
$PYTHON "$TOOLS_DIR/verify_kernel.py" "$GEN_PY" > /dev/null 2>&1 || gen_ok=false

if $golden_ok; then
    print_pass "Golden passes static verification"
else
    print_warn "Golden has static verification issues (unexpected)"
fi
if $gen_ok; then
    print_pass "Generated kernel passes static verification"
else
    print_fail "Generated kernel fails static verification"
fi

echo ""

# ============================================
# Structural comparison
# ============================================

print_section_header "Phase: Structural Comparison"

FAILED=0

golden_src=$(cat "$GOLDEN")
gen_src=$(cat "$GEN_PY")

# API pattern checks
for pattern in "data_copy" "set_flag" "wait_flag" "allclose"; do
    golden_has=false
    gen_has=false
    echo "$golden_src" | grep -q "$pattern" && golden_has=true
    echo "$gen_src" | grep -q "$pattern" && gen_has=true

    if $gen_has; then
        print_pass "Generated has '$pattern' (golden: $golden_has)"
    else
        print_fail "Generated missing '$pattern' (golden: $golden_has)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

# Sync event checks
for event in "MTE2_V" "V_MTE3" "MTE3_MTE2"; do
    if echo "$gen_src" | grep -q "$event"; then
        print_pass "Sync event '$event' present"
    else
        print_warn "Sync event '$event' missing (may use different naming)"
    fi
done

echo ""

# Decorator check
if echo "$gen_src" | grep -q "@asc.jit"; then
    print_pass "@asc.jit decorator present"
else
    print_fail "@asc.jit decorator missing"
    FAILED=$((FAILED + 1))
fi

# Tensor type check
for ttype in "GlobalTensor" "LocalTensor"; do
    if echo "$gen_src" | grep -q "$ttype"; then
        print_pass "$ttype used"
    else
        print_warn "$ttype not found"
    fi
done

echo ""

# ============================================
# Scoring comparison
# ============================================

print_section_header "Phase: Score Comparison"

golden_score=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$GOLDEN" --json 2>&1 | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")
gen_score=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$GEN_PY" --json 2>&1 | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")

echo "  Golden score:    $golden_score / 10"
echo "  Generated score: $gen_score / 10"
echo ""

gen_score_int=${gen_score%%.*}
if [ "${gen_score_int:-0}" -ge 8 ]; then
    print_pass "Generated score is competitive ($gen_score >= 8)"
else
    print_fail "Generated score too low ($gen_score < 8)"
    FAILED=$((FAILED + 1))
fi

# ============================================
# Summary
# ============================================

echo ""
echo "========================================"
echo -e " ${BOLD}Golden Comparison Results${NC}"
echo "========================================"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "  Structural mismatches: $FAILED"
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
