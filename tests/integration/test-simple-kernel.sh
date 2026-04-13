#!/usr/bin/env bash
# =============================================================================
# L3 Integration Test: Simple kernel generation scenario
# Tests the end-to-end path: init -> design -> implement -> verify
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "=== Integration Test: Simple Kernel Scenario ==="

ERRORS=0

# Test 1: Golden set exists
echo ""
echo "--- Test 1: Golden set files ---"

assert_file_exists "$SKILLS_DIR/golden/tutorials/01_add.py" "Golden 01_add.py exists" || ERRORS=$((ERRORS + 1))
assert_file_exists "$SKILLS_DIR/golden/docs/architecture_introduction.md" "Golden architecture doc exists" || ERRORS=$((ERRORS + 1))
assert_file_exists "$SKILLS_DIR/golden/docs/python_syntax_support.md" "Golden syntax doc exists" || ERRORS=$((ERRORS + 1))

# Test 2: Init script works
echo ""
echo "--- Test 2: Init kernel project ---"

TEST_KERNEL_NAME="test_add_$$"
TEAM_DIR="$SKILLS_DIR/teams/pyasc-kernel-dev-team"

bash "$SKILLS_DIR/skills/pyasc-codegen-workflow/scripts/init_kernel_project.sh" "$TEST_KERNEL_NAME"
KERNEL_DIR="$TEAM_DIR/kernels/$TEST_KERNEL_NAME"

if [ -d "$KERNEL_DIR" ]; then
    print_pass "Kernel directory created"
else
    print_fail "Kernel directory not created"
    ERRORS=$((ERRORS + 1))
fi

assert_file_exists "$KERNEL_DIR/README.md" "README.md created" || ERRORS=$((ERRORS + 1))

if [ -d "$KERNEL_DIR/docs" ]; then
    print_pass "docs/ directory created"
else
    print_fail "docs/ directory not created"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "$KERNEL_DIR/test" ]; then
    print_pass "test/ directory created"
else
    print_fail "test/ directory not created"
    ERRORS=$((ERRORS + 1))
fi

# Test 3: Verify environment script runs
echo ""
echo "--- Test 3: Verify environment ---"

if bash "$SKILLS_DIR/skills/pyasc-codegen-workflow/scripts/verify_environment.sh" "$TEST_KERNEL_NAME"; then
    print_pass "Environment verification completed"
else
    print_warn "Environment verification had issues (non-blocking)"
fi

if [ -f "$KERNEL_DIR/docs/environment.json" ]; then
    print_pass "environment.json created"
else
    print_fail "environment.json not created"
    ERRORS=$((ERRORS + 1))
fi

# Test 4: Golden kernel syntax check
echo ""
echo "--- Test 4: Golden kernel content check ---"

GOLDEN_ADD="$SKILLS_DIR/golden/tutorials/01_add.py"
if [ -f "$GOLDEN_ADD" ]; then
    if grep -qE '@asc2\.jit|@asc\.jit' "$GOLDEN_ADD"; then
        print_pass "Golden kernel has @asc2.jit or @asc.jit decorator"
    else
        print_fail "Golden kernel missing @asc2.jit / @asc.jit"
        ERRORS=$((ERRORS + 1))
    fi

    if grep -q "asc2.load" "$GOLDEN_ADD" && grep -q "asc2.store" "$GOLDEN_ADD"; then
        print_pass "Golden kernel has asc2.load and asc2.store"
    else
        print_fail "Golden kernel missing asc2.load or asc2.store"
        ERRORS=$((ERRORS + 1))
    fi

    if grep -q "torch.allclose" "$GOLDEN_ADD"; then
        print_pass "Golden kernel has torch.allclose verification"
    else
        print_fail "Golden kernel missing verification"
        ERRORS=$((ERRORS + 1))
    fi

    if grep -q "asc2.range" "$GOLDEN_ADD"; then
        print_pass "Golden kernel has asc2.range tile loop"
    else
        print_fail "Golden kernel missing asc2.range"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Test 5: Template content check
echo ""
echo "--- Test 5: Template content check ---"

TEMPLATE="$SKILLS_DIR/skills/pyasc-codegen-workflow/templates/kernel-template.py"
if [ -f "$TEMPLATE" ]; then
    if grep -qE '@asc2\.jit|@asc\.jit' "$TEMPLATE"; then
        print_pass "Kernel template has @asc2.jit or @asc.jit"
    else
        print_fail "Kernel template missing @asc2.jit / @asc.jit"
        ERRORS=$((ERRORS + 1))
    fi

    if grep -q "torch.allclose" "$TEMPLATE"; then
        print_pass "Kernel template has verification"
    else
        print_fail "Kernel template missing verification"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Cleanup
rm -rf "$KERNEL_DIR"

echo ""
echo "=== Summary ==="
if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] $ERRORS error(s) found"
    exit 1
else
    echo "[PASS] All integration checks passed"
fi
