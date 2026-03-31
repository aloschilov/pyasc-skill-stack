#!/usr/bin/env bash
# =============================================================================
# L3 Integration Test: Workflow execution validation
# Verifies the workflow skill enforces correct phasing and checkpoints
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "=== Integration Test: Workflow Execution ==="

ERRORS=0
WORKFLOW="$SKILLS_DIR/skills/pyasc-codegen-workflow/SKILL.md"

# Test 1: Workflow has all phases
echo ""
echo "--- Test 1: Phase completeness ---"

for phase in "Phase 0" "Phase 1" "Phase 2" "Phase 3"; do
    if grep -q "$phase" "$WORKFLOW"; then
        print_pass "Workflow contains $phase"
    else
        print_fail "Workflow missing $phase"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 2: Workflow has all checkpoints
echo ""
echo "--- Test 2: Checkpoint completeness ---"

for cp in "CP-0" "CP-1" "CP-2" "CP-3"; do
    if grep -q "$cp" "$WORKFLOW"; then
        print_pass "Workflow contains $cp"
    else
        print_fail "Workflow missing $cp"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 3: Workflow enforces order
echo ""
echo "--- Test 3: Order enforcement ---"

if grep -qiE "forced|forbidden|prohibited|banned" "$WORKFLOW"; then
    print_pass "Workflow contains order enforcement language"
else
    print_fail "Workflow missing order enforcement"
    ERRORS=$((ERRORS + 1))
fi

# Test 4: Workflow references pyasc-specific concepts
echo ""
echo "--- Test 4: pyasc domain specificity ---"

for concept in "@asc.jit" "torch.allclose" "set_flag" "GlobalTensor" "LocalTensor"; do
    if grep -q "$concept" "$WORKFLOW"; then
        print_pass "Workflow references $concept"
    else
        print_warn "Workflow does not reference $concept"
    fi
done

# Test 5: References exist
echo ""
echo "--- Test 5: Reference file integrity ---"

REF_DIR="$SKILLS_DIR/skills/pyasc-codegen-workflow/references"
for ref in phase0-environment.md phase1-design.md phase2-implementation.md phase3-testing.md code-review-checklist.md; do
    assert_file_exists "$REF_DIR/$ref" "Reference: $ref" || ERRORS=$((ERRORS + 1))
done

# Test 6: Scripts exist and are executable content
echo ""
echo "--- Test 6: Script integrity ---"

SCRIPT_PATH="$SKILLS_DIR/skills/pyasc-codegen-workflow/scripts"
for script in init_kernel_project.sh verify_environment.sh; do
    if [ -f "$SCRIPT_PATH/$script" ]; then
        if grep -q "#!/usr/bin/env bash" "$SCRIPT_PATH/$script"; then
            print_pass "Script $script has bash shebang"
        else
            print_fail "Script $script missing bash shebang"
            ERRORS=$((ERRORS + 1))
        fi
    else
        print_fail "Script $script not found"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 7: Templates exist
echo ""
echo "--- Test 7: Template integrity ---"

TMPL_DIR="$SKILLS_DIR/skills/pyasc-codegen-workflow/templates"
assert_file_exists "$TMPL_DIR/design-template.md" "Design template" || ERRORS=$((ERRORS + 1))
assert_file_exists "$TMPL_DIR/kernel-template.py" "Kernel template" || ERRORS=$((ERRORS + 1))

echo ""
echo "=== Summary ==="
if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] $ERRORS error(s) found"
    exit 1
else
    echo "[PASS] All workflow execution checks passed"
fi
