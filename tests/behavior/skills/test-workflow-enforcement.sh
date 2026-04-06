#!/usr/bin/env bash
# =============================================================================
# L2 Behavior Test: Workflow Enforcement (Agent-in-the-Loop)
# =============================================================================
# Runs OpenCode with a kernel development prompt and verifies the response
# demonstrates a phased workflow (environment -> design -> implement -> verify)
# rather than jumping straight to code.
#
# Requires: opencode CLI on PATH
# Estimated time: 2-3 minutes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L2 Behavior: Workflow Enforcement${NC}"
echo "========================================"
echo ""
echo "Tests that the agent follows a phased workflow."
echo "Requires: opencode CLI"
echo ""

if ! check_opencode; then
    echo -e "${RED}[ERROR]${NC} opencode CLI not found on PATH"
    exit 1
fi

FAILED=0

# ============================================
# Scenario: Elementwise multiply kernel
# ============================================

print_section_header "Scenario: Elementwise Multiply Kernel"

PROMPT="Develop a pyasc elementwise multiply kernel with full verification.
Explain your approach step by step before writing any code.
What phases should you follow?"

echo "Running OpenCode..."
OUTPUT=$(run_opencode "$PROMPT" 90 2>&1) || true

echo ""
print_section_header "Response Analysis"

# Check for phased workflow indicators
phase_checks=0
phase_total=0

((phase_total++))
if echo "$OUTPUT" | grep -qiE "phase 0|environment|env.*check|setup|prerequisite"; then
    print_pass "Mentions environment/setup phase"
    ((phase_checks++))
else
    print_fail "No environment/setup phase mentioned"
    FAILED=$((FAILED + 1))
fi

((phase_total++))
if echo "$OUTPUT" | grep -qiE "phase 1|design|api.*select|buffer.*strat|architecture"; then
    print_pass "Mentions design phase"
    ((phase_checks++))
else
    print_fail "No design phase mentioned"
    FAILED=$((FAILED + 1))
fi

((phase_total++))
if echo "$OUTPUT" | grep -qiE "phase 2|implement|code|kernel.*func|@asc\.jit"; then
    print_pass "Mentions implementation phase"
    ((phase_checks++))
else
    print_fail "No implementation phase mentioned"
    FAILED=$((FAILED + 1))
fi

((phase_total++))
if echo "$OUTPUT" | grep -qiE "phase 3|verif|test|torch\.allclose|numpy|check.*output"; then
    print_pass "Mentions verification phase"
    ((phase_checks++))
else
    print_fail "No verification phase mentioned"
    FAILED=$((FAILED + 1))
fi

echo ""

# Check that response doesn't immediately dump code without explanation
((phase_total++))
first_20_lines=$(echo "$OUTPUT" | head -20)
if echo "$first_20_lines" | grep -qE "^(import |from |def |@asc)"; then
    print_fail "Response starts with code (no planning evident)"
    FAILED=$((FAILED + 1))
else
    print_pass "Response begins with explanation, not raw code"
    ((phase_checks++))
fi

# Check for verification mention
((phase_total++))
if echo "$OUTPUT" | grep -qiE "allclose|verify|verification|correctness|golden|reference"; then
    print_pass "Response discusses verification strategy"
    ((phase_checks++))
else
    print_warn "No verification strategy discussion found"
fi

# ============================================
# Summary
# ============================================

echo ""
echo "========================================"
echo -e " ${BOLD}Workflow Enforcement Results${NC}"
echo "========================================"
echo ""
echo "  Phase checks passed: $phase_checks / $phase_total"
echo ""

if [ "$FAILED" -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
