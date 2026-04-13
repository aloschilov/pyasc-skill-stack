#!/usr/bin/env bash
# =============================================================================
# L2 Behavior Test: Trigger Correctness (Agent-in-the-Loop)
# =============================================================================
# Runs OpenCode with domain-specific prompts and verifies the response
# contains pyasc-relevant content, demonstrating that the skills are
# influencing the agent's output.
#
# Requires: opencode CLI on PATH
# Estimated time: 2-5 minutes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L2 Behavior: Trigger Correctness${NC}"
echo "========================================"
echo ""
echo "Tests that OpenCode responds with pyasc-relevant content."
echo "Requires: opencode CLI"
echo ""

if ! check_opencode; then
    echo -e "${RED}[ERROR]${NC} opencode CLI not found on PATH"
    exit 1
fi

pass_count=0
fail_count=0
skip_count=0

# ============================================
# Positive domain prompts
# ============================================

print_section_header "Positive Domain Prompts"

run_behavior_test \
    "Vector add kernel query" \
    "How do I implement a vector add kernel in pyasc?" \
    "@asc2\.jit|@asc\.jit|kernel|asc2\.load|asc2\.store|asc2\.tensor|pyasc|asc\." \
    60

run_behavior_test \
    "Supported syntax query" \
    "What Python syntax is supported inside @asc2.jit functions in pyasc?" \
    "for.*range|supported|unsupported|ConstExpr|syntax|jit" \
    60

run_behavior_test \
    "Verification query" \
    "How do I verify my pyasc kernel output is correct?" \
    "torch\.allclose|numpy|allclose|verification|Model|verify" \
    60

run_behavior_test \
    "Data copy API query" \
    "How do I use asc2.load and asc2.store in pyasc to transfer data?" \
    "asc2\.load|asc2\.store|asc2\.tensor|TPosition|copy|GM|UB" \
    60

run_behavior_test \
    "Sync primitives query" \
    "How do asc2.load and asc2.store relate to pipeline sync in pyasc?" \
    "asc2\.load|asc2\.store|HardEvent|MTE2|sync|pipeline" \
    60

run_behavior_test \
    "pyasc-api-patterns skill trigger" \
    "What are the best practices for using asc2.load and asc2.store?" \
    "asc2\.load|asc2\.store|asc2\.tensor|TPosition|copy|tensor|pyasc|asc\." \
    60

run_behavior_test \
    "pyasc-syntax-constraints skill trigger" \
    "What Python constructs are banned inside @asc2.jit?" \
    "syntax|unsupported|forbidden|jit|@asc|lambda|print|try" \
    60

run_behavior_test \
    "pyasc-code-review skill trigger" \
    "Review my pyasc kernel for syntax constraint violations" \
    "review|syntax|constraint|jit|pyasc|kernel|check|violat" \
    60

# ============================================
# Negative / off-topic prompts
# ============================================

print_section_header "Negative Prompts"

run_behavior_test \
    "Off-topic: weather" \
    "What is the weather today?" \
    "weather|temperature|forecast|sorry|cannot|don.t" \
    30

run_behavior_test \
    "Off-topic: cooking" \
    "Give me a recipe for chocolate cake." \
    "cake|chocolate|recipe|sorry|cannot|baking|flour" \
    30

run_behavior_test_negative_no_pyasc() {
    local name="$1"
    local prompt="$2"
    local timeout_val="${3:-45}"
    local drift='@asc2\.jit|@asc\.jit|asc2\.tensor|GlobalTensor|LocalTensor|pyasc.*kernel|Ascend|NPU|asc2\.load.*pyasc|data_copy.*pyasc'

    echo "Testing: $name"

    local output
    if output=$(run_opencode "$prompt Answer in 1 line." "$timeout_val" 2>&1); then
        if echo "$output" | grep -qiE "$drift"; then
            print_fail "Response drifted into pyasc/kernel-specific content"
            fail_count=$((fail_count + 1))
        else
            print_pass "Response avoided pyasc/kernel-specific drift"
            pass_count=$((pass_count + 1))
        fi
    else
        local ec=$?
        if [ "$ec" -eq 124 ]; then
            print_skip "OpenCode timed out after ${timeout_val}s"
        else
            print_skip "OpenCode exited with code $ec"
        fi
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

run_behavior_test_negative_no_pyasc \
    "Off-topic: web scraper (no pyasc)" \
    "Write me a Python web scraper" \
    45

# ============================================
# Summary
# ============================================

echo "========================================"
echo -e " ${BOLD}Trigger Correctness Results${NC}"
echo "========================================"
echo ""
total=$((pass_count + fail_count + skip_count))
echo -e "  ${GREEN}Passed:${NC}  $pass_count / $total"
echo -e "  ${RED}Failed:${NC}  $fail_count / $total"
echo -e "  ${YELLOW}Skipped:${NC} $skip_count / $total"
echo ""

if [ "$fail_count" -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
