#!/usr/bin/env bash
# =============================================================================
# L2 Behavior Test: Agent / kernel-dev trigger correctness
# =============================================================================
# Domain prompts framed for pyasc kernel development (team agent context).
# Verifies OpenCode responses stay on pyasc/kernel topics for in-domain
# prompts and remain appropriately off-topic for unrelated asks.
#
# Requires: opencode CLI on PATH
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}L2 Behavior: Agent Trigger Correctness${NC}"
echo "========================================"
echo ""
echo "Tests kernel-development prompts produce pyasc-relevant guidance."
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
# Positive: kernel dev / team-context prompts
# ============================================

print_section_header "Positive Kernel-Dev Prompts"

run_behavior_test \
    "Kernel dev team: start new operator" \
    "I am using the pyasc kernel development team workflow. What is the first step before writing any @asc.jit kernel code?" \
    "workflow|phase|skill|codegen|document|read|design|pyasc|kernel" \
    60

run_behavior_test \
    "Team context: vector add" \
    "Under the pyasc-kernel-dev-team setup, how do I implement a minimal float32 vector add kernel?" \
    "@asc\.jit|kernel|data_copy|GlobalTensor|pyasc|asc\.|LocalTensor" \
    60

run_behavior_test \
    "Kernel dev: verification" \
    "As a kernel dev agent user, how should I verify my pyasc operator output when I only have the Model backend?" \
    "torch\.allclose|numpy|Model|verify|output|pyasc" \
    60

run_behavior_test \
    "Forced workflow mention" \
    "What does the pyasc kernel dev team require regarding pyasc-codegen-workflow and phases?" \
    "phase|workflow|codegen|skill|kernel|pyasc" \
    60

run_behavior_test \
    "Sync in pipeline" \
    "For pyasc kernel development on Ascend, explain set_flag and wait_flag briefly." \
    "set_flag|wait_flag|HardEvent|sync|pipeline|pyasc|kernel" \
    60

# ============================================
# Negative / off-topic
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

# ============================================
# Summary
# ============================================

echo "========================================"
echo -e " ${BOLD}Agent Trigger Correctness Results${NC}"
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
