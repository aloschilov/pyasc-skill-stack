#!/usr/bin/env bash
# =============================================================================
# Evaluation Report Generator
# =============================================================================
# Aggregates results from all eval layers into a structured summary.
# Can be run standalone or piped into after a full test run.
#
# Usage:
#   bash eval-report.sh [--json] [--kernel <path>] [--session <path>]
#
# Without arguments, runs the Python verification tools against the
# first-scenario kernel and prints a consolidated report.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

USE_JSON=false
KERNEL_PATH=""
SESSION_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --json)    USE_JSON=true; shift ;;
        --kernel)  KERNEL_PATH="$2"; shift 2 ;;
        --session) SESSION_PATH="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--json] [--kernel <path>] [--session <path>]"
            exit 0
            ;;
        *) shift ;;
    esac
done

# Default kernel: first-scenario add kernel
if [ -z "$KERNEL_PATH" ]; then
    KERNEL_PATH="$SKILLS_DIR/teams/pyasc-kernel-dev-team/kernels/add/kernel.py"
fi

# ============================================
# Gather results
# ============================================

echo "========================================"
echo -e " ${BOLD}pyasc Evaluation Report${NC}"
echo "========================================"
echo ""
echo "  Date:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python:  $PYTHON"
echo "  Kernel:  $KERNEL_PATH"
echo ""

# --- L1: Structure ---
print_section_header "L1: Skill Structure"

l1_total=0
l1_pass=0
for skill_file in "$SKILLS_DIR"/skills/*/SKILL.md; do
    [ -f "$skill_file" ] || continue
    l1_total=$((l1_total + 1))
    if validate_skill_structure "$skill_file" > /dev/null 2>&1; then
        l1_pass=$((l1_pass + 1))
    fi
done
echo "  Skills: $l1_pass / $l1_total pass structure validation"

team_ok=true
for team_file in "$SKILLS_DIR"/teams/*/AGENTS.md; do
    [ -f "$team_file" ] || continue
    validate_team_structure "$team_file" > /dev/null 2>&1 || team_ok=false
done
echo "  Teams:  $( $team_ok && echo 'PASS' || echo 'FAIL' )"

agent_total=0
agent_pass=0
for agent_file in "$SKILLS_DIR"/teams/*/AGENTS.md; do
    [ -f "$agent_file" ] || continue
    agent_total=$((agent_total + 1))
    if validate_agent_structure "$agent_file" > /dev/null 2>&1; then
        agent_pass=$((agent_pass + 1))
    fi
done
echo "  Agents: $agent_pass / $agent_total pass structure validation"

echo ""

# --- Static verification ---
print_section_header "L2/L3: Static Kernel Verification"

if [ -f "$KERNEL_PATH" ]; then
    echo "  verify_kernel.py:"
    $PYTHON "$TOOLS_DIR/verify_kernel.py" "$KERNEL_PATH" 2>&1 || true
    verify_exit=0
    $PYTHON "$TOOLS_DIR/verify_kernel.py" "$KERNEL_PATH" > /dev/null 2>&1 || verify_exit=$?

    echo ""
    echo "  score_kernel.py:"
    score_json=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$KERNEL_PATH" --json 2>&1) || true
    score_val=$(echo "$score_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")
    accepted=$(echo "$score_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('accepted',False))" 2>/dev/null || echo "False")
    echo "    Score: $score_val / 10"
    echo "    Accepted: $accepted"
else
    echo "  Kernel file not found, skipping."
    verify_exit=1
    score_val="0"
    accepted="False"
fi

echo ""

# --- Runtime verification ---
print_section_header "Runtime Verification"

runtime_status="SKIP"
if [ -f "$KERNEL_PATH" ]; then
    rt_exit=0
    $PYTHON "$TOOLS_DIR/run_and_verify.py" "$KERNEL_PATH" 2>&1 || rt_exit=$?
    if [ "$rt_exit" -eq 0 ]; then
        runtime_status="PASS"
    elif [ "$rt_exit" -eq 2 ]; then
        runtime_status="SKIP"
    else
        runtime_status="FAIL"
    fi
else
    echo "  No kernel to verify at runtime."
fi

echo "  Runtime: $runtime_status"
echo ""

# --- JIT verification ---
print_section_header "JIT Verification"

jit_status="SKIP"
PYTEST_JIT="$TOOLS_DIR/pytest_verify_kernel.py"
if [ ! -f "$PYTEST_JIT" ]; then
    echo "  pytest_verify_kernel.py not found, skipping."
elif [ ! -f "$KERNEL_PATH" ]; then
    echo "  No kernel file for JIT verification."
else
    echo "  pytest_verify_kernel.py:"
    jit_exit=0
    $PYTHON "$PYTEST_JIT" "$KERNEL_PATH" 2>&1 || jit_exit=$?
    if [ "$jit_exit" -eq 0 ]; then
        jit_status="PASS"
    elif [ "$jit_exit" -eq 2 ]; then
        jit_status="SKIP"
    else
        jit_status="FAIL"
    fi
    echo "  JIT: $jit_status"
fi

echo ""

# --- Golden comparison ---
print_section_header "Golden Comparison"

GOLDEN_DIR="$SKILLS_DIR/golden/tutorials"
if [ -d "$GOLDEN_DIR" ]; then
    printf '  %-26s  %8s  %12s  %8s\n' "Tutorial" "Golden" "Generated" "Delta"
    printf '  %-26s  %8s  %12s  %8s\n' "--------------------------" "--------" "-----------" "--------"
    while IFS= read -r golden_py; do
        [ -f "$golden_py" ] || continue
        base=$(basename "$golden_py" .py)
        # score_kernel exits 1 when score < threshold; with pipefail, A|B || echo would merge outputs — capture JSON first.
        golden_json=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$golden_py" --json 2>/dev/null) || true
        gs=$(printf '%s' "$golden_json" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('score',0))" 2>/dev/null || echo "0")
        if [ -f "$KERNEL_PATH" ]; then
            delta=$($PYTHON -c "print(round(float('$gs') - float('$score_val'), 2))" 2>/dev/null || echo "n/a")
        else
            delta="n/a"
        fi
        printf '  %-26s  %8s  %12s  %8s\n' "$base" "$gs / 10" "$score_val / 10" "$delta"
    done < <(find "$GOLDEN_DIR" -maxdepth 1 -name '*.py' -type f 2>/dev/null | sort)
else
    echo "  Golden tutorials directory not found: $GOLDEN_DIR"
fi

echo ""

# --- Session analysis ---
print_section_header "Session Analysis"

if [ -n "$SESSION_PATH" ] && [ -f "$SESSION_PATH" ]; then
    if [ -f "$TOOLS_DIR/analyze-session.sh" ]; then
        echo "  analyze-session.sh (--brief):"
        bash "$TOOLS_DIR/analyze-session.sh" "$SESSION_PATH" --brief 2>&1 || true
        echo ""
    fi

    echo "  Tool chain:"
    analyze_tool_chain "$SESSION_PATH"
    echo ""

    for tool in Read Write Edit Shell Skill Grep Glob; do
        count=$(count_tool_invocations "$SESSION_PATH" "$tool")
        [ "$count" -gt 0 ] && echo "    $tool: $count"
    done
else
    echo "  No session file provided (use --session <path>)"
fi

echo ""

# --- Platform availability ---
print_section_header "Environment"

echo "  opencode: $(check_opencode && echo 'available' || echo 'not found')"
echo "  pyasc import: $(check_pyasc_import && echo 'OK' || echo 'not available')"
echo "  pyasc runtime: $(check_pyasc_runtime && echo 'OK' || echo 'not available (CANN simulator missing)')"
echo ""

# ============================================
# Overall
# ============================================

echo "========================================"
echo -e " ${BOLD}Summary${NC}"
echo "========================================"
echo ""
echo "  L1 structure:      skills $l1_pass / $l1_total; agents $agent_pass / $agent_total"
echo "  Teams (structure): $( $team_ok && echo 'PASS' || echo 'FAIL' )"
echo "  Static verify:     $([ $verify_exit -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  Score:             $score_val / 10 (accepted: $accepted)"
echo "  Runtime:           $runtime_status"
echo "  JIT:               $jit_status"
echo "  Golden comparison: generated kernel score $score_val / 10 (see table above)"
echo ""

# L1: teams + agent structure on teams/*/AGENTS.md
l1_ok=false
if $team_ok; then
    if [ "$agent_total" -eq 0 ] || [ "$agent_pass" -eq "$agent_total" ]; then
        l1_ok=true
    fi
fi

jit_ok=true
[ "$jit_status" = "FAIL" ] && jit_ok=false

overall_pass=false
if [ "$verify_exit" -eq 0 ] && [ "$accepted" = "True" ] && $l1_ok && $jit_ok; then
    overall_pass=true
    print_status_passed
else
    print_status_failed
fi

if $USE_JSON; then
    cat <<EOJSON
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "$( $overall_pass && echo 'passed' || echo 'failed' )",
  "l1_skills": {"pass": $l1_pass, "total": $l1_total},
  "l1_agents": {"pass": $agent_pass, "total": $agent_total},
  "teams_structure": "$( $team_ok && echo 'pass' || echo 'fail' )",
  "static_verify": "$([ $verify_exit -eq 0 ] && echo 'pass' || echo 'fail')",
  "score": $score_val,
  "accepted": $( [ "$accepted" = "True" ] && echo 'true' || echo 'false' ),
  "runtime": "$runtime_status",
  "jit": "$jit_status",
  "kernel": "$KERNEL_PATH"
}
EOJSON
fi

if ! $overall_pass; then
    exit 1
fi
