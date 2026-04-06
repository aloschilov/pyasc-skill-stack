#!/usr/bin/env bash
# =============================================================================
# Workflow Analysis Tool for OpenCode sessions
# =============================================================================
# Checks skill load order, phase progression, premature actions
# Usage: analyze-workflow.sh <session.json>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/test-helpers.sh
source "$SCRIPT_DIR/../lib/test-helpers.sh" 2>/dev/null || true

SESSION_FILE="${1:?Usage: $0 <session.json>}"

if [ ! -f "$SESSION_FILE" ]; then
    echo "[ERROR] Session file not found: $SESSION_FILE"
    exit 1
fi

echo "=== Workflow Analysis ==="
echo ""
echo "File: $SESSION_FILE"
echo ""

ISSUES=0

# Check skill load order
echo "--- Skill Load Order ---"
skill_lines=$(grep -n "pyasc-" "$SESSION_FILE" 2>/dev/null | head -20 || true)
if [ -n "$skill_lines" ]; then
    while IFS=: read -r line_num content; do
        [ -z "${line_num:-}" ] && continue
        skill=$(printf '%s\n' "$content" | grep -oE 'pyasc-[a-z0-9-]+' | head -1 || true)
        if [ -n "$skill" ]; then
            printf "  Line %5s: %s\n" "$line_num" "$skill"
        fi
    done <<<"$skill_lines"
else
    echo "  No pyasc skill references found"
fi
echo ""

# Check phase progression
echo "--- Phase Progression ---"
prev_phase=-1
for phase_num in 0 1 2 3; do
    line=$(grep -n -iE "phase[[:space:]]*${phase_num}|Phase[[:space:]]*${phase_num}|CP-${phase_num}" "$SESSION_FILE" 2>/dev/null | head -1 | cut -d: -f1 || true)
    if [ -n "$line" ]; then
        echo "  Phase $phase_num: first mention at line $line"
        if [ "$prev_phase" -ge 0 ] && [ "$line" -lt "$prev_phase" ]; then
            echo "    [WARN] Out-of-order phase detected"
            ISSUES=$((ISSUES + 1))
        fi
        prev_phase=$line
    else
        echo "  Phase $phase_num: not found"
    fi
done
echo ""

# Check premature actions
echo "--- Premature Action Check ---"
first_skill_line=$(grep -n "pyasc-codegen-workflow" "$SESSION_FILE" 2>/dev/null | head -1 | cut -d: -f1 || true)
if [ -n "$first_skill_line" ]; then
    echo "  Workflow skill first referenced at line: $first_skill_line"
    write_before=$(head -n "$first_skill_line" "$SESSION_FILE" 2>/dev/null | \
        grep -ciE '"(write|edit|patch|bash|shell)"' 2>/dev/null || echo "0")
    write_before=${write_before:-0}
    if [ "$write_before" -gt 0 ]; then
        echo "  [WARN] $write_before write/edit/patch/shell-like actions BEFORE workflow skill line"
        ISSUES=$((ISSUES + 1))
    else
        echo "  [PASS] No premature write-like JSON hints before workflow skill line"
    fi
else
    echo "  [INFO] Workflow skill not found in session"
fi
echo ""

# Summary
echo "--- Summary ---"
if [ "$ISSUES" -gt 0 ]; then
    echo "  Issues found: $ISSUES"
    echo "  Status: NEEDS REVIEW"
else
    echo "  Status: OK"
fi
