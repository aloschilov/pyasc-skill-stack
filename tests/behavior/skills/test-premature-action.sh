#!/usr/bin/env bash
# =============================================================================
# L2 Behavior Test: No premature action
# Verifies that skills do not instruct premature Write/Edit/Bash before loading
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: No Premature Action in Skills ==="

ERRORS=0
TESTED=0

for skill_dir in "$SKILLS_DIR"/skills/*/; do
    skill_file="$skill_dir/SKILL.md"
    if [ -f "$skill_file" ]; then
        TESTED=$((TESTED + 1))
        skill_name=$(basename "$skill_dir")

        # Check that workflow skill enforces ordered phases
        if [ "$skill_name" = "pyasc-codegen-workflow" ]; then
            if grep -qiE "forbidden|prohibited|banned|skip" "$skill_file"; then
                print_pass "$skill_name: contains skip prevention rules"
            else
                print_fail "$skill_name: missing skip prevention rules"
                ERRORS=$((ERRORS + 1))
            fi
        fi

        # Check that review skills require parameters before execution
        if echo "$skill_name" | grep -qE "review"; then
            if grep -qiE "required.*parameter|parameter.*required|must.*provide" "$skill_file"; then
                print_pass "$skill_name: requires parameters before execution"
            else
                print_warn "$skill_name: may not enforce parameter requirements"
            fi
        fi
    fi
done

echo ""
echo "Tested: $TESTED skills"

if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] $ERRORS skill(s) have premature action issues"
    exit 1
else
    echo "[PASS] All $TESTED skills pass premature action checks"
fi
