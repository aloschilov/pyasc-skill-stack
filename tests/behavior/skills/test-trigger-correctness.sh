#!/usr/bin/env bash
# =============================================================================
# L2 Behavior Test: Skill trigger correctness
# Verifies that skills contain appropriate trigger keywords and conditions
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Trigger Correctness ==="

ERRORS=0
TESTED=0

# Expected trigger patterns for each skill
declare -A SKILL_TRIGGERS=(
    ["pyasc-codegen-workflow"]="workflow|development|kernel|operator"
    ["pyasc-docs-search"]="search|documentation|resource|tutorial|API"
    ["pyasc-api-patterns"]="API|usage|pattern|best.practice"
    ["pyasc-syntax-constraints"]="syntax|constraint|support|restrict"
    ["pyasc-build-run-verify"]="build|run|verify|JIT|diagnostic"
    ["pyasc-code-review"]="review|code|security|syntax"
    ["pyasc-env-check"]="environment|check|install|CANN|Python"
    ["pyasc-task-focus"]="task|focus|attention|todo"
)

for skill_name in "${!SKILL_TRIGGERS[@]}"; do
    skill_file="$SKILLS_DIR/skills/$skill_name/SKILL.md"
    if [ -f "$skill_file" ]; then
        TESTED=$((TESTED + 1))
        description=$(grep "^description:" "$skill_file" | head -1 | cut -d: -f2-)
        expected_pattern="${SKILL_TRIGGERS[$skill_name]}"

        if echo "$description" | grep -qiE "$expected_pattern"; then
            print_pass "$skill_name: trigger keywords present"
        else
            print_fail "$skill_name: missing trigger keywords (expected: $expected_pattern)"
            ERRORS=$((ERRORS + 1))
        fi
    else
        print_skip "$skill_name: SKILL.md not found"
    fi
done

echo ""
echo "Tested: $TESTED skills"

if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] $ERRORS skill(s) have trigger issues"
    exit 1
else
    echo "[PASS] All $TESTED skills have correct triggers"
fi
