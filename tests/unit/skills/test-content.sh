#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: Skill content validation
# Rules: S-CON-01, S-CON-02, S-CON-04
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Content Validation ==="

ERRORS=0
TESTED=0

for skill_dir in "$SKILLS_DIR"/skills/*/; do
    skill_file="$skill_dir/SKILL.md"
    if [ -f "$skill_file" ]; then
        TESTED=$((TESTED + 1))
        if ! validate_skill_content "$skill_file"; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

# Validate team structure
echo ""
echo "--- Team validation ---"

for team_dir in "$SKILLS_DIR"/teams/*/; do
    team_file="$team_dir/AGENTS.md"
    if [ -f "$team_file" ]; then
        TESTED=$((TESTED + 1))
        if ! validate_team_structure "$team_file"; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

echo ""
echo "Tested: $TESTED items"

if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] $ERRORS item(s) have content errors"
    exit 1
else
    echo "[PASS] All $TESTED items pass content validation"
fi
