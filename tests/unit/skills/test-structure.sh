#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: Skill structure validation
# Rules: S-STR-01 to S-STR-06
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Structure Validation ==="

ERRORS=0
TESTED=0

for skill_dir in "$SKILLS_DIR"/skills/*/; do
    skill_file="$skill_dir/SKILL.md"
    if [ -f "$skill_file" ]; then
        TESTED=$((TESTED + 1))
        if ! validate_skill_structure "$skill_file"; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

echo ""
echo "Tested: $TESTED skills"

if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] $ERRORS skill(s) have structure errors"
    exit 1
else
    echo "[PASS] All $TESTED skills pass structure validation"
fi
