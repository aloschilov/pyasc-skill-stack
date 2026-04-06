#!/usr/bin/env bash
# L1 Unit Test: Team content validation
# Rules: T-CON-01 to T-CON-05
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Team Content Validation ==="
ERRORS=0
TESTED=0

for team_dir in "$SKILLS_DIR"/teams/*/; do
    team_file="$team_dir/AGENTS.md"
    if [ -f "$team_file" ]; then
        TESTED=$((TESTED + 1))
        if ! validate_team_content "$team_file"; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

echo ""
echo "Tested: $TESTED team(s)"
if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] $ERRORS team(s) have content errors"
    exit 1
else
    echo "[PASS] All $TESTED team(s) pass content validation"
fi
