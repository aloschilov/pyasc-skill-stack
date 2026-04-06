#!/usr/bin/env bash
# L1 Unit Test: Agent/Team structure validation
# Rules: A-STR-01 to A-STR-07 (adapted for teams/*/AGENTS.md)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Agent Structure Validation ==="
ERRORS=0
TESTED=0

for team_dir in "$SKILLS_DIR"/teams/*/; do
    agent_file="$team_dir/AGENTS.md"
    if [ -f "$agent_file" ]; then
        TESTED=$((TESTED + 1))
        if ! validate_agent_structure "$agent_file"; then
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

echo ""
echo "Tested: $TESTED agent(s)"
if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] $ERRORS agent(s) have structure errors"
    exit 1
else
    echo "[PASS] All $TESTED agent(s) pass structure validation"
fi
