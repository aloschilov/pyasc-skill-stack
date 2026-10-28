#!/usr/bin/env bash
# =============================================================================
# CI Gate Runner
# =============================================================================
# Single entry point for the PR gate: static checks + L1 unit tests, no Docker
# and no CANN. Finishes in seconds.
#
#   bash tests/ci-gate.sh --tier pr
#
# There is no merge or nightly tier here any more. Simulator verification of
# the upstream AscTile target suite runs inline in .github/workflows/ci.yml
# (job `merge-gate`), because it needs the pyasc-sim container rather than a
# bare shell. The campaign-era perf, nightly, stability and skills-value tiers
# were retired along with their evidence tree; they remain reachable from the
# `archive/campaign-2026` branch.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIER="pr"

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier) TIER="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--tier pr]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$TIER" != "pr" ]; then
    echo "Unknown tier: $TIER (only 'pr' is supported; see the comment at the"
    echo "top of this file for where the other tiers went)"
    exit 1
fi

echo "========================================"
echo " CI Gate: $TIER"
echo "========================================"
echo ""

FAILED=0

echo "--- L1 Unit Tests ---"
if bash "$SCRIPT_DIR/run-tests.sh" --fast; then
    echo ""
else
    FAILED=$((FAILED + 1))
fi

echo ""
echo "========================================"
echo " CI Gate Result: $TIER"
echo "========================================"
if [ "$FAILED" -gt 0 ]; then
    echo "  FAILED ($FAILED check(s))"
    exit 1
else
    echo "  PASSED"
    exit 0
fi
