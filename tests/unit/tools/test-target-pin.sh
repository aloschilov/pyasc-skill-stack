#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: reference/asctile/ matches its pin
#
# The AscTile target suite is the correctness reference, vendored from upstream
# PyAsc. Upstream moves it between versions -- it has lived at
# python/test/asc2/target/, then python/test/asctile/target/, and was deleted
# outright when AscTile was relocated under experimental/. A snapshot that
# drifts silently is worse than no snapshot, because it still looks
# authoritative.
#
# This guards the cheap half: that nobody hand-edited the vendored files
# instead of re-running scripts/sync-target-suite.sh. It is offline, so it is
# safe in --fast and in CI without network.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== Test: AscTile target suite pin ==="

if ! command -v jq >/dev/null 2>&1; then
    echo "[SKIP] jq not installed"
    exit 0
fi

if [ ! -f "$REPO_ROOT/reference/asctile/PIN.json" ]; then
    echo "[FAIL] reference/asctile/PIN.json is missing"
    exit 1
fi

if "$REPO_ROOT/scripts/check-target-pin.sh"; then
    echo "[PASS] vendored suite matches reference/asctile/PIN.json"
    exit 0
else
    echo "[FAIL] reference/asctile/ drifted from its pin"
    echo "       Re-sync rather than hand-editing: scripts/sync-target-suite.sh"
    exit 1
fi
