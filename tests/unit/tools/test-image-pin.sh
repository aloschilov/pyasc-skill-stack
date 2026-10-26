#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: the simulator image pin is internally consistent
#
# This repository consumes a simulator image built and smoke-tested upstream by
# compiler-team-ru/pyasc-images. The digest is recorded twice -- once in
# reference/images/ENV.json, once as PYASC_SIM_IMAGE in the CI workflow,
# because GitHub Actions cannot read a file into `env:`. Duplicated values
# drift, so a check is the only thing keeping them equal.
#
# It also asserts that the image and the vendored target suite derive from the
# same upstream commit. Compile-checking the suite against a runtime it was not
# written for produces a green gate that means nothing.
#
# Offline, so it is safe in --fast and in CI without network.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== Test: simulator image pin ==="

if ! command -v jq >/dev/null 2>&1; then
    echo "[SKIP] jq not installed"
    exit 0
fi

if [ ! -f "$REPO_ROOT/reference/images/ENV.json" ]; then
    echo "[FAIL] reference/images/ENV.json is missing"
    exit 1
fi

if "$REPO_ROOT/scripts/check-image-pin.sh"; then
    echo "[PASS] image pin, CI workflow and target-suite pin agree"
    exit 0
else
    echo "[FAIL] the image pin is inconsistent"
    echo "       Re-pin rather than hand-editing: scripts/sync-image-pin.sh"
    exit 1
fi
