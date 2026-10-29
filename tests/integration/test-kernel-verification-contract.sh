#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# A new directory is required so evidence cannot silently overwrite a prior run.
RESULTS="${PYASC_RESULTS_DIR:-${TMPDIR:-/tmp}/pyasc-contract-$(date +%Y%m%dT%H%M%S)-$$}"
exec "${PYASC_PYTHON:-python3}" "$ROOT/tests/tools/verification_contract.py" --results-dir "$RESULTS"
