#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${PYASC_PYTHON:-python3}" -m unittest discover -s "$ROOT/tests/unit/python" -p 'test_verification_contract.py' -v
