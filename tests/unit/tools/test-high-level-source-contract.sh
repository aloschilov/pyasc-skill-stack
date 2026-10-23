#!/usr/bin/env bash
# CPU-only new-promotion contract tests; no runtime or remote credentials.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -m unittest discover -s "${script_dir}/../../tools" \
  -p test_high_level_source_contract.py -v
