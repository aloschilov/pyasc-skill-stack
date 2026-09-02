#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
credentials="${repo_root}/.secrets/cannbench.env"
mcp_python="${repo_root}/.tools/benchsite-mcp/bin/python"

if [[ ! -r "${credentials}" ]]; then
  echo "CANNBench credentials not found: ${credentials}" >&2
  exit 1
fi
if [[ ! -x "${mcp_python}" ]]; then
  echo "CANNBench MCP environment not found: ${mcp_python}" >&2
  exit 1
fi

# Local-only credentials; .secrets/ is gitignored and mode 0700/0600.
# shellcheck disable=SC1090
source "${credentials}"
exec "${mcp_python}" -m benchsite_mcp.server
