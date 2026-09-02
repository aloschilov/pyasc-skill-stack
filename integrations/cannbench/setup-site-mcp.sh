#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="${repo_root}/.tools/benchsite-mcp"

download_url="$(${PYTHON:-python3} - <<'PY'
import json
import urllib.request

with urllib.request.urlopen(
    "https://cannbench.com/api/meta/mcp-version", timeout=30
) as response:
    print(json.load(response)["download_url"])
PY
)"

"${PYTHON:-python3}" -m venv "${venv}"
"${venv}/bin/pip" install --upgrade "${download_url}" "mcp<2"
echo "Installed BenchSite MCP in ${venv}"
echo "OpenCode entry: cann-bench-site"
