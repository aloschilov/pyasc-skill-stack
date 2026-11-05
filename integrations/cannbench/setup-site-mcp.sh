#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="${CANNBENCH_VENV:-${repo_root}/.tools/benchsite-mcp}"

download_url="$(${PYTHON:-python3} - <<'PY'
import json
import time
import urllib.error
import urllib.request

for attempt, delay in enumerate((15, 30, 60, 60, 0)):
    try:
        with urllib.request.urlopen(
            "https://cannbench.com/api/meta/mcp-version", timeout=30
        ) as response:
            print(json.load(response)["download_url"])
        break
    except urllib.error.HTTPError as exc:
        if exc.code not in (502, 503, 504) or not delay:
            raise
        time.sleep(delay)
PY
)"

"${PYTHON:-python3}" -m venv "${venv}"
"${venv}/bin/pip" install --upgrade "${download_url}" "mcp<2"
echo "Installed BenchSite MCP in ${venv}"
echo "OpenCode entry: cann-bench-site"
