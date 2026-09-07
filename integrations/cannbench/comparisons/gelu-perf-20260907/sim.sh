#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd "$(dirname "$0")" && pwd)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$task_dir/native-source/python:${PYTHONPATH:-}"
export PYASC_CACHE_DIR=/tmp/pyasc-gelu-perf-20260907-cache
task_run=$(mktemp -d /tmp/gelu-sim-XXXXXXXX)
cd "$task_run"
exec python3 "$task_dir/probe.py" "$@"
