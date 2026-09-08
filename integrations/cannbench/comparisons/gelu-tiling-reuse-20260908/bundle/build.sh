#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum --check wheel.sha256.txt
mkdir -p dist
cp wheel/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl dist/
