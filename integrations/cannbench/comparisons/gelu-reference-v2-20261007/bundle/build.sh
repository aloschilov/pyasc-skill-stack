#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash metadata/build_payload.sh
