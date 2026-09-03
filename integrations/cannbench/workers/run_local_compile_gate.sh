#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
image="pyasc-cannbench-local-eval:0a631f70-v2"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
    docker build \
        --platform linux/amd64 \
        --file "${repo_root}/integrations/cannbench/Dockerfile.local-eval" \
        --tag "${image}" \
        "${repo_root}/integrations/cannbench" >&2
fi

exec docker run --rm --platform linux/amd64 \
    --volume "${repo_root}:/workspace:ro" \
    --workdir /workspace \
    "${image}" "$@"
