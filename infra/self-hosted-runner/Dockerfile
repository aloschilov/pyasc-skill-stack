# Custom GitHub Actions self-hosted runner image for pyasc-skill-stack CI.
#
# Base: myoung34/github-runner (multi-arch: linux/amd64 + linux/arm64). The base
# already handles runner registration, ephemeral re-registration from a PAT, and
# Docker-out-of-Docker (it ships the Docker CLI). It now serves only the arm64
# MacBook runner (198.18.0.1) -- the sole self-hosted runner. The x86_64 RTX 4090
# GPU pool has been retired from CI; pr-gate/merge-gate moved to GitHub-hosted
# ubuntu-latest and nightly/cloud-dashscope to this arm64 Mac.
#
# IMPORTANT: pin to the ubuntu-jammy (22.04) tag, NOT :latest (= ubuntu-focal /
# 20.04). actions/setup-python's python-versions has NO prebuilt Python 3.10 for
# 20.04/arm64 (perf-gate on the Mac fails outright) and on 20.04/x64 it falls back
# to a fragile from-source build (intermittent `tar`/extract corruption). 22.04
# has prebuilt 3.10 for both arm64 and x64, so setup-python is fast + reliable.
#
# We only layer the toolchain the CI jobs assume a GitHub-hosted ubuntu runner
# provides and which the base image lacks on the user PATH:
#   - python3 + pip
#   - jq / curl / ca-certificates -> misc steps
# Node.js 24 is retained but no longer strictly required by CI: opencode now
# runs INSIDE the pyasc-sim image (--agent-backend docker) rather than on the
# runner. Ollama is the macOS host's native (Metal) daemon, reached via
# host.docker.internal; nothing here needs a GPU.
FROM myoung34/github-runner:ubuntu-jammy

USER root

ARG NODE_MAJOR=24
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl jq ca-certificates gnupg python3 python3-pip \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && node --version && npm --version \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Inherit the base image ENTRYPOINT (registration + run loop); no override.
