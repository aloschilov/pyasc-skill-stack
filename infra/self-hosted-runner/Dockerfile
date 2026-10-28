# Custom GitHub Actions self-hosted runner image for pyasc-skill-stack CI.
#
# Base: myoung34/github-runner (multi-arch: linux/amd64 + linux/arm64). The base
# already handles runner registration, ephemeral re-registration from a PAT, and
# Docker-out-of-Docker (it ships the Docker CLI). It serves the arm64 MacBook
# runner, which is the only runner there is: every job in ci.yml targets
# `[self-hosted, arm64]`. The x86_64 RTX 4090 GPU pool was retired.
#
# Self-hosted is not a preference here. The gates run the pinned simulator image
# as a sibling container with the workspace bind-mounted at its host path, which
# needs a daemon and a work dir the job controls -- see compose.arm64.yml.
#
# REBUILD THIS IMAGE PERIODICALLY. GitHub retires old runner binaries, and a
# retired one fails in a way that reads as a credential problem rather than a
# version problem: it connects, registers, prints "Listening for Jobs", then
# "Runner version vX is deprecated and cannot receive messages", deregisters
# and loops. The runner never appears online. DISABLE_AUTO_UPDATE does not
# save you and neither does turning it off -- the update is delivered over the
# message channel the deprecated runner is barred from. Only a rebuild on a
# current base fixes it. 2026-10-26: v2.335.1 was refused, rebuilding gave
# v2.338.2.
#
# Docker Desktop's keychain credstore blocks pulls and builds over SSH, so
# build on another arm64 Linux host and ship the result:
#   DOCKER_CONFIG=$(mktemp -d) && echo '{}' > "$DOCKER_CONFIG/config.json"
#   docker build --pull -t pyasc-ci-runner:latest infra/self-hosted-runner/
#   docker save pyasc-ci-runner:latest | ssh <mac> docker load
# --pull matters: without it the stale base is reused and nothing changes.
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
