#!/usr/bin/env bash
# Build (and push) the base `pyasc-sim` image and publish an honest multiarch
# manifest under a single tag (default `py3.11`).
#
# docker/Dockerfile is already arch-agnostic (multi-arch CANN base + LLVM
# archive auto-selected by `uname -m`), so the only thing missing is a
# publish path. Cross-building the arm64 leg under QEMU emulation on an amd64
# host is impractical for the CANN+pyasc+LLVM toolchain, so each architecture
# is built NATIVELY on its own host and the two single-arch tags are then
# stitched into a multiarch manifest with `docker buildx imagetools create`.
#
# Typical workflow (run on the respective native hosts, like build-perf-image.sh
# — CI only `docker pull`s the published tag):
#
#   # On the amd64 box (e.g. the RTX 4090 host):
#   docker/build-sim-image.sh --arch amd64 --push
#
#   # On the arm64 box (the Apple-silicon Mac, Docker Desktop logged in to ghcr):
#   docker/build-sim-image.sh --arch arm64 --push
#
#   # On either host, once both single-arch tags are pushed:
#   docker/build-sim-image.sh --merge
#
# After --merge, `docker buildx imagetools inspect ghcr.io/<owner>/pyasc-sim:py3.11`
# lists both linux/amd64 and linux/arm64, and PYASC_SIM_IMAGE resolves to the
# right arch on both the amd64 `gpu` jobs and the arm64 Mac `local-stability-gate`.
#
# Env overrides:
#   OWNER       ghcr owner used to derive the image (default: aloschilov)
#   IMAGE       multiarch manifest tag (default: ghcr.io/${OWNER}/pyasc-sim:py3.11)
#   PYASC_GIT_URL / PYASC_GIT_REV / PYASC_GIT_REF   forwarded to Dockerfile build args
#   GIT_CREDENTIALS_FILE  optional path to a git-credentials file (e.g. a line
#                         https://user:token@gitcode.com); passed as a BuildKit
#                         secret for the clone when the host egress requires auth.
#   PYASC_SETUP_JOBS  optional cap on native-build parallelism (cmake --parallel N).
#                     Set low (e.g. 2) on small-RAM build hosts to avoid OOM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OWNER="${OWNER:-aloschilov}"
IMAGE="${IMAGE:-ghcr.io/${OWNER}/pyasc-sim:py3.11}"

ARCH=""
PUSH=0
MERGE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --arch)  ARCH="${2:-}"; shift 2 ;;
        --push)  PUSH=1; shift ;;
        --merge) MERGE=1; shift ;;
        -h|--help)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: unknown arg '$1'" >&2; exit 1 ;;
    esac
done

# The per-arch tags that --merge stitches together.
TAG_AMD64="${IMAGE}-amd64"
TAG_ARM64="${IMAGE}-arm64"

build_arch() {
    local arch="$1" tag
    case "$arch" in
        amd64) tag="$TAG_AMD64" ;;
        arm64) tag="$TAG_ARM64" ;;
        *) echo "ERROR: --arch must be amd64 or arm64 (got '$arch')" >&2; exit 1 ;;
    esac

    local host_arch
    host_arch="$(uname -m)"
    case "$arch:$host_arch" in
        amd64:x86_64|arm64:aarch64|arm64:arm64) : ;;
        *)
            echo "ERROR: refusing to build --arch $arch on a $host_arch host." >&2
            echo "       Build each architecture natively (QEMU emulation of the" >&2
            echo "       CANN+pyasc+LLVM toolchain is impractical)." >&2
            exit 1 ;;
    esac

    echo "== build-sim-image (arch=$arch) =="
    echo "  tag : $tag"
    local build_args=()
    [ -n "${PYASC_GIT_URL:-}" ] && build_args+=(--build-arg "PYASC_GIT_URL=${PYASC_GIT_URL}")
    [ -n "${PYASC_GIT_REV:-}" ] && build_args+=(--build-arg "PYASC_GIT_REV=${PYASC_GIT_REV}")
    [ -n "${PYASC_GIT_REF:-}" ] && build_args+=(--build-arg "PYASC_GIT_REF=${PYASC_GIT_REF}")
    [ -n "${PYASC_SETUP_JOBS:-}" ] && build_args+=(--build-arg "PYASC_SETUP_JOBS=${PYASC_SETUP_JOBS}")

    # Optional git credentials for the clone (passed as a BuildKit secret, so the
    # token is mounted only during the clone RUN and never baked into a layer).
    # Needed when gitcode.com requires auth on the build host's egress path.
    [ -n "${GIT_CREDENTIALS_FILE:-}" ] && build_args+=(--secret "id=git_credentials,src=${GIT_CREDENTIALS_FILE}")

    # NB: "${build_args[@]+...}" guards against the empty-array-under-`set -u`
    # error on macOS's stock bash 3.2 (where a bare "${build_args[@]}" is
    # treated as an unbound variable when the array is empty).
    DOCKER_BUILDKIT=1 docker build \
        -f "$SCRIPT_DIR/Dockerfile" \
        "${build_args[@]+"${build_args[@]}"}" \
        -t "$tag" \
        "$SCRIPT_DIR"

    echo "  built: $tag"
    if [ "$PUSH" -eq 1 ]; then
        echo "  pushing $tag ..."
        docker push "$tag"
    fi
}

merge_manifest() {
    echo "== build-sim-image (merge multiarch manifest) =="
    echo "  manifest : $IMAGE"
    echo "  sources  : $TAG_AMD64 , $TAG_ARM64"
    docker buildx imagetools create -t "$IMAGE" "$TAG_AMD64" "$TAG_ARM64"
    echo "  inspecting ..."
    docker buildx imagetools inspect "$IMAGE"
}

if [ "$MERGE" -eq 1 ]; then
    merge_manifest
elif [ -n "$ARCH" ]; then
    build_arch "$ARCH"
else
    echo "ERROR: nothing to do — pass --arch <amd64|arm64> [--push] or --merge." >&2
    echo "       See --help for the full workflow." >&2
    exit 1
fi

echo "done."
