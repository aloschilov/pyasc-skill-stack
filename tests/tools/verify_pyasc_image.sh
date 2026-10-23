#!/usr/bin/env bash
# Fail fast when a CI runner image does not match the skill-stack pyasc pin.
set -euo pipefail

EXPECTED_REV="${PYASC_GIT_REV:-030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d}"
IMAGE="${1:?usage: verify_pyasc_image.sh IMAGE}"

ACTUAL_REV="$(docker run --rm --entrypoint git "$IMAGE" -C /opt/pyasc rev-parse HEAD)"
if [ "$ACTUAL_REV" != "$EXPECTED_REV" ]; then
    echo "ERROR: $IMAGE contains pyasc $ACTUAL_REV; expected $EXPECTED_REV" >&2
    exit 1
fi

docker run --rm --entrypoint python3.11 "$IMAGE" -c \
    'import asc, asctile; print("pyasc runtime OK: asc + asctile")'
echo "Image pin verified: $IMAGE -> $ACTUAL_REV"
