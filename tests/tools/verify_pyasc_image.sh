#!/usr/bin/env bash
# Fail fast when a CI runner image is not the one reference/images/ENV.json pins.
#
# Two independent assertions, because each catches a different mistake:
#
#   1. Identity. The image's repo digest must match ENV.json -- either the
#      multi-arch manifest digest, or the per-arch child digest that `docker
#      pull` resolves a manifest list to. This catches a stale or hand-edited
#      PYASC_SIM_IMAGE in the workflow.
#   2. Usability. `asc.experimental.asctile` must import and expose `jit`. A
#      digest can be right while the runtime inside is unusable, and an empty
#      namespace package imports perfectly happily without having `jit`.
#
# When ENV.json carries a dev_image, that is what CI runs and what is checked
# here, and usability additionally means torch and pytest at the recorded
# versions. The repository no longer installs either: it relies on the overlay
# image shipping them, so "the image has them" is now load-bearing and belongs
# in the gate rather than in a comment.
#
# Deliberately NOT a `git -C /opt/pyasc rev-parse` check: the upstream images
# install the published wheel, so they carry no source checkout.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PIN="${PYASC_ENV_PIN:-${REPO_ROOT}/reference/images/ENV.json}"
IMAGE="${1:?usage: verify_pyasc_image.sh IMAGE}"

[ -f "$ENV_PIN" ] || { echo "ERROR: no image pin at $ENV_PIN" >&2; exit 1; }

if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required to read $ENV_PIN" >&2
    exit 1
fi

# The overlay is what CI runs when it exists; fall back to the minimal image.
if [ "$(jq -r 'has("dev_image")' "$ENV_PIN")" = "true" ]; then
    sel=".dev_image"
    want_torch="$(jq -r '.dev_image.torch' "$ENV_PIN")"
    want_pytest="$(jq -r '.dev_image.pytest' "$ENV_PIN")"
else
    sel=".image"
    want_torch=""
    want_pytest=""
fi

want_manifest="$(jq -r "${sel}.digest" "$ENV_PIN")"
want_children="$(jq -r "${sel}.children | to_entries[] | .value" "$ENV_PIN")"
python_bin="$(jq -r '.image.python_bin' "$ENV_PIN")"

# RepoDigests is what the registry served, independent of any local tag.
mapfile -t got < <(docker image inspect "$IMAGE" \
    --format '{{range .RepoDigests}}{{println .}}{{end}}' | sed 's/.*@//' | grep .)

if [ "${#got[@]}" -eq 0 ]; then
    echo "ERROR: $IMAGE has no repo digest; it was built locally, not pulled" >&2
    exit 1
fi

matched=0
for d in "${got[@]}"; do
    if [ "$d" = "$want_manifest" ]; then matched=1; break; fi
    for c in $want_children; do
        [ "$d" = "$c" ] && { matched=1; break 2; }
    done
done

if [ "$matched" -ne 1 ]; then
    {
        echo "ERROR: $IMAGE does not match the pin in $ENV_PIN"
        echo "  pinned manifest  $want_manifest"
        for c in $want_children; do echo "  pinned child     $c"; done
        for d in "${got[@]}"; do echo "  image has        $d"; done
        echo "  Re-pin with scripts/sync-image-pin.sh if this is intentional."
    } >&2
    exit 1
fi

docker run --rm --entrypoint "$python_bin" \
    -e WANT_TORCH="$want_torch" -e WANT_PYTEST="$want_pytest" "$IMAGE" -c '
import os, sys
from asc.experimental import asctile
assert hasattr(asctile, "jit"), (
    "asc.experimental.asctile has no jit (%r)" % (getattr(asctile, "__file__", None),))
print("pyasc runtime OK: asc.experimental.asctile on %s" % sys.version.split()[0])

want_torch = os.environ.get("WANT_TORCH") or ""
want_pytest = os.environ.get("WANT_PYTEST") or ""
if want_torch or want_pytest:
    import torch, pytest
    if want_torch and torch.__version__ != want_torch:
        sys.exit("torch is %s, pin says %s" % (torch.__version__, want_torch))
    if want_pytest and pytest.__version__ != want_pytest:
        sys.exit("pytest is %s, pin says %s" % (pytest.__version__, want_pytest))
    print("overlay OK: torch %s | pytest %s" % (torch.__version__, pytest.__version__))
'

echo "Image pin verified: $IMAGE"
