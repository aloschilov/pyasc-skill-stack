#!/usr/bin/env bash
# Verify the image pin is internally consistent. Offline; safe in --fast CI.
#
# Three things have to agree, and nothing but a check keeps them agreeing:
#
#   1. reference/images/ENV.json is well formed and pins a digest, not a tag.
#      A tag is mutable, so a tagged runtime silently invalidates every result
#      recorded beneath it.
#   2. .github/workflows/ci.yml pulls exactly that digest. The workflow needs a
#      literal because GitHub Actions cannot read a file into `env:`, so the
#      value is duplicated -- and duplicated values drift.
#   3. ENV.json.source_sha equals reference/asctile/PIN.json.commit. The image,
#      the wheel inside it and the vendored target suite all derive from one
#      upstream commit. If they diverge, the suite is being compile-checked
#      against a runtime it was not written for, and a green gate means nothing.
#
# Everything here is local file reading. No network, no Docker.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PIN="${REPO_ROOT}/reference/images/ENV.json"
TARGET_PIN="${REPO_ROOT}/reference/asctile/PIN.json"
WORKFLOW="${REPO_ROOT}/.github/workflows/ci.yml"

fail=0
say() { echo "[image-pin] $*"; }

[[ -f "${ENV_PIN}" ]] || { echo "[image-pin] FAIL: no pin at ${ENV_PIN}" >&2; exit 1; }

if ! command -v jq >/dev/null 2>&1; then
  say "SKIP: jq not installed"
  exit 0
fi

jq -e . "${ENV_PIN}" >/dev/null 2>&1 || {
  echo "[image-pin] FAIL: ${ENV_PIN} is not valid JSON" >&2; exit 1; }

version="$(jq -r '.version' "${ENV_PIN}")"
env_sha="$(jq -r '.source_sha' "${ENV_PIN}")"
digest="$(jq -r '.image.digest' "${ENV_PIN}")"
ref="$(jq -r '.image.ref' "${ENV_PIN}")"
cell="$(jq -r '.image.cell' "${ENV_PIN}")"

say "version       ${version}"
say "cell          ${cell}"
say "source_sha    ${env_sha}"
say "digest        ${digest}"

# --- 1. Shape ----------------------------------------------------------------
for field in .version .source_sha .recipe .scope .image.digest .image.ref \
             .image.python_bin .image.children.amd64 .image.children.arm64; do
  v="$(jq -r "${field} // empty" "${ENV_PIN}")"
  if [[ -z "${v}" || "${v}" == "null" ]]; then
    echo "[image-pin] FAIL: ${field} is missing from ${ENV_PIN}" >&2
    fail=1
  fi
done

if [[ ! "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "[image-pin] FAIL: .image.digest is not a sha256 digest: ${digest}" >&2
  fail=1
fi

if [[ "${ref}" != *"@${digest}" ]]; then
  echo "[image-pin] FAIL: .image.ref does not end in @.image.digest" >&2
  echo "           ref    ${ref}" >&2
  echo "           digest ${digest}" >&2
  fail=1
fi

# --- 2. The workflow pulls that digest ---------------------------------------
if [[ -f "${WORKFLOW}" ]]; then
  wf_ref="$(sed -n 's/^[[:space:]]*PYASC_SIM_IMAGE:[[:space:]]*//p' "${WORKFLOW}" | head -1)"
  if [[ -z "${wf_ref}" ]]; then
    echo "[image-pin] FAIL: no PYASC_SIM_IMAGE in ${WORKFLOW#${REPO_ROOT}/}" >&2
    fail=1
  elif [[ "${wf_ref}" != *"@sha256:"* ]]; then
    echo "[image-pin] FAIL: PYASC_SIM_IMAGE is pinned by tag, not digest:" >&2
    echo "           ${wf_ref}" >&2
    fail=1
  elif [[ "${wf_ref}" != "${ref}" ]]; then
    echo "[image-pin] FAIL: ${WORKFLOW#${REPO_ROOT}/} and the pin disagree" >&2
    echo "           workflow ${wf_ref}" >&2
    echo "           pin      ${ref}" >&2
    echo "           Re-pin with scripts/sync-image-pin.sh, then copy .image.ref" >&2
    fail=1
  else
    say "workflow      agrees"
  fi
fi

# --- 3. Image and vendored target suite share one upstream commit ------------
if [[ -f "${TARGET_PIN}" ]]; then
  suite_sha="$(jq -r '.commit' "${TARGET_PIN}")"
  if [[ "${suite_sha}" == "unresolved" ]]; then
    say "target suite  pin is unresolved; resolve with scripts/sync-target-suite.sh"
  elif [[ "${suite_sha}" != "${env_sha}" ]]; then
    echo "[image-pin] FAIL: image and target suite come from different commits" >&2
    echo "           image        ${env_sha}" >&2
    echo "           target suite ${suite_sha}" >&2
    echo "           Re-sync both: scripts/sync-image-pin.sh and" >&2
    echo "           scripts/sync-target-suite.sh --ref ${env_sha}" >&2
    fail=1
  else
    say "target suite  ${suite_sha} (matches)"
  fi
fi

if [[ ${fail} -ne 0 ]]; then
  echo "[image-pin] FAILED" >&2
  exit 1
fi
say "OK"
