#!/usr/bin/env bash
# Verify the image pin is internally consistent. Offline; safe in --fast CI.
#
# Three things have to agree, and nothing but a check keeps them agreeing:
#
#   1. reference/images/ENV.json is well formed and pins a digest, not a tag.
#      A tag is mutable, so a tagged runtime silently invalidates every result
#      recorded beneath it. Its matrix report URL names a specific release for
#      the same reason: "latest" is a pointer, and pointers move.
#   2. .github/workflows/ci.yml pulls exactly the digest CI is supposed to run,
#      which is the development overlay when one is pinned. The workflow needs
#      a literal because GitHub Actions cannot read a file into `env:`, so the
#      value is duplicated -- and duplicated values drift. The overlay must
#      also declare the base it was built on, since an overlay from a
#      different base would pair this suite with a runtime built from another
#      upstream commit.
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

# CI runs the overlay when there is one; the minimal image stays recorded as
# the base of record.
has_dev="$(jq -r 'has("dev_image")' "${ENV_PIN}")"
if [[ "${has_dev}" == "true" ]]; then
  run_ref="$(jq -r '.dev_image.ref' "${ENV_PIN}")"
  run_digest="$(jq -r '.dev_image.digest' "${ENV_PIN}")"
  run_label="dev_image"
else
  run_ref="${ref}"
  run_digest="${digest}"
  run_label="image"
fi

say "version       ${version}"
say "cell          ${cell}"
say "source_sha    ${env_sha}"
say "base digest   ${digest}"
if [[ "${has_dev}" == "true" ]]; then
  say "dev digest    ${run_digest}"
  say "dev torch     $(jq -r '.dev_image.torch' "${ENV_PIN}") | pytest $(jq -r '.dev_image.pytest' "${ENV_PIN}")"
fi

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

# --- 1b. The overlay, when pinned --------------------------------------------
if [[ "${has_dev}" == "true" ]]; then
  for field in .dev_image.digest .dev_image.ref .dev_image.base_report \
               .dev_image.torch .dev_image.pytest \
               .dev_image.children.amd64 .dev_image.children.arm64; do
    v="$(jq -r "${field} // empty" "${ENV_PIN}")"
    if [[ -z "${v}" || "${v}" == "null" ]]; then
      echo "[image-pin] FAIL: ${field} is missing from ${ENV_PIN}" >&2
      fail=1
    fi
  done

  if [[ ! "${run_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "[image-pin] FAIL: .dev_image.digest is not a sha256 digest: ${run_digest}" >&2
    fail=1
  fi

  if [[ "${run_ref}" != *"@${run_digest}" ]]; then
    echo "[image-pin] FAIL: .dev_image.ref does not end in @.dev_image.digest" >&2
    fail=1
  fi

  # The overlay is the base plus two packages. If it was built on a different
  # base, the wheel inside it comes from a different upstream commit than the
  # vendored suite, and a green gate would mean nothing.
  dev_base="$(jq -r '.dev_image.base_report' "${ENV_PIN}")"
  pin_base="$(jq -r '.matrix.release_tag' "${ENV_PIN}")"
  if [[ "${dev_base}" != "${pin_base}" ]]; then
    echo "[image-pin] FAIL: the overlay was built on a different base" >&2
    echo "           overlay base  ${dev_base}" >&2
    echo "           pinned base   ${pin_base}" >&2
    echo "           Re-sync with scripts/sync-image-pin.sh --release ${pin_base}" >&2
    fail=1
  else
    say "overlay base  ${dev_base} (matches)"
  fi
fi

# The report URL must name the release it came from. GitHub's /releases/latest/
# resolves to whichever release was published most recently, and upstream
# publishes dev-overlay and dev-test releases into the same repository -- so a
# "latest" URL silently stops pointing at a matrix report at all. It did:
# releases/latest/download/matrix.json now 404s because the newest release is
# a dev-tests-* one. Provenance that 404s is not provenance, and this is
# checkable offline by requiring the tag to appear in the URL.
report="$(jq -r '.matrix.report // empty' "${ENV_PIN}")"
release_tag="$(jq -r '.matrix.release_tag // empty' "${ENV_PIN}")"
if [[ -z "${report}" || -z "${release_tag}" ]]; then
  echo "[image-pin] FAIL: .matrix.report and .matrix.release_tag are required" >&2
  fail=1
elif [[ "${report}" == *"/releases/latest/"* ]]; then
  echo "[image-pin] FAIL: .matrix.report uses the mutable 'latest' pointer:" >&2
  echo "           ${report}" >&2
  echo "           Re-sync with scripts/sync-image-pin.sh to record the" >&2
  echo "           per-release asset URL for ${release_tag}." >&2
  fail=1
elif [[ "${report}" != *"${release_tag}"* ]]; then
  echo "[image-pin] FAIL: .matrix.report does not name .matrix.release_tag" >&2
  echo "           report       ${report}" >&2
  echo "           release_tag  ${release_tag}" >&2
  fail=1
else
  say "report        pinned to ${release_tag}"
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
  elif [[ "${wf_ref}" != "${run_ref}" ]]; then
    echo "[image-pin] FAIL: ${WORKFLOW#${REPO_ROOT}/} and the pin disagree" >&2
    echo "           workflow ${wf_ref}" >&2
    echo "           pin      ${run_ref}" >&2
    echo "           Re-pin with scripts/sync-image-pin.sh, then copy .${run_label}.ref" >&2
    fail=1
  else
    say "workflow      agrees (.${run_label}.ref)"
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
