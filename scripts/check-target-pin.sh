#!/usr/bin/env bash
# Verify reference/asctile/ still matches its pin. Offline; safe in --fast CI.
#
# Exits non-zero only on real drift: a vendored file whose content no longer
# matches the hash recorded in PIN.json, or a file added/removed without the
# pin being updated. An unresolved commit is reported but is NOT a failure --
# resolving it needs network access that CI may not have.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/reference/asctile"
PIN="${DEST}/PIN.json"
PYTHON_BIN="${PYASC_PYTHON:-python3}"

fail=0
say() { echo "[pin] $*"; }

[[ -f "${PIN}" ]] || { echo "[pin] FAIL: no pin at ${PIN}" >&2; exit 1; }

if ! command -v jq >/dev/null 2>&1; then
  say "SKIP: jq not installed"
  exit 0
fi

pinned_commit="$(jq -r '.commit' "${PIN}")"
resolved_path="$(jq -r '.resolved_path' "${PIN}")"
say "upstream      $(jq -r '.upstream' "${PIN}")"
say "path          ${resolved_path}"
say "commit        ${pinned_commit}"

# --- Content drift -----------------------------------------------------------
while IFS=$'\t' read -r name want; do
  f="${DEST}/target/${name}"
  if [[ ! -f "$f" ]]; then
    echo "[pin] FAIL: ${name} is in the pin but missing from the tree" >&2
    fail=1
    continue
  fi
  got="$(sha256sum "$f" | awk '{print $1}')"
  if [[ "${got}" != "${want}" ]]; then
    echo "[pin] FAIL: ${name} content differs from the pin" >&2
    fail=1
  fi
done < <(jq -r '.files | to_entries[] | "\(.key)\t\(.value)"' "${PIN}")

# Untracked additions: a file present in the suite but absent from the pin
# means someone edited the snapshot by hand instead of re-syncing.
for f in "${DEST}"/target/*.py; do
  base="$(basename "$f")"
  [[ "${base}" == "conftest.py" || "${base}" == "__init__.py" ]] && continue
  if [[ "$(jq -r --arg k "${base}" '.files[$k] // "missing"' "${PIN}")" == "missing" ]]; then
    echo "[pin] FAIL: ${base} is in the tree but not in the pin" >&2
    fail=1
  fi
done

# --- Runtime agreement (advisory) --------------------------------------------
# The suite should match the runtime it will be run against. Report, never fail:
# a developer may legitimately hold a pin while upgrading the wheel.
runtime_commit="$("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import re
import importlib.metadata as md
for dist in ("pyasc-dev", "pyasc"):
    try:
        desc = md.metadata(dist).get("Description") or ""
    except md.PackageNotFoundError:
        continue
    m = re.search(r"Source: .([0-9a-f]{40}).", desc)
    if m:
        print(m.group(1))
        break
PY
)"

if [[ -z "${runtime_commit}" ]]; then
  say "runtime       not installed or no source commit recorded"
elif [[ "${pinned_commit}" == "unresolved" ]]; then
  say "runtime       ${runtime_commit}"
  say "NOTE: pin commit is 'unresolved'. Resolve it with:"
  say "      scripts/sync-target-suite.sh"
elif [[ "${pinned_commit}" != "${runtime_commit}" ]]; then
  say "runtime       ${runtime_commit}"
  say "NOTE: pin and installed runtime differ. Target tests move between"
  say "      versions; re-sync with scripts/sync-target-suite.sh"
else
  say "runtime       ${runtime_commit} (matches pin)"
fi

if [[ ${fail} -ne 0 ]]; then
  echo "[pin] FAILED" >&2
  exit 1
fi
say "OK"
