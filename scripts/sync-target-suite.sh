#!/usr/bin/env bash
# Refresh reference/asctile/ from upstream PyAsc.
#
# Usage:
#   scripts/sync-target-suite.sh [--ref COMMIT] [--repo URL] [--source DIR]
#                                [--python BIN] [--dry-run]
#
#   --ref COMMIT   upstream commit to vendor. Defaults to the source commit of
#                  the PyAsc runtime currently installed, read from its own
#                  package metadata, so a wheel upgrade needs no argument.
#   --repo URL     upstream repository (default: the pin's recorded upstream).
#   --source DIR   use an existing checkout instead of cloning. --ref is then
#                  only recorded, not checked out.
#   --python BIN   interpreter to read the installed runtime version from.
#   --dry-run      report what would change; write nothing.
#
# Target tests move between PyAsc versions. Upstream has carried them at
# python/test/asc2/target/, then python/test/asctile/target/, and deleted the
# latter when AscTile was relocated under experimental/. This script therefore
# finds the suite by CONTENT rather than by a remembered path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/reference/asctile"
PIN="${DEST}/PIN.json"

# Upstream commit that renamed asc2 -> asctile. Anything older describes an API
# no skill in this repository documents, so vendoring it would be a regression
# dressed up as an update.
RENAME_COMMIT="ba6bb847"

REF=""
SOURCE_DIR=""
PYTHON_BIN="python3"
DRY_RUN=false
REPO_URL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="${2:?--ref requires a commit}"; shift 2 ;;
    --ref=*) REF="${1#--ref=}"; shift ;;
    --repo) REPO_URL="${2:?--repo requires a URL}"; shift 2 ;;
    --repo=*) REPO_URL="${1#--repo=}"; shift ;;
    --source) SOURCE_DIR="${2:?--source requires a directory}"; shift 2 ;;
    --source=*) SOURCE_DIR="${1#--source=}"; shift ;;
    --python) PYTHON_BIN="${2:?--python requires an interpreter}"; shift 2 ;;
    --python=*) PYTHON_BIN="${1#--python=}"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }
note() { echo "[sync] $*"; }

command -v jq >/dev/null 2>&1 || die "jq is required"

if [[ -z "${REPO_URL}" ]]; then
  if [[ -f "${PIN}" ]]; then
    REPO_URL="$(jq -r '.upstream' "${PIN}")"
  else
    die "no --repo given and no existing pin at ${PIN}"
  fi
fi

# --- Resolve the ref from the installed runtime ------------------------------
# The wheel records the commit it was built from in its own metadata. That
# makes "which version of the tests do I need?" a question the environment can
# answer, instead of a number someone has to remember.
installed_source_commit() {
  "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
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
}

if [[ -z "${REF}" ]]; then
  REF="$(installed_source_commit)"
  if [[ -n "${REF}" ]]; then
    note "ref defaulted to the installed runtime's source commit: ${REF}"
  else
    die "could not read a source commit from the installed runtime; pass --ref"
  fi
fi

# --- Obtain a checkout -------------------------------------------------------
WORKDIR=""
cleanup() { [[ -n "${WORKDIR}" && -d "${WORKDIR}" ]] && rm -rf "${WORKDIR}"; }
trap cleanup EXIT

if [[ -n "${SOURCE_DIR}" ]]; then
  [[ -d "${SOURCE_DIR}" ]] || die "no such directory: ${SOURCE_DIR}"
  CHECKOUT="$(cd "${SOURCE_DIR}" && pwd)"
  note "using existing checkout ${CHECKOUT} (not fetching ${REF})"
else
  WORKDIR="$(mktemp -d)"
  CHECKOUT="${WORKDIR}/pyasc"
  note "cloning ${REPO_URL}"
  git clone --quiet --filter=blob:none --no-checkout "${REPO_URL}" "${CHECKOUT}" \
    || die "clone failed; pass --source DIR to use a local checkout"
  git -C "${CHECKOUT}" fetch --quiet origin "${REF}" 2>/dev/null \
    || git -C "${CHECKOUT}" fetch --quiet origin
  git -C "${CHECKOUT}" checkout --quiet "${REF}" \
    || die "cannot check out ${REF}"

  # Refuse anything predating the asc2 -> asctile rename.
  if git -C "${CHECKOUT}" merge-base --is-ancestor "${RENAME_COMMIT}" HEAD 2>/dev/null; then
    :
  else
    die "${REF} predates the asc2->asctile rename (${RENAME_COMMIT}); it describes an API the skills do not document"
  fi
fi

# --- Locate the suite by content --------------------------------------------
# A file belongs to the AscTile target suite if it exercises the asctile API.
# Matching on the path or the filename would be wrong: test_vadd.py exists
# three times in a typical checkout, and the other copies are low-level asc.jit
# kernels that would import cleanly and verify the wrong thing.
note "locating the asctile suite by content"
mapfile -t HITS < <(
  cd "${CHECKOUT}" &&
  grep -rl --include='test_*.py' -e 'asctile\.jit' . 2>/dev/null |
  xargs -r grep -l 'asctile\.global_tensor' 2>/dev/null |
  sed 's|^\./||' | sort
)

[[ ${#HITS[@]} -gt 0 ]] || die "no asctile test files found in ${CHECKOUT}"

# The suite directory is the one holding the most matches.
SUITE_REL="$(printf '%s\n' "${HITS[@]}" | xargs -n1 dirname | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')"
SUITE_ABS="${CHECKOUT}/${SUITE_REL}"
note "suite: ${SUITE_REL} ($(printf '%s\n' "${HITS[@]}" | grep -c "^${SUITE_REL}/") files)"

# The suite needs whichever ancestor conftest.py defines require_c310.
PARENT_CONFTEST=""
probe="$(dirname "${SUITE_ABS}")"
while [[ "${probe}" == "${CHECKOUT}"* && "${probe}" != "/" ]]; do
  if [[ -f "${probe}/conftest.py" ]] && grep -q 'def require_c310' "${probe}/conftest.py"; then
    PARENT_CONFTEST="${probe}/conftest.py"
    break
  fi
  probe="$(dirname "${probe}")"
done
[[ -n "${PARENT_CONFTEST}" ]] || note "WARNING: no ancestor conftest.py defines require_c310; the suite may not run"

# --- Report the change -------------------------------------------------------
sha() { sha256sum "$1" | awk '{print $1}'; }

echo
echo "--- changes against the current pin ---"
changed=0
for f in "${SUITE_ABS}"/*.py; do
  base="$(basename "$f")"
  if [[ -f "${DEST}/target/${base}" ]]; then
    if [[ "$(sha "$f")" != "$(sha "${DEST}/target/${base}")" ]]; then
      echo "  modified  ${base}"; changed=$((changed+1))
    fi
  else
    echo "  added     ${base}"; changed=$((changed+1))
  fi
done
for f in "${DEST}"/target/*.py; do
  base="$(basename "$f")"
  [[ -f "${SUITE_ABS}/${base}" ]] || { echo "  removed   ${base}"; changed=$((changed+1)); }
done
[[ ${changed} -eq 0 ]] && echo "  (none)"
echo

if [[ "${DRY_RUN}" == true ]]; then
  note "dry run; nothing written"
  exit 0
fi

# --- Write ------------------------------------------------------------------
# conftest.py in the destination carries a local alias and is regenerated by
# hand, not overwritten here; everything else is replaced wholesale so that
# upstream deletions actually take effect.
note "writing ${DEST}/target/"
mkdir -p "${DEST}/target"
find "${DEST}/target" -maxdepth 1 -name '*.py' ! -name 'conftest.py' -delete
cp "${SUITE_ABS}"/*.py "${DEST}/target/"
[[ -f "${SUITE_ABS}/conftest.py" ]] && cp "${SUITE_ABS}/conftest.py" "${DEST}/target/conftest.py"

if [[ -n "${PARENT_CONFTEST}" ]]; then
  note "parent conftest available at ${PARENT_CONFTEST#${CHECKOUT}/}"
  note "NOT overwriting ${DEST}/conftest.py -- it carries the local asctile alias."
  note "Re-apply the alias by hand if upstream's version changed:"
  note "  diff ${PARENT_CONFTEST} ${DEST}/conftest.py"
fi

RESOLVED_COMMIT="$(git -C "${CHECKOUT}" rev-parse HEAD 2>/dev/null || echo "${REF}")"

note "writing ${PIN}"
{
  echo '{'
  echo "  \"upstream\": \"${REPO_URL}\","
  echo "  \"commit\": \"${RESOLVED_COMMIT}\","
  echo "  \"resolved_path\": \"${SUITE_REL}\","
  echo "  \"synced_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo '  "files": {'
  first=true
  for f in $(ls "${DEST}/target"/*.py | sort); do
    base="$(basename "$f")"
    [[ "${base}" == "conftest.py" ]] && continue
    ${first} || echo ','
    first=false
    printf '    "%s": "%s"' "${base}" "$(sha "$f")"
  done
  echo
  echo '  }'
  echo '}'
} > "${PIN}"

note "done. Review the diff before committing."
