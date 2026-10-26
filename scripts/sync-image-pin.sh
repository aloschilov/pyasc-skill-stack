#!/usr/bin/env bash
# Refresh reference/images/ENV.json from the upstream image matrix.
#
# Usage:
#   scripts/sync-image-pin.sh [--ubuntu 22.04] [--cann 9.2.0] [--python 3.11]
#                             [--release TAG] [--matrix FILE] [--dry-run]
#
#   --ubuntu/--cann/--python  select one cell of the matrix. Defaults come from
#                             the current pin, so a plain run just moves the
#                             existing selection forward to the newest matrix.
#   --release TAG             a specific pyasc-images release. Default: the
#                             newest release tagged matrix-*.
#   --matrix FILE             use a local matrix.json instead of downloading.
#   --dry-run                 report what would change; write nothing.
#
# This repository does not build simulator images. compiler-team-ru/pyasc-images
# builds the full Ubuntu x CANN x Python x arch matrix, smoke-tests each cell,
# and publishes matrix.json alongside the images.
#
# The matrix MOVES between versions, which is the whole reason this is a script
# and not a table in the README. On 2026-10-23 eight cells were incompatible
# because upstream used dataclass(slots=True), which needs Python 3.10; on
# 2026-10-24 that was gone and all 32 cells passed. Anything transcribed by
# hand is wrong within a day.
#
# Note the release is resolved by tag prefix, not by GitHub's "latest" pointer.
# That pointer belongs to whichever release was published last, and upstream
# now publishes dev-overlay and dev-test releases into the same repository, so
# releases/latest/download/matrix.json resolves to a release with no such
# asset and 404s. Filtering on matrix-* is what keeps this pointing at a base
# matrix report.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="${REPO_ROOT}/reference/images/ENV.json"
MATRIX_REPO="compiler-team-ru/pyasc-images"

UBUNTU=""; CANN=""; PYTHON_VER=""; RELEASE=""; MATRIX_FILE=""; DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ubuntu)  UBUNTU="${2:?--ubuntu requires a value}"; shift 2 ;;
    --ubuntu=*) UBUNTU="${1#--ubuntu=}"; shift ;;
    --cann)    CANN="${2:?--cann requires a value}"; shift 2 ;;
    --cann=*)  CANN="${1#--cann=}"; shift ;;
    --python)  PYTHON_VER="${2:?--python requires a value}"; shift 2 ;;
    --python=*) PYTHON_VER="${1#--python=}"; shift ;;
    --release) RELEASE="${2:?--release requires a tag}"; shift 2 ;;
    --release=*) RELEASE="${1#--release=}"; shift ;;
    --matrix)  MATRIX_FILE="${2:?--matrix requires a file}"; shift 2 ;;
    --matrix=*) MATRIX_FILE="${1#--matrix=}"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }
note() { echo "[image-pin] $*"; }

command -v jq >/dev/null 2>&1 || die "jq is required"

# The newest release whose tag starts with matrix-, newest first by publish
# time. Deliberately not GitHub's /releases/latest: that is simply whatever was
# published most recently, and the dev-overlay and dev-test releases upstream
# now publishes share this repository, so the latest release is routinely one
# that carries no matrix.json at all.
#
# dev-matrix-* is excluded for free -- it does not start with matrix-.
# Drafts are skipped; prereleases are not, since a prerelease matrix report is
# still a matrix report.
newest_matrix_release() {
  local auth=() token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  # Spelled as an if, not `[[ ... ]] && ...`: that idiom returns the test's
  # status, which under set -e has already cost this repository one green run
  # reported as a failure.
  if [[ -n "${token}" ]]; then
    auth=(-H "Authorization: Bearer ${token}")
  fi
  curl -fsSL "${auth[@]}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${MATRIX_REPO}/releases?per_page=100" \
    | jq -r '
        [ .[]
          | select(.draft | not)
          | select(.tag_name | startswith("matrix-"))
        ]
        | sort_by(.published_at // .created_at)
        | reverse
        | .[0].tag_name // empty'
}

# Defaults from the existing pin: a bare run refreshes the same selection.
if [[ -f "${PIN}" ]]; then
  [[ -z "${UBUNTU}"     ]] && UBUNTU="$(jq -r '.image.ubuntu' "${PIN}")"
  [[ -z "${CANN}"       ]] && CANN="$(jq -r '.image.cann' "${PIN}")"
  [[ -z "${PYTHON_VER}" ]] && PYTHON_VER="$(jq -r '.image.python' "${PIN}")"
fi
[[ -n "${UBUNTU}" && -n "${CANN}" && -n "${PYTHON_VER}" ]] \
  || die "no existing pin; pass --ubuntu, --cann and --python"

# --- Obtain matrix.json ------------------------------------------------------
WORKDIR=""
# Must return 0: on a normal exit bash adopts the trap's last status as the
# script's, so a short-circuited && here would fail an otherwise clean run.
cleanup() {
  if [[ -n "${WORKDIR}" && -d "${WORKDIR}" ]]; then rm -rf "${WORKDIR}"; fi
  return 0
}
trap cleanup EXIT

if [[ -n "${MATRIX_FILE}" ]]; then
  [[ -f "${MATRIX_FILE}" ]] || die "no such file: ${MATRIX_FILE}"
  MATRIX="${MATRIX_FILE}"
  note "using local ${MATRIX}"
else
  WORKDIR="$(mktemp -d)"
  MATRIX="${WORKDIR}/matrix.json"
  if [[ -z "${RELEASE}" ]]; then
    note "resolving the newest matrix-* release in ${MATRIX_REPO}"
    RELEASE="$(newest_matrix_release)" \
      || die "could not list releases of ${MATRIX_REPO}"
    [[ -n "${RELEASE}" ]] \
      || die "no release tagged matrix-* in ${MATRIX_REPO}; pass --release TAG"
    note "resolved ${RELEASE}"
  fi
  URL="https://github.com/${MATRIX_REPO}/releases/download/${RELEASE}/matrix.json"
  note "fetching ${URL}"
  curl -fsSL "${URL}" -o "${MATRIX}" || die "could not fetch the matrix report"
fi

jq -e '.cells and .manifests' "${MATRIX}" >/dev/null 2>&1 \
  || die "${MATRIX} does not look like a matrix report"

# --- Select the cell and rewrite the pin ------------------------------------
note "selecting ubuntu=${UBUNTU} cann=${CANN} python=${PYTHON_VER}"

RELEASE_TAG="${RELEASE}" DRY_RUN="${DRY_RUN}" PIN_PATH="${PIN}" \
UBUNTU="${UBUNTU}" CANN="${CANN}" PYTHON_VER="${PYTHON_VER}" \
MATRIX_REPO="${MATRIX_REPO}" \
python3 - "${MATRIX}" <<'PY'
import json, os, sys, datetime

matrix = json.load(open(sys.argv[1]))
ubuntu = os.environ["UBUNTU"]
cann = os.environ["CANN"]
py = os.environ["PYTHON_VER"]
pin_path = os.environ["PIN_PATH"]
dry = os.environ["DRY_RUN"] == "true"
repo = os.environ["MATRIX_REPO"]
release_tag = os.environ.get("RELEASE_TAG") or ""

prefix = f"u{ubuntu}-c{cann}-p{py}-"
cells = [c for c in matrix["cells"] if c["key"].startswith(prefix)]
if not cells:
    keys = sorted({c["key"].rsplit("-", 1)[0] for c in matrix["cells"]})
    sys.exit("error: no cell u%s-c%s-p%s-*.\n  available: %s"
             % (ubuntu, cann, py, ", ".join(keys)))

bad = [c for c in cells if c.get("status") != "pass"]
if bad:
    sys.exit("error: %s did not pass upstream: %s"
             % (prefix, ", ".join(f"{c['key']}={c.get('status')}" for c in bad)))

# The multi-arch manifest is what CI should pull: one ref, both architectures.
want = f"-cann{cann}-u{ubuntu}-py{py}-"
mans = [m for m in matrix["manifests"] if want in m["tag"]]
if len(mans) != 1:
    sys.exit("error: expected exactly one manifest matching %r, found %d"
             % (want, len(mans)))
man = mans[0]
if man.get("status") != "pass":
    sys.exit("error: manifest %s status=%s" % (man["tag"], man.get("status")))

by_arch = {c["environment"]["arch"]: c for c in cells}
version = matrix["version"]

if not release_tag:
    release_tag = "matrix-%s-%s" % (version, matrix["recipe"][:16])

# Per-release asset URL, never releases/latest/download/. "latest" follows
# whatever upstream published most recently -- today that is a dev-test
# release with no matrix.json -- so a pin that recorded it would cite a 404
# as its provenance. check-image-pin.sh enforces that this URL names the
# release tag.
report_url = "https://github.com/%s/releases/download/%s/matrix.json" % (
    repo, release_tag)

pin = {
    "schema_version": 1,
    "note": ("Pin for the upstream-built simulator image. This repository does "
             "not build simulator images; it consumes one cell of the matrix "
             "published by %s. Refresh with scripts/sync-image-pin.sh; verify "
             "offline with scripts/check-image-pin.sh." % repo),
    "matrix": {
        "repository": repo,
        "release_tag": release_tag,
        "report": report_url,
    },
    "version": version,
    "source_repository": "https://github.com/compiler-team-ru/pyasc.git",
    "source_sha": matrix["source_sha"],
    "recipe": matrix["recipe"],
    "scope": matrix["scope"],
    "image": {
        "cell": f"u{ubuntu}-c{cann}-p{py}",
        "ubuntu": ubuntu,
        "cann": cann,
        "python": py,
        "python_bin": "python%s" % py,
        "simulator": cells[0]["environment"]["simulator"],
        "anonymous_pull": all(
            c.get("public_registry_check", {}).get("anonymous_registry_access")
            for c in cells),
        "tag": man["tag"],
        "digest": man["digest"],
        "ref": "ghcr.io/compiler-team-ru/pyasc-sim@%s" % man["digest"],
        "children": man["children"],
    },
    "wheel": {
        **{arch: {"filename": c["wheel"]["filename"],
                  "sha256": c["wheel"]["sha256"]}
           for arch, c in sorted(by_arch.items())},
        "manifest": ("https://github.com/compiler-team-ru/asctile/releases/"
                     "download/wheel-%s/wheel-manifest.json" % version),
    },
    "synced_at": datetime.datetime.now(datetime.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
}

old = {}
if os.path.exists(pin_path):
    old = json.load(open(pin_path))

for field in ("version", "source_sha", "recipe"):
    if old.get(field) != pin[field]:
        print("  %-14s %s -> %s" % (field, old.get(field, "(none)"), pin[field]))
if old.get("image", {}).get("digest") != pin["image"]["digest"]:
    print("  %-14s %s -> %s" % ("digest",
                                old.get("image", {}).get("digest", "(none)"),
                                pin["image"]["digest"]))

if dry:
    print("[image-pin] dry run; nothing written")
    sys.exit(0)

with open(pin_path, "w") as fh:
    json.dump(pin, fh, indent=2)
    fh.write("\n")
print("[image-pin] wrote %s" % pin_path)
PY

if [[ "${DRY_RUN}" == false ]]; then
  note "now update PYASC_SIM_IMAGE in .github/workflows/ci.yml to:"
  note "  $(jq -r '.image.ref' "${PIN}")"
  note "scripts/check-image-pin.sh enforces that they agree."
fi
