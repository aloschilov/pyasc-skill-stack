#!/usr/bin/env bash
# Install all pyasc-skill-stack skills globally for Cursor and opencode via symlinks.
#
# Usage:
#   ./install.sh              # Cursor + opencode (default)
#   ./install.sh --claude     # also link ~/.claude/skills/<skill>
#   ./install.sh --agents     # also link ~/.agents/skills/<skill>
#   ./install.sh --target DIR # link into DIR only, leaving the defaults alone
#
# Each skill folder stays in this repository; symlinks point at it.
# Re-run after moving the clone to refresh links.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="${REPO_ROOT}/skills"

LINK_CLAUDE=false
LINK_AGENTS=false
# An explicit --target replaces the default destinations rather than adding to
# them, so a test run cannot disturb the user's real Cursor/opencode setup.
TARGET_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --claude) LINK_CLAUDE=true; shift ;;
    --agents) LINK_AGENTS=true; shift ;;
    --target)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "error: --target requires a directory" >&2
        exit 1
      fi
      TARGET_DIR="$2"; shift 2 ;;
    --target=*) TARGET_DIR="${1#--target=}"; shift ;;
    -h|--help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${TARGET_DIR}" ]]; then
  mkdir -p "${TARGET_DIR}"
  TARGET_DIR="$(cd "${TARGET_DIR}" && pwd)"
fi

if [[ ! -d "${SKILLS_DIR}" ]]; then
  echo "error: skills directory not found at ${SKILLS_DIR}" >&2
  exit 1
fi

link_skill() {
  local target_parent="$1"
  local skill_src="$2"
  local skill_name="$3"
  local link_path="${target_parent}/${skill_name}"
  mkdir -p "${target_parent}"
  if [[ -L "${link_path}" ]]; then
    rm -f "${link_path}"
  elif [[ -e "${link_path}" ]]; then
    echo "error: ${link_path} exists and is not a symlink; remove manually" >&2
    exit 1
  fi
  ln -s "${skill_src}" "${link_path}"
  echo "linked ${link_path} -> ${skill_src}"
}

# Remove links we previously created that now dangle (skill removed/renamed in
# the repo). Only links pointing into THIS repo's skills dir are considered, so
# foreign links (e.g. another repo's gitcode-api) are never touched.
prune_stale() {
  local target_parent="$1"
  [[ -d "${target_parent}" ]] || return 0
  local link dest
  for link in "${target_parent}"/*; do
    [[ -L "${link}" ]] || continue
    dest="$(readlink "${link}")"
    case "${dest}" in
      "${SKILLS_DIR}/"*)
        if [[ ! -e "${link}" ]]; then
          rm -f "${link}"
          echo "pruned stale ${link} -> ${dest}"
        fi ;;
    esac
  done
}

# Expand non-matching globs to nothing instead of the literal pattern, so an
# empty skills dir (or the prune_stale globs below) does not loop on "*/".
shopt -s nullglob

linked=0
for skill_src in "${SKILLS_DIR}"/*/; do
  skill_src="${skill_src%/}"
  skill_name="$(basename "${skill_src}")"

  if [[ ! -f "${skill_src}/SKILL.md" ]]; then
    echo "skip ${skill_name}: no SKILL.md"
    continue
  fi

  if [[ -n "${TARGET_DIR}" ]]; then
    link_skill "${TARGET_DIR}" "${skill_src}" "${skill_name}"
  else
    link_skill "${HOME}/.cursor/skills" "${skill_src}" "${skill_name}"
    link_skill "${HOME}/.config/opencode/skills" "${skill_src}" "${skill_name}"

    if [[ "${LINK_CLAUDE}" == true ]]; then
      link_skill "${HOME}/.claude/skills" "${skill_src}" "${skill_name}"
    fi

    if [[ "${LINK_AGENTS}" == true ]]; then
      link_skill "${HOME}/.agents/skills" "${skill_src}" "${skill_name}"
    fi
  fi

  if command -v skills-ref >/dev/null 2>&1; then
    skills-ref validate "${skill_src}"
  fi

  linked=$((linked + 1))
done

if [[ -n "${TARGET_DIR}" ]]; then
  prune_stale "${TARGET_DIR}"
else
  prune_stale "${HOME}/.cursor/skills"
  prune_stale "${HOME}/.config/opencode/skills"
  if [[ "${LINK_CLAUDE}" == true ]]; then
    prune_stale "${HOME}/.claude/skills"
  fi
  if [[ "${LINK_AGENTS}" == true ]]; then
    prune_stale "${HOME}/.agents/skills"
  fi
fi

if command -v skills-ref >/dev/null 2>&1; then
  echo "skills-ref validate: OK"
else
  echo "skills-ref not installed; skipped validation"
fi

echo "install complete (${linked} skill(s))"
