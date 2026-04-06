#!/usr/bin/env bash
# =============================================================================
# Session Analysis Tool for OpenCode exports
# =============================================================================
# Usage: analyze-session.sh <session.json> [--brief|--full|--json|--tools]
# Default: --brief. Flags may appear before or after the session path.
# Uses jq when available and the file is valid JSON; otherwise grep fallbacks.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/test-helpers.sh
source "$SCRIPT_DIR/../lib/test-helpers.sh" 2>/dev/null || true

SESSION_FILE=""
MODE="--brief"

usage() {
    echo "Usage: $0 <session.json> [--brief|--full|--json|--tools]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --brief|--full|--json|--tools) MODE="$1"; shift ;;
        -h|--help) usage; exit 0 ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$SESSION_FILE" ]]; then
                echo "[ERROR] Unexpected argument: $1" >&2
                usage >&2
                exit 1
            fi
            SESSION_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$SESSION_FILE" ]]; then
    echo "[ERROR] Usage: $0 <session.json> [--brief|--full|--json|--tools]" >&2
    exit 1
fi

if [[ ! -f "$SESSION_FILE" ]]; then
    echo "[ERROR] Session file not found: $SESSION_FILE" >&2
    exit 1
fi

SESSION_BYTES=$(($(wc -c < "$SESSION_FILE")))

have_jq() {
    command -v jq &>/dev/null && jq -e . "$SESSION_FILE" &>/dev/null
}

# --- grep fallbacks ----------------------------------------------------------

extract_tools_grep() {
    grep -oE '"name"[[:space:]]*:[[:space:]]*"[^"]*"' "$SESSION_FILE" 2>/dev/null \
        | sed 's/"name"[[:space:]]*:[[:space:]]*"//; s/"$//' \
        | sort | uniq -c | sort -rn || true
}

extract_skills_grep() {
    grep -oiE '"(skill|name)"[[:space:]]*:[[:space:]]*"[^"]*pyasc[^"]*"' "$SESSION_FILE" 2>/dev/null \
        | sed -E 's/.*"([^"]*pyasc[^"]*)".*/\1/i' \
        | sort -u || true
}

extract_file_ops_grep() {
    grep -oE '"(path|filePath|file)"[[:space:]]*:[[:space:]]*"[^"]*"' "$SESSION_FILE" 2>/dev/null \
        | sed -E 's/.*"([^"]+)".*/\1/' \
        | sort -u || true
}

# --- jq paths (OpenCode export: messages[].parts[]) --------------------------

extract_tools_jq() {
    jq -r '
      [.messages[]? | .parts[]?
        | select(type == "object" and has("tool")
            and ((((.type // "") | tostring) | ascii_downcase | gsub("_";"-")) == "tool"))
        | .tool]
      | map(select(type == "string" and . != ""))
      | group_by(.)
      | map("\(length) \(.[0])")
      | .[]
    ' "$SESSION_FILE" 2>/dev/null | sort -k1,1nr || true
}

extract_skills_jq() {
    jq -r '
      [.messages[]? | .parts[]?
        | select(type == "object" and (.tool == "skill"))
        | (.state // {}) as $s
        | (($s | .input // .Input // {}) | .name // .Name // empty)
        | strings
        | select(test("pyasc"; "i"))]
      | unique
      | .[]
    ' "$SESSION_FILE" 2>/dev/null || true
}

extract_file_ops_jq() {
    jq -r '
      [.messages[]? | .parts[]?
        | select(type == "object")
        | (
            if (.tool == "read" or .tool == "write" or .tool == "edit") then
              (.state // {}) | (.input // .Input // {})
              | (.filePath // .path // .file // empty)
            elif ((.type // "") | tostring | ascii_downcase | gsub("_";"-")) == "patch" then
              (.files // [])[]
            else
              empty
            end
          )
        | select(type == "string" and . != "")]
      | unique
      | .[]
    ' "$SESSION_FILE" 2>/dev/null || true
}

extract_tools() {
    if have_jq; then
        extract_tools_jq
    else
        extract_tools_grep
    fi
}

extract_skills() {
    if have_jq; then
        extract_skills_jq
    else
        extract_skills_grep
    fi
}

extract_file_ops() {
    if have_jq; then
        extract_file_ops_jq
    else
        extract_file_ops_grep
    fi
}

json_tools_object_grep() {
    echo -n "{"
    local first=1 n
    while read -r count name; do
        [[ -z "${name:-}" ]] && continue
        n=$(printf '%s' "$name" | jq -Rs . 2>/dev/null || printf '"%s"' "${name//\"/\\\"}")
        if [[ "$first" -eq 1 ]]; then
            first=0
        else
            echo -n ","
        fi
        printf '\n  %s: %s' "$n" "$count"
    done < <(extract_tools_grep)
    echo ""
    echo -n "}"
}

json_skills_array_grep() {
    echo -n "["
    local first=1 s enc
    while IFS= read -r s; do
        [[ -z "$s" ]] && continue
        enc=$(printf '%s' "$s" | jq -Rs . 2>/dev/null || printf '"%s"' "${s//\"/\\\"}")
        if [[ "$first" -eq 1 ]]; then
            first=0
        else
            echo -n ", "
        fi
        echo -n "$enc"
    done < <(extract_skills_grep)
    echo "]"
}

workflow_phase_lines() {
    local phases=("Phase 0" "Phase 1" "Phase 2" "Phase 3" "environment" "design" "implement" "verif")
    local ph count
    for ph in "${phases[@]}"; do
        count=$(grep -ci "$ph" "$SESSION_FILE" 2>/dev/null || true)
        if [[ "${count:-0}" -gt 0 ]]; then
            echo "  $ph: $count mention(s)"
        fi
    done
}

case "$MODE" in
    --brief)
        echo "=== Session Analysis (Brief) ==="
        echo ""
        echo "File: $SESSION_FILE"
        echo "Size: $SESSION_BYTES bytes"
        if have_jq; then
            echo "Parser: jq (structured)"
        else
            echo "Parser: grep (fallback; install jq for OpenCode message/part parsing)"
        fi
        echo ""
        echo "Tool invocations:"
        extract_tools | head -10 | while read -r count name; do
            printf "  %4s  %s\n" "$count" "$name"
        done
        echo ""
        echo "Skills referenced (pyasc):"
        extract_skills | while IFS= read -r skill; do
            [[ -z "$skill" ]] && continue
            echo "  - $skill"
        done
        ;;

    --full)
        echo "=== Session Analysis (Full) ==="
        echo ""
        echo "File: $SESSION_FILE"
        echo "Size: $SESSION_BYTES bytes"
        if have_jq; then
            echo "Parser: jq (structured)"
        else
            echo "Parser: grep (fallback)"
        fi
        echo ""
        echo "--- Tool Invocations ---"
        extract_tools | while read -r count name; do
            printf "  %4s  %s\n" "$count" "$name"
        done
        echo ""
        echo "--- Skills Referenced (pyasc) ---"
        extract_skills | while IFS= read -r skill; do
            [[ -z "$skill" ]] && continue
            echo "  - $skill"
        done
        echo ""
        echo "--- File Operations (paths) ---"
        extract_file_ops | head -50 | while IFS= read -r fpath; do
            [[ -z "$fpath" ]] && continue
            echo "  - $fpath"
        done
        echo ""
        echo "--- Workflow Phase Evidence ---"
        workflow_phase_lines
        ;;

    --json)
        if have_jq; then
            tools_json=$(
                jq -c '
                  [.messages[]? | .parts[]?
                    | select(type == "object" and has("tool")
                        and ((((.type // "") | tostring) | ascii_downcase | gsub("_";"-")) == "tool"))
                    | .tool]
                  | map(select(type == "string" and . != ""))
                  | group_by(.)
                  | map({key: .[0], value: length})
                  | from_entries
                ' "$SESSION_FILE" 2>/dev/null || echo "{}"
            )
            [[ -z "$tools_json" || "$tools_json" == "null" ]] && tools_json="{}"
            skills_json=$(
                jq -c '
                  [.messages[]? | .parts[]?
                    | select(type == "object" and (.tool == "skill"))
                    | (.state // {}) | (.input // .Input // {}) | (.name // .Name // empty)
                    | strings
                    | select(test("pyasc"; "i"))]
                  | unique
                ' "$SESSION_FILE" 2>/dev/null || echo "[]"
            )
            [[ -z "$skills_json" || "$skills_json" == "null" ]] && skills_json="[]"
            jq -n \
                --arg file "$SESSION_FILE" \
                --argjson size "$SESSION_BYTES" \
                --argjson tools "$tools_json" \
                --argjson skills "$skills_json" \
                '{file: $file, size_bytes: $size, tools: $tools, skills: $skills}'
        else
            tools_json=$(json_tools_object_grep)
            skills_json=$(json_skills_array_grep)
            cat <<EOF
{
  "file": "$SESSION_FILE",
  "size_bytes": $SESSION_BYTES,
  "tools": $tools_json,
  "skills": $skills_json
}
EOF
        fi
        ;;

    --tools)
        echo "=== Tool Call Details ==="
        echo ""
        extract_tools | while read -r count name; do
            printf "  %4s  %s\n" "$count" "$name"
        done
        total=$(extract_tools | awk '{sum+=$1} END{print sum+0}')
        echo ""
        echo "  Total: $total tool calls"
        ;;

    *)
        usage >&2
        exit 1
        ;;
esac
