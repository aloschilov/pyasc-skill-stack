#!/usr/bin/env bash
# =============================================================================
# Token Usage Analysis for OpenCode sessions
# =============================================================================
# Usage: analyze-tokens.sh <session.json>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SESSION_FILE="${1:?Usage: $0 <session.json>}"

if [ ! -f "$SESSION_FILE" ]; then
    echo "[ERROR] Session file not found: $SESSION_FILE"
    exit 1
fi

# Try Python analysis first
PYTHON="${PYASC_PYTHON:-python3.10}"
PY_SCRIPT="$SCRIPT_DIR/analyze-token-usage.py"

if [ -f "$PY_SCRIPT" ] && command -v "$PYTHON" &>/dev/null; then
    "$PYTHON" "$PY_SCRIPT" "$SESSION_FILE"
    exit $?
fi

# Fallback: try python3 if PYASC_PYTHON missing
if [ -f "$PY_SCRIPT" ] && command -v python3 &>/dev/null; then
    python3 "$PY_SCRIPT" "$SESSION_FILE"
    exit $?
fi

# Fallback: basic grep / jq analysis
echo "=== Token Usage Analysis (basic) ==="
echo ""
echo "File: $SESSION_FILE"
echo "Size: $(wc -c < "$SESSION_FILE") bytes"
echo ""

if command -v jq &>/dev/null; then
    echo "--- Token counts (from jq) ---"
    input_tokens=$(jq -r '.. | .input_tokens? // empty' "$SESSION_FILE" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    output_tokens=$(jq -r '.. | .output_tokens? // empty' "$SESSION_FILE" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    cache_tokens=$(jq -r '.. | .cache_read_input_tokens? // empty' "$SESSION_FILE" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    echo "  Input tokens:  $input_tokens"
    echo "  Output tokens: $output_tokens"
    echo "  Cache tokens:  $cache_tokens"
    echo "  Total tokens:  $((input_tokens + output_tokens))"
else
    echo "--- Token counts (grep estimate) ---"
    input_matches=$(grep -oE '"input_tokens"[[:space:]]*:[[:space:]]*[0-9]+' "$SESSION_FILE" 2>/dev/null | grep -oE '[0-9]+$' | awk '{sum+=$1} END{print sum+0}' || true)
    output_matches=$(grep -oE '"output_tokens"[[:space:]]*:[[:space:]]*[0-9]+' "$SESSION_FILE" 2>/dev/null | grep -oE '[0-9]+$' | awk '{sum+=$1} END{print sum+0}' || true)
    cache_matches=$(grep -oE '"cache_read_input_tokens"[[:space:]]*:[[:space:]]*[0-9]+' "$SESSION_FILE" 2>/dev/null | grep -oE '[0-9]+$' | awk '{sum+=$1} END{print sum+0}' || true)
    echo "  Input tokens:  ${input_matches:-0}"
    echo "  Output tokens: ${output_matches:-0}"
    echo "  Cache tokens:  ${cache_matches:-0}"
    echo "  Total tokens:  $((${input_matches:-0} + ${output_matches:-0}))"
    echo ""
    echo "  (Install jq for broader JSON traversal)"
fi
