#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: no absolute home-directory paths in tracked files
#
# A path like /home/<someone>/workspace/ops-math baked into a default is a
# defect that only shows on someone else's machine, and it shows badly: the
# tool fails with a FileNotFoundError naming a stranger's account, which reads
# as a broken tool rather than as missing configuration. Documentation has the
# same problem one step removed -- a reader follows an absolute path that
# cannot exist for them.
#
# So: external trees are named by environment variable and resolved through
# tests/tools/repo_paths.py:external_root(), which says what is missing and
# what to set. This test is what keeps that true.
#
# Deliberately NOT flagged:
#   - `~/...`, which is portable by construction (install targets, caches).
#   - github.com/<user>/... URLs, where the name is an org, not a home dir.
#   - /home or /Users with no user component.
#
# Offline and pure text, so it is safe in --fast.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Test: no absolute home paths in tracked files ==="

if ! command -v git >/dev/null 2>&1; then
    echo "[SKIP] git not installed"
    exit 0
fi

# A home path is /home/<name>/ or /Users/<name>/ with a real name component.
# Anchoring on the trailing slash keeps bare /home and /Users out of it.
PATTERN='(/home/|/Users/)[A-Za-z0-9._-]+/'

# This file necessarily contains the pattern it searches for.
SELF="tests/unit/tools/test-portability.sh"

hits=""
while IFS= read -r f; do
    [ "$f" = "$SELF" ] && continue
    [ -f "$f" ] || continue
    # Skip binaries; grep -I does it per file.
    if match="$(grep -InIE "$PATTERN" "$f" 2>/dev/null \
                | grep -vE 'github\.com/[A-Za-z0-9._-]+/' || true)"; then
        if [ -n "$match" ]; then
            hits="${hits}${f}
$(echo "$match" | sed 's/^/    /')
"
        fi
    fi
done < <(git ls-files)

if [ -n "$hits" ]; then
    echo "[FAIL] tracked files contain absolute home-directory paths:"
    printf '%s' "$hits"
    echo
    echo "       For code: take the location from an environment variable and"
    echo "       resolve it with external_root() in tests/tools/repo_paths.py,"
    echo "       which fails with a readable diagnostic when it is unset."
    echo "       For docs: name the variable, or use a <placeholder>."
    exit 1
fi

echo "[PASS] no absolute home-directory paths in tracked files"
exit 0
