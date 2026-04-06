#!/usr/bin/env bash
# =============================================================================
# CI Gate Runner
# =============================================================================
# Single entry point for CI pipelines. Selects checks based on --tier.
#
# Usage:
#   bash tests/ci-gate.sh --tier pr       # PR checks (< 30s)
#   bash tests/ci-gate.sh --tier merge    # Merge checks (< 5min)
#   bash tests/ci-gate.sh --tier nightly  # Nightly checks (15-30min)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/test-helpers.sh"

TIER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier) TIER="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 --tier pr|merge|nightly"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$TIER" ]; then
    echo "Usage: $0 --tier pr|merge|nightly"
    exit 1
fi

FAILED=0

echo "========================================"
echo " CI Gate: $TIER"
echo "========================================"
echo ""

# ============================================
# PR tier: L1 + JIT
# ============================================

run_pr_gate() {
    echo "--- L1 Unit Tests ---"
    if bash "$SCRIPT_DIR/run-tests.sh" --fast; then
        echo ""
    else
        FAILED=$((FAILED + 1))
    fi

    echo ""
    echo "--- JIT Verification (golden kernels) ---"
    for kernel in "$SKILLS_DIR"/golden/kernels/*.py; do
        if [ -f "$kernel" ]; then
            base=$(basename "$kernel")
            jit_exit=0
            $PYTHON "$TOOLS_DIR/pytest_verify_kernel.py" "$kernel" 2>&1 || jit_exit=$?
            if [ "$jit_exit" -eq 0 ]; then
                echo "  [PASS] JIT: $base"
            elif [ "$jit_exit" -eq 2 ]; then
                echo "  [SKIP] JIT: $base (pyasc not available)"
            else
                echo "  [FAIL] JIT: $base"
                FAILED=$((FAILED + 1))
            fi
        fi
    done
}

# ============================================
# Merge tier: PR + simulator
# ============================================

run_merge_gate() {
    run_pr_gate

    echo ""
    echo "--- Simulator Verification (golden kernels) ---"

    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        echo "  [SKIP] ASCEND_HOME_PATH not set (source set_env.sh first)"
        echo "  Merge gate requires CANN simulator. See docs/cann-setup.md."
        exit 2
    fi

    for kernel in "$SKILLS_DIR"/golden/kernels/*.py; do
        if [ -f "$kernel" ]; then
            base=$(basename "$kernel")
            sim_exit=0
            $PYTHON "$TOOLS_DIR/run_and_verify.py" "$kernel" --mode simulator 2>&1 || sim_exit=$?
            if [ "$sim_exit" -eq 0 ]; then
                echo "  [PASS] Simulator: $base"
            elif [ "$sim_exit" -eq 2 ]; then
                echo "  [SKIP] Simulator: $base (env not available)"
            else
                echo "  [FAIL] Simulator: $base"
                FAILED=$((FAILED + 1))
            fi
        fi
    done
}

# ============================================
# Nightly tier: merge + L2/L3
# ============================================

run_nightly_gate() {
    run_merge_gate

    echo ""
    echo "--- Full Test Suite (L1 + L2 + L3) ---"
    if bash "$SCRIPT_DIR/run-tests.sh" --all; then
        echo ""
    else
        FAILED=$((FAILED + 1))
    fi
}

# ============================================
# Dispatch
# ============================================

case "$TIER" in
    pr)      run_pr_gate ;;
    merge)   run_merge_gate ;;
    nightly) run_nightly_gate ;;
    *)
        echo "Unknown tier: $TIER (use: pr, merge, nightly)"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo " CI Gate Result: $TIER"
echo "========================================"
if [ "$FAILED" -gt 0 ]; then
    echo "  FAILED ($FAILED check(s))"
    exit 1
else
    echo "  PASSED"
    exit 0
fi
