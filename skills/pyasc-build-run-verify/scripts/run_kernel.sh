#!/usr/bin/env bash
set -euo pipefail

KERNEL_PATH="${1:?Usage: $0 <kernel.py> [backend] [platform]}"
BACKEND="${2:-Model}"
PLATFORM="${3:-}"

if [ ! -f "$KERNEL_PATH" ]; then
    echo "[ERROR] Kernel file not found: $KERNEL_PATH"
    exit 1
fi

echo "[INFO] Running kernel: $KERNEL_PATH"
echo "[INFO] Backend: $BACKEND"
if [ -n "$PLATFORM" ]; then
    echo "[INFO] Platform: $PLATFORM"
fi

CMD="python3 $KERNEL_PATH -r $BACKEND"
if [ -n "$PLATFORM" ]; then
    CMD="$CMD -v $PLATFORM"
fi

echo "[INFO] Command: $CMD"
echo ""

if eval "$CMD"; then
    echo ""
    echo "[PASS] Kernel execution successful"
else
    echo ""
    echo "[FAIL] Kernel execution failed (exit code: $?)"
    exit 1
fi
