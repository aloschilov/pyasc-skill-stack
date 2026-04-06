#!/usr/bin/env bash
set -euo pipefail

KERNEL_PATH="${1:?Usage: $0 <kernel.py> [backend] [platform]}"
BACKEND="${2:-Model}"
PLATFORM="${3:-Ascend910B1}"

if [ ! -f "$KERNEL_PATH" ]; then
    echo "[ERROR] Kernel file not found: $KERNEL_PATH"
    exit 1
fi

echo "[INFO] Running kernel: $KERNEL_PATH"
echo "[INFO] Backend: $BACKEND"
echo "[INFO] Platform: $PLATFORM"

if [ "$BACKEND" = "Model" ] && [ -n "${ASCEND_HOME_PATH:-}" ]; then
    SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/${PLATFORM}/lib"
    if [ -d "$SIM_LIB" ]; then
        export LD_LIBRARY_PATH="${SIM_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[INFO] LD_LIBRARY_PATH prepended with simulator libs: $SIM_LIB"
    else
        echo "[WARN] Simulator lib directory not found: $SIM_LIB (LD_LIBRARY_PATH unchanged)"
    fi
fi

CMD=(python3.10 "$KERNEL_PATH" -r "$BACKEND" -v "$PLATFORM")

echo "[INFO] Command: ${CMD[*]}"
echo ""

if "${CMD[@]}"; then
    echo ""
    echo "[PASS] Kernel execution successful"
else
    ec=$?
    echo ""
    echo "[FAIL] Kernel execution failed (exit code: $ec)"
    exit 1
fi
