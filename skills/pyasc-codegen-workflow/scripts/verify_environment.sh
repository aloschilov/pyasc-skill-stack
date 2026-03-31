#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_NAME="${1:?Usage: $0 <kernel_name>}"

TEAM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/teams/pyasc-kernel-dev-team"
KERNEL_DIR="$TEAM_DIR/kernels/$KERNEL_NAME"
ENV_FILE="$KERNEL_DIR/docs/environment.json"

if [ ! -d "$KERNEL_DIR" ]; then
    echo "[ERROR] Kernel directory not found: $KERNEL_DIR"
    echo "[INFO] Run init_kernel_project.sh first."
    exit 1
fi

echo "[INFO] Verifying environment for kernel: $KERNEL_NAME"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' || echo "not found")

PYASC_VERSION=$(python3 -c "import asc; print(getattr(asc, '__version__', 'installed'))" 2>/dev/null || echo "not installed")

NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")

TORCH_NPU=$(python3 -c "import torch_npu; print('available')" 2>/dev/null || echo "not available")

CANN_PATH="${ASCEND_HOME_PATH:-not set}"
if [ "$CANN_PATH" != "not set" ] && [ -d "$CANN_PATH" ]; then
    CANN_STATUS="found"
else
    CANN_STATUS="not found"
fi

NPU_AVAILABLE=$(npu-smi info 2>/dev/null | head -1 || echo "not available")

MODEL_BACKEND="unknown"
if [ "$PYASC_VERSION" != "not installed" ]; then
    MODEL_BACKEND=$(python3 -c "
import asc.runtime.config as config
try:
    print('available')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
fi

cat > "$ENV_FILE" << EOF
{
  "kernel_name": "$KERNEL_NAME",
  "timestamp": "$(date -Iseconds)",
  "python": {
    "version": "$PYTHON_VERSION"
  },
  "pyasc": {
    "version": "$PYASC_VERSION"
  },
  "numpy": {
    "version": "$NUMPY_VERSION"
  },
  "torch": {
    "version": "$TORCH_VERSION",
    "torch_npu": "$TORCH_NPU"
  },
  "cann": {
    "ascend_home_path": "$CANN_PATH",
    "status": "$CANN_STATUS"
  },
  "npu": {
    "available": "$NPU_AVAILABLE"
  },
  "backend": {
    "model": "$MODEL_BACKEND"
  }
}
EOF

echo "[PASS] Environment saved to: $ENV_FILE"
echo ""
echo "Summary:"
echo "  Python:     $PYTHON_VERSION"
echo "  pyasc:      $PYASC_VERSION"
echo "  numpy:      $NUMPY_VERSION"
echo "  torch:      $TORCH_VERSION"
echo "  torch_npu:  $TORCH_NPU"
echo "  CANN:       $CANN_STATUS ($CANN_PATH)"
echo "  NPU:        $NPU_AVAILABLE"

ERRORS=0
if [ "$PYASC_VERSION" = "not installed" ]; then
    echo ""
    echo "[WARN] pyasc is not installed. Install with: pip install pyasc"
    ERRORS=$((ERRORS + 1))
fi
if [ "$NUMPY_VERSION" = "not installed" ]; then
    echo "[WARN] numpy is not installed. Install with: pip install 'numpy<2'"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "[WARN] $ERRORS issue(s) found. Fix before proceeding to Phase 1."
else
    echo ""
    echo "[PASS] Environment looks good. Ready for Phase 1."
fi
