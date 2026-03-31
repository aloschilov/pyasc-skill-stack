#!/usr/bin/env bash
set -euo pipefail

echo "=== pyasc Environment Check ==="
echo ""

ERRORS=0
WARNINGS=0

# Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' || echo "not found")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_VERSION" = "not found" ]; then
    echo "[FAIL] Python3 not found"
    ERRORS=$((ERRORS + 1))
elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ] && [ "$PYTHON_MINOR" -le 12 ]; then
    echo "[PASS] Python: $PYTHON_VERSION"
else
    echo "[WARN] Python: $PYTHON_VERSION (recommended: 3.9-3.12)"
    WARNINGS=$((WARNINGS + 1))
fi

# pyasc
PYASC_VERSION=$(python3 -c "import asc; print(getattr(asc, '__version__', 'installed'))" 2>/dev/null || echo "not installed")
if [ "$PYASC_VERSION" = "not installed" ]; then
    echo "[FAIL] pyasc not installed (pip install pyasc)"
    ERRORS=$((ERRORS + 1))
else
    echo "[PASS] pyasc: $PYASC_VERSION"
fi

# numpy
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")
if [ "$NUMPY_VERSION" = "not installed" ]; then
    echo "[FAIL] numpy not installed (pip install 'numpy<2')"
    ERRORS=$((ERRORS + 1))
else
    NUMPY_MAJOR=$(echo "$NUMPY_VERSION" | cut -d. -f1)
    if [ "$NUMPY_MAJOR" -ge 2 ]; then
        echo "[WARN] numpy: $NUMPY_VERSION (recommended: < 2.0)"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "[PASS] numpy: $NUMPY_VERSION"
    fi
fi

# torch (optional)
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
if [ "$TORCH_VERSION" = "not installed" ]; then
    echo "[INFO] torch: not installed (optional)"
else
    echo "[PASS] torch: $TORCH_VERSION"
fi

# torch_npu (optional)
TORCH_NPU=$(python3 -c "import torch_npu; print('available')" 2>/dev/null || echo "not available")
if [ "$TORCH_NPU" = "not available" ]; then
    echo "[INFO] torch_npu: not available (optional, for NPU tensors)"
else
    echo "[PASS] torch_npu: available"
fi

# CANN
CANN_PATH="${ASCEND_HOME_PATH:-not set}"
if [ "$CANN_PATH" = "not set" ]; then
    echo "[WARN] ASCEND_HOME_PATH not set"
    echo "       Run: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    WARNINGS=$((WARNINGS + 1))
elif [ -d "$CANN_PATH" ]; then
    echo "[PASS] CANN: $CANN_PATH"
else
    echo "[FAIL] CANN path does not exist: $CANN_PATH"
    ERRORS=$((ERRORS + 1))
fi

# NPU (optional)
NPU_STATUS=$(npu-smi info 2>/dev/null | head -1 || echo "not available")
if [ "$NPU_STATUS" = "not available" ]; then
    echo "[INFO] NPU: not available (Model backend still usable)"
else
    echo "[PASS] NPU: $NPU_STATUS"
fi

# Summary
echo ""
echo "=== Summary ==="
echo "Errors:   $ERRORS"
echo "Warnings: $WARNINGS"
if [ $ERRORS -gt 0 ]; then
    echo "[FAIL] Fix $ERRORS error(s) before proceeding"
    exit 1
else
    echo "[PASS] Environment ready"
fi
