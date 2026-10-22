#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
PYASC_COMMIT="030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d"
PYBIND11_WHEEL_SHA256="237c41e29157b962835d356b370ededd57594a26d5894a795960f0047cb5caf5"
BASE_DIST_DIR="${SCRIPT_DIR}/.base-dist"
RUNTIME_SHA256="${SCRIPT_DIR}/vendor/runtime-wheels/pyasc.sha256.txt"
PYBIND11_WHEEL="${SCRIPT_DIR}/vendor/runtime-wheels/pybind11-2.13.6-py3-none-any.whl"

# Validate authored sources before changing build outputs. Worker staging records
# the newly promoted operator; a direct package build checks every operator.
source_contract_args=(--package "${SCRIPT_DIR}/cann_bench")
promotion_args=()
if [[ -f "${SCRIPT_DIR}/source-contract-promotions.json" ]]; then
  promotion_args=(--promotions "${SCRIPT_DIR}/source-contract-promotions.json")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/source_contract.py" "${source_contract_args[@]}" "${promotion_args[@]}"

mapfile -t runtime_wheels < <(
  find "${SCRIPT_DIR}/vendor/runtime-wheels" -maxdepth 1 -type f \
    -name 'pyasc-*-cp312-cp312-linux_x86_64.whl' -print | sort
)
test "${#runtime_wheels[@]}" -eq 1
RUNTIME_WHEEL="${runtime_wheels[0]}"

cd "${SCRIPT_DIR}"
rm -rf "${BASE_DIST_DIR}" build dist cann_bench.egg-info
mkdir -p "${BASE_DIST_DIR}" dist

echo "[cann-bench] using vendored pyasc v2 runtime at ${PYASC_COMMIT}"
# The submission shell is pure Python.  The final CPython ABI is supplied by
# the pinned x86_64 runtime wheel and checked again by verify_wheel.py.  This
# repack step is intentionally cross-architecture and performs no native link.
test -f "${RUNTIME_WHEEL}"
test -f "${RUNTIME_SHA256}"
(cd "$(dirname "${RUNTIME_WHEEL}")" && sha256sum --check "$(basename "${RUNTIME_SHA256}")")
test -f "${PYBIND11_WHEEL}"
test "$(sha256sum "${PYBIND11_WHEEL}" | awk '{print $1}')" = "${PYBIND11_WHEEL_SHA256}"

echo "[cann-bench] building submission shell wheel"
"${PYTHON_BIN}" setup.py bdist_wheel --dist-dir "${BASE_DIST_DIR}"
base_wheel="$(find "${BASE_DIST_DIR}" -maxdepth 1 -type f -name 'cann_bench-*.whl' -print -quit)"
test -n "${base_wheel}"

"${PYTHON_BIN}" merge_wheels.py \
  "${promotion_args[@]}" \
  --base-wheel "${base_wheel}" \
  --runtime-wheel "${RUNTIME_WHEEL}" \
  --pybind11-wheel "${PYBIND11_WHEEL}" \
  --license "${SCRIPT_DIR}/vendor/PYASC-LICENSE" \
  --output-dir "${SCRIPT_DIR}/dist"

final_wheel="$(find dist -maxdepth 1 -type f -name 'cann_bench-*.whl' -print -quit)"
test -n "${final_wheel}"
"${PYTHON_BIN}" verify_wheel.py "${final_wheel}" "${PYASC_COMMIT}"
echo "[cann-bench] self-contained wheel: ${final_wheel}"
