#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
PYASC_COMMIT="ac1222a48c8914d3f81297c7570d1a84f0f26778"
PYBIND11_WHEEL_SHA256="237c41e29157b962835d356b370ededd57594a26d5894a795960f0047cb5caf5"
BASE_DIST_DIR="${SCRIPT_DIR}/.base-dist"
RUNTIME_SHA256="${SCRIPT_DIR}/vendor/runtime-wheels/pyasc.sha256.txt"
PYBIND11_WHEEL="${SCRIPT_DIR}/vendor/runtime-wheels/pybind11-2.13.6-py3-none-any.whl"

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
test "$(${PYTHON_BIN} -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')" = "cp312"
test "$(uname -m)" = "x86_64"
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
  --base-wheel "${base_wheel}" \
  --runtime-wheel "${RUNTIME_WHEEL}" \
  --pybind11-wheel "${PYBIND11_WHEEL}" \
  --license "${SCRIPT_DIR}/vendor/PYASC-LICENSE" \
  --output-dir "${SCRIPT_DIR}/dist"

final_wheel="$(find dist -maxdepth 1 -type f -name 'cann_bench-*.whl' -print -quit)"
test -n "${final_wheel}"
"${PYTHON_BIN}" verify_wheel.py "${final_wheel}" "${PYASC_COMMIT}"
echo "[cann-bench] self-contained wheel: ${final_wheel}"
