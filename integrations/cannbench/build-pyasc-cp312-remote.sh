#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOME="${REMOTE_HOME:-/home/l00958488}"
PYTHON_VERSION="3.12.13"
PYTHON_SHA256="c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684"
PYTHON_PREFIX="${REMOTE_HOME}/python312-cannbench"
BUILD_ROOT="${REMOTE_HOME}/python312-cannbench-build"
PYASC_SOURCE="${REMOTE_HOME}/pyasc-v2-cannbench"
PYASC_BUILD="${REMOTE_HOME}/pyasc-v2-cp312-build"
PYASC_DIST="${REMOTE_HOME}/pyasc-v2-cp312-dist"

mkdir -p "${BUILD_ROOT}"
if [[ ! -x "${PYTHON_PREFIX}/bin/python3.12" ]]; then
  archive="${BUILD_ROOT}/Python-${PYTHON_VERSION}.tar.xz"
  if [[ ! -f "${archive}" ]]; then
    curl --insecure --fail --location --retry 4 \
      --output "${archive}.part" \
      "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz"
    mv "${archive}.part" "${archive}"
  fi
  echo "${PYTHON_SHA256}  ${archive}" | sha256sum --check --strict
  rm -rf "${BUILD_ROOT}/Python-${PYTHON_VERSION}"
  tar -xJf "${archive}" -C "${BUILD_ROOT}"
  (
    cd "${BUILD_ROOT}/Python-${PYTHON_VERSION}"
    ./configure \
      --prefix="${PYTHON_PREFIX}" \
      --enable-shared \
      --with-ensurepip=install \
      LDFLAGS="-Wl,-rpath,${PYTHON_PREFIX}/lib"
    make -j16
    make install
  )
fi

PYTHON_BIN="${PYTHON_PREFIX}/bin/python3.12"
test "$(${PYTHON_BIN} -c 'import sys; print(sys.version_info[:2])')" = "(3, 12)"
test "$(git -C "${PYASC_SOURCE}" rev-parse HEAD)" = \
  "ac1222a48c8914d3f81297c7570d1a84f0f26778"

# Python 3.12 no longer ships distutils. Seed its site-packages with the
# build host's already-installed pure-Python packaging toolchain, including
# setuptools' maintained distutils compatibility layer.
PYTHON_SITE="$(${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
for package in \
  setuptools setuptools-*.dist-info _distutils_hack distutils-precedence.pth \
  wheel wheel-*.dist-info packaging packaging-*.dist-info \
  pybind11 pybind11-*.dist-info setuptools_scm setuptools_scm-*.dist-info; do
  for source_path in /usr/local/lib/python3.10/site-packages/${package}; do
    [[ -e "${source_path}" ]] || continue
    cp -a "${source_path}" "${PYTHON_SITE}/"
  done
done
"${PYTHON_BIN}" -c 'import distutils, packaging, pybind11, setuptools, setuptools_scm, wheel'

rm -rf "${PYASC_BUILD}" "${PYASC_DIST}"
mkdir -p "${PYASC_BUILD}" "${PYASC_DIST}"

export PYASC_SETUP_JOBS=16
export PYASC_SETUP_VERSION=1.1.1
export PYASC_SETUP_VERSION_SUFFIX=.cannbench.v2.ac1222a
export PYASC_SETUP_BUILD_DIR="${PYASC_BUILD}"
export PYASC_SETUP_CLANG_LLD=ON
export PYASC_SETUP_CMAKE_APPEND=-DCMAKE_CXX_FLAGS_RELEASE=-O1

cd "${PYASC_SOURCE}"
"${PYTHON_BIN}" setup.py bdist_wheel --dist-dir "${PYASC_DIST}"
wheel_path="$(find "${PYASC_DIST}" -maxdepth 1 -type f -name 'pyasc-*.whl' -print -quit)"
test -n "${wheel_path}"
"${PYTHON_BIN}" - "${wheel_path}" <<'PY'
import sys, zipfile
wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as zf:
    names = zf.namelist()
    assert any(n.startswith("asc/_C/libpyasc") and n.endswith(".so") for n in names)
print(wheel)
PY
