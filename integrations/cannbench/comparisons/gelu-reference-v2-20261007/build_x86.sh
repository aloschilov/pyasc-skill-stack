#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd "$(dirname "$0")" && pwd)
test -f "$task_dir/llvm-x86-complete.tar.gz"
mkdir -p "$task_dir/llvm-x86" "$task_dir/x86-build" "$task_dir/x86-wheels"
if [[ ! -f "$task_dir/llvm-x86/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
  tar --no-same-owner -xzf "$task_dir/llvm-x86-complete.tar.gz" --strip-components=1 -C "$task_dir/llvm-x86"
fi
if [[ ! -d "$task_dir/x86-source" ]]; then
  git clone --local --no-hardlinks /home/aloschilov/workspace/pyasc-gelu-reference-20261007 "$task_dir/x86-source"
fi
test "$(git -C "$task_dir/x86-source" rev-parse HEAD)" = 9069108e323746187c48d78fec4929c4afa2efc4
docker run --rm --platform linux/amd64 --name pyasc-gelu-reference-9069108-build \
  --user "$(id -u):$(id -g)" \
  -v "$task_dir/x86-source:/src/pyasc" -v "$task_dir/x86-build:/build" \
  -v "$task_dir/x86-wheels:/out" -v "$task_dir/llvm-x86:/llvm:ro" \
  -e LLVM_INSTALL_PREFIX=/llvm -e PYASC_SETUP_EXPERIMENTAL=1 \
  -e PYASC_SETUP_JOBS=6 -e PYASC_SETUP_BUILD_DIR=/build \
  -e PYASC_SETUP_CLANG_LLD=1 -e PYASC_SETUP_CMAKE_APPEND=-DCMAKE_CXX_FLAGS_RELEASE=-O1 \
  -w /src/pyasc sha256:9fdec1126a3d95148c3c5c2cbe643dac232690be4230cc06c7bc3867eba0c40d \
  sh -c 'python -m venv /build/venv && /build/venv/bin/pip install "setuptools<81" wheel packaging pybind11==2.13.6 setuptools_scm && /build/venv/bin/python setup.py bdist_wheel --dist-dir /out'
