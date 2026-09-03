# Vendored pyasc v2 runtime

The submission evaluates `compiler-team/pyasc`, branch `v2`, at the immutable
commit `ac1222a48c8914d3f81297c7570d1a84f0f26778` (2026-08-31).

- Source: <https://gitcode.com/compiler-team/pyasc/tree/ac1222a48c8914d3f81297c7570d1a84f0f26778>
- Runtime wheel: `runtime-wheels/pyasc-1.1.1-cp312-cp312-linux_x86_64.whl`
- Wheel SHA-256: `27e675fe2446b9ff8b2841688218c2d6b3ace877a3c4be491d0cba68af4dfcdd`
- Runtime build helper: `runtime-wheels/pybind11-2.13.6-py3-none-any.whl`
- pybind11 SHA-256: `237c41e29157b962835d356b370ededd57594a26d5894a795960f0047cb5caf5`
- Build ABI: CPython 3.12 / x86_64, matching the CANNBench evaluator.
- Reproducible build inputs: `../../Dockerfile.cp312-x86` and
  `../../build-runtime-qemu.sh`. The build uses LLVM 20 commit
  `86b69c31642e98f8357df62c09d118ad1da4e16a`
  (`llvm-86b69c31-ubuntu-x64`) and does not use a Python package repository for
  pyasc.
- Transport encoding: the vendored runtime wheel is stripped and uses
  ZIP-LZMA to keep private submission uploads bounded. `merge_wheels.py`
  extracts it and creates a regular final wheel, so this internal encoding is
  not exposed to the evaluator's Python installer.
- License: CANN Open Software License Agreement Version 2.0, retained as
  `PYASC-LICENSE` and embedded in the final wheel.

The final CANNBench wheel has no external Python package dependency on pyasc
or pybind11 and contains `asc`, `asc2`, `asc._C.libpyasc`, and the pybind11
headers/runtime helper needed by pyasc's host-binding compiler.
