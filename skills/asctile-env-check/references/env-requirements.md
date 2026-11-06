# PyASC/AscTile environment requirements

## Required

| Component | Version | Install |
|-----------|---------|---------|
| Python | Match the selected wheel ABI; prefer 3.12 for a new setup | System package manager |
| PyASC with AscTile | Select and record a `pyasc-dev` version and wheel hash | `pip install --pre pyasc-dev` for exploration; exact version for qualification |
| numpy | < 2.0 | pulled in by `pyasc-dev` |
| CANN Toolkit | 9.2.0 for a new setup; retain a qualification's exact version | Huawei Ascend installer |

Choose a published wheel for the actual Python/architecture/glibc combination;
availability is per version. The maintained public [wheel recipe](https://github.com/compiler-team-ru/asctile)
uses Linux manylinux_2_28 (glibc >= 2.28) for x86_64 and aarch64. Check the
selected release's target status; a profile or source-build flag does not
establish compatibility. No macOS/Windows wheel is implied.

## Optional

| Component | Purpose | Install |
|-----------|---------|---------|
| torch | Host tensor management | `pip install torch` |
| torch_npu | NPU tensor support | Huawei PyTorch adapter |
| NPU hardware | NPU backend execution | Atlas A2/A3 |

## Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ASCEND_HOME_PATH` | Yes | - | CANN Toolkit installation path |
| `LD_LIBRARY_PATH` | Yes (runtime) | - | Must include CANN libs |
| `PYASC_HOME` | No | `$HOME` | JIT cache root |
| `PYASC_CACHE_DIR` | No | `$PYASC_HOME/.pyasc/cache` | JIT cache directory |
| `PYASC_DUMP_PATH` | No | - | Dump generated IR/code |

## Setup steps

1. Install CANN Toolkit
2. Source the installed toolkit environment, for example
   `source /usr/local/Ascend/cann-9.2.0/set_env.sh`; adjust to its actual path.
3. `python3.12 -m venv .venv && source .venv/bin/activate && pip install --pre pyasc-dev`
4. (Optional) `pip install torch torch_npu`
5. Verify: `python3 -c "from asc.experimental import asctile; print(asctile.__file__)"`
   (legacy source builds expose it as a top-level `import asctile` instead)

## Model backend setup

For simulator (no NPU hardware):
- CANN Toolkit must be installed
- Simulator libraries must be in `LD_LIBRARY_PATH`
- Some environments require `LD_PRELOAD` for camodel
