# pyasc Environment Requirements

## Required

| Component | Version | Install |
|-----------|---------|---------|
| Python | 3.9-3.12 | System package manager |
| pyasc | Latest | `pip install pyasc` or build from source |
| numpy | < 2.0 | `pip install "numpy<2"` |
| CANN Toolkit | 8.0.0+ | Huawei Ascend installer |

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
2. `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
3. `pip install pyasc "numpy<2"`
4. (Optional) `pip install torch torch_npu`
5. Verify: `python3 -c "import asc; print('OK')"`

## Model backend setup

For simulator (no NPU hardware):
- CANN Toolkit must be installed
- Simulator libraries must be in `LD_LIBRARY_PATH`
- Some environments require `LD_PRELOAD` for camodel
