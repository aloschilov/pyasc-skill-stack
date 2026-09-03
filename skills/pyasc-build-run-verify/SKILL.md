---
name: pyasc-build-run-verify
description: pyasc asc2 kernel build, run, and verification skill. Provides JIT compilation diagnostics, runtime execution guidance, and output verification patterns. Trigger — after kernel implementation, when running pyasc kernels, debugging JIT errors, or verifying kernel output correctness.
---

# pyasc Build, Run, and Verify (asc2)

> Current-v2 CANNBench override: commit `030e9b2c` packages `asctile` instead
> of `asc2`. The campaign-specific wheel, compile gate, and candidates must all
> use `asctile`; the `asc2` commands below describe older snapshots only.

## Runtime profiles

- **Standalone simulator/golden:** use the Python/toolkit instructions below
  and numpy or the operator-specific torch reference.
- **CANNBench pinned v2:** load `pyasc-cannbench-kernel` and run
  `integrations/cannbench/workers/run_local_compile_gate.sh`. That QEMU gate
  uses CPython 3.12 and the exact vendored self-contained wheel; the host's
  default `import asc2` is not proof that the candidate targets the submitted
  runtime. Its result is compile/lowering evidence only.
- **Real NPU:** only an official CANNBench run establishes benchmark accuracy
  and performance. Never relabel a compile-only result as a numerical pass.

The `python3.10` and numpy-only directions below apply to the standalone
simulator profile, not to CANNBench modules whose public interface is torch.

## Overview

pyasc uses JIT (Just-In-Time) compilation: Python -> ASC-IR -> Ascend C -> Bisheng compiler -> NPU binary. The asc2 API simplifies this by handling synchronization and memory management automatically.

## Workflow

```
Kernel implementation complete
    |
    +-- JIT compilation (automatic on first call)
    |       |
    |       +-- Success -> Run kernel
    |       |
    |       +-- Failure -> Check diagnostics
    |
    +-- Run kernel
    |       |
    |       +-- Model backend (simulator, always available)
    |       |
    |       +-- NPU backend (requires hardware)
    |
    +-- Verify output
            |
            +-- np.testing.assert_allclose
```

## Running a pyasc asc2 kernel

### Basic execution

> **IMPORTANT**: Use `python3.10` (not `python` or `python3`). The pyasc packages are installed under python3.10.

```bash
# Set up simulator environment (required for Model backend). The toolkit often lives at
# /usr/local/Ascend/cann-9.0.0 (NOT the ~/Ascend path in .env.pyasc.example).
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYASC_COMPILER=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng
# bisheng ships its OWN ld.lld that knows `-m aicorelinux`; put its bin FIRST on PATH or the
# system /usr/bin/ld.lld is picked and link fails with `unknown emulation: aicorelinux`.
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH
# Put the simulator `camodel` dir BEFORE `lib`: both ship libstars.so but with different symbol
# case (`STARS_TOP` in camodel vs `stars_top` in lib); wrong order => `undefined symbol:
# _ZN9STARS_TOP27ext_write_ffts_plus_contextEjjPv` or a GetC2cCtrlAddrWrapper hang at run.
SIM=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599
export LD_LIBRARY_PATH=$SIM/camodel:$SIM/lib:$SIM/lib64:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH

# Run with Model backend (simulator) — specify platform explicitly
python3.10 kernel.py -r Model -v Ascend950PR_9599

# Run with NPU backend (requires hardware)
python3.10 kernel.py -r NPU -v Ascend950PR_9599
```

> The `-v Ascend950PR_9599` flag matches the only platform the stack targets. Do NOT use `-v Ascend950PR` (missing version suffix).

> **Run pytest target files through `pytest`, not raw `python`.** Test files under
> `python/test/asc2/target/` rely on the `conftest.py` `set_platform` fixture to initialise the
> backend/platform; invoking them with raw `python3` skips that setup and the simulator fails with
> `GetC2cCtrlAddrWrapper returned ...`. Only standalone golden kernels (with their own `__main__`)
> run via `python3.10 kernel.py`.

> **Tile-size budgets: align with CANN.** The source of truth for UB tile sizing is the CANN op
> tiling (`ops-math/.../op_host/arch35/*_tiling_arch35.cpp`). For concat that is
> `maxAvaliableUb = (UB_CAPACITY − INDEX_USE_UB[=1024]) / dtypeSize`, divided by `BUFFER_NUM[=2]`
> on the non-aligned double-buffered path — prefer these over ad-hoc byte constants.

### Running via pytest

```bash
pytest kernel.py --backend Model --platform Ascend950PR_9599
```

This requires a `conftest.py` with `backend` and `platform` fixtures (see the kernel template).

### Script tool

```bash
bash scripts/run_kernel.sh {kernel_path} [backend] [platform]
```

## JIT Diagnostics

### Environment variables for debugging

| Variable | Purpose | Example |
|----------|---------|---------|
| `PYASC_DUMP_PATH` | Save generated ASC-IR and Ascend C files | `export PYASC_DUMP_PATH=/tmp/pyasc_dump` |
| `PYASC_HOME` | JIT cache root directory | `export PYASC_HOME=$HOME` |
| `PYASC_CACHE_DIR` | Specific cache directory | `export PYASC_CACHE_DIR=$HOME/.pyasc/cache` |

### Compile options for debugging

| Option | Purpose | Usage |
|--------|---------|-------|
| `always_compile=True` | Force recompilation (bypass cache) | `@asc2.jit(always_compile=True)` — **standard for development** |
| `opt_level=0` | Disable optimizations for debugging | `@asc2.jit(opt_level=0)` |

Note: `insert_sync=True` and `run_asc2_passes=True` are defaults for `@asc2.jit` — do not disable them unless debugging.

### Common JIT errors

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `SyntaxError` in AST visitor | Unsupported Python syntax | Check `pyasc-syntax-constraints` |
| Type error in IR builder | Wrong parameter type | Check type constraints for kernel params |
| Bisheng compilation error | Invalid generated Ascend C | Check `PYASC_DUMP_PATH` output for generated code |
| `ImportError: asc` or `asc2` | pyasc not installed | Run `pip install pyasc` or build from source |
| `RuntimeError` on launch | Wrong core count | Verify `CORE_NUM` value |
| Used `range()` instead of `asc2.range()` | Wrong loop construct inside kernel | Replace with `asc2.range()` |
| `RuntimeError: UB overflow: N bytes are available, M bytes are used` (also `L1`/`L0A`/`L0B`/`L0C`) | `Launcher.check_memory_overflow` rejects the kernel **before** any numerics: the sum of live on-chip buffers exceeds capacity. Common cause: a tile sized to a full row, one buffer per input, and/or `unroll_factor` × `parallel` double-buffering multiplying the live set. Note `M` is often a clean multiple of the per-buffer size (e.g. `786432 = 8 × 98304`) — the multiplier tells you how many buffers are live | Shrink the live set: tile/column-chunk wide data through **one reused buffer**; lower `unroll_factor`; pipeline only the loop whose buffer you can halve. Size the chunk for the worst case `arity × unroll_factor × chunk_bytes ≤ capacity`. See `pyasc-api-patterns` → "Multi-input / multi-axis copy". The **static-dims** variant is the worst case (sibling loops not reused) — always compile-check it |
| `RuntimeError: Compiler executable is not found, check PYASC_COMPILER environment variable` | `bisheng` is not on `PATH` — the CANN env was not sourced (toolkit may live at `/usr/local/Ascend/cann-9.0.0`, not the `.env` example path) | `source /usr/local/Ascend/cann-9.0.0/set_env.sh` (or set `PYASC_COMPILER`) so `bisheng` resolves; `--compile-only` still needs it |

## Verification Patterns

### numpy verification (REQUIRED for asc2)

> **CRITICAL**: Always use **numpy** for host-side verification and data preparation.
> Do NOT use `scipy`, `torch`, or any other library — they are not available in
> the simulator Docker environment and will cause runtime verification failures.
> For reference implementations (e.g., softmax, gelu), write a pure numpy version.

> **CRITICAL**: numpy's `default_rng().random()` does NOT support `dtype=np.float16`.
> Always generate test data as `float32`, then cast:
> `x = (rng.random(shape, dtype=np.float32) * 10 - 5).astype(np.float16)`

```python
import numpy as np
rng = np.random.default_rng(seed=2026)
x = (rng.random(size, dtype=np.float32) * 10 - 5).astype(np.float16)
result = kernel_launch(x)
expected = np.abs(x)
np.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-3)
```

For operations that need `erf`, use `math.erf` with `np.vectorize`:

```python
import math
_verf = np.vectorize(math.erf)
expected = 0.5 * x * (1.0 + _verf(x / np.sqrt(2.0)))
```

For softmax, implement a pure numpy reference:

```python
def softmax_numpy(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)
```

### Verification script

```bash
python scripts/verify_output.py {kernel_path} [--backend Model] [--atol 1e-5]
```

## Backend selection

| Backend | When to use | Availability |
|---------|-------------|-------------|
| `Model` | Development, CI, no NPU hardware | Always (requires CANN simulator libs) |
| `NPU` | Final verification, performance testing | Requires Atlas A2/A3 hardware |

**If runtime execution is unavailable**: Perform static verification (syntax check, ASC-IR dump inspection) and state the limitation explicitly in the delivery.

### Simulator throughput — pick runnable shapes; compile-only the rest

The Model simulator is cycle-accurate and **~1000× slower than NPU**, and cost
scales with the number of *kernel iterations*, not just element count. A kernel
whose 2-D reduction yields a huge **row/iteration count** (e.g. a concat that
reduces to ~160K rows) can take many minutes *per launch* even when total bytes
are modest — a small handful of such launches will blow past any reasonable
wait. This is a **simulator-throughput** limit, not a kernel bug.

Practical policy when validating a large case matrix:

- **Compile-only** every case to prove it lowers and fits UB (no simulator):

  ```bash
  pytest --compile-only python/test/asc2/target/test_<op>.py -q   # needs bisheng on PATH
  ```

- **Run on the simulator** only a representative subset with **low iteration
  counts** (few rows / few blocks), including any case that previously failed
  (e.g. a wide-row case you just fixed), and verify numerically vs torch/numpy.
- For high-iteration shapes, state in the delivery that they are **compile-only
  validated** (consistent with how large shapes are handled elsewhere). Do **not**
  drop a shape from coverage just because the simulator is slow.

> A piped `pytest ... | tail -N` shows **no output until the pipe closes**, so a
> long sim run looks "hung". Confirm progress by watching the terminal/output
> file directly, and size your wait to the expected per-launch cost.

### Simulator feasibility: verify only *feasible* shapes, not all of them

When a kernel must cover **many shapes** (e.g. a production case list pulled from
a CANN regression CSV), do **not** push every shape through the Model simulator.
Apply an explicit **feasibility predicate** first and only sim-verify the shapes
that pass; the rest are **compile-only** (they still must lower and fit UB). This
generalises the older per-op notes ("Softmax/RMSNorm: test ONLY the shape in the
prompt — the simulator is extremely slow for large shapes") into one reusable
rule.

A shape `S` for dtype `D` (flatten-1D elementwise / row-tiled reductions) is
**simulator-feasible** when **all** hold:

- **dtype supported by the sim build** — `float32`/`float16` always; `bfloat16`
  lowers on both the x86_64 and aarch64 simulator images (do **not** add false
  arch guards), but confirm the active image supports it.
- **element-count ceiling (sim time budget)** — `numel(S) ≤ ELEM_CEILING`. Sim
  cost scales with *kernel iterations*, so this is a wall-clock guard, not a
  correctness one. A practical default is `ELEM_CEILING ≈ 4,194,304` elements
  for a single flatten-1D pass; lower it for high-iteration 2-D reductions
  (row/iteration count, not bytes, is what hurts — see the concat example above).
- **minimum size** — `numel(S) ≥ 32 / sizeof(D)` (a tile must fill a 32-byte
  cache line).
- **UB tile fits** — a flatten-1D kernel tiles the buffer so a single UB tile
  always fits; for fixed-tile kernels check
  `chunk_elems = ((UB_CAPACITY[=253952] − 1024) / sizeof(D)) / BUFFER_NUM[=2] / sibling_regions`.
- **alignment is NOT a feasibility blocker** — a size that is not a multiple of
  `TILE_SIZE × CORE_NUM` is still feasible *provided the kernel declares a
  cache-line-aligned tail path*; otherwise treat it as `aligned_only` and skip.

```python
UB_CAPACITY, INDEX_RESERVE, CORE_NUM = 253952, 1024, 16
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
ELEM_CEILING = 4_194_304  # tune down for high-iteration 2-D reductions

def sim_feasible(shape, dtype):
    if dtype not in DTYPE_BYTES:
        return False, "dtype unsupported by simulator build"
    n = 1
    for d in shape:
        n *= d
    if n < 32 // DTYPE_BYTES[dtype]:
        return False, "below 32-byte minimum"
    if n > ELEM_CEILING:
        return False, "exceeds sim-time element ceiling"
    return True, "ok"   # UB tile fits (kernel tiles); tail handles non-alignment
```

Policy: **sim-verify** the feasible subset numerically vs the numpy reference;
**compile-only** the infeasible ones and say so in the delivery. Never drop a
shape from coverage just because it is sim-infeasible.

## References

- [JIT Diagnostics Guide](references/jit-diagnostics.md)
- [Verification Patterns](references/verification-patterns.md)
