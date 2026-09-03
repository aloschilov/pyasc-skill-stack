# Local validation ladder

## Exact-v2 compile gate

Run from the repository root:

```bash
integrations/cannbench/workers/run_local_compile_gate.sh \
  --candidate path/inside/repository/candidate.py --op <operator>
```

The current integration installs the self-contained CPython 3.12
x86_64 wheel built from `compiler-team/pyasc` branch `v2`, commit
`0a631f70968c3cb7c33ce45330a85768dd5a6f06`. It replays all 20 case shapes,
dtypes, and attrs, captures host-selected JIT launches, and runs codegen,
compiler passes, AscendC translation, and the 950PR UB check for every unique
specialization.

The gate does not execute the generated kernel. It cannot detect wrong math,
NaN/Inf-position differences, NPU runtime failures, or bad performance.

The report also records legalized kernel argument kinds. A low-level C310
kernel is not submission-ready unless every specialization has
`has_ffts_arg=false`; the base compiler at this commit otherwise adds an FFTS
argument that fails through `c2c_ctrl_addr()` on CANNBench 950PR.

## Full matrix

```bash
python3 integrations/cannbench/workers/run_local_matrix.py \
  --candidate-root <directory-containing-current-modules> \
  --output-dir <evidence-directory>
```

For the current 11-task checkout, pass criteria are 11/11 operators, 220/220
host dispatches, and 220/220 compile/lowering routes. Derive this count from
the task directories rather than embedding it in tooling. A controlled host
fallback may leave a rejected
specialization in the report while all case routes still pass; retain that
failure as evidence rather than hiding it.

## Numerical and remote gates

- Camodel: compare outputs against each task's `golden.py`, preserving its
  dtype, tolerance, special-value, shape, and list-output rules. Mark large or
  infeasible cases compile-only, not numerically passed.
- CANNBench: the only authoritative source for real-NPU correctness and
  profiler speed. Remote evaluation is opt-in and consumes a credit.

For a representative smoke on this AArch64 host, build the vendored exact-v2
source with `LLVM_INSTALL_PREFIX=/home/aloschilov/workspace/llvm`. The CMake
graph has an observed generated-header race: if a parallel clean build misses
`AscendCDialect.h.inc`, run once with `PYASC_SETUP_JOBS=1`, then continue the
same build directory in parallel. GCC `-O3` may use roughly 12 GB RSS and about
15 minutes for `Translation.cpp`.

After installing the resulting native wheel into an isolated Python 3.10
environment with torch, run:

```bash
python3.10 integrations/cannbench/workers/run_camodel_smoke.py \
  --candidate-root <generated-bundle>/cann_bench \
  --output <evidence.json>
```

Use `--suite critical` for selected dtype/control-flow branches and split
`--ops` into short groups when the simulator's process time budget is limited.
Use `--suite adversarial --ops gelu,foreach_addcdiv_scalar` before promoting
either of those operators: it covers both GELU modes, wide negative ranges,
BF16, runtime scalars, and NaN/Inf positions with task-derived tolerances.
Inspect simulator diagnostics as well as output comparisons: padded lanes may
execute invalid arithmetic even though `real_shape` prevents them being stored.

This smoke is `verified-camodel-smoke`, not `verified-camodel`, unless its
declared routes cover the complete benchmark task.
