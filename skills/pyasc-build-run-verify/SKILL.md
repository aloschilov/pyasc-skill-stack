---
name: pyasc-build-run-verify
description: Build, compile, execute, and verify current-v2 asctile kernels on exact-source QEMU gates, camodel, NPU, and CANNBench.
---

# Build, run, and verify current pyasc v2

Always keep source, wheel, Python ABI, compiler, and platform provenance
together. For the current CANNBench campaign use
`pyasc v2@0a631f70968c3cb7c33ce45330a85768dd5a6f06`, CPython 3.12 for
the submitted x86_64 wheel, and `Ascend950PR_9599`.

## Validation ladder

1. **Syntax/static:** parse and `python3 -m py_compile`; enforce callable,
   anti-cheat, API, and host/JIT-boundary rules.
2. **Exact-v2 compile:** replay every case dispatch and lower every unique JIT
   specialization through pyasc/AscendC, including memory-budget checks and
   legalized kernel-argument kinds.
3. **Camodel:** execute ordinary and adversarial numerical routes against the
   golden with task tolerances.
4. **Real NPU/CANNBench:** authoritative correctness, profiler time, score, and
   anti-cheat result.

Never claim numerics or performance from a QEMU compile-only gate.

## CANNBench compile gate

From this repository root:

```bash
integrations/cannbench/workers/run_local_compile_gate.sh \
  --candidate path/inside/repository/candidate.py --op <operator>

python3 integrations/cannbench/workers/run_local_matrix.py \
  --candidate-root <bundle>/cann_bench \
  --output-dir <evidence-dir>
```

The matrix must cover all current operator task directories, not a hard-coded
legacy nine-operator list.

## Native camodel setup

Use a wheel built from the selected commit. On this machine the local LLVM
reference is `/home/aloschilov/workspace/llvm`. A typical environment is:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYASC_COMPILER=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH
SIM=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599
export LD_LIBRARY_PATH=$SIM/camodel:$SIM/lib:$SIM/lib64:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
```

Run target tests through pytest so fixtures configure backend/platform:

```bash
python3 -m pytest python/test/asctile/target/test_<op>.py \
  --compile-only --backend Model --platform Ascend950PR_9599 -q
python3 -m pytest python/test/asctile/target/test_<op>.py \
  --backend Model --platform Ascend950PR_9599 -q
```

For CANNBench modules use `run_camodel_smoke.py` and state its suite/operator
scope. `critical` or `adversarial` is `verified-camodel-smoke`, not a full
20-case pass unless every contract route was executed.

## JIT diagnostics

- Set `PYASC_DUMP_PATH` to inspect generated IR and AscendC.
- Use `always_compile=True` only while debugging cache effects.
- Lower `opt_level` when a clearer compiler failure is useful.
- `insert_sync=True` is a current AscTile default.
- Valid allocation modes are `reuse_alloc=0/1/2`; test the mode used in the
  submitted candidate. At `0a631f70`, use a concrete-options adapter or
  upstream repair before assuming AscTile-specific decorator options took
  effect.
- Do not use the removed `run_asc2_passes` option.

Common failures:

- rank mismatch: `global_tensor`, `offsets`, physical shape, and `real_shape`
  do not have the same rank
- alignment error: final physical transfer dimension is not 32-byte aligned
- UB/L1/L0 overflow: tile or unroll factor creates too many live buffers
- generated C++ declaration conflict: f16/bf16 temporaries share a scalar name
- dispatch passes but compile fails: host fallback needs to observe the same
  compile-time exception and select a supported specialization
- low-level C310 specialization contains `FftsAddr`: the base compiler will
  call `c2c_ctrl_addr()` and fail on CANNBench 950PR before launch
- large VF/reuse loop times out on NPU despite fitting UB: classify it as a
  runtime/compiler failure and fall back to a bounded non-fused route

## Numerical verification

Compare the full logical output, shape, dtype, and list arity. Use the task's
atol/rtol, compare NaN/Inf positions, exercise runtime scalars and both sides
of attribute dispatch, and inspect simulator diagnostics for invalid padded
lane arithmetic. A finite random smoke is not enough for exp/GELU/norm/softmax.

## Remote evidence

For every CANNBench result record submission ID, job ID, immutable URL,
operator/case counts, passed cases, score/geomean, anti-cheat failures, exact
wheel hash, and pyasc commit. A queued job is not a completed run.

## References

- [JIT diagnostics](references/jit-diagnostics.md)
- [Verification patterns](references/verification-patterns.md)
- `pyasc-cannbench-kernel/references/local-validation.md`
