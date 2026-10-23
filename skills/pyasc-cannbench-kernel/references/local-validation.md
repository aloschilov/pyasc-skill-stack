# Local validation ladder

## Exact-v2 compile gate

The CANNBench campaign requires a source/runtime pair at
`0a631f70968c3cb7c33ce45330a85768dd5a6f06` (the campaign runtime authority).
Local compilation and CaModel are available, but confirm the selected
interpreter/image, imported module paths, and build provenance before counting
any result. The native exact-pin runtime is built; the x86 build must be
qualified separately before its results count. Historical runs at other pins —
including the GeLU campaign pin `adadd7d66ed0ee16d33d79487bf584899a26ef1e` —
stay evidence for those pins and are not CANNBench authority.

Workers use only already provisioned commands within their assigned time/tool
budget. Do not install packages, build a compiler/image, or launch an unbounded
simulator suite unless asked. A worker without the configured runtime performs
syntax/static checks and reports pending validation to the coordinator.

The existing wrapper below is a historical integration entrypoint. Inspect its
image pin and whether it would auto-build an absent image before using it;
an older image cannot qualify the new campaign. From the repository root:

```bash
integrations/cannbench/workers/run_local_compile_gate.sh \
  --candidate path/inside/repository/candidate.py --op <operator>
```

The CANNBench compile gate uses the self-contained CPython 3.12
x86_64 wheel built from `compiler-team/pyasc` branch `v2`, commit
`0a631f70968c3cb7c33ce45330a85768dd5a6f06` (the campaign runtime authority). It replays all 20 case shapes,
dtypes, and attrs, captures host-selected JIT launches, and runs codegen,
compiler passes, AscendC translation, and the 950PR UB check for every unique
specialization.

The gate does not execute the generated kernel. It cannot detect wrong math,
NaN/Inf-position differences, NPU runtime failures, or bad performance.

The gate replays the host dispatch path with shape/dtype-only tensor stubs.
Host code that reads tensor *values* — `.item()`, `int(t[0])`, branching on
data — raises `AttributeError: 'FakeTensor' object has no attribute 'item'`
and every case is marked `dispatch: failed` even when all kernel
specializations compile. Classify this signature as a host/gate-contract
defect, not a kernel defect: the candidate must derive host logic from
shapes/metadata or guard genuinely data-dependent reads with a fallback to a
shape-derived bound (measured on pinned-runtime nms, 0/20 with 6/6
specializations passed). The guard must catch only the replay signature
(`AttributeError`), must be unreachable on the real evaluator, and may only
supply shape metadata — never numerical values. A dispatch pass under the
fallback is not evidence for the data-dependent path; CaModel numerical
qualification remains required.

The historical report also records legalized kernel argument kinds: its
low-level C310 experiments required `has_ffts_arg=false` because that base
compiler otherwise added an FFTS argument failing through `c2c_ctrl_addr()`.
This does not authorize low-level kernels or an ABI/compiler adapter in the
high-level campaign. Use the public `@asctile.jit` path.

## Decorator and JIT matrix qualification

Establish bare `@asctile.jit` and `@asctile.jit(reuse_alloc=1)` baselines.
Then plan `reuse_alloc={0,1,2}` × `static_alloc={None,False,True}` ×
`vf_fusion={False,True}`, always with synchronization ON (`insert_sync=True`).
Follow the [JIT diagnostics](../../pyasc-api-patterns/references/api-jit-options.md)
for fresh processes, both import orders, separate task-owned artifact caches,
and requested/effective options. A gate that directly constructs compiler
options does not prove the real decorator or cache path works. Do not patch
option discovery, use a concrete-options adapter/custom JIT, or invent
`run_asctile_passes`.

The pinned base-first reproduction calls
`asc.runtime.jit.JITFunction.get_config_keywords()` before importing AscTile
and fails decorator construction with unknown `reuse_alloc`. Cold AscTile-first
native CaModel probes pass. This is an inherited discovery-cache issue;
extraction uses `self.compiler.options_cls`. Keep that narrow distinction and
the linked diagnostic case scope when recording results.

For each bounded experiment retain source/pin/options, geometry, UB usage,
compile status, numerical scope, and hardware timings if available. Reuse and
VF fusion may fit UB while changing correctness, synchronization, and runtime
stability. Record failures and untested matrix cells; keep selected-candidate
case coverage complete even when a worker only tests a subset locally.

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

Use the configured native runtime for representative smoke tests. The following
setup notes are historical provisioning evidence, not worker instructions to
install or rebuild: the AArch64 build used `LLVM_INSTALL_PREFIX` pointing at a
local LLVM install (set it to your own path; there is no standard location).
The CMake
graph has an observed generated-header race: if a parallel clean build misses
`AscendCDialect.h.inc`, run once with `PYASC_SETUP_JOBS=1`, then continue the
same build directory in parallel. GCC `-O3` may use roughly 12 GB RSS and about
15 minutes for `Translation.cpp`.

In an already provisioned native environment with matching pyasc and torch,
the smoke entrypoint is:

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
