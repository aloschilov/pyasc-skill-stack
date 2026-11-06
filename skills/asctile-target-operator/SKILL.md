---
name: asctile-target-operator
description: Procedure for authoring and verifying a selected-runtime handwritten asctile target operator in the selected upstream target suite. Use for target tests, handwritten reference kernels, simulator verification, and CANN operator parity work.
---

# AscTile target operator workflow

A target operator is a handwritten `@asctile.jit` kernel plus pytest coverage
inside the pyasc repository. Locate the suite using the selected source revision. The repository snapshot
and its upstream path are recorded in `reference/asctile/PIN.json`; historical
`python/test/asctile/target/` paths are not universal.

## Version discipline

Record the exact upstream base and use a separate development checkout.
For the repository runtime, read the upstream URL, commit and `resolved_path`
from `reference/asctile/PIN.json`. Branch from that revision when extending its
tests; do not automatically switch to a moving branch or overwrite another task.

An explicitly selected campaign may use its own source/runtime pair. Preserve
historical results at their original revisions. Build from source only when
the selected revision is not available as a matching runtime artifact.

Use prebuilt local compile/CaModel environments within a worker's assigned
budget. Missing dependencies are a reported limitation; workers do not install
packages or start source builds unless asked.

## Source layout and reference selection

New work is authored in a pyasc checkout. **Confirm the current path before
assuming one** — upstream has carried this suite at `python/test/asc2/target/`,
then `python/test/asctile/target/`, and relocated it under `experimental/`
when AscTile was localised there. Locate it by content rather than memory:

```bash
grep -rl --include='test_*.py' 'asctile\.jit' . | xargs grep -l 'asctile\.global_tensor'
```

- New test: `<suite>/test_<op>.py`, at whatever path the command above reports.
- Shared fixtures: the suite's own `conftest.py` plus the ancestor one that
  defines `require_c310`.
- Start from the closest current sibling: `test_vadd.py` for elementwise,
  `test_reduce_sum.py`/`test_reduce_max.py` for reductions,
  `test_softmax.py` for normalization, and `test_transpose.py` for movement.

### Reading siblings without a checkout

This repository vendors a pinned copy of the suite at
[`reference/asctile/target/`](../../reference/asctile/target/), which is enough
to read the canonical patterns. It is a **snapshot, not the upstream tree**:
check [`reference/asctile/PIN.json`](../../reference/asctile/PIN.json) for which
version it is, and never author a new test there. Refresh it with
`scripts/sync-target-suite.sh`; verify it with `scripts/check-target-pin.sh`.
- Use the matching `gitcode.com/cann/ops-*` implementation as semantic and
  tiling reference. Do not copy its C++ as the pyasc implementation.

Use public `asctile` annotations, dtypes, JIT, tile math, copy operations, and
range. New high-level campaign source must not import low-level `asc`/`asc2`,
author `inline`/`inline_vf` code, or implement a long Numerical Recipes/Horner
polynomial. Compiler-generated register code from `vf_fusion=True` is allowed.

## Canonical elementwise structure

```python
from asc.experimental import asctile
import pytest
import torch

@asctile.jit(reuse_alloc=1)
def op_kernel(x_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress,
              size: int, tile_size: asctile.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for tile_id in asctile.range(asctile.block_idx(),
                                 asctile.ceildiv(size, tile_size),
                                 asctile.block_num(), unroll_factor=2):
        offset = tile_id * tile_size
        valid = min(size - offset, tile_size)
        x = asctile.copy_in(
            x_gm, [offset], [tile_size], real_shape=[valid], pad_value=1
        )
        one = asctile.full([tile_size], 1.0, dtype=x.dtype)
        out = one / x
        asctile.copy_out(out, out_gm, [offset], real_shape=[valid])

def test_op(profiler, runs):
    x = torch.randn(1025, dtype=torch.float32)
    out = torch.empty_like(x)
    with profiler.profile():
        for _ in range(runs):
            op_kernel[16](x, out, x.numel(), 1024)
    torch.testing.assert_close(out, torch.reciprocal(x), atol=1e-3, rtol=1e-3)
```

Keep input/output descriptors, local shapes, offsets, and `real_shape` ranks
consistent. Tail padding must be neutral for the operation. Pass tensors
directly to the launch; never use `data_ptr()`.
For runtime tails, use the supported JIT builtin `min(size - offset, tile_size)`.
Do not assign `size - offset` to a variable and then replace it with the
compile-time tile size inside an `if`: that changes its type from PlainValue
to Python `int` and fails codegen at the campaign pin.

## JIT options and baselines

Compare bare `@asctile.jit` defaults with `@asctile.jit(reuse_alloc=1)` on the
same source and geometry. The latter's "legacy implementation" names an
internal reuse pass; it remains the public AscTile API. Keep synchronization
ON (`insert_sync=True`). Plan `reuse_alloc={0,1,2}` ×
`static_alloc={None,False,True}` × `vf_fusion={False,True}`; record unsupported,
failed, and untested cells as well as successful ones. Static allocation,
reuse, and VF fusion interact with UB, unrolling, correctness, and runtime
performance; do not infer speed or safety from a successful UB check.

Follow the [JIT matrix and diagnostics](../asctile-api-patterns/references/api-jit-options.md)
to test actual decorators/effective options in fresh processes and both import
orders. Separate in-process option-discovery caches from compiled artifacts.
At the GeLU pin, base `JITFunction.get_config_keywords()` before AscTile
discovery triggers an inherited-cache rejection of `reuse_alloc`; cold
AscTile-first probes pass. Extraction uses `self.compiler.options_cls`.
Do not assume JIT is broken, monkeypatch it, or introduce a custom JIT or
unsupported `run_asctile_passes` option.

## Coverage and performance

- Parameterize production shapes, dtypes, attributes, boundary sizes, and
  special values—not only a teaching shape.
- Select core count from workload size and platform capacity. Tiny shapes must
  not launch empty work on most cores; large shapes should use the full grid.
- Derive tile size from dtype, live local tensors, `unroll_factor`, and UB
  capacity. A compile pass at one shape is not a UB proof for all cases.
- For reductions, use the correct padding identity and pack contiguous rows;
  see `asctile-api-patterns/references/reduction-tiling.md`.
- Compare every logical output element with the pure-torch golden using
  dtype-appropriate tolerance and explicit NaN/Inf handling where required.

## Verification

From the pyasc repository root, using a runtime built from the same commit:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
export PYASC_COMPILER=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH
SIM=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599
export LD_LIBRARY_PATH=$SIM/camodel:$SIM/lib:$SIM/lib64:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH

python3 -m pytest <resolved-target-path>/test_<op>.py \
  --compile-only --backend Model --platform Ascend950PR_9599 -q
python3 -m pytest <resolved-target-path>/test_<op>.py \
  --backend Model --platform Ascend950PR_9599 -q
```

Use `--compile-only` for the whole matrix first. Run feasible numerical cases
on camodel and record any omitted routes. Only real-NPU/CANNBench measurements
establish production performance.

## Definition of done

- Exact PyASC/AscTile commit and runtime provenance recorded.
- Target file is based on a current sibling and covers the requested domain.
- Whole matrix compiles; feasible model cases match the golden.
- CANN reference and remaining compiler/runtime blockers are documented.
- Commit only the intended target/test files when a commit was requested.
