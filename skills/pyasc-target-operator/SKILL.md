---
name: pyasc-target-operator
description: Procedure for authoring and verifying a current-v2 handwritten asctile target operator under python/test/asctile/target. Use for target tests, handwritten reference kernels, simulator verification, and CANN operator parity work.
---

# Current-v2 target operator workflow

A target operator is a handwritten `@asctile.jit` kernel plus pytest coverage
inside the pyasc repository. For the current v2 tree the location is
`python/test/asctile/target/`; `python/test/asc2/target/` is obsolete.

## Version discipline

Record the exact base before editing:

```bash
git fetch origin v2
git rev-parse origin/v2
git switch -c <op>-target origin/v2
```

If a campaign intentionally pins another revision, use that recorded revision
instead of the fetch/switch example above. The CANNBench campaign is pinned to
`0a631f70968c3cb7c33ce45330a85768dd5a6f06` (runtime authority). The historical
high-level GeLU campaign pinned `adadd7d66ed0ee16d33d79487bf584899a26ef1e` and
is not CANNBench authority; preserve historical results at their original
revisions. Never mix source from one commit with
a wheel/runtime from another. Build from source when the selected v2 commit is
not distributed as a Python package; do not replace it with a repository wheel
from another revision.

Use prebuilt local compile/CaModel environments within a worker's assigned
budget. Missing dependencies are a reported limitation; workers do not install
packages or start source builds unless asked.

## Source layout and reference selection

- New test: `python/test/asctile/target/test_<op>.py`.
- Shared fixtures: `python/test/asctile/conftest.py` and
  `python/test/asctile/target/conftest.py`.
- Start from the closest current sibling: `test_vadd.py` for elementwise,
  `test_reduce_sum.py`/`test_reduce_max.py` for reductions,
  `test_softmax.py` for normalization, and `test_transpose.py` for movement.
- Use the matching `gitcode.com/cann/ops-*` implementation as semantic and
  tiling reference. Do not copy its C++ as the pyasc implementation.

Use public `asctile` annotations, dtypes, JIT, tile math, copy operations, and
range. New high-level campaign source must not import low-level `asc`/`asc2`,
author `inline`/`inline_vf` code, or implement a long Numerical Recipes/Horner
polynomial. Compiler-generated register code from `vf_fusion=True` is allowed.

## Canonical elementwise structure

```python
import asctile
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

Follow the [JIT matrix and diagnostics](../pyasc-api-patterns/references/api-jit-options.md)
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
  see `pyasc-api-patterns/references/reduction-tiling.md`.
- Compare every logical output element with the pure-torch golden using
  dtype-appropriate tolerance and explicit NaN/Inf handling where required.

## Verification

From the pyasc repository root, using a runtime built from the same commit:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYASC_COMPILER=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH
SIM=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599
export LD_LIBRARY_PATH=$SIM/camodel:$SIM/lib:$SIM/lib64:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH

python3 -m pytest python/test/asctile/target/test_<op>.py \
  --compile-only --backend Model --platform Ascend950PR_9599 -q
python3 -m pytest python/test/asctile/target/test_<op>.py \
  --backend Model --platform Ascend950PR_9599 -q
```

Use `--compile-only` for the whole matrix first. Run feasible numerical cases
on camodel and record any omitted routes. Only real-NPU/CANNBench measurements
establish production performance.

## Definition of done

- Exact v2 commit and runtime provenance recorded.
- Target file is based on a current sibling and covers the requested domain.
- Whole matrix compiles; feasible model cases match the golden.
- CANN reference and remaining compiler/runtime blockers are documented.
- Commit only the intended target/test files when a commit was requested.
