---
name: pyasc-target-operator
description: Fixed procedure for authoring a pyasc2 "target operator" (a hand-written @asc2.jit kernel + pytest target test that mirrors a CANN ops-* operator on production shapes) inside the pyasc compiler repo. Covers the target test layout, the canonical elementwise template, the branch-off-v2 + commit routine, ops-* reference discovery, and the exact run/verify command. Trigger — implementing a new operator as a target test in python/test/asc2/target/, or running/verifying target tests on the simulator.
---

# pyasc Target Operator (asc2)

A **target operator** is a hand-written `@asc2.jit` kernel plus a pytest suite that
reproduces the behavior (and production shapes/tiling) of a CANN `ops-*` operator,
living **inside the pyasc compiler repo** under `python/test/asc2/target/`. It is the
ground-truth, on-simulator implementation a kernel author commits on a feature branch.

This skill is the **fixed running procedure**: follow it exactly. The only things that
vary per operator are the math, the dtype, and the shape list — everything else below is
identical for every target operator.

## 1. Version-control routine (do this first)

The repo is a fresh clone with `v2` already checked out at the exact commit that
matches the installed `asc2`. **Branch off the CURRENT checkout — do not change which
commit is checked out:**

```bash
git switch -c <op>-target         # e.g. reciprocal-target ; branches from current HEAD
```

> **Do NOT `git fetch`, `git pull`, `git checkout <other-rev>`, or fast-forward `v2`.**
> The checked-out commit is pinned to the installed/built `asc2`; moving to a newer
> `origin/v2` will desync the API (e.g. the `asc2.load` signature) from the runtime and
> your tests will fail to compile. Always branch from what is already checked out.

Author your work, run it green (section 5), then commit **only the operator files** you
added (the test file, and any helper module). Do not commit injected tooling
(`opencode.json`, `skills/`, `.agent-prompt.txt`, `agent-output.txt`) — they are
git-excluded; verify with `git status` that your commit is clean.

```bash
git add python/test/asc2/target/test_<op>.py
git commit -m "Add <Op> target operator (float32) for production shapes"
```

## 2. Where the operator goes

- Test file: `python/test/asc2/target/test_<op>.py` (one file per operator).
- Fixtures are provided by the shared `python/test/asc2/conftest.py` +
  `python/test/asc2/target/conftest.py`: `backend`, `platform`, `set_platform`
  (autouse), `require_c310`, `profiler`, `runs`, `--compile-only`. You do **not** write
  these — just use `profiler` and `runs` in the test signature.
- **Model your file on the closest existing sibling target test.** List the directory
  (`ls python/test/asc2/target/`) and open the nearest match: for a **unary elementwise**
  op use `test_vadd.py` / `test_gelu.py`; for reductions `test_reduce_sum.py`; for
  normalization `test_softmax.py`, etc. Mirror that file's structure, imports, kernel
  shape, host launcher, parametrization, and golden style. The template below is the
  canonical elementwise form (matches `test_vadd.py`).

## 3. Canonical elementwise target test (template)

Adapt the math (`zt = ...`) and arity for your operator; keep the tiling/launch scaffold.

```python
import asc2
import pytest
import torch


@asc2.jit(static_alloc=True, reuse_ub=True)
def <op>_kernel_1D(input_ptr: asc2.GlobalAddress, output_ptr: asc2.GlobalAddress,
                   input_shape: asc2.ConstExpr, output_shape: asc2.ConstExpr,
                   block_loop_num: asc2.ConstExpr, block_loop_num_tail: asc2.ConstExpr,
                   tile_length: asc2.ConstExpr, block_length: asc2.ConstExpr,
                   UNROLL_FACTOR: asc2.ConstExpr):
    x = asc2.tensor(input_ptr, input_shape)
    z = asc2.tensor(output_ptr, output_shape)

    block_offset = asc2.block_idx() * block_length
    loop_count = block_loop_num
    if asc2.block_idx() == (asc2.block_num() - 1):
        loop_count = block_loop_num_tail

    for i in asc2.range(loop_count, unroll_factor=UNROLL_FACTOR, parallel=True):
        current_offset = block_offset + i * tile_length
        xt = asc2.load(x, [tile_length], offsets=[current_offset])
        zt = 1.0 / xt                      # <-- operator math (out = 1/x); see section 4
        asc2.store(zt, z, offsets=[current_offset])


@pytest.mark.parametrize(
    "core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype, tiling_key, tiling_values", [
        # one row per shape: tiling_values = [length, 0, tile_length, core_num]
        (16, 2, [1024], torch.float32, [1024], torch.float32, 8, [1024, 0, 128, 16]),
        # ... remaining shapes ...
    ])
def test_<op>(profiler, runs, core_num, unroll_factor, input_shape, input_dtype, output_shape, output_dtype,
              tiling_key, tiling_values):
    _, _, tile_length, core_num = tiling_values

    input_shape_1d = [torch.prod(torch.tensor(input_shape[:])).item()]
    length = input_shape_1d[0]

    # 32-byte alignment for tile_length
    ALIGNMENT_ELEMENTS = 32 // input_dtype.itemsize
    tile_length = asc2.ceildiv(tile_length, ALIGNMENT_ELEMENTS) * ALIGNMENT_ELEMENTS

    block_loop_num = asc2.ceildiv(asc2.ceildiv(length, core_num), tile_length)
    block_length = tile_length * block_loop_num
    block_loop_num_tail = asc2.ceildiv(length - block_length * (core_num - 1), tile_length)
    padded_length = block_length * (core_num - 1) + tile_length * block_loop_num_tail
    padded_shape = [padded_length]

    in_tensor = torch.full(padded_shape, dtype=input_dtype, fill_value=1.0)
    in_tensor[:length] = torch.randn(input_shape_1d, dtype=input_dtype)
    out_tensor = torch.zeros(padded_shape, dtype=output_dtype)

    with profiler.profile():
        for _ in range(runs):
            <op>_kernel_1D[core_num](in_tensor, out_tensor, padded_shape, padded_shape,
                                     block_loop_num, block_loop_num_tail, tile_length, block_length, unroll_factor)

    expected = torch.reciprocal(in_tensor)        # <-- torch golden for the SAME math
    torch.testing.assert_close(out_tensor[:length], expected[:length], atol=1e-3, rtol=1e-3)
```

Notes that matter:
- **Flatten any shape to 1D** with `torch.prod(...)`; tile over the flat length.
- **Pad** the flat length up to `block_length*(core_num-1) + tile_length*tail` so every
  core/tile is full; only compare the first `length` elements against the golden.
- Pick `core_num` ≤ the platform core count and a `tile_length` (UB-bounded, then
  32-byte aligned by the host code). A safe default `tile_length` is a few KB worth of
  elements (e.g. 128–10496 for f32); choose smaller `core_num` for tiny shapes so each
  core gets ≥1 tile. The host code derives all loop counts from `(length, core_num,
  tile_length)`.
- **Golden is pure torch**, compared with `torch.testing.assert_close(atol=1e-3,
  rtol=1e-3)`. Avoid inputs at/near singularities (e.g. for `1/x`, fill padding with 1.0
  and draw `randn`, which is ~never 0).
- The launch form is `kernel[core_num](*args)`; pass `asc2.ConstExpr` host scalars as
  plain ints here (they bind to the `ConstExpr` params).

## 4. Operator math + ops-* reference

- The operator math comes from the task (e.g. `out = 1 / x`). asc2 has **no
  `reciprocal`** builtin — express `1/x` as `1.0 / xt` (tile rdiv) or via `asc2.div`.
  Browse `dir(asc2)` for available elementwise builtins (`add, sub, mul, div, exp, log,
  sqrt, rsqrt, abs, tanh, erf, relu, maximum, minimum, where, ...`).
- **Discover the matching CANN `ops-*` reference yourself** using the remote operator map
  in the `pyasc-docs-search` skill (the `gitcode.com/cann/ops-*` repositories). Pick the
  repo/op whose semantics match (for `reciprocal`: `ops-math` → `math/reciprocal`; also
  see `div` / `real_div`). Use it only as the semantic/tiling ground truth — do not copy
  CANN C++; the deliverable is the asc2 kernel + torch golden above. Cite the chosen
  reference in the commit message.

## 5. Run / verify (the fixed command)

Source the CANN simulator env (see `pyasc-build-run-verify`), then from the **repo root**:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYASC_COMPILER=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng
export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH
SIM=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599
export LD_LIBRARY_PATH=$SIM/camodel:$SIM/lib:$SIM/lib64:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH

python3 -m pytest python/test/asc2/target/test_<op>.py \
    --backend Model --platform Ascend950PR_9599 -p no:cacheprovider -q
```

- Run from the repo root so `import asc2` resolves to the **installed/built** compiler,
  not this clone's unbuilt `python/asc2` source. `asc2` is already importable here.
- **NEVER build pyasc from source.** Do not run `cmake`, `ninja`, `pip install .`,
  `pip install -e .`, or any LLVM/MLIR build — `asc2` is already installed and importable.
  Confirm once with `python3 -c "import asc2; print(asc2.__file__)"`. If (and only if)
  that import fails, STOP and report the failure; do not attempt to build.
- The Model simulator is slow (~30–60 s per case). **Verify only *feasible* shapes** on
  the simulator and `--compile-only` the rest — apply the feasibility predicate from
  `pyasc-build-run-verify`.
- Iterate until all selected shapes pass, then commit (section 1).

## 6. Definition of done

- Branch `<op>-target` created off `v2`.
- `python/test/asc2/target/test_<op>.py` present, modeled on the closest sibling, all
  requested shapes parametrized, pure-torch golden with `torch.testing.assert_close`.
- The chosen `ops-*` reference is identified and cited in the commit message.
- Feasible shapes pass on `--backend Model`; the work is committed with a clean
  `git status` (no injected tooling in the commit).
