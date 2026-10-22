# pyasc v2 AscTile JIT compile options

The CANNBench campaign uses
`compiler-team/pyasc@0a631f70968c3cb7c33ce45330a85768dd5a6f06` with a matching
runtime. The API notes below were measured for the historical high-level GeLU
campaign at `adadd7d66ed0ee16d33d79487bf584899a26ef1e`; treat them as historical
behavior notes, not as CANNBench runtime authority. Other reports, including
those at `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`, retain their original pins. An option failure in one process is not evidence
that public AscTile JIT is generally broken: test the actual decorator,
effective options, and caches as described below.

## Decorator syntax

```python
@asctile.jit                         # default baseline (uses cache)
@asctile.jit(reuse_alloc=1)           # second baseline; public AscTile API
@asctile.jit(reuse_alloc=2, static_alloc=True, vf_fusion=True,
            insert_sync=True, always_compile=True)  # development experiment
```

These are alternative decorators for separate kernel variants, not stacked
decorators. Use public `asctile.GlobalAddress`, `asctile.ConstExpr`, dtypes,
tile math, copy operations, and range in their bodies, following the target
tests. No authored `inline`/`inline_vf`, low-level `asc`/`asc2`, compiler
monkeypatch, or custom JIT belongs in the high-level campaign. Compiler-created
register code from `vf_fusion=True` is permitted.

## Compile parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `always_compile` | `bool` | `False` | Force recompilation, bypass cache |
| `opt_level` | `int` (0-3) | Compiler default | Bisheng optimization level |
| `matmul_cube_only` | `bool` | `False` | Pure cube mode (matrix compute only) |
| `reuse_alloc` | `int` (0-2) | `0` | Which UB-reuse pass runs. `0` none, `1` `ReuseUBAllocation`, `2` `ReuseTensorAllocation`. `2` packs buffers differently rather than strictly tighter — see the hazard below |
| `static_alloc` | `bool` or `None` | `None` | `None` selects the platform default (enabled on 950PR); `True` requests static tile allocation; `False` uses the AscendC TPipe allocation path |
| `vf_fusion` | `bool` | `False` | Fuse eligible vector operations into compiler-generated register/VF blocks; supported on C310 |
| `insert_sync` | `bool` | `True` | Insert synchronization; keep ON throughout the campaign |

`insert_sync=True` is the current AscTile default. There is no
`run_asc2_passes` or `run_asctile_passes` option in current v2; do not copy it
from older snapshots.

## Explicit JIT experiment matrix

Plan all 18 combinations of `reuse_alloc=(0, 1, 2)`,
`static_alloc=(None, False, True)`, and `vf_fusion=(False, True)`, always with
`insert_sync=True`. Begin with bare defaults and `reuse_alloc=1`, leaving the
other options at defaults, on identical source, tile size, unroll, and inputs.
The source calls `reuse_alloc=1` a recommended "legacy implementation": that
describes its internal pass, not the public API. `reuse_alloc=2` is experimental.

`static_alloc=None` records platform selection, even if it resolves like
`True` on 950PR. Reuse changes buffer layout/lifetimes and may reduce UB or
regress speed; static allocation can reduce scalar allocation overhead;
VF fusion can remove UB transfers but changes register pressure and scheduling.
Their interactions with unrolling and tile size can change UB usage, numerical
behavior, synchronization, and large-case runtime stability. A smaller UB
number does not establish correctness or speed. Preserve the compare/select
hazard below; a passing `reuse_alloc=1` is not proof that the overrun is absent.

For each attempted combination record requested and effective options, pin,
source hash, dtype/shape/mode, tile/core/unroll choices, compiled UB, compile
result, numerical case coverage, and real-NPU timings when authorized.
Record unsupported/failed combinations and untested combinations explicitly.
Workers may test a bounded subset; the matrix is an experiment plan, not a
requirement to run 18 full simulator suites inside one worker budget. Tune
tile/unroll/core choices after comparing options at fixed geometry, and rerun
correctness for the selected combination.

## Diagnose the public JIT path before blaming it

At the historical GeLU pin `adadd7d66ed0ee16d33d79487bf584899a26ef1e`, a
fresh-process diagnostic
reproduced a narrow discovery-cache inheritance bug: calling
`asc.runtime.jit.JITFunction.get_config_keywords()` before importing AscTile
populates the base class's `_config_keywords`. The AscTile subclass inherits
that cached list through `getattr`, so `@asctile.jit(reuse_alloc=1)` then raises
`The following option names are unknown: reuse_alloc` during construction.
Simply importing `asc` is not the measured trigger; base keyword discovery is.
See the [base-first result](../../../integrations/cannbench/comparisons/gelu-asctile-jit-20260907/evidence/jit-base-first/result.json).

Call-time extraction uses `self.compiler.options_cls`; it is not hardwired to
the base compiler. The [cold AscTile-first result](../../../integrations/cannbench/comparisons/gelu-asctile-jit-20260907/evidence/jit-cold/result.json)
records successful native CaModel execution of a 257-element FP32 arithmetic
probe with defaults, decorator `reuse_alloc=1`, a call-time
`reuse_alloc=2, static_alloc=True, vf_fusion=True` override, a cached repeat,
and an `always_compile=True` repeat. These are diagnostic routes, not GeLU
coverage or proof of all 18 combinations. Run stock AscTile in a fresh process
before base keyword discovery; do not patch or delete the inherited cache.

1. Record the source commit, wheel/build provenance, imported module paths,
   architecture, and actual decorated kernel. Test bare `@asctile.jit`,
   `@asctile.jit(reuse_alloc=1)`, then the exact failing option combination.
   Merely constructing `asctile.CompileOptions` does not test the decorator or
   launch-time extraction.
2. Run each probe in a fresh process with a new task-owned `PYASC_CACHE_DIR`.
   Compare importing `asctile` first with `from asc.runtime.jit import
   JITFunction; JITFunction.get_config_keywords()` then importing `asctile`,
   in diagnostic probes only. Apply the same decorator. Record requested/effective compile options
   and cache keys; inherited option-discovery caches may make order matter.
3. Check both an isolated cold compile and a same-options cache hit, then
   change an option and verify it reaches compilation and separates the cache
   entries. `always_compile=True` bypasses compiled artifacts; it does not
   reset in-process option-discovery/extraction caches. Fresh processes do.
4. Reproduce through the real public decorator and supported compile/launch
   entrypoint, not only a gate that directly constructs compiler options.
   Retain exception, logs, and omitted routes if the installed environment
   cannot exercise that path. Report a narrowly scoped blocker; do not
   monkeypatch option discovery/compiler classes or introduce a custom JIT.

<!-- BEGIN: temporary, delete once pyasc issue #2 is fixed -->
### `asctile.where` writes past its destination tile (open compiler defect)

**This is a current compiler defect, not a property of the model.** Delete this
section once [pyasc issue #2](https://gitcode.com/compiler-team/pyasc/issues/2)
is fixed.

A loop whose body produces a mask with a comparison and consumes it with
`asctile.where` is miscompiled at `reuse_alloc=2`:

```python
for i in asctile.range(n):
    idx_scalar = asctile.copy_in(idx_gm, [i])
    mask = asctile.equal(ref_tile, idx_scalar)
    result = asctile.where(mask, asctile.cast(ON, out_ptr.dtype), asctile.cast(OFF, out_ptr.dtype))
    asctile.copy_out(result, out_gm, [i * depth])
```

The compare/select pair lowers to `CompareScalar` and `Select` with
`repeatTimes = ceil(elems / lanes)` and a **full mask**, so it writes a whole
number of 256-byte vector repeats regardless of how small the tile is. A
3-element `int32` tile occupies 32 bytes of UB but the `Select` writes 256 —
up to 255 bytes past the destination. The reuse setting only decides whether
that overrun is destructive: `reuse_alloc=1` happens to place the destination
last in UB, so the overrun falls off the end harmlessly, while `reuse_alloc=2`
places it in the middle, where it clobbers the tiles the next iteration reads.

#### The trigger condition

A `where` destination is **exposed** — the compiler emits a write past its end —
exactly when

```
align_to(numel * itemsize, 32)  is NOT a multiple of 256
```

The controlling quantity is the tile's byte size, not the loop, the dtype or the
reuse setting. Measured on `Ascend950PR_9599` / NPU at `reuse_alloc=2`, `int32`:

| `depth` | allocated bytes | bytes written | result |
|---|---|---|---|
| 3, 8, 16, 32 | 32, 32, 64, 128 | 256 | corrupt |
| 63, 64 | 256 | 256 | correct |
| 65, 96 | 288, 384 | 512 | corrupt |
| 128 | 512 | 512 | correct |
| 129 | 544 | 768 | corrupt |

Note `depth=63` survives only because 252 bytes rounds up to a 256-byte
allocation.

Exposure is necessary but not sufficient. To produce **wrong numbers** the
overrun also has to land on a tile that is not rewritten before its next read.
Two things therefore protect a kernel, and only the first is robust:

- **Sizing.** Give the `where` destination a tile whose aligned byte size is a
  multiple of 256. Then there is no overrun at all.
- **No loop-invariant tiles.** If every tile the loop reads is `copy_in`-ed
  *inside* the loop, the overrun is repaired before it is read and the kernel is
  self-healing. A tile hoisted **above** the loop — a reference tile, an
  `arange`, a splat constant — is the vulnerable kind, because nothing rewrites
  it. This is why `one_hot` (hoisted `arange_tile`) corrupts while `select`
  (all three inputs copied in per iteration) does not, even at tile sizes that
  are equally exposed.

#### Which operations are affected

Only three lowerings use the full-mask form, all in the compare/select family:

| Emitted | Reached from |
|---|---|
| `Select` | `asctile.where` |
| `Compare` | comparison of two tiles |
| `CompareScalar` | comparison of a tile against a scalar |

Everything else lowers to the *counted* form, which is given the exact element
count and is unaffected: `add`, `sub`, `mul`, `div`, `maximum`, `minimum`,
`abs`, `exp`, `log`, `sqrt`, `relu`, `leaky_relu`, `cast`, `duplicate`, the
reductions, the shifts and the bitwise ops. So `asctile.maximum(x, eps)` is a safe
way to express a clamp, and a plain elementwise loop is never affected.

Three rewrites also fold `where`-shaped code into a counted op before it can
reach `Select`, which makes those spellings safe at any tile size:

| Source | Folded to |
|---|---|
| `asctile.where(x >= 0, x, 0)` | `relu` |
| `asctile.where(x >= 0, x, x * alpha)` | `leaky_relu` |
| `asctile.maximum(x, 0)` | `relu` |

The folds require `float16`/`float32`, so the same spelling on an integer tile
does **not** fold and is exposed.

#### Signature when it does bite

- the **first** iteration is correct, every later one is wrong (a one-iteration
  launch therefore passes and hides the bug);
- the output holds values neither `where` operand can produce — the clobbered
  input tile verbatim, or `0x3F800000` / `0x40000000`, which are floats `1.0`
  and `2.0` from the `float32` cast that `asctile.equal` inserts before comparing;
- `reuse_alloc=1` on the same source is correct;
- there is **no diagnostic**: it fails silently, as wrong numbers only.

Across 20 measured `one_hot` cases at `reuse_alloc=2`, the sizing rule predicted
the outcome in 19: the single case with `depth=64, int32` (exactly 256 bytes) was
the only geometrically safe one and the only clean pass, 17 exposed cases
corrupted, and two (`depth=7, float32`, in both sweeps) were exposed but did not
manifest. Note that single-element inputs are **not** safe — `depth=1` corrupted.

**What to do when writing a kernel.** Prefer, in order:

1. size the `where` destination so `align_to(bytes, 32) % 256 == 0`;
2. use one of the folded spellings above, or `asctile.maximum`/`asctile.minimum`,
   when the intent is a clamp or a relu;
3. keep no loop-invariant tile live across the `where` — copy inputs in inside
   the loop;
4. failing all of those, `reuse_alloc=1`, which usually places the destination
   last so the overrun falls off the end of the used region.

Only the first removes the out-of-bounds write. The others merely make it
harmless, and they stop being true when the shape or the surrounding code
changes.

Because the overrun is in the lowering rather than the allocator, it is latent
at `reuse_alloc=1` too — it writes into unallocated UB there rather than into a
live tile. Do not read a passing `reuse_alloc=1` run as evidence that a
compare/select kernel is free of it.

**When measuring performance**, run both settings and treat a `reuse_alloc=2`
miscompare as a recorded failure rather than something to work around: a kernel
whose numbers only look good at `reuse_alloc=2` is not a kernel that works.
<!-- END: temporary, delete once pyasc issue #2 is fixed -->

### `int64` and the comparison ops

Comparisons (`asctile.equal` and friends) accept `int8, int16, int32, float16,
bfloat16, float32` — there is no `int64` form, so an `int64` index tile fails at
codegen ([pyasc issue #3](https://gitcode.com/compiler-team/pyasc/issues/3)).
Narrow to `int32` for the compare only, and record the precondition that makes
it exact — the values must fit in `int32`. See Common Mistakes in the main
skill for the full pattern.

## Launch syntax (asctile)

```python
kernel[core_num](arg1, arg2, ...)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `core_num` | `int` | Yes | Number of cores to use |

At the campaign pin, `asctile.runtime.jit.JITFunction` inherits the base
`__getitem__`, which constructs `LaunchOptions(core_num, stream, ...)`.
`kernel[core_num, stream](...)` is supported public syntax. Use a qualified
current-device/current-stream context when required by the evaluator; do not
mistake the default-stream shorthand above for a stream prohibition.

## JIT cache behavior

- **Cache location**: `${PYASC_HOME}/.pyasc/cache` (or `${PYASC_CACHE_DIR}`)
- **Cache key**: compile options + kernel parameters + global variables + source code
- **Force rebuild**: Set `always_compile=True` or use a fresh task-owned cache
  directory; neither resets in-process option-discovery caches. Do not delete
  shared caches for a campaign probe.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `PYASC_HOME` | Cache root directory (default: user home) |
| `PYASC_CACHE_DIR` | Specific cache directory |
| `PYASC_DUMP_PATH` | Save generated ASC-IR and Ascend C code for inspection |

## Kernel vs device function

| Aspect | Kernel function | Device function |
|--------|----------------|-----------------|
| Called from | Host: `kernel[cores](...)` | Other `@asctile.jit` functions |
| Compile options | Effective | Ignored |
| `return` | Not allowed | Allowed (top-level only) |
