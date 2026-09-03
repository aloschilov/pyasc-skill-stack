# pyasc v2 AscTile JIT compile options

At evaluated commit `0a631f70968c3cb7c33ce45330a85768dd5a6f06`, these options
exist on `asctile.runtime.compiler.CompileOptions`, but inherited JIT option
discovery/extraction still uses the base `asc` option class. The plain
decorator forms below are therefore API intent, not a working CANNBench path,
until an upstream fix or the repository's concrete-options adapter is loaded.

## Decorator syntax

```python
@asctile.jit(always_compile=True)        # standard for development
@asctile.jit                              # defaults (uses cache)
```

## Compile parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `always_compile` | `bool` | `False` | Force recompilation, bypass cache |
| `opt_level` | `int` (0-3) | Compiler default | Bisheng optimization level |
| `matmul_cube_only` | `bool` | `False` | Pure cube mode (matrix compute only) |
| `reuse_alloc` | `int` (0-2) | `0` | Which UB-reuse pass runs. `0` none, `1` `ReuseUBAllocation`, `2` `ReuseTensorAllocation`. `2` packs buffers differently rather than strictly tighter — see the hazard below |

`insert_sync=True` is the current AscTile default. There is no
`run_asc2_passes` or `run_asctile_passes` option in current v2; do not copy it
from older snapshots.

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

**asctile does NOT use a stream argument.** The v1 syntax `kernel[core_num, stream](...)` must not be used.

## JIT cache behavior

- **Cache location**: `${PYASC_HOME}/.pyasc/cache` (or `${PYASC_CACHE_DIR}`)
- **Cache key**: compile options + kernel parameters + global variables + source code
- **Force rebuild**: Set `always_compile=True` or delete cache directory

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
