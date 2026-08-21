# pyasc asc2 JIT Compile Options

## Decorator syntax

```python
@asc2.jit(always_compile=True)        # standard for development
@asc2.jit                              # defaults (uses cache)
```

## Compile parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `always_compile` | `bool` | `False` | Force recompilation, bypass cache |
| `opt_level` | `int` (0-3) | Compiler default | Bisheng optimization level |
| `matmul_cube_only` | `bool` | `False` | Pure cube mode (matrix compute only) |
| `reuse_alloc` | `int` (0-2) | `0` | Which UB-reuse pass runs. `0` none, `1` `ReuseUBAllocation`, `2` `ReuseTensorAllocation`. `2` packs buffers differently rather than strictly tighter — see the hazard below |

Note: `insert_sync=True` and `run_asc2_passes=True` are defaults for `@asc2.jit`.
Do not disable them unless debugging a specific issue.

<!-- BEGIN: temporary, delete once pyasc issue #2 is fixed -->
### `asc2.where` writes past its destination tile (open compiler defect)

**This is a current compiler defect, not a property of the model.** Delete this
section once [pyasc issue #2](https://gitcode.com/compiler-team/pyasc/issues/2)
is fixed.

A loop whose body produces a mask with a comparison and consumes it with
`asc2.where` is miscompiled at `reuse_alloc=2`:

```python
for i in asc2.range(n):
    idx_scalar = asc2.copy_in(idx_gm, [i])
    mask = asc2.equal(ref_tile, idx_scalar)
    result = asc2.where(mask, asc2.cast(ON, out_ptr.dtype), asc2.cast(OFF, out_ptr.dtype))
    asc2.copy_out(result, out_gm, [i * depth])
```

The compare/select pair lowers to `CompareScalar` and `Select` with
`repeatTimes = ceil(elems / lanes)` and a **full mask**, so it writes a whole
number of 256-byte vector repeats regardless of how small the tile is. A
3-element `int32` tile occupies 32 bytes of UB but the `Select` writes 256 —
up to 255 bytes past the destination. The reuse setting only decides whether
that overrun is destructive: `reuse_alloc=1` happens to place the destination
last in UB, so the overrun falls off the end harmlessly, while `reuse_alloc=2`
places it in the middle, where it clobbers the tiles the next iteration reads.

The controlling quantity is the tile's **byte size**, not the loop or the
dtype. Measured on `Ascend950PR_9599` / NPU at `reuse_alloc=2`, `int32`,
256-byte repeats:

| `depth` | allocated bytes | bytes written | result |
|---|---|---|---|
| 3, 8, 16, 32 | 32, 32, 64, 128 | 256 | corrupt |
| 63, 64 | 256 | 256 | correct |
| 65, 96 | 288, 384 | 512 | corrupt |
| 128 | 512 | 512 | correct |
| 129 | 544 | 768 | corrupt |

It is correct exactly when the allocation is a whole number of repeats — note
`depth=63` survives only because 252 bytes rounds up to a 256-byte allocation.

The signature is distinctive:

- the **first** iteration is correct, every later one is wrong (a one-iteration
  launch therefore passes and hides the bug);
- the output holds values neither `where` operand can produce — the clobbered
  input tile verbatim, or `0x3F800000` / `0x40000000`, which are floats `1.0`
  and `2.0` from the `float32` cast that `asc2.equal` inserts before comparing;
- `reuse_alloc=1` on the same source is correct;
- there is **no diagnostic**: it fails silently, as wrong numbers only.

In two operator sweeps it hit 3 of 4 and 15 of 16 cases; the survivors had
tiles that happened to fill whole repeats.

**What to do.** Use `reuse_alloc=1` for any kernel where a comparison feeds
`asc2.where`. A plain elementwise loop is unaffected, so this is not a reason
to avoid `reuse_alloc=2` generally. Padding the tile to a multiple of 256 bytes
also avoids it, and is worth knowing for diagnosis, but do not ship it as a
workaround: it silences the symptom by making the out-of-bounds write land
inside the allocation, and it breaks again the moment the shape changes.

Because the overrun is in the lowering rather than the allocator, it is latent
at `reuse_alloc=1` too — it writes into unallocated UB there rather than into a
live tile. Do not read a passing `reuse_alloc=1` run as evidence that a
compare/select kernel is free of it.

**When measuring performance**, run both settings and treat a `reuse_alloc=2`
miscompare as a recorded failure rather than something to work around: a kernel
whose numbers only look good at `reuse_alloc=2` is not a kernel that works.
<!-- END: temporary, delete once pyasc issue #2 is fixed -->

### `int64` and the comparison ops

Comparisons (`asc2.equal` and friends) accept `int8, int16, int32, float16,
bfloat16, float32` — there is no `int64` form, so an `int64` index tile fails at
codegen ([pyasc issue #3](https://gitcode.com/compiler-team/pyasc/issues/3)).
Narrow to `int32` for the compare only, and record the precondition that makes
it exact — the values must fit in `int32`. See Common Mistakes in the main
skill for the full pattern.

## Launch syntax (asc2)

```python
kernel[core_num](arg1, arg2, ...)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `core_num` | `int` | Yes | Number of cores to use |

**asc2 does NOT use a stream argument.** The v1 syntax `kernel[core_num, stream](...)` must not be used.

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
| Called from | Host: `kernel[cores](...)` | Other `@asc2.jit` functions |
| Compile options | Effective | Ignored |
| `return` | Not allowed | Allowed (top-level only) |
