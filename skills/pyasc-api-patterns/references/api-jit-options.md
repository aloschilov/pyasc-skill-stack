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
| `reuse_alloc` | `int` | `1` | How aggressively `ReuseTensorAllocation` shares UB buffers between values. `2` reuses more and fits larger tiles, but see the hazard below |

Note: `insert_sync=True` and `run_asc2_passes=True` are defaults for `@asc2.jit`.
Do not disable them unless debugging a specific issue.

<!-- BEGIN: temporary, delete once pyasc issue #2 is fixed -->
### `reuse_alloc=2` corrupts comparison masks (open compiler defect)

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

The allocator understates the mask buffer's live range, so it is reused inside
the same iteration. The signature is distinctive:

- the **first** iteration is correct, every later one is wrong (a one-iteration
  launch therefore passes and hides the bug);
- the output holds values neither `where` operand can produce, most often
  `0x3F800000` — float `1.0`, the mask's true value — and sometimes other live
  tiles of the kernel verbatim;
- `reuse_alloc=1` on the same source is correct;
- there is **no diagnostic**: it fails silently, as wrong numbers only.

Verified on `Ascend950PR_9599` / NPU. In two operator sweeps it hit 3 of 4 and
15 of 16 cases; the only survivors were single-element inputs.

**What to do.** Use `reuse_alloc=1` for any kernel where a comparison feeds
`asc2.where`. A plain elementwise loop is unaffected, so this is not a reason to
avoid `reuse_alloc=2` generally. If a kernel needs both the reuse and the
select, materialising the `where` operands as real tiles instead of
`asc2.cast(scalar, dtype)` splats reduces the damage but does **not** fix it —
the mask itself is still corrupted on some iterations. Do not ship that as a
workaround.

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
