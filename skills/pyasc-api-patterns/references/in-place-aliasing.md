# In-place / aliasing pattern (output aliases an input buffer)

Guidance for writing an **in-place** elementwise kernel — one whose output
**aliases an input buffer** rather than writing to a freshly allocated output.
The canonical example is a PyTorch-style `a <- op(a, ...)` (`add_`, `mul_`,
`sub_`, `div_`, `relu_`, `clamp_`, ...). The kernel reads a GM tensor, computes,
and `copy_out`s the result straight back into the **same** GM tensor; the host
passes the **same** tensor as input `a` and as the destination, not a separate
`out` allocation.

This is a small but load-bearing variation on the [elementwise tiling
pattern](elementwise-tiling.md): the tiling, the rank rules, the selector, and
the `asc2.range` choice are all unchanged — only the *number of GM buffers* and
the *`copy_out` destination* differ.

> **API note.** This reference uses the **fork target-test API**
> (`asc2.global_tensor` / `asc2.copy_in` / `asc2.copy_out`, `torch` tensors,
> the `profiler` / `runs` fixtures) — the API the pyasc fork's
> `python/test/asc2/target/*.py` tests are written in (see `test_vadd.py`,
> `test_reciprocal.py`, `test_addcdiv.py`). The **aliasing contract** (one GM
> handle is both `copy_in` source and `copy_out` destination; no separate
> output) is identical on the v2-mainline tile API (`asc2.tensor` / `asc2.load`
> / `asc2.store` + numpy) that `golden/kernels/add_inplace_f32.py` and the
> `capabilities.yaml` `add_inplace` cell use — only the surface calls differ.
> When writing a fork target test, match the fork's `global_tensor` surface; when
> writing a v2-mainline kernel (the skill-stack default), use
> `tensor`/`load`/`store` with `asc2.range(..., unroll_factor=2, parallel=True)`.

## The rule

- Wrap the aliased pointer **once** with `asc2.global_tensor` and use that
  single `a_gm` handle as both the `copy_in` source and the `copy_out`
  destination.
- Do **not** declare a separate output GM tensor (no extra `output_ptr`, no
  `asc2.global_tensor(output_ptr, ...)`).
- On the host, pass the **same** torch tensor as input `a` and as the buffer you
  read the result back from — no `torch.zeros_like` output tensor. `a` is
  mutated in place.
- Everything else (1-D flatten, the UB-budget tile selector, `asc2.copy_in(a_gm,
  [offset], [tile_length])`, `asc2.copy_out(zt, a_gm, [offset])`,
  `asc2.range(..., unroll_factor=unroll_factor)`) is identical to the
  out-of-place elementwise pattern.

## Single-tensor in-place kernel: `a <- a + b`

`a_ptr` is **both** an input and the destination; `b_ptr` is a read-only addend.
There is exactly one output — and it *is* `a`. The kernel mirrors the canonical
`test_vadd.py` `add`, but with **two** GM pointers instead of three and the
`copy_out` targeting `a_gm`:

```python
@asc2.jit(reuse_alloc=1)
def add_inplace(a_ptr: asc2.GlobalAddress, b_ptr: asc2.GlobalAddress, input_length,
                tile_length: asc2.ConstExpr, unroll_factor: asc2.ConstExpr):
    a_gm = asc2.global_tensor(a_ptr, [input_length])   # a_gm is BOTH input and output
    b_gm = asc2.global_tensor(b_ptr, [input_length])

    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length

    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        at = asc2.copy_in(a_gm, [current_offset], [tile_length])
        bt = asc2.copy_in(b_gm, [current_offset], [tile_length])
        zt = at + bt
        asc2.copy_out(zt, a_gm, [current_offset])   # store back into a_gm, NOT a new output_gm
```

- `@asc2.jit(reuse_alloc=1)` only — the same decorator the fork's
  `reciprocal` / `addcdiv` target kernels use. `static_alloc` defaults to `True`
  on C310. On this fork surface `asc2.range` takes `gm_barrier` (not `parallel`);
  pass only `unroll_factor` (typically `2`) for overlap — `gm_barrier` defaults
  to `False` (overlap enabled). On v2 mainline the equivalent is
  `asc2.range(..., unroll_factor=2, parallel=True)` (`parallel=` is **not**
  removed there).
- No host padding and no tail branch: `copy_in` past the extent auto-pads and
  `copy_out` clamps to the declared `global_tensor` shape, so
  `block_length * block_num` may exceed `input_length` safely (see
  [elementwise-tiling.md](elementwise-tiling.md) "No host padding").

## Host / target test — one tensor for input `a` **and** the result

Mirror the fork target-test shape (`test_vadd.py` / `test_addcdiv.py`): a
`_select_elementwise_tile` selector, the `profiler` / `runs` fixtures, torch
tensors, STATIC/DYNAMIC parametrization, and `torch.testing.assert_close`. The
aliasing lives in two places: (1) the kernel has no third pointer, and (2) the
host passes the **same** tensor for `a` and reads the result back from it.

```python
def test_add_inplace(profiler, runs, kernel_type, test_name, input_shape, input_dtype, tiling):
    length, tile_length, block_num, unroll_factor = tiling

    a = torch.randn([length], dtype=input_dtype)
    b = torch.randn([length], dtype=input_dtype)
    # Snapshot the ORIGINAL `a` BEFORE any launch mutates it. Each launch does
    # a <- a + b, so after `runs` launches a == a0 + runs * b. Reconstruct that
    # exactly so the assert stays correct for both --runs 1 and --profile --runs N.
    a0 = a.clone()
    expected = a0 + runs * b

    params = [a, b]                       # NO separate output tensor — `a` is the output
    if kernel_type == STATIC:
        params.append(asc2.ConstExpr(length))
    else:
        params.append(length)
    params.extend([tile_length, unroll_factor])

    with profiler.profile():
        for _ in range(runs):
            add_inplace[block_num](*params)   # mutates `a` in place each launch

    torch.testing.assert_close(a, expected, atol=1e-3, rtol=1e-3)
```

- **Live tensors = 2** (`a` tile + `b` tile) — the output aliases `a`, so it is
  *not* a third live buffer. Pass `live_tensors=2` to
  `_select_elementwise_tile` (same as `reciprocal`; `addcdiv` uses 4).
- **Snapshot `a0` first, reconstruct the accumulation.** `a0 = a.clone()` must be
  taken *before* the launch loop, because each launch overwrites `a`. Because the
  op is genuinely in-place (`a <- a + b`), N launches leave `a == a0 + N*b`, so
  the reference is `expected = a0 + runs * b`. This is the honest reconciliation:
  it keeps the perf loop (identical-cost launches, median under `--profile
  --runs`) *and* a correct correctness assert. Do **not** assert against a bare
  `a + b` when `runs > 1` — that only matches the first launch. Do **not** re-seed
  `a` inside the profiler loop — that adds a host copy to every measured
  iteration and corrupts the timing.
- **fp precision under accumulation.** Reconstructing `a0 + runs * b` (a single
  scaled add) vs the kernel's `runs` sequential adds can differ by a few ULP;
  `atol=rtol=1e-3` with a small `runs` (≤ 10) absorbs it. For fp16 widen to
  `4e-3` as elsewhere.

## Multi-tensor in-place: `apply_adam` (contrast)

`apply_adam` is the same idea applied to **several** buffers at once: `var`, `m`,
and `v` are each `copy_in`'d, updated, and `copy_out`'d back into their **own** GM
tensors in the same kernel (`grad` is read-only). So `apply_adam` aliases *three*
input buffers as three outputs, whereas `add_inplace` aliases *one*:

```python
# apply_adam: each of var_gm / m_gm / v_gm is copy_in'd AND copy_out'd back
asc2.copy_out(m_new, m_gm, [current_offset])
asc2.copy_out(v_new, v_gm, [current_offset])
asc2.copy_out(var_new, var_gm, [current_offset])
```

| Kernel | Aliased buffers (in == out) | Read-only inputs | Output count |
|---|---|---|---|
| `add_inplace` (`a <- a + b`) | `a` | `b` | 1 (is `a`) |
| `apply_adam` (`var/m/v <- ...`) | `var`, `m`, `v` | `grad` | 3 (are var/m/v) |

Both are marked `in_place: true` in `capabilities.yaml`; the checker requires
`output_shape: same_as_input` on any `in_place: true` cell (an in-place op
writes into an existing buffer, so its output is shaped like that buffer). See
[`docs/glossary.md` §6](../../../docs/glossary.md) `in_place`.

## Why this is safe

Within one tile iteration the value is `copy_in`'d to UB, transformed on the AIV
vector pipeline, and `copy_out` back to the same GM offset — there is no
read-after-write hazard *across* iterations because each iteration touches a
disjoint `[offset, offset + tile_length)` slice, so the loop carries no
dependency through the aliased buffer and `unroll_factor=2` double-buffering is
still correct. This is the same reason the out-of-place elementwise loop overlaps
safely.

## Performance note (memory-bound floor)

In-place add moves the same bytes as out-of-place add: **2 reads (`a`, `b`) + 1
write (`a`)** per element. It is purely memory-bound, so its CANN reference is
the elementwise `Add` (`aclnnInplaceAdd` lowers to the same AICORE `Add` kernel).
Tune it exactly like any other elementwise op — use every AI core and size the
tile to the UB budget with `live_tensors=2` (see
[elementwise-tiling.md](elementwise-tiling.md) "The selector"). A fixed small
core count with tiny fixed tiles leaves most of the grid idle and lands well
below CANN on the large shapes.

## Anti-patterns

- **Allocating a separate output** (`out = torch.zeros_like(a)`; `output_gm =
  asc2.global_tensor(output_ptr, [size])`) for an op the caller asked to run *in
  place* — that is out-of-place; it defeats the aliasing contract and adds GM
  traffic.
- **`copy_out` into the wrong handle** — `copy_in` from `a_gm` but `copy_out`
  into a second `output_gm` when only `a_ptr` was passed; the sum never lands in
  `a`.
- **Emitting the legacy `asc2.tensor` / `asc2.load` / `asc2.store` + numpy form**
  for a fork target test — the fork uses `global_tensor` / `copy_in` /
  `copy_out` + torch + `profiler`/`runs`. Match the fork API.
- **Building `expected` from a post-launch read of `a`** — snapshot with
  `expected = a + b` *before* the launch loop.
