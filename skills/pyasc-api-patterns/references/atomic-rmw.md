# Atomic read-modify-write into shared GM (multi-core reduce-to-GM)

Guidance for writing a **multi-core atomic RMW** kernel — one where several
cores atomically read-modify-write a **shared** global-memory buffer so their
overlapping writes combine deterministically. The canonical example is a
multi-core **atomic-add reduction**:

    out[j] = sum_i in[i, j]      (i over the cores, j over the shared width N)

Each of `CORE_NUM` blocks owns one row of a `[CORE_NUM, N]` input, `copy_in`s
its `[N]` slice, and `asc2.atomic_add`s it into the **same** `[N]` output at
`offsets=[0]`. The hardware serialises the colliding adds, so the shared buffer
ends up holding the column sum. This is the M3 "Atomic" family primitive and
the building block for **scatter-add / histogram / segment-sum** and any
cross-core reduce-to-GM.

> **API note.** This reference uses the **fork target-test API**
> (`asc2.global_tensor` / `asc2.copy_in` / `asc2.atomic_add`, `torch` tensors,
> the `profiler` / `runs` fixtures). It is the API the pyasc fork's
> `python/test/asc2/target/*.py` and `python/test/asc2/operations/test_atomic_ops.py`
> tests are written in. Do **not** emit the legacy `asc2.tensor` / `asc2.load`
> / `asc2.store` + numpy form for a fork target test — the reviewer will reject
> it.

## The op

`asc2.atomic_add(src, dst, offsets=[...])` performs an atomic
read-modify-write: each element of the UB tile `src` is atomically added to
`dst[offsets + ...]` in global memory. Defined in the fork
`python/asc/language/tile/atomic_ops.py` (exported via `asc2`).

- `src` — a **UB `LocalTensor`**, i.e. the result of an `asc2.copy_in` (or any
  tile computed on the vector pipeline). It must live in UB.
- `dst` — a **`GlobalTensor`** (`asc2.global_tensor(ptr, shape)`).
- `offsets` — the destination offsets, **rank-consistent with `dst`** (one
  entry per `dst` dimension). For a 1-D `[N]` destination use `offsets=[0]`.
- **Supported dtypes:** `int16`, `int32`, `float16`, `bfloat16`, `float32`.
  `src.dtype` must equal `dst.dtype`.
- Siblings `asc2.atomic_max` / `asc2.atomic_min` have identical signatures and
  the same dtype set — only the RMW op differs.

## The rule

- Wrap the shared destination pointer **once** with `asc2.global_tensor(out_ptr,
  [N])` and have **every** core `atomic_add` into it at the **same** overlapping
  `offsets` — the overlap is intentional; it is what produces the reduction.
- `src` must be a UB tile: `copy_in` the per-core slice first, then
  `atomic_add`. Do not pass a `GlobalTensor` as `src`.
- Keep **ranks consistent**: a 1-D input tensor → 1-D `copy_in` shape → 1-D
  `copy_in` offsets, and a 1-D `[N]` dst → 1-D `offsets=[0]`. Never mix ranks
  (v2's strict rank check rejects a 2-D dst with a 1-D `offsets`, etc.).
- **Host MUST zero-init the shared destination** before launch — see below.

## Multi-core atomic-add-into-shared-GM kernel

`in_ptr` packs `[CORE_NUM, N]` flat as `[CORE_NUM * N]`; core `block_idx()`
owns the slice `[block_idx()*N : (block_idx()+1)*N]` and accumulates it into the
single shared `[N]` output:

```python
@asc2.jit(always_compile=True)
def atomic_add_kernel(in_ptr: asc2.GlobalAddress, out_ptr: asc2.GlobalAddress,
                      in_length: asc2.ConstExpr, tile_length: asc2.ConstExpr):
    in_gm = asc2.global_tensor(in_ptr, [in_length])     # packed [CORE_NUM * N]
    out_gm = asc2.global_tensor(out_ptr, [tile_length]) # SHARED [N] destination
    offset = asc2.block_idx() * tile_length
    src = asc2.copy_in(in_gm, [offset], [tile_length])  # this core's [N] slice -> UB
    asc2.atomic_add(src, out_gm, offsets=[0])           # RMW into the SAME [N] region
```

- No per-tile reduce and no `for` loop over tiles for the basic demonstrator —
  each core contributes its whole `[N]` slice in one atomic RMW. (Tile the slice
  with an `asc2.range(..., unroll_factor=2)` loop only when `N` exceeds the UB
  budget; each iteration atomic_adds a disjoint `[tile]` chunk into the matching
  `offsets=[chunk_off]`.)
- **Do not** allocate a per-core private output and sum afterwards — that defeats
  the point of the atomic. The shared buffer + atomic RMW *is* the reduction.

## Host / target test — zero-init the shared output

```python
def atomic_add_launch(x):                 # x: [CORE_NUM, N] torch tensor
    in_flat = x.reshape(-1).contiguous()
    out = torch.zeros([N], dtype=x.dtype) # MANDATORY zero-init (atomic accumulates)
    atomic_add_kernel[CORE_NUM](in_flat, out, CORE_NUM * N, N)
    return out

# Reference: column-wise sum across cores.
expected = x.reshape(CORE_NUM, N).sum(0)
torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
```

- **MANDATORY host zero-init.** `atomic_add` *accumulates* into `dst`, so the
  destination must start at zero. Passing a dirty / uninitialised buffer adds
  its prior contents to the result. This is the single most common atomic bug.
  (For `atomic_max` seed with `-inf`; for `atomic_min` seed with `+inf` — the
  op's identity, not zero.)
- **Torch tensors on C310.** The `Ascend950PR_9599` (C310) simulator path
  expects `torch` tensors; numpy is silently zeroed on this path (same property
  as matmul / rms_norm). Verify with `torch.testing.assert_close`.
- **`--profile --runs N` caution.** Like the in-place add loop, N launches into
  the *same* buffer accumulate. Either re-zero the output between profiled
  launches (a host copy — corrupts timing) or reconstruct the reference as
  `runs * expected` from a single zero-init. The honest pattern mirrors
  [in-place-aliasing.md](in-place-aliasing.md): snapshot the invariant before
  the loop and reconstruct the accumulation, do not re-seed inside the timed
  loop.

## Generalisation

| Op | `asc2` call | Identity (host seed) | torch reference |
|---|---|---|---|
| atomic-add | `asc2.atomic_add(src, dst, offsets)` | `0` | `x.sum(0)` |
| atomic-max | `asc2.atomic_max(src, dst, offsets)` | `-inf` | `x.amax(0)` |
| atomic-min | `asc2.atomic_min(src, dst, offsets)` | `+inf` | `x.amin(0)` |

The **overlapping-`offsets`** shape above is the general template for the whole
atomic family:

- **scatter-add** — instead of every core writing `offsets=[0]`, each core (or
  each row) writes a *data-dependent* destination offset `offsets=[idx]` loaded
  from an index tensor; colliding indices accumulate atomically. `out[idx[k]] +=
  src[k]`.
- **histogram** — scatter-add of a tile of ones (or counts) into per-bin
  offsets; the bin index is the destination offset.
- **segment-sum** — scatter-add where the offset is a segment id, so each
  segment's rows accumulate into one output slot.

All four are the same primitive: `copy_in` a UB tile, then `atomic_*` it into a
shared GM destination at a (possibly runtime, possibly overlapping) offset, with
the destination host-seeded to the op's identity.

## Performance — contention floor + the pre-reduction lever

A multi-core atomic-add-into-**one** shared region is **serialisation-bound, not
bandwidth-bound**: every colliding `atomic_add` into `out[0:N]` is serialised by
the hardware, so latency grows with the **number of atomic writes**, not just the
bytes moved. The naive one-`atomic_add`-per-row kernel above therefore issues
`num_rows` serialised RMWs into the shared region.

Measured on the real NPU (`Ascend950PR_9599`, `--profile --runs 10`, single-shot
skills-on generated `test_atomic_add.py`, one `atomic_add` per row):

| shape `[num_rows, width]` | median latency |
|---|---|
| `[16, 4096]`  | 1.82 μs |
| `[32, 4096]`  | 1.94 μs |
| `[64, 4096]`  | 2.81 μs |
| `[128, 4096]` | 5.34 μs |
| `[256, 4096]` | 12.79 μs |
| `[512, 4096]` | 27.37 μs |
| `[16, 8192]`  | 2.29 μs |
| `[64, 8192]`  | 3.94 μs |

Latency is roughly flat while `num_rows <= core count` (the atomics run one wave,
different cores, little collision) and then grows **~linearly** with `num_rows`
once each core owns several rows and all cores hammer the same `[N]` region
(`128→256→512` rows ≈ `5.3→12.8→27.4` μs). That linear growth is the **honest
atomic-contention floor** of reducing into a single shared destination — it is
intrinsic to the RMW-into-one-region semantics, not a codegen miss.

**The perf lever: per-core partial pre-reduction (fewer atomics).** When a core
owns several rows, sum them in a **UB accumulator first** and issue **one**
`atomic_add` per core instead of one per row — collapsing the serialised RMWs
from `num_rows` down to `min(num_rows, core_count)`:

```python
# Launch with block_num = min(num_rows, core_count) so every core owns >= 1 row.
for r in asc2.range(asc2.block_idx(), num_rows, asc2.block_num(), unroll_factor=2):
    tile = asc2.copy_in(in_gm, [r * width], [width])
    acc = tile if <first owned row> else acc + tile   # sum this core's rows in UB
# one atomic per core, not per row:
asc2.atomic_add(acc, out_gm, offsets=[0])
```

The carried UB accumulator (`acc = acc + tile`) is a genuine loop-carried
dependency, so the accumulation loop sets `gm_barrier=True` (overlap must be
disabled — the default `gm_barrier=False` would let iterations overlap and
corrupt the running accumulator), and the UB
vector adds double as the copy_in→atomic sync (no `+bias` trick needed). This is
the first optimisation to reach for when the heavy-`num_rows` shapes dominate;
it trades `num_rows` serialised GM atomics for `num_rows` UB vector adds + one
atomic per core. For a genuine scatter (data-dependent, non-overlapping offsets)
the collision is lower and the naive one-atomic-per-tile form is already close to
the floor.

## Anti-patterns

- **Forgetting the host zero-init** (or seeding max/min with `0` instead of
  `-inf`/`+inf`) — the destination's prior contents leak into the result. Always
  seed the shared buffer to the op's identity.
- **Passing a `GlobalTensor` as `src`** — `src` must be a UB `LocalTensor`
  (`copy_in` first). The op raises if `src` is not in UB.
- **Rank-mismatched `offsets`** — a 1-D `[N]` dst needs a 1-D `offsets=[0]`; a
  2-D dst needs 2-D offsets. Do not pair a 2-D `global_tensor` with a scalar
  `offsets`.
- **Emitting the legacy `asc2.tensor` / `asc2.load` / `asc2.store` + numpy form**
  for a fork target test — the fork uses `global_tensor` / `copy_in` /
  `atomic_add` + torch + `profiler`/`runs`. Match the fork API.
- **Per-core private outputs + a host-side sum** — that is not the atomic
  pattern; the whole point is the shared destination and hardware-serialised
  RMW.
