# Reduction tiling selection (last-axis reduce)

Guidance for choosing the host-side tiling of a last-axis reduction
(`reduce_max`, `reduce_min`, `reduce_sum`, `reduce_mean`, `reduce_prod`) so the
kernel is **performant**, not merely correct. Correctness comes from padding with
the reduction identity (see the reductions section of `pyasc-target-operator`);
this file is about the tile *shape* and how the row block is loaded.

## Three levers that decide reduction performance

A last-axis reduction is memory-bound. Against the hand-written CANN operator the
outcome is set by three levers, in order of impact:

1. **Use every AI core.** Spread the `R` rows across the full core grid
   (`core_num = platform core count`, e.g. `72` on `Ascend950PR_9599`), not a
   handful. Under-parallelizing to 4-8 cores is a 9-18x deficit on the large-`R`
   shapes and is the single biggest mistake. `rows_per_block = ceildiv(R, core_num)`
   is the **primary** input to tile sizing.
2. **Keep the reduce axis contiguous and unpadded — `tile_cols = C`.** Load a
   `[tile_rows, C]` block as one contiguous run of `tile_rows * C` elements and pad
   only the flat total to 32 B, **never per-row**. Do *not* align `tile_cols` up to
   32 B: when `C` is not a multiple of `32/itemsize` that pads every row
   (`C=4 -> 8` doubles traffic; `10 -> 16`; `18 -> 24`), and since the kernel is
   memory-bound the wasted bytes map straight to the ratio.
3. **Size `tile_rows` to the per-core block, then the UB budget.** Clear a core's
   `rows_per_block` rows in a few large tiles; the tile only has to fit UB. Do
   **not** cap `tile_rows` to multiples of 8 and do **not** shrink it to manufacture
   double-buffer iterations — contiguity and core count matter more than pipeline
   overlap here.

The anti-pattern all three catch: a fixed tiny tile (`[8, 8]` = 256 B, ~0.4% of
UB, ~100x slower than CANN) run on a few cores. If `tile_rows`, `tile_cols`, or the
core count do not scale with the shape, the tiling is wrong. Reference point: the
degenerate `[8, 8]` tile on 4-8 cores lands at geomean ~0.01-0.43 of CANN; the
three levers above bring a last-axis reduce to ~1.0 (parity).

### The most common miss: one row per iteration

Do **not** reduce one row per loop iteration (the `reduce_sum` `[32, 4096]`
teaching example does this because it has only 32 wide rows). Copying that pattern
here is the mistake:

```python
# WRONG for large R: thousands of tiny [1, C] loads, memory-latency bound, loses to CANN
for i in asc2.range(asc2.block_idx(), num_rows, asc2.block_num(), ...):
    row = asc2.load(x_gm, [1, C], offsets=[i, 0])   # one row -> one reduce
    m = asc2.reduce_max(row)
```

```python
# RIGHT: pack many rows into one contiguous [tile_rows, C] tile, reduce axis 1
for t in asc2.range(row_iters_per_block, unroll_factor=2):
    tile = asc2.load(x_gm, [tile_rows, C], offsets=[row0 + t * tile_rows, 0])  # tile_rows rows
    out = asc2.reduce_max(tile, 1)                                             # [tile_rows] in one op
```

Most target shapes have `R` in the thousands-to-hundreds-of-thousands. With one
row per iteration each core issues thousands of tiny `[1, C]` DMAs; the win is
issuing a few large `[tile_rows, C]` contiguous DMAs instead.

Also **do not hard-code the core count** to the `reduce_sum` example's `16` (or
any literal). Launch on the platform's full core count (`72` on
`Ascend950PR_9599`) so `R` is spread across every core; treat the example's number
as illustrative, not a target.

## UB budget numbers

- Physical UB on `Ascend950PR_9599` (C310) is `192 * 1024` bytes. Size tiles
  against the **real** budget: with double buffering the per-buffer budget is
  `per_buffer = (UB_PHYS - reserve) // itemsize // BUFFER_NUM` with
  `BUFFER_NUM = 2` and a small `reserve` (~1 KB). A conservative `64 * 1024`
  budget leaves most of UB idle and yields tiles that are too small — prefer the
  physical budget and only back off if `UB overflow` fires.
- CANN formula (from `concat_tiling_arch35.cpp`, `TilingUb`):
  `max_elems = ((UB_CAPACITY - 1024) // dtype.itemsize) // BUFFER_NUM // sibling_regions`.
  A single-region reduction has `sibling_regions = 1`.

## Double buffering (secondary)

`asc2.range(..., unroll_factor=2)` double-buffers the row loop, and
the overlap only materializes with `>= 2 * unroll_factor` iterations. For a
reduction this is a **secondary** concern: with all cores active each core owns few
rows, so 1-5 large contiguous tiles per block already beat a many-small-tile
schedule. Keep double buffering when it is free (the block naturally splits into
several tiles), but never shrink a tile below the UB fill just to add iterations.

## Step 1 — flatten and simplify

Flatten the input to a 2-D `[R, C]` where `C` is the last-axis reduce width and
`R = prod(shape[:-1])`. First simplify:

- **Reshape short-circuit.** Drop size-1 dims before flattening (e.g.
  `[1500, 1, 61] -> [1500, 61]`). If `C == 1` the reduction is the identity —
  emit a copy/reshape, not a vector reduce.

## Step 2 — pick the regime by C

Let `align = 32 // dtype.itemsize` (fp32 -> 8) and
`per_buffer = (UB_PHYS - reserve) // dtype.itemsize // BUFFER_NUM`.

### Small C (whole row fits) — the common regime

When a whole row fits (`C <= per_buffer`), reduce a **block of many rows at once**:

- `tile_cols = C` (the true row width, unpadded — this keeps the block contiguous).
- `rows_per_block = ceildiv(R, core_num)` across all cores.
- `ub_cap = per_buffer // C` (max rows per tile that fit one buffer).
- `tile_rows = min(ub_cap, rows_per_block)` — one tile per core when it fits;
  otherwise split the block into `ceildiv(rows_per_block, ub_cap)` even, large
  tiles.
- Load `[tile_rows, C]` as one contiguous buffer (flat total padded to 32 B), then
  a single `asc2.reduce_max(tile, 1)` reduces `tile_rows` rows in one wide op.

### Tiny C (e.g. 4, 8) — still row-pack, do not pad

The win is packing many rows into one contiguous buffer, exactly as above with
`tile_cols = C`. A transpose -> reduce -> transpose variant is rarely worth it once
the load is already contiguous; measure before adding one.

### Large C (row wider than the budget)

When `C > per_buffer`, the row does not fit — tile the C axis:

- `tile_cols = align_down(per_buffer, align)` (fill the per-buffer budget).
- `tile_rows` small; maintain a per-row max accumulator across column tiles,
  folding with `asc2.maximum(acc, part)`.
- Keep `>= 2 * unroll_factor` column iterations so the column loop double-buffers.

## Step 3 — host-side tiling selector

Compute the tile on the host from shape + dtype + budget + **core count**:

```python
def select_reduce_tiling(shape, itemsize, core_num,
                         ub_phys=192 * 1024, reserve=1024, buffer_num=2):
    dims = [d for d in shape if d != 1]           # reshape short-circuit
    R = 1
    for d in dims[:-1]:
        R *= d
    C = dims[-1] if dims else 1
    if C == 1:
        return "reshape", None                    # identity, no reduce
    align = 32 // itemsize
    per_buffer = (ub_phys - reserve) // itemsize // buffer_num
    rows_per_block = -(-R // core_num)            # ceildiv: spread across ALL cores
    if C <= per_buffer:                           # small-C: pack rows, tile_cols = C
        ub_cap = max(1, per_buffer // C)
        n_tiles = -(-rows_per_block // ub_cap)    # tiles needed to cover the block
        tile_rows = -(-rows_per_block // n_tiles) # even, large tiles
        tile_cols = C                             # unpadded -> contiguous block
    else:                                          # large-C: tile the column axis
        tile_rows = 1
        tile_cols = (per_buffer // align) * align
    return "reduce", (tile_rows, tile_cols)
```

Launch with `core_num` blocks and load each `[tile_rows, C]` tile as one
contiguous run (pad only the flat total to 32 B).

## Verify

Validate the **static** path with `pytest --compile-only` (worst case for UB),
then run `--backend Model` for numerics against `torch.amax(x, dim=-1)` (or the
matching torch reduction). UB overflow shows as
`RuntimeError: UB overflow: N available, M used`; back off `tile_rows` (or the UB
budget) if it fires. Confirm the reduce axis stays contiguous (`tile_cols == C`)
and that all cores are used (`core_num == platform core count`).
