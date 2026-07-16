# Reduction tiling selection (last-axis reduce)

Guidance for choosing the host-side tiling of a last-axis reduction
(`reduce_max`, `reduce_min`, `reduce_sum`, `reduce_mean`, `reduce_prod`) so the
kernel is **performant**, not merely correct. Correctness comes from padding with
the reduction identity (see the reductions section of `pyasc-target-operator`);
this file is about the tile *shape*.

## The quality metric: UB utilization

A last-axis reduction is memory-bound. The single biggest performance lever is
**UB utilization** — the fraction of the on-core Unified Buffer that the resident
working tiles actually occupy each iteration. Maximize it, subject to the
double-buffering constraint below.

Anti-pattern that this metric catches (the mistake to avoid): a fixed tiny tile
regardless of shape. A `tile_shape = [8, 8]` fp32 tile is 256 bytes against a
64 KB usable budget — about 0.4% utilization — and runs ~100x slower than the
hand-written CANN operator. If your `tile_rows`/`tile_cols` do not scale with the
shape and the UB budget, the tiling is wrong.

## UB budget numbers

- Physical UB on `Ascend950PR_9599` (C310) is `192 * 1024` bytes. The goldens use
  a conservative usable budget `UB_BUDGET_BYTES = 64 * 1024` because live UB is
  multiplied by double buffering and any non-liveness-merged sibling regions
  (see the "UB budgeting rule" in `SKILL.md`).
- CANN formula (from `concat_tiling_arch35.cpp`, `TilingUb`):
  `max_elems = ((UB_CAPACITY - 1024) // dtype.itemsize) // BUFFER_NUM // sibling_regions`
  with `BUFFER_NUM = 2` (double buffer). A single-region reduction has
  `sibling_regions = 1`, so per-buffer budget is roughly `UB_BUDGET_BYTES / 2`.

## The double-buffering constraint ("2 * (1+) iterations")

`asc2.range(..., parallel=True, unroll_factor=2)` double-buffers the loop, but the
overlap only materializes if the loop actually runs **at least `2 * unroll_factor`
iterations**. If a tile is grown so large the loop degenerates to a single
iteration, there is nothing to overlap and the prologue/epilogue dominate.

Rule: grow the tile to fill UB, but stop growing once the surviving loop drops
below `2 * unroll_factor` iterations; keep enough iterations for the pipeline to
fill.

## Step 1 — flatten and simplify

Flatten the input to a 2-D `[R, C]` where `C` is the last-axis reduce width and
`R = prod(shape[:-1])`. First simplify:

- **Reshape short-circuit.** Drop size-1 dims before flattening (e.g.
  `[1500, 1, 61] -> [1500, 61]`). If `C == 1` the reduction is the identity —
  emit a copy/reshape, not a vector reduce.

## Step 2 — pick the regime by C

Let `align = 32 // dtype.itemsize` (fp32 -> 8), `C_aligned = ceildiv(C, align) * align`,
and `per_buffer_elems = (UB_BUDGET_BYTES // dtype.itemsize) // BUFFER_NUM`.

### Small C (narrow rows) — the common regime

When a whole row fits comfortably (`C_aligned <= per_buffer_elems`), reduce a
**block of many rows at once** rather than one narrow row:

- `tile_cols = C_aligned` (the whole row width).
- `tile_rows = max(align, floor(per_buffer_elems / tile_cols))`, i.e. pack as many
  rows as the per-buffer budget allows, so the `[tile_rows, C]` tile fills UB.
- One `asc2.reduce_max(tile, 1)` reduces `tile_rows` rows in a single wide vector
  op; spread `R` across cores (`row_per_core`).
- Preserve double buffering: ensure `ceildiv(rows_per_core, tile_rows) >= 2 * unroll_factor`.
  If not, shrink `tile_rows` (trade a little UB fill for pipeline overlap).

This replaces the degenerate `[8, 8]` tile with a `[tile_rows, C]` block sized to
the budget.

### Tiny C (C very small, about <= 16) — consider transpose

When `C` is tiny, a per-row reduce along axis 1 wastes vector lanes (only `C`
elements reduced per lane group). Evaluate a
**transpose -> compute -> transpose** alternative: transpose the `[tile_rows, C]`
block so the long `tile_rows` dimension is contiguous, then reduce, so the vector
engine works on wide contiguous data. Keep whichever variant measures faster;
row-packing (above) is the safe default when transpose does not help.

### Large C (row wider than the budget)

When `C_aligned > per_buffer_elems`, the row does not fit — tile the C axis:

- `tile_cols = align_down(per_buffer_elems, align)` (fill the per-buffer budget).
- `tile_rows` small (e.g. `align`), maintain a per-row max accumulator across
  column tiles, folding with `asc2.maximum(acc, part)`.
- Keep `>= 2 * unroll_factor` column iterations so the column loop double-buffers.

## Step 3 — host-side tiling selector

Compute the tile on the host from shape + dtype + budget rather than hard-coding
it. Sketch:

```python
def select_reduce_tiling(shape, itemsize, ub_budget=64 * 1024,
                         buffer_num=2, unroll_factor=2):
    dims = [d for d in shape if d != 1]           # reshape short-circuit
    R = 1
    for d in dims[:-1]:
        R *= d
    C = dims[-1] if dims else 1
    align = 32 // itemsize
    C_aligned = -(-C // align) * align            # ceildiv * align
    per_buffer = (ub_budget // itemsize) // buffer_num
    if C == 1:
        return "reshape", None                    # identity, no reduce
    if C_aligned <= per_buffer:                    # small-C: pack rows
        tile_rows = max(align, per_buffer // C_aligned)
        tile_cols = C_aligned
    else:                                          # large-C: tile columns
        tile_rows = align
        tile_cols = (per_buffer // align) * align
    return "reduce", (tile_rows, tile_cols)
```

Then cap `tile_rows` so the per-core row loop keeps `>= 2 * unroll_factor`
iterations for double buffering.

## Verify

Validate the **static** path with `pytest --compile-only` (worst case for UB),
then run `--backend Model` for numerics against `torch.amax(x, dim=-1)` (or the
matching torch reduction). UB overflow shows as
`RuntimeError: UB overflow: N available, M used` where `M` is a multiple of the
per-buffer tile bytes — that multiple tells you how many live buffers you have.
