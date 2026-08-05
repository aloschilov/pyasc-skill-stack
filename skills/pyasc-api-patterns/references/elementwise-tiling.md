# Elementwise tiling selection (map/zip ops)

Guidance for choosing the host-side tiling of a 1-D elementwise operator
(`reciprocal`, `addcdiv`, `add`/`vadd`, `mul`, `cast`, `where`, ...) so the kernel
is **performant**, not merely correct. Elementwise ops are purely memory-bound:
every element is read once, transformed, and written once, so the tiling only has
to (a) use every AI core and (b) keep each core streaming large contiguous tiles
through the UB with double buffering. There is no reduction identity to pad with,
so bounds are handled entirely in-kernel — see "No host padding" below.

> **API surface.** The canonical kernel below uses the fork target-test API
> (`asc2.global_tensor` / `asc2.copy_in` / `asc2.copy_out`). The v2-mainline tile
> API (`asc2.tensor` / `asc2.load` / `asc2.store`), used by most
> `capabilities.yaml` goldens, is equivalent for tiling purposes — the selector
> and the three levers below apply identically to both surfaces.

## The canonical kernel (mirror `test_vadd.py`)

```python
@asc2.jit(reuse_alloc=1)                       # no static_alloc; fork surface
def op(in_ptr, out_ptr, input_length, tile_length: asc2.ConstExpr,
       unroll_factor: asc2.ConstExpr):
    x = asc2.global_tensor(in_ptr, [input_length])
    z = asc2.global_tensor(out_ptr, [input_length])
    # Uniform block loop derived in-kernel; every core runs the same count.
    block_loop_num = asc2.ceildiv(asc2.ceildiv(input_length, asc2.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = asc2.block_idx() * block_length
    for i in asc2.range(block_loop_num, unroll_factor=unroll_factor):
        off = block_offset + i * tile_length
        xt = asc2.copy_in(x, [off], [tile_length])
        asc2.copy_out(f(xt), z, [off])
```

- `@asc2.jit(reuse_alloc=1)` only. `static_alloc` defaults to `True` on C310 so it
  is dropped; keep `reuse_alloc=1` (its default is `0`).
- Overlap via `unroll_factor` (typically `2`). This fork target-test surface
  renamed `asc2.range`'s `parallel=` to `gm_barrier` (inverted: `gm_barrier=True`
  *disables* overlap, default `False`), so on this surface pass only
  `unroll_factor` (do not pass `parallel=`). On v2 mainline the equivalent is
  `asc2.range(..., unroll_factor=2, parallel=True)` — `parallel=` is **not**
  removed there; it is the software-pipelining flag the `capabilities.yaml`
  goldens use.

## No host padding, no tail branch

Do **not** pad the input on the host and do **not** special-case the last block
(`if block_idx() == block_num()-1: ...`). The runtime is bounds-safe:

- `copy_in` reading past the source extent auto-pads the overflow;
- `copy_out` clamps writes to the declared global-tensor shape.

So `block_length * block_num` may exceed `input_length` — the trailing over-read /
over-write is handled by the framework, exactly as `test_vadd.py` relies on. This
directly addresses the reviewer request to remove host padding + tail branches.

## The selector

```python
_UB_BUDGET_BYTES = 192 * 1024
_UB_RESERVE_BYTES = 1024
_CORE_NUM = 72
_MIN_TILE_ELEMS = 128
_TILES_PER_CORE = 2

def _select_elementwise_tile(shape, itemsize, live_tensors, unroll_factor=2, ...):
    length = math.prod(shape)
    align = 32 // itemsize
    # Lever 1: largest tile that fits UB with double buffering, sized against the
    # number of tiles live at once = live_tensors * unroll_factor.
    per_buffer = (ub_budget - reserve) // itemsize // (live_tensors * unroll_factor)
    ub_tile = max(align, (per_buffer // align) * align)
    # Lever 2/3: ~tiles_per_core tiles per core across the full grid, floored to a
    # useful size and capped by the UB tile and the length.
    per_core = -(-length // core_num)
    tile = -(-per_core // tiles_per_core)
    tile = -(-tile // align) * align
    tile = max(min_tile, min(tile, ub_tile))
    tile = max(align, min(tile, -(-length // align) * align))
    block_num = min(core_num, -(-length // tile))
    return (length, tile, block_num, unroll_factor)
```

Three levers, in order of impact:

1. **Use every AI core.** `block_num = min(72, ceildiv(length, tile_length))` — big
   tensors spread across the full grid, tiny ones collapse to a few cores so they
   are not dominated by launch overhead.
2. **Size the tile to the UB budget.** The largest tile that fits UB with double
   buffering is `per_buffer = (UB - reserve) / itemsize / (live_tensors *
   unroll_factor)`. `live_tensors` counts the tiles simultaneously resident:
   `reciprocal` = 2 (in + out), `addcdiv` = 4 (three inputs + output). Undercount
   and the four ping-pong tiles overflow UB; overcount and tiles shrink needlessly.
3. **Aim for ~2 tiles per core.** With `unroll_factor=2`, two tiles per core give
   the pipeline something to overlap. A `_MIN_TILE_ELEMS` floor (128) stops tiny
   tensors from being sprayed across 72 cores in 8-element slivers.

## dtype notes

- `asc2.div` (used by `reciprocal` as `div(1.0, x)` and by `addcdiv` as `x1 / x2`)
  supports `int16/int32/int64/float16/float32` but **not** `bfloat16`. Cover fp16;
  skip bf16 for division-based ops.
- Division amplifies low-precision error and can overflow the fp16 range when the
  divisor is near zero, diverging from the CPU golden. In fp16/bf16 tests bound the
  divisor (`|x| >= 1`) and widen `atol/rtol` (fp16 ~4e-3).

## Anti-patterns this replaces

A fixed small core count (8/16/32) with tiny fixed tiles (128/2048) — the original
`reciprocal`/`addcdiv` tables — leaves most of the grid idle and streams
UB-sized fragments, landing well below CANN on the large shapes. Scale `block_num`
and `tile_length` with the shape and the UB budget instead.
