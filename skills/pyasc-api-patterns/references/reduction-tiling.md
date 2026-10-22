# Reduction tiling: preserve semantics, then measure geometry

For last-axis reduce_max/min/sum/mean/prod, choose geometry using the
[top-down tuning reference](cann-performance-tuning.md). Reductions are not
universally memory-bound: short rows can be dominated by DMA/VF setup, long
rows by accumulation dependencies, and small row counts by available work.

## Preserve the logical reduction axis

Flatten only leading dimensions: R=prod(shape[:-1]), C=shape[-1].
Do not remove all singleton dimensions first: [32, 1] reduces 32 independent
one-element rows, not one 32-element row. Keep output shape/keepdim/dtype and
empty-reduction identities/errors from the task golden. A C=1 mathematical
identity may still need a cast or output copy; it is not permission to return
an input alias or skip required dtype conversion.

This host-only shape helper performs no tiling, allocation or launch:

```python
from math import prod


def last_axis_geometry(shape):
    dims = tuple(shape)
    if not dims or any(type(d) is not int or d < 0 for d in dims):
        raise ValueError("expected a nonempty shape of nonnegative integers")
    return prod(dims[:-1]), dims[-1]
```

For a general axis, normalize it before deriving [outer, axis_size, inner].
Do not reinterpret a non-last-axis reduction as a contiguous last-axis one.

## Work distribution

Query available vector resources for the active device, respecting applicable
limits. Choose no more useful row groups than the work supports, unless a
separately qualified column-split algorithm supplies more parallel work.
Do not hard-code 72/64/32 blocks or infer resources from a SoC suffix.
A labeled one-block fallback is compatible, not evidence of tuned performance.

Compare contiguous balanced row ranges and cyclic row tiles. Verify coverage,
non-overlapping stores, tails, idle blocks and address integer range.
Record min/max rows and tiles per block; raw block count alone does not show
load balance. Many narrow rows often benefit from batching; a few wide rows
can legitimately use one row per iteration.

## Physical versus logical layout

For supported 2-D copies use physical [tile_rows, aligned_cols], logical
real_shape=[valid_rows, C], and the operation's identity padding. The local
final dimension must satisfy the pinned copy API's alignment constraints.
For max/min use -inf/+inf where supported; for sum use 0, product 1. For mean,
divide by the logical count, not padded width.

A flat contiguous load followed by a reshape is an alternative only after
proving the pinned API supports the physical layout and masks. Do not emit
an unaligned [tile_rows, C] copy merely because the total byte count is aligned.
Padding in UB is not automatically extra GM traffic; inspect emitted copies
and actual byte counters before quantifying bandwidth waste.

Test tiny C (including 1, 4 and non-aligned widths), aligned rows, and odd tails.
For broadcast/compare/select outputs retain the destination-size hazard in
[the JIT reference](api-jit-options.md); extending a tensor's lifetime by
hoisting can make a latent alias/overrun problem destructive.

## UB budget and candidate search

Use the compiler's measured peak allocation at the exact pin, dtype,
tile shape and effective options. Include input/output casts, reduction
scratch, alignment, resident gamma/scale data, live intermediates and actual
unrolled buffers. The local 950PR compiler budget of 253952 B is not a
universal hardware capacity. UB divided by two buffers is an initial estimate,
not proof of fit; a concat-specific formula is not a reduction memory model.

- Whole row fits: compare one row with batches of independent rows. Bound
  batch size by useful rows per block and measured UB. Larger DMA can amortize
  setup but reduce concurrency or leave a costly tail.
- Row exceeds budget: tile columns and maintain correctly typed partial
  reductions. Seed from a supported reduction/local value, not a Python float
  later reassigned to a PlainValue. Use correct merge math: max/min, sum/product,
  and mean's total logical divisor differ.
- Few huge rows: cross-core partials may help, but count workspace,
  synchronization and the second reduction kernel; revalidate accumulation
  order, exceptional values and task determinism.
- Shared gamma/broadcast input: compare reuse across rows against the resident
  allocation and longer lifetime. Do not assume hoisting is free.

## Unrolling and buffering are distinct

Compare unroll=1/2 at matched geometry and effective JIT settings.
AscTile unrolling does not guarantee distinct ping-pong storage or overlapping
MTE2/V execution. There is no universally proven minimum iteration count of
2*unroll. Inspect generated buffers, dependencies and timestamped intervals.
A single large tile can win by DMA efficiency or lose by eliminating overlap;
keep both as measured alternatives instead of always filling UB.

For long dependent sums, compare independent row chains with partial/balanced
sums as separate experiments. Do not stack them assuming additive speedups.
FP32 accumulation, pairwise summation and rsqrt refinement alter numerical
behavior; validate them against the actual normalization/quantization contract.

## Qualification and reporting

Compile every selected specialization before simulation. Reject UB overflow
or unknown allocation, then execute logical outputs against the golden:
singleton and empty axes per contract, unaligned rows, tails, NaN/Inf, repeated
execution, multicore partitioning and guard regions. Static inspection leaves
undemonstrated synchronization cases unknown.

Record tile_rows/physical_cols/logical_cols, launched/useful/available blocks,
rows and tiles per block, JIT settings, UB, numerical scope and repeated timing.
Confirm observed bottleneck changes in traces, then use authorized hardware
runs to establish speed. None of these rules promises parity with CANN.
