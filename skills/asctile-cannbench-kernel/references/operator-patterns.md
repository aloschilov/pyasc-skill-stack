# CANNBench operator patterns

Use the task’s public signature and case attributes to select a pattern.

At the host/JIT boundary pass torch tensors directly. Do not call
`Tensor.data_ptr()`: the pinned JIT derives pointer dtype specialization from
the tensor, while a raw integer loses that contract. Host-side output casts or
slice assignments are also numerical device operations and are not an
acceptable substitute for a final asctile kernel.

## Common elementwise path

`sigmoid`, `exp`, `mish`, and `gelu` flatten arbitrary-rank contiguous input to
one dimension. Use grid-stride tiles, `real_shape=[n]` tails, f32 intermediates
for f16/bf16 where needed, and `cores = min(available_vector_blocks, num_tiles)`
using a qualified active-device resource query. Stable forms must
preserve NaN and Inf behavior from `golden.py`. GELU has both `none` and `tanh`
attributes and requires distinct kernels or tile widths.

On the archive-bundled evaluator import `from asc.experimental import asctile`.
Generate the kernel with `@asctile.jit(reuse_alloc=2)`; allocate with
`out = x.new_empty(x.shape)` on the caller's device and invoke
`kernel[vector_blocks(work_items)](x, out, ...)`. Read the
[ranked copy example](../../asctile-api-patterns/SKILL.md#surface-to-use)
before writing the body. The evaluator has no `asc.tile` module.

For Exp, specialize the public scalar attributes at compile time. Evaluate
`scale*x + shift` in the golden compute dtype; for a positive base multiply
by its natural logarithm before the exponential. Compute the scalar logarithm
once outside the JIT, at the precision of the public golden. Elide identity
multiplications and a zero shift. Do not fold a zero multiplier into a
constant: infinities and NaNs still participate. Use one contiguous copy in
and one copy out per tile. Bound the last transfer in the same launch.
Benchmark a direct one-tile body when the whole input fits one wave, and
a grid-stride pipelined loop when there are enough tiles for multiple waves.

### High-level GeLU campaign (historical)

Use `compiler-team/pyasc@adadd7d66ed0ee16d33d79487bf584899a26ef1e` — the
historical GeLU pin, not CANNBench runtime authority — and public
`asctile` types, JIT, math, copy operations, and range, as in the target tests.
The upstream GeLU test is a structural reference, not a complete CANNBench
implementation: its one FP32 shape and sigmoid-equivalent tanh approximation
do not cover both `approximate` modes, all dtypes, tails, or special values.

Explore concise public tile expressions: `asctile.erf` for exact-mode
semantics and `asctile.tanh` or a stable exponential identity for tanh mode.
For the latter, with `s = (x + x*x*x * 0.044715) * (2*sqrt(2/pi))`, the
cancellation-free form is `x * exp(minimum(s, 0)) / (exp(-abs(s)) + 1)`.
Precompute Python math constants outside JIT and use `asctile.float32`
intermediates where needed. Direct `1 + erf/tanh` can lose negative tails;
validate against the task's dtype-specific error and NaN/Inf rules rather
than treating a compact formula as proven. If the public exact-mode route
cannot meet those tolerances, record that measured blocker.

Do not author `inline`, `inline_vf`, low-level `asc`/`asc2` kernels, a custom
JIT/compiler adapter, or the long Numerical Recipes polynomial below for this
campaign. Compiler-generated VF/register code from public tile operations is
allowed. Establish default and `reuse_alloc=1` baselines, then explore the
18-combination [JIT matrix](../../asctile-api-patterns/references/api-jit-options.md)
with synchronization ON. Measure UB and numerical behavior at fixed geometry
before widening tiles; only real-NPU timings establish speed.

Measured local qualification at this pin (CANN 9.0 / 950PR_9599) rejects the
public-erf FP32 exact path on dense negative-tail inputs for **all 18 JIT
configurations**, despite successful compilation and smaller smoke tests.
Read the minimal reproduction and campaign report (`integrations/cannbench/comparisons/gelu-asctile-jit-20260907/README.md` on the `archive/campaign-2026` branch)
before promoting GeLU. Increasing tile size or changing allocation does not
repair this measured numerical failure. Do not substitute a polynomial or
silently relabel tanh as exact.

Use the actual Torch golden for special values: in this harness GeLU(-Inf)
is NaN in both modes, not the mathematical limiting value zero. Worker claims
about limits or precision must be checked against executed reference results.

### Historical GeLU recipes and measurements — not valid campaign guidance

The following NR/Horner and low-level routes are retained as historical
evidence at their original pins (for example `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`).
They belong to the historical high-level GeLU era; `0a631f70968c3cb7c33ce45330a85768dd5a6f06`
was the archived full-catalog campaign's runtime. The current run manifest,
not these historical notes, determines runtime authority.
Their successful cases do not qualify them for the new high-level campaign.
The historical 950PR-verified 20/20 recipe was:

- exact mode: set `z = abs(x) / sqrt(2)`, then construct the reciprocal with
  pyasc-valid operand order: `den = z*0.5 + 1.0`,
  `one = asctile.full([tile_size], 1.0, dtype=asc.float32)`, and
  `t = one/den`. Compute the
  Numerical Recipes fit exactly as `p=t*0.17087277-0.82215223`, followed by
  Horner updates with coefficients `+1.48851587`, `-1.13520398`,
  `+0.27886807`, `-0.18628806`, `+0.09678418`, `+0.37409196`,
  `+1.00002368`, `-1.26551223`; then `erfc=t*exp(p-z*z)`. Do not substitute
  a different erf approximation. Form the result without cancellation as
  `where(x >= 0, x - x*0.5*erfc, x*0.5*erfc)`;
- tanh mode: let `s = 2*sqrt(2/pi)*(x + 0.044715*x^3)`, set
  `den = exp(-abs(s)) + 1.0`, and compute `x * exp(min(s, 0)) / den`;
- perform both in FP32 and cast once at output. These formulas preserve the
  benchmark's NaN/Inf positions, unlike host special-casing, and avoid the
  negative-tail cancellation observed with direct `1 + erf/tanh` forms.

The exact Horner chain has a much larger static UB footprint than its source
suggests because every SSA update remains live during allocation. On the
pinned compiler, `tile_size=2048`, `unroll_factor=2` consumes roughly
796--820 KB and is rejected against the 253952-byte UB limit. The measured
safe historical baseline was `tile_size=512`, `unroll_factor=2` for exact mode.
Tanh used a separate kernel/tile; `tile_size=1024`, `unroll_factor=2` measured at
172--184 KB. Do not infer a larger exact tile solely from a source-level
temporary count; only the exact-runtime compile report establishes the UB budget.

`real_shape` limits DMA but not the vector instructions consuming the local
tile. Supply neutral padding explicitly: `pad_value=0` for additive operands,
`pad_value=1` for divisors, and a finite safe input for logarithm/reciprocal
chains. This prevents invalid arithmetic in inactive lanes without changing
the logical output.

The historical performance investigation used low-level `asc.adv.erfc` to
form exact GeLU without the Horner chain. A plain `@asc.jit` C310 kernel at
that revision had a hidden FFTS argument and failed in `c2c_ctrl_addr()` on
CANNBench. The GeLU deep-dive adapter changed only
kernel-argument legalization and verifies `has_ffts_arg=false`. Its FP32 and
BF16 exact routes passed the official cases that ran, but measured only 0.2968x
and 0.1462x respectively; it is not yet a >=1x performance pattern.

Avoid large `vf_fusion=True, reuse_alloc=1` tanh loops merely because they fit
the static UB budget. A 72-core, tile-13,824 GeLU route compiled at 221,184
bytes and then hit vector-core timeout 507034 on the 67M-element official case.

Do not replace the cancellation-free tanh route with
`0.5*x*(1+asc.adv.tanh(u))` based on source size alone. A native FP16 probe
lowers to much smaller AscendC, but CPU formula validation violates the
CANNBench MARE bound on the `[-10, 10]` case because the negative tail rounds
`1+tanh(u)` to zero. Native BF16 basic multiplication does not lower at all in
commit `0a631f70`, so retain FP32 promotion for BF16. The installed CANN API
has a fused `AscendC::Gelu`; PyASC/AscTile does not expose it, which should be filed
as a performance/API blocker rather than worked around with an inaccurate
formula.

### Shared tile constraints

Older snapshots required LocalTensor branches for `asctile.where`; the later
historical GeLU snapshot `adadd7d6` added numeric scalar operands. Verify the
actually supported operand kinds against the current run's pinned runtime
before relying on either form, and do not require
extra full tiles solely to reproduce an older API restriction. Alignment
and measured compare/select lowering limitations still need verification.

## StridedSlice

For a contiguous 1-D input with a positive stride `s`, a selected residue
can be read as one column of a logical GM tensor with `s` columns. For an
aligned start and complete rows, use a destination with one column. Tile
along the row dimension. For fp16 the local tile's last dimension needs 16
lanes (32 bytes), even when only one element in each row is valid. Set
`real_shape=[valid_rows, 1]` on both the padded `copy_in` and the
`copy_out`. This keeps adjacent output elements contiguous.

Case 01 uses zero start, stride two and no masks. The padded column copy
passed the official CaModel comparator on both seeds. A local tile shaped
`[rows, 1]` failed lowering; reshaping contiguous pairs and transposing them
failed numerical comparison. Recompute the offsets and bounds for other
slices. Negative strides and masked axes need their own checks.

## Gather

Check the indexing formula in the task. For 2-D `dim=0` elementwise gather,
each index selects a row **at its own column**. The pinned AscTile
`gather` takes a rank-one local index vector and copies trailing subtensors
from GM. For a flattened `[input_numel, 1]` source the call is
`asctile.gather(src_gm, [0], 0, flat_indices)`. The second argument is a
list of base offsets through the selected dimension; the fourth is the local
index vector. Pass `num_indices=valid` when only a prefix of the tile is valid.
The original 2-D index cannot be passed directly. A scalar GM read/write loop
did not finish the first seed in 600 seconds on the original case 01 shapes.

One device-only route transposes the index into a scratch tensor in square
tiles. For column `c`, load a contiguous vector of row indices and compute
`row * input_width + c` in the tile. View the input as GM
`[input_numel, 1]` and gather along dimension zero, retaining a static
trailing dimension. Copy the valid column into the output at its original
row and column. Preserve the declared integer index dtype and account for
padding when choosing a tile size. This route matched Torch on a reduced
fp16/int32 case with two real kernel calls per seed. Full-size checks and
runtime costs must be recorded separately.

## MaskedScale

Cover every cross-product present in `cases.yaml`: x is f16/bf16/f32; mask is
int8/uint8/f16/bf16/f32. Direct vector arithmetic on int8 is unsupported in the
observed runtime; cast int8 to f16 and then f32. Torch uint8 can be reinterpreted
as int8 without copying, but values 128..255 then require an in-kernel f32 +256
correction. Avoid f16 and bf16 temporaries coexisting in one specialization;
use a two-kernel f32 bridge for cross-half combinations.

## SwiGLU

Split the selected even-sized dimension and compute `silu(x0) * x1`. Map the
contiguous N-D tensor to `[outer, 2 * half_cols]` without materializing aligned
halves. Two-dimensional copy operations require a 32-byte-aligned final tile
dimension. Degenerate `half_cols` therefore needs a separately evidenced path.
The first reference smoke passed 18/20: case 9 failed around a host
narrow/contiguous copy and case 12 had a bf16 NaN-position mismatch. A repaired
reference using direct padded 2D PyASC loads/stores passed all 20 cases on
2026-11-02 at runtime `030e9b2c`; see the [recorded qualification](../../../docs/hardware-lane.md).
This supports that memory-access pattern for that runtime, not arbitrary
generated candidates. The passing repair does not isolate the original NaN
mismatch's cause. Avoid host copy fallbacks; independently qualify short rows,
tails and NaN/Inf positions for every new candidate.

For SwiGLU, normalize the split axis first. When the contiguous BF16 input
is split on its final axis of length two, each row is a neighbor pair
`(gate, value)`. Select the implementation for the active build using the
[BF16 pair pattern](../../asctile-api-patterns/references/packed-bf16-pairs.md).
On the qualified archive-bundled public 950PR build, admit these pairs into
the existing short-half reindex route before any transpose shortcut. Derive
tile and unroll from the slab geometry. Confirm the actual kernel, grid,
bounded final copy and compiled UB allocation.
For an already validated contiguous
tensor the guard is `dtype == torch.bfloat16 and d == rank - 1 and shape[d] == 2`.
Here `d` is the normalized **axis index**, not its extent. Verify the launched
kernel on a rank-two [R, 2] input; golden values alone also pass on the old
row route. A last dimension of 14, for example,
contains two adjacent **halves** of length seven, not seven interleaved pairs.
Admit wider short BF16 halves to the same slab eligibility check. Use the
UB-sized row path when that check declines the gather. Dispatch from layout and dtype,
never the case ID or a particular outer shape.

For other contiguous splits, express the input as `[R, 2H]`, where
`R = prod(shape[:d])` and `H = (shape[d] // 2) * prod(shape[d+1:])`.
Choose the DMA layout from the half-row byte count and available independent
tiles. The [950PR archive replay](../../../docs/evidence/swiglu-vector-grid-20261110/README.md#register-swiglu-and-archived-submission-replay)
exercises three useful candidates:

- For very short halves, load a contiguous slab of complete rows. On pins
  supporting public `asctile.reindex`, compare two local gathers with
  padded strided copies. For output lane `r * H + h`, the two indices are
  `r * 2 * H + h` and `r * 2 * H + H + h`. They also work for H greater
  than one. The archive loads two reusable int32 index tiles per block.
  Include their transfers and UB allocation, and use an index-construction
  method allowed by the candidate contract. On the CANNBench 950PR image,
  SwiGLU case 9 stopped before its gather kernel because device-side
  `torch.arange` required a missing `aclnnArange` binary. For a shape-only
  address table, construct the two int32 arrays on the host, upload them
  once per layout and device, and keep all input-dependent math in the
  generated kernel. With the same generated source, this host change let the
  H=7 slab run in 14.77 µs and all 20 cases pass.
  [CI-generated source and service measurements](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-swiglu-performance-43061369765).
  For H=1, compare this public
  route with the register specialization when native code is enabled.
  For this slab route, derive height and loop unroll from the input DMA
  bytes, compiled UB footprint and useful tiles per AIV. Apply the
  [reindex slab geometry](../../asctile-api-patterns/references/reindex-slabs.md)
  for the matching build.
- When several half-rows fit in UB, batch them into a two-dimensional tile
  with a 32-byte-aligned physical width. Bound both the row count and the
  logical width on reads and stores. The archive targets roughly 8–16 KiB
  per half-tile for short rows. For the public 950PR kernel, choose height
  from the logical transfer size and choose unroll from the padded UB
  footprint of the emitted arithmetic and work per AIV. For FP16, widen
  the halves, divide the gate directly by its sigmoid denominator and use
  the qualified FP16 coefficient in the [2D row geometry](../../asctile-api-patterns/references/strided-row-tiles.md).
- When a half-row is already a long contiguous stream, split it into flat
  chunks. Load gate and value from offsets separated by H, and include
  useful column chunks when choosing the grid. Compare loop unroll 1, 2
  and 3 using compiled UB for the exact dtype, tile and JIT options.
  See [flat SwiGLU streams](../../asctile-api-patterns/references/streamed-halves.md)
  for whole-row specialization and block scheduling.

Use the active vector-resource query for these routes. Recheck warm and
cache-flushed timings under the runner's actual sampling convention before
transferring a threshold or unroll choice from another runtime.

Check which guard wins when the BF16 packed-pair and generic gather routes
overlap. In the matched 950PR replay, the generated H=7 gather took 15.229 µs
versus 6.385 µs for the archived candidate. The H=1 dispatch took 10.625 µs
versus 6.345 µs. Retain the selected kernel name, tile height, unroll, compiled
UB and warmed samples for both shapes before promoting a short-row default.
[Per-case replay and source](../../../docs/evidence/swiglu-vector-grid-20261110/README.md)

For a reproducible public `asctile.reindex` check, use CPython 3.12, PyASC
`ed793e45` and
`ghcr.io/compiler-team-ru/pyasc-sim-dev@sha256:1efdcc8585b769715f640e9c9078c2e03cdf2157082d05b0c4b87974beea7407`.
The arm64 CaModel probe gathered both halves of 16 and 4,096 BF16 pairs with
exact results. Recheck the complete generated kernel on the submission runtime
before comparing device times across wheels.
[Probe, image and per-architecture wheel hashes](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-swiglu-retile-20261110/reindex-cp312-camodel.tar.gz)

For the padded SwiGLU kernel on PyASC `6b3447b4`, the BF16 specialization
at height 512 and physical width 16 requested 655,360 UB bytes on the 950PR.
Budget at least 1,280 bytes per physical row (80 bytes per padded lane) for
this path. The 128-row specialization used 163,840 bytes in both the full
and tail launches. Query the active SoC capacity after platform setup. Requalify the
allocation when changing the kernel, dtype, padding, compiler or JIT flags.

On a wide split, budget the tile for the rows that actually contain data.
The eight-block parent used a physical `16 × 64` tile even when only one
or four rows were live. An OpenCode-generated host change retained the
1,024-element budget, used the live row count as tile height and rounded the
width down to a multiple of 16 elements. It also bounded the width so that
eight full column tiles remained, then recomputed the tail offset and owner.
On Ascend950PR_9599 with CANN 9.2.0 and PyASC `6b3447b4`, case 6 with
one BF16 row improved from 1,294.781 to 74.646 µs, and case 19 with four
BF16 rows and an uneven tail improved from 1,504.377 to 334.773 µs.
These are medians of three warm device calls after a full-output golden check.

On CANNBench 950PR (CANN 9.2.0), that tile change at eight blocks sped up
all six wide cases by **13.0×** geometrically. The other 14 retained the
same emitted kernels and grid (1.0004×). All 20 passed. The generated
candidate scored 32.7712, versus 32.2349 for its parent and 42.8164 for
the PyASC reference. Check live lanes and the number of independent
column tiles before raising the block cap. The torch_npu op-api dispatches
to `aclnnSwiGlu`; its published CANN tiler considers UB and available
vector cores, then prefers fewer tiles at equal occupancy.
[Matched service profiles](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-swiglu-retile-20261110) ·
[op-plugin dispatch](https://github.com/Ascend/op-plugin/blob/eee88cf400f796c150db5a6203c4218b5af7ea18/op_plugin/config/op_plugin_functions.yaml#L6996-L7004) ·
[CANN tiler](https://gitcode.com/cann/ops-nn/blob/f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307/activation/swi_glu/op_host/swi_glu_tiling.cpp)

For a shape-dependent constexpr, count the emitted specializations before
shipping an AOT package. The full SwiGLU tile family produced 1,266 device
sources. Building common layouts eagerly and compiling the others at first
use preserved the official cases’ machine code and passed the service build.

Measure the grid separately at fixed geometry. With the parent's tile and
the same inputs, 8 → 32 → 64 blocks took 1,294.781 → 307.247 → 162.471 µs
on case 6 and 1,504.377 → 377.400 → 193.149 µs on case 19. The retained
PyASC reference uses a `1 × 1024` wide tile, grid-strides over row/column
tiles and launches up to 72 blocks. When an operator uses separate full-row
and row-tail launches, sum their NPU task durations for each call before
taking the median. The [20-case grid comparison](../../../docs/evidence/swiglu-vector-grid-20261110/README.md) records the actual launch sizes for the adaptive variant.

## ForeachAddcdivScalar

Loop over tensor lists on the host and launch one elementwise kernel for each
triple. Preserve scalar zero, Inf, and NaN behavior. Compute division in f32
for FP16 and BF16 inputs and cast once on output. Direct target-dtype division
is not a valid generic adapter: the pinned compiler rejects BF16 division, and an FP16
intermediate produced a NaN-position mismatch in the scalar-zero CANNBench
route. The FP32-internal pattern is verified 20/20 on 950PR.

## ForeachNorm

Return one 0-D result per input tensor. Specialize host dispatch by p, including
negative p and Inf. Cross-core sum/max needs `asctile.atomic_add` or
`asctile.atomic_max`; pad scalar storage to at least 32 bytes. A loop-carried
accumulator must be seeded with a pyasc scalar/tile result, not a Python float.
Use only `asctile.range` kwargs confirmed by the pinned source.
`reduce_max` returns a PlainValue; `asctile.maximum(acc, m)` therefore fails when
both operands are PlainValue. Keep the maximum in a LocalTensor-compatible
accumulator or use a reduction form that the pinned scalar algebra supports.
Return the final typed scalar as a 0-D metadata view of an aligned output
buffer; perform dtype conversion in the final JIT kernel.

## RMSNorm

Normalize each last-dimension row and multiply by gamma. The pinned source
contains `asctile.rms_norm`; the current canonical module uses it for the main
path, with an f16 hop for bf16, and a manual fallback for shapes that fail its
constraints. Preserve f32 accumulation where the manual path is used. Cover D
from 2 through 8192, including prime/non-aligned tails and special values.

Reduction scalars are PlainValue results. `asctile.rsqrt(plain_value)` rejects the
input because `rsqrt` requires LocalTensor. Use scalar arithmetic supported by
the pinned API (for example `1.0 / asctile.sqrt(plain_value)`) or broadcast the
value into an aligned local tensor before a tensor-only unary op. Budget that
broadcast as a live f32 tile: a generated `tile_d=2048` manual RMSNorm used
279–329 KB and overflowed the 253952 B UB.

## Softmax

The benchmark axis is arbitrary, not always the last dimension. Normalize the
possibly-negative host `dim`, then describe the contiguous tensor as
`[outer, axis_size, inner]`. When `inner == 1`, use the upstream full-row
`asctile.softmax` target path directly. Otherwise load a bounded
`[1, axis_size, inner_tile]` region, transpose it locally to
`[1, inner_tile, axis_size]`, apply `asctile.softmax` on the final dimension,
transpose back, and store with exact tails. Never materialize a device
`torch.permute(...).contiguous()` as a host shortcut. Shifted/stable softmax
must preserve NaN/Inf behavior from `golden.py`; all three floating dtypes and
axis sizes through 8193 are mandatory. Compile-check the largest axis in f32
because the softmax implementation's internal live buffers, not only the
input tile, determine UB use.

## Transpose

Start from the current upstream
`python/test/asctile/target/test_transpose.py`; its kernel bodies and shape
simplification are the handwritten reference. The CANNBench wrapper must
cover ranks 2–5, arbitrary listed permutations, and f16/bf16/f32 plus signed
integer dtypes through int64. Allocate the output with the permuted shape and
perform all movement in `@asctile.jit` kernels—`torch.permute` is golden-only,
not an implementation path. Collapse adjacent dimensions that remain
adjacent under the permutation before selecting one-axis/two-axis tiling.
The selected-runtime transpose signature is variadic: call
`tile.transpose(*permute)` or `tile.transpose(1, 0)`. Do not pass a list as one
argument (`asctile.transpose(tile, [1, 0])`), because the compiler interprets
that list as a single axis value and rejects it.
Both the input tile's last physical dimension and the permuted output tile's
last physical dimension need 32-byte alignment; preserve logical extents with
`real_shape`. Identity-after-simplification must still launch the handwritten
copy kernel rather than returning the input alias.
