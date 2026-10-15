# CANNBench operator patterns

These patterns describe the eleven vendored operators. Task files remain the
normative source.

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
18-combination [JIT matrix](../../pyasc-api-patterns/references/api-jit-options.md)
with synchronization ON. Measure UB and numerical behavior at fixed geometry
before widening tiles; only real-NPU timings establish speed.

Measured local qualification at this pin (CANN 9.0 / 950PR_9599) rejects the
public-erf FP32 exact path on dense negative-tail inputs for **all 18 JIT
configurations**, despite successful compilation and smaller smoke tests.
Read the [minimal reproduction and campaign report](../../../../integrations/cannbench/comparisons/gelu-asctile-jit-20260907/README.md)
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
is the current CANNBench runtime authority, not a historical GeLU pin.
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
pinned v2 compiler, `tile_size=2048`, `unroll_factor=2` consumes roughly
796--820 KB and is rejected against the 253952-byte UB limit. The measured
safe historical baseline was `tile_size=512`, `unroll_factor=2` for exact mode.
Tanh used a separate kernel/tile; `tile_size=1024`, `unroll_factor=2` measured at
172--184 KB. Do not infer a larger exact tile solely from a source-level
temporary count; only the exact-v2 compile report establishes the UB budget.

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
has a fused `AscendC::Gelu`; pyasc v2 does not expose it, which should be filed
as a performance/API blocker rather than worked around with an inaccurate
formula.

### Shared tile constraints

Older snapshots required LocalTensor branches for `asctile.where`; the later
historical GeLU snapshot `adadd7d6` added numeric scalar operands. Verify the
actually supported operand kinds against the CANNBench runtime pin `0a631f70`
before relying on either form, and do not require
extra full tiles solely to reproduce an older API restriction. Alignment
and measured compare/select lowering limitations still need verification.

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
The current official result is 18/20: case 9 is an NPU runtime failure around
the host narrow/contiguous fallback and case 12 is a bf16 NaN-position
mismatch. Treat both roots as suspected until reproduced on hardware; a prior
replacement fallback regressed to 0/20 and must not be copied.

## ForeachAddcdivScalar

Loop over tensor lists on the host and launch one elementwise kernel for each
triple. Preserve scalar zero, Inf, and NaN behavior. Compute division in f32
for FP16 and BF16 inputs and cast once on output. Direct target-dtype division
is not a valid generic adapter: pinned v2 rejects BF16 division, and an FP16
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

Normalize each last-dimension row and multiply by gamma. The pinned v2 source
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
The current-v2 transpose signature is variadic: call
`tile.transpose(*permute)` or `tile.transpose(1, 0)`. Do not pass a list as one
argument (`asctile.transpose(tile, [1, 0])`), because the compiler interprets
that list as a single axis value and rejects it.
Both the input tile's last physical dimension and the permuted output tile's
last physical dimension need 32-byte alignment; preserve logical extents with
`real_shape`. Identity-after-simplification must still launch the handwritten
copy kernel rather than returning the input alias.
