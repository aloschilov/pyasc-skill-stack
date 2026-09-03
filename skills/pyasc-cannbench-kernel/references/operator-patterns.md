# CANNBench operator patterns

These patterns describe the nine vendored operators. Task files remain the
normative source.

At the host/JIT boundary pass torch tensors directly. Do not call
`Tensor.data_ptr()`: the pinned JIT derives pointer dtype specialization from
the tensor, while a raw integer loses that contract. Host-side output casts or
slice assignments are also numerical device operations and are not an
acceptable substitute for a final asctile kernel.

## Common elementwise path

`sigmoid`, `exp`, `mish`, and `gelu` flatten arbitrary-rank contiguous input to
one dimension. Use grid-stride tiles, `real_shape=[n]` tails, f32 intermediates
for f16/bf16 where needed, and `cores = min(72, num_tiles)`. Stable forms must
preserve NaN and Inf behavior from `golden.py`. GELU has both `none` and `tanh`
attributes and requires distinct kernels or tile widths.

For GELU, do not copy the narrow upstream target test as a complete
CANNBench implementation: that test covers one FP32 shape and only the
sigmoid-equivalent tanh approximation. The 950PR-verified 20/20 pattern is:

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
796--820 KB and is rejected against the 253952-byte UB limit. Use the measured
safe baseline `tile_size=512`, `unroll_factor=2` for exact mode. Keep tanh on a
separate kernel/tile; `tile_size=1024`, `unroll_factor=2` is measured at
172--184 KB. Do not infer a larger exact tile solely from a source-level
temporary count; only the exact-v2 compile report establishes the UB budget.

`real_shape` limits DMA but not the vector instructions consuming the local
tile. Supply neutral padding explicitly: `pad_value=0` for additive operands,
`pad_value=1` for divisors, and a finite safe input for logarithm/reciprocal
chains. This prevents invalid arithmetic in inactive lanes without changing
the logical output.

Pinned-v2 scalar rule: both data branches of `asctile.where(condition, a, b)`
must be LocalTensor values. A Python scalar branch such as
`asctile.where(x >= 0, 1.0, tile)` fails with `AttributeError: 'float' object has
no attribute 'dtype'`. Prefer algebra that avoids the select, or build an
aligned tensor branch with `asctile.full`.

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
