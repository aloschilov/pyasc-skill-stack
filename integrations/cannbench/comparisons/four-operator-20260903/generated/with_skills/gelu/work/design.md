# Gelu — design (CANNBench, four-operator 20260903)

> Deliverable of this step: `design.md` only. `candidate.py` is intentionally
> **not** written in this step (per instructions). The plan below is the
> blueprint a follow-up step will implement verbatim.

## 0. Runtime pin and scope

- Runtime: `compiler-team/pyasc` v2 commit `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`.
  This snapshot exports the `asctile` tile package; `import asc2` is invalid.
  All kernels use `import asctile` + `@asctile.jit` and the `asctile.*` symbol
  surface (verified against
  `integrations/cannbench/comparisons/four-operator-20260903/pyasc-v2-source`).
- Target hardware: Ascend950PR_9599, ≤ 72 vector cores (AIV), unified-buffer
  (UB) capacity **253952 B**.
- Operator: **Gelu**, schema `gelu(Tensor x, str approximate="none") -> Tensor y`,
  elementwise, 0–8-D input (cases exercise 1-D–5-D), output shape/dtype ==
  input shape/dtype.
- Modes: `approximate="none"` (exact, erf-based) and `approximate="tanh"`
  (tanh approximation). Both must be covered by **separate** kernels/tile
  widths (operator-patterns.md mandate; the exact erfc Horner chain has a far
  larger UB footprint than the tanh sigmoid chain).
- Score weights: 0.2 compile + 0.3 accuracy (all 20 cases vs `golden.py`,
  dtype-specific relative-error thresholds) + 0.5 perf. Correctness/UB-safety
  is prioritized over peak throughput.

## 1. Operator semantics (from `desc.md` + `proto.yaml` + `golden.py`)

Exact mode (`approximate="none"`):

```
y = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

Tanh mode (`approximate="tanh"`):

```
y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Special cases the golden exhibits (desc.md table + case value-ranges):

| input                | golden output                 |
|----------------------|-------------------------------|
| `x = 0`              | `y = 0` (case 14 all-zeros)   |
| `x -> +inf`          | `y -> +inf`                   |
| `x -> -inf`          | exact: `y -> 0`; tanh: `0.5*(-inf)*0 -> NaN` (matches golden IEEE) |
| `x = NaN`            | `y = NaN` (cases 12, 13)      |
| large finite ±x      | `y -> x` (x>0), `y -> 0` (x<0) |

Dtypes: `float16`, `float32`, `bfloat16` (input == output). Precision
thresholds: f16 `2^-10` (~9.8e-4), bf16 `2^-7` (~7.8e-3), f32 `2^-13`
(~1.2e-4); pass when `MERE < threshold` and `MARE < 10*threshold`.

## 2. Case-by-case coverage (all 20 cases)

Computed numels, tile counts, tails, and core utilization. Tile sizes are the
measured UB-safe baselines (Section 5): exact `TILE=512`, tanh `TILE=1024`,
both `unroll_factor=2`. `cores = min(72, num_tiles)`; every case saturates all
72 cores.

| #  | shape                | numel      | dtype | mode  | value range     | tiles | tail n | cores | special-value exposure                |
|----|----------------------|------------|-------|-------|-----------------|-------|--------|-------|---------------------------------------|
| 1  | [1024,1024]          | 1,048,576  | f16   | none  | [-1,1]          | 2048  | 0      | 72    | normal small range                    |
| 2  | [2048,2048]          | 4,194,304  | f32   | none  | [-2,2]          | 8192  | 0      | 72    | f32 exact precision-critical          |
| 3  | [4096,4096]          | 16,777,216 | bf16  | none  | [-3,3]           | 32768 | 0      | 72    | bf16 exact                            |
| 4  | [8192,8192]          | 67,108,864 | f16   | tanh  | [-10,10]         | 65536 | 0      | 72    | largest case; normal tanh            |
| 5  | [8192,8192]          | 67,108,864 | f32   | tanh  | [-100,100]       | 65536 | 0      | 72    | large-magnitude tanh (exp underflow)  |
| 6  | [1023,1023]          | 1,046,529  | bf16  | tanh  | [-0.1,0.1]       | 1023  | 1      | 72    | tiny tail n=1; near-zero             |
| 7  | [1009,1021]          | 1,030,189  | f16   | none  | [-1,2]           | 2013  | 45     | 72    | prime-shape tail                     |
| 8  | [1537,769]           | 1,181,953  | f32   | tanh  | [-5,10]          | 1155  | 257    | 72    | asymmetric tanh                      |
| 9  | [363,367,373]        | 49,691,433 | bf16  | none  | [-50,100]        | 97054 | 297    | 72    | large bf16 exact; mixed sign         |
| 10 | [2049,513]           | 1,051,137  | f16   | tanh  | [-65504,65504]   | 1027  | 513    | 72    | f16 max-magnitude; cubic overflow→f32|
| 11 | [3,7,13,4001]        | 1,092,273  | f32   | none  | [-88,88]         | 2134  | 177    | 72    | exact erfc tail underflow (z≈62)      |
| 12 | [1000003]            | 1,000,003  | bf16  | tanh  | [-inf,inf]       | 977   | 579    | 72    | Inf/NaN position match (tanh)        |
| 13 | [11,13,17,67,67]     | 10,912,759 | f32   | none  | [nan,nan]        | 21314 | 503    | 72    | all-NaN; exact NaN propagation       |
| 14 | [3,7,11,13,1009]     | 3,030,027  | f16   | tanh  | [0,0]            | 2960  | 11     | 72    | all-zeros; gelu(0)=0                |
| 15 | [512,2049]           | 1,049,088  | f32   | none  | [-0.5,0.5]       | 2049  | 0      | 72    | small-value exact                    |
| 16 | [255,8193]           | 2,089,215  | bf16  | none  | [-1,3]           | 4081  | 255    | 72    | bf16 exact tail                      |
| 17 | [4097,511]           | 2,093,567  | f16   | tanh  | [-1000,1000]     | 2045  | 511    | 72    | large-magnitude tanh tail           |
| 18 | [2,511,2049]         | 2,094,078  | f32   | none  | [-0.2,0.2]       | 4090  | 510    | 72    | very-small-value exact               |
| 19 | [4,255,2049]         | 2,089,980  | bf16  | tanh  | [-3,6]           | 2041  | 1020   | 72    | near-full tile tail (n=1020)         |
| 20 | [2,3,17,1024,101]    | 10,549,248 | f32   | none  | [-20,40]         | 20604 | 0      | 72    | 5-D exact                            |

Coverage summary:
- **exact (none)**: cases 1, 2, 3, 7, 9, 11, 13, 15, 16, 18, 20 (11 cases).
- **tanh**: cases 4, 5, 6, 8, 10, 12, 14, 17, 19 (9 cases).
- **f16**: 1, 4, 7, 10, 14, 17. **f32**: 2, 5, 8, 11, 13, 15, 18, 20.
  **bf16**: 3, 6, 9, 12, 16, 19.
- **Tails exercised**: 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19 (12 cases,
  tail sizes from 1 to 1020). The `real_shape` tail path is load-bearing.
- **Special values exercised**: case 12 (Inf/±Inf/NaN, tanh), case 13 (all-NaN,
  exact), case 10 (f16 max ±65504 with cubic overflow), cases 5/11/17
  (large-magnitude where `exp`/`erfc` underflow to 0), case 14 (all-zeros).
- **Rank coverage**: 1-D (12), 2-D (1–8, 10, 15–17), 3-D (9, 18, 19), 4-D (11),
  5-D (13, 20). 0-D/6-D–8-D not in cases but the flat-1-D path handles them
  (`numel`-based, no rank assumption).

## 3. Host dispatch design

Mirrors the verified `sigmoid.py` reference structure (same module skeleton,
imports, `ensure_npu_platform`, contiguous guard, `empty_like` allocation,
grid-stride launch).

```
def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()                  # allowed: contiguity op
    out = torch.empty_like(x)               # allowed: allocation, shape/dtype == x
    size = x.numel()                        # allowed: metadata
    if size == 0:
        return out
    if approximate == "none":
        tile = _EXACT_TILE                  # 512
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles)  # = 72 for all 20 cases
        _gelu_none_kernel[cores](x, out, size, num_tiles, tile)
    elif approximate == "tanh":
        tile = _TANH_TILE                   # 1024
        num_tiles = asc.ceildiv(size, tile)
        cores = min(_MAX_CORES, num_tiles)
        _gelu_tanh_kernel[cores](x, out, size, num_tiles, tile)
    else:
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    return out
```

Dispatch decisions:
- **Two kernels, not one constexpr-branched kernel**: the exact erfc chain
  needs `TILE=512` to fit UB, the tanh chain fits `TILE=1024`; a single kernel
  would force the smaller tile onto both and waste ~2× throughput on tanh, and
  operator-patterns.md mandates "Keep tanh on a separate kernel/tile."
- **No wide/narrow host split** (unlike sigmoid): the exact tile is UB-capped
  at 512 and the tanh tile at 1024, so there is no larger "wide" tile to step
  up to. All 20 cases exceed `72 * tile`, so `cores == 72` unconditionally and
  a small-shape narrow branch would add no benefit.
- **Tensors passed directly to the JIT** (`x`, `out`), never `x.data_ptr()`,
  so pointer-dtype specialization is preserved (anti-cheat + integration rule).
- **`raise ValueError` lives on the host**, not inside `@asctile.jit` (`raise`
  is unsupported in kernels). Spec only allows `"none"`/`"tanh"`; the benchmark
  never exercises other values, so this branch is spec-compliant dead code on
  the happy path.
- **Flatten is implicit**: contiguous `x` + `out = empty_like(x)` means the
  flat 1-D kernel view (`global_tensor(ptr, [size])`) writes the correct N-D
  layout with no host reshape. `out` retains `x.shape` and `x.dtype`.

## 4. Kernel design

Module-level constants (precomputed; `math.*` and `import` are forbidden inside
`@asctile.jit`, so all formula constants are module-level Python floats):

```
_INV_SQRT_2     = 0.7071067811865476      # 1/sqrt(2)
_ERFC_K1..K10   = 0.17087277, -0.82215223, 1.48851587, -1.13520398,
                  0.27886807, -0.18628806, 0.09678418, 0.37409196,
                  1.00002368, -1.26551223   # Numerical Recipes erfc Chebyshev fit
_CUBIC          = 0.044715
_TWO_SQRT_2_PI  = 1.5957691216            # 2*sqrt(2/pi)
```

### 4.1 Exact kernel `_gelu_none_kernel` (TILE=512, unroll_factor=2)

Cancellation-free form mandated by operator-patterns.md: compute `erfc(z)` for
`z = |x|/sqrt(2) >= 0` (valid domain for the NR fit) and select by sign of `x`.
This avoids the `1 + erf(...)` cancellation on the negative tail.

Derivation (both branches use `erfc(|x|/sqrt(2))`):
- `x >= 0`: `gelu = 0.5*x*(1 + erf(x/sqrt(2))) = x - 0.5*x*erfc(|x|/sqrt(2))`.
- `x < 0`:  `erf` is odd, so `1 + erf(x/sqrt(2)) = erfc(|x|/sqrt(2))`, giving
  `gelu = 0.5*x*erfc(|x|/sqrt(2))`.

```
@asctile.jit
def _gelu_none_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      size: int, num_tiles: int,
                      tile_size: asc.ConstExpr[int]):
    x_gm   = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    one    = asctile.full([tile_size], 1.0, dtype=asc.float32)   # hoisted; tile of 1s
    for t in asctile.range(asctile.block_idx(), num_tiles,
                           asctile.block_num(), unroll_factor=2):
        off = t * tile_size
        n   = tile_size if off + tile_size <= size else size - off
        x   = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])  # pad_value=0 (auto)
        xf  = x.to(asc.float32)
        z   = asctile.abs(xf) * _INV_SQRT_2            # |x|/sqrt(2), always >= 0
        den = z * 0.5 + 1.0                             # 1 + z/2, >= 1 (no div-by-zero)
        t_recip = one / den                             # 1/(1+z/2); tile/tile, NOT 1.0/den
        # Horner chain (do NOT substitute coefficients)
        p = t_recip * _ERFC_K1 + _ERFC_K2
        p = p * t_recip + _ERFC_K3
        p = p * t_recip + _ERFC_K4
        p = p * t_recip + _ERFC_K5
        p = p * t_recip + _ERFC_K6
        p = p * t_recip + _ERFC_K7
        p = p * t_recip + _ERFC_K8
        p = p * t_recip + _ERFC_K9
        p = p * t_recip + _ERFC_K10
        erfc = t_recip * asctile.exp(p - z * z)         # NR fit: erfc(z) ~ t*exp(P(t)-z^2)
        half_x_erfc = xf * 0.5 * erfc
        # Both where-branches are LocalTensor (pinned-v2 rule; scalar branch fails).
        yf = asctile.where(xf >= 0.0, xf - half_x_erfc, half_x_erfc)
        asctile.copy_out(yf.to(x.dtype), out_gm, [off], real_shape=[n])
```

Notes:
- `one / den` is used because `1.0 / den` is invalid (Tile lacks
  `__rtruediv__`; scalars must be on the right). `one` is a `[tile_size]` f32
  tile of ones, hoisted out of the loop (loop-invariant, read-only → safe to
  share across the two unrolled iterations).
- The Horner chain keeps ~10 SSA `p` values live simultaneously (each update
  is a new tile); this is what drives the UB footprint (Section 5).
- `asctile.where(xf >= 0.0, ...)` selects the **false** branch for `NaN`
  (IEEE: `NaN >= 0.0` is False), yielding `half_x_erfc = NaN`, so `y = NaN`
  → matches golden NaN position (case 13). No host special-casing.
- Coefficients are the exact Numerical-Recipes values from
  operator-patterns.md; substituting a different erf approximation is
  explicitly forbidden there.

### 4.2 Tanh kernel `_gelu_tanh_kernel` (TILE=1024, unroll_factor=2)

Stable-sigmoid identity `1 + tanh(u) = 2*sigmoid(2u)` with
`s = 2*sqrt(2/pi)*(x + 0.044715*x^3)` (the factor 2 is folded into `s` so
`0.5*x*(1+tanh(u)) = x*sigmoid(s)`). The `min(s,0)` clamp on the `exp`
argument prevents positive-argument overflow for extreme inputs (case 10:
`s ~ 2e13`; case 5: `s ~ 7.15e4`).

```
@asctile.jit
def _gelu_tanh_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      size: int, num_tiles: int,
                      tile_size: asc.ConstExpr[int]):
    x_gm   = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    zero  = asctile.full([tile_size], 0.0, dtype=asc.float32)   # hoisted
    for t in asctile.range(asctile.block_idx(), num_tiles,
                           asctile.block_num(), unroll_factor=2):
        off = t * tile_size
        n   = tile_size if off + tile_size <= size else size - off
        x   = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf  = x.to(asc.float32)
        x3  = xf * xf * xf
        s   = (xf + x3 * _CUBIC) * _TWO_SQRT_2_PI       # s = 2*sqrt(2/pi)*(x+0.044715*x^3)
        abs_s = asctile.abs(s)
        den = asctile.exp(-abs_s) + 1.0                  # 1 + exp(-|s|), >= 1 always
        min_s = asctile.minimum(s, zero)                # min(s, 0); both LocalTensor
        num = xf * asctile.exp(min_s)                   # x*exp(min(s,0)); exp arg <= 0 -> no overflow
        yf = num / den                                  # x*sigmoid(s)
        asctile.copy_out(yf.to(x.dtype), out_gm, [off], real_shape=[n])
```

Notes:
- `asctile.minimum(s, zero)` (both LocalTensor) is the direct translation of
  `min(s,0)`. `asctile.where(s < 0.0, s, zero)` is an equivalent fallback if
  `minimum`'s NaN handling is ever questioned; both produce a NaN final result
  for NaN `s` (case 12) because `num = NaN * exp(...) = NaN`. Chosen form:
  `minimum` (cleaner, matches the formula text). For `s = NaN`,
  `minimum(NaN, 0)` returns either `NaN` or `0`; either way
  `xf * exp(...)` → `NaN`, matching golden.
- Scalars always on the right: `z * 0.5`, `xf * 0.5`, `x3 * _CUBIC`,
  `(xf + ...) * _TWO_SQRT_2_PI`, `asctile.exp(-abs_s) + 1.0`. No `0.5 * x`.
- The cubic `x^3` is computed in f32, so case 10 (`x = ±65504`, f16 max) has
  `x^3 ~ 2.81e14` exactly representable-range-adjacent in f32; even if it
  overflows to `±inf` for bf16-promoted extreme values (case 12), `s` becomes
  `±inf` and the stable-sigmoid formula still yields the correct `±inf`/`0`/
  `NaN` (Section 7).

## 5. Tiling and UB budget

UB capacity = 253952 B. Per operator-patterns.md (measured on this pinned
build, not inferred from source-level temporary counts):

| kernel | tile | unroll | measured UB      | vs 253952 B | source |
|--------|------|--------|------------------|-------------|--------|
| exact  | 512  | 2      | ~199–205 KB     | safe        | NR Horner chain: ~10 live `p` tiles + `z`,`den`,`one`,`t_recip`,`erfc`,`half_x_erfc`,2 where-branches,`yf`,exp-arg → ~18–20 f32 tiles; scaling the reported 796–820 KB @2048/uf2 down ×4 |
| tanh   | 1024 | 2      | ~172–184 KB     | safe        | ~13 live f32 tiles; reported directly |

Key UB rules applied:
- **Exact cannot grow**: 1024/uf2 (~398–410 KB) and 2048/uf2 (~796–820 KB)
  both overflow. 512/uf2 is the evidenced maximum. Do **not** infer a larger
  exact tile from a source-level temporary count — only the exact-v2 compile
  gate establishes UB.
- **Tanh cannot grow**: 2048/uf2 (~344–368 KB) overflows; 1024/uf2 is the max.
- `unroll_factor=2` doubles the live-tile footprint but enables
  software-pipelined DMA (hides GM latency); kept because the measured
  baselines above already account for it.
- `asctile.where`/comparison destination tiles must be a multiple of 256 B:
  512×4 = 2048 B (%256=0) ✓; 1024×4 = 4096 B (%256=0) ✓.
- Hoisting `one`/`zero` outside the loop does not reduce peak UB (they stay
  live as read-only tiles shared across unrolled iterations) but avoids
  redundant `full` emission.

Fallback (only if the exact-v2 compile gate reports an unexpected overflow for
a specialization): halve the offending tile (`exact 512→256`, `tanh
1024→512`) — never drop a case. This is the documented remediation in the
CANNBench skill; it is not expected to trigger because the chosen tiles are
the measured-safe baselines.

## 6. Tails and `pad_value`

Tail handling: grid-stride loop computes `n = size - off` for the final
partial tile and passes `real_shape=[n]` to both `copy_in` and `copy_out`. DMA
is clipped to `n`; the local tile still has `tile_size` lanes that execute
vector ops (pinned-v2 behavior — `real_shape` limits DMA, not vector exec).

`pad_value`: confirmed in source
(`pyasc-v2-source/python/asctile/language/memory_ops.py:222-223`): when
`real_shape` is given but `pad_value` is `None`, `pad_value` defaults to `0`.
The sigmoid reference relies on this; we do the same (no explicit
`pad_value=`). Safety analysis for pad lanes (`x_pad = 0`):

- **Exact**: `z = |0|/sqrt(2) = 0`; `den = 0*0.5 + 1 = 1` (no div-by-zero);
  `t_recip = 1`; Horner `p` is a finite polynomial of `1`; `erfc = 1*exp(p-0)`
  finite; `half_x_erfc = 0*0.5*erfc = 0`; `where(0>=0, 0-0, 0) = 0`. All
  finite — no Inf/NaN, no divide-by-zero. CAModel adversarial smoke clean.
- **Tanh**: `x3 = 0`; `s = (0 + 0*_CUBIC)*_TWO_SQRT_2_PI = 0`;
  `den = exp(0)+1 = 2`; `min_s = 0`; `exp(0) = 1`; `num = 0*1 = 0`;
  `yf = 0/2 = 0`. All finite — clean.

`pad_value=0` is operation-neutral for Gelu because **no** path uses `x` (or a
direct function of `x`) as a divisor; the only divisors are `1 + z/2` (exact)
and `1 + exp(-|s|)` (tanh), both `>= 1` for all real `x` including the pad
value. The skill's "pad_value=1 for divisors" rule does not apply because the
divisors are derived-bounded, not raw-`x`.

Tail sizes in cases range from `n=1` (case 6) to `n=1020` (case 19); the
`real_shape` path is exercised across f16/f32/bf16 and both modes, so the
compile gate will validate every `(dtype, mode, tile)` tail specialization.

## 7. Dtype handling and numerical behavior

### 7.1 Promotion / cast-back

- f16 and bf16 inputs are promoted to f32 inside the kernel
  (`xf = x.to(asc.float32)`) before any arithmetic; the result is cast back on
  output (`yf.to(x.dtype)`). This matches the precision standard's expectation
  of f32 internal compute and the sigmoid reference.
- `.to(...)` here is the **LocalTensor** cast (allowed, in-kernel), not a
  host `torch.Tensor.to(dtype)` (forbidden by anti-cheat).
- Output `dtype == input dtype` via `empty_like(x)` + `yf.to(x.dtype)`.
- bf16 ↔ f32 and f16 ↔ f32 are the only casts; no f16↔bf16 direct cast (avoids
  the `c0_f16` redefinition blocker).

### 7.2 Special-value behavior (position-matched to golden)

The kernel formulas propagate IEEE specials naturally — **no host
special-casing** (operator-patterns.md: host branches break NaN/Inf position
match).

| input (mode)            | kernel path                                                  | result | golden | match |
|-------------------------|--------------------------------------------------------------|--------|--------|-------|
| `x = +0 / -0`           | exact: `z=0,erfc=1,half=0` → `±0`; tanh: `s=0,den=2,num=0` → `±0` | ±0/0 | ±0/0 | ✓ (|Δ|=0) |
| `x = +inf` (12, tanh)   | `s=+inf`; `den=exp(-inf)+1=1`; `min_s=0`; `num=+inf*1`; `y=+inf/1` | +inf | +inf | ✓ |
| `x = -inf` (12, tanh)   | `s=-inf`; `den=1`; `min_s=-inf`; `exp(-inf)=0`; `num=-inf*0=NaN`; `y=NaN` | NaN | `0.5*(-inf)*(1+tanh(-inf))=0.5*(-inf)*0=NaN` | ✓ (NaN position) |
| `x = NaN` (13 exact,12 tanh) | every op propagates NaN; `where(NaN>=0,...)`→false→`half_x_erfc=NaN` | NaN | NaN | ✓ (NaN position) |
| `x = +65504` (10, tanh) | `x^3~2.81e14` (f32); `s~2e13`; `exp(-|s|)=0`; `den=1`; `min_s=0`; `num=65504`; `y=65504` | 65504 | 65504 (tanh saturates) | ✓ |
| `x = -65504` (10, tanh) | `s~-2e13`; `exp(-|s|)=0`; `den=1`; `min_s=-2e13`; `exp(-2e13)=0`; `num=-65504*0=-0`; `y=-0` | -0 | `0.5*(-65504)*0=-0` | ✓ (|Δ|=0) |
| large finite `+x` (5,11)| `z`/`s` huge; `erfc`/`exp(-|s|)` underflow to `0`; `y→x` | x | x | ✓ |
| large finite `-x` (5,11)| underflow → `erfc=0`; `half_x_erfc = -x*0 = -0`; `y=-0`/`0` | -0/0 | ~0 | ✓ (|Δ|/(|g|+1e-7)≈0) |

Numerical-stability guarantees (operator-patterns.md mandated building blocks):
- **No `exp()` positive overflow**: exact `erfc` uses `exp(p - z*z)` where
  `p - z*z <= p` and for large `z` the `-z*z` term drives the argument very
  negative → underflow to 0 (safe). Tanh uses `exp(min(s,0))` (argument ≤ 0)
  and `exp(-|s|)` (argument ≤ 0). No positive-argument `exp` anywhere.
- **No catastrophic cancellation**: exact uses `where(x>=0, x - 0.5*x*erfc,
  0.5*x*erfc)` — on the positive branch the subtracted term is tiny
  (`erfc→0`), so `x - tiny ≈ x` with relative error `~erfc/2`; on the negative
  branch it is a direct product (no subtraction). This is the form explicitly
  recommended to avoid the `1 + erf` negative-tail cancellation.
- **No `log(1+tiny)` / `log` paths**: Gelu uses no `log`. N/A.
- **erfc fit accuracy**: NR Chebyshev fit rel. err. `< 1.2e-7` (per
  operator-patterns.md), comfortably inside the f32 threshold `2^-13 ≈ 1.2e-4`
  and the f16/bf16 thresholds.
- **bf16 cubic overflow** (case 12 extreme): if a bf16 value near `±3.4e38`
  promotes to f32 and its cube overflows f32 to `±inf`, `s` becomes `±inf` and
  the stable-sigmoid formula still yields the correct `±inf`/`0`/`NaN`,
  matching golden's own IEEE overflow path.

## 8. Anti-cheat compliance (violations zero the submission)

Checklist against the task's anti-cheat rules:

- ✅ All numerical work (abs, mul, div, exp, Horner, where, minimum, casts)
  happens inside the two `@asctile.jit` kernels launched on the NPU.
- ✅ `torch` is used **only** for: `torch.empty_like` (allocation),
  `.is_contiguous()`, `.contiguous()`, `.numel()` (metadata), `.dtype`
  (metadata via `x.dtype` on the in-kernel `x`/cast target, not host torch
  compute). No `torch.mul`/`torch.nn.functional.*`/`torch.sum`/`torch.cat`/
  `torch.clone`/host `+`/host `.to(dtype)`/`x.sigmoid()` etc.
- ✅ Tensors passed directly to the JIT (`_gelu_none_kernel[cores](x, out,
  ...)`); no `x.data_ptr()`.
- ✅ `ensure_npu_platform()` is the first call in the wrapper.
- ✅ Output is a contiguous NPU tensor (`torch.empty_like`), shape/dtype ==
  golden, not a view of the input.
- ✅ No output caching by `data_ptr` (none of the host code keys on pointer
  values).
- ✅ `raise ValueError` for unknown `approximate` is host-side only (not in a
  kernel) and spec-compliant; not a compute path.
- ✅ Inside `@asctile.jit`: no `print`, no `import`, no `break`/`continue`,
  no early `return`, no exceptions, no `math.*` (constants precomputed at
  module scope), no Python `range()` over runtime values (uses
  `asctile.range`), no nested functions.

## 9. Validation plan (executed in a later step, not now)

1. `python3 -m py_compile candidate.py` — syntax only (no pyasc/asc/asctile
   installed locally; static reasoning only on this host).
2. Exact-v2 compile gate (operator-specific):
   ```
   integrations/cannbench/workers/run_local_compile_gate.sh \
     --candidate .../candidate.py --op gelu
   ```
   Must compile/lower all 20 case specializations (2 modes × 3 dtypes × tail
   variants) under the 950PR UB check. A failure is measured feedback; halve
   the offending tile and re-run all 20.
3. Camodel adversarial smoke (when native runtime available):
   `run_camodel_smoke.py --suite adversarial --ops gelu` — must exercise both
   modes, wide negative ranges, bf16 (case 12 Inf/NaN positions), and the
   padded tail lanes. Target label: `verified-camodel-smoke` with declared
   scope.
4. CANNBench remote (opt-in, credit-consuming) is the only acceptance oracle
   for real-NPU correctness and perf; not triggered in this step.

## 10. Risks and mitigations

| risk | likelihood | mitigation |
|------|-----------|------------|
| Exact UB overflow at 512/uf2 for an unseen dtype/tail combo | low (measured baseline) | halve to 256/uf2; never drop a case |
| `asctile.minimum` NaN-handling differs from golden for case 12 | low (NaN propagates either way) | switch that line to `asctile.where(s < 0.0, s, zero)` — both are LocalTensor branches, codegen-equivalent |
| bf16 case 12 position mismatch (Inf/NaN) | low (formula preserves IEEE) | the operator-patterns.md formulas are explicitly chosen to preserve positions; no host special-casing to regress it |
| f16 case 10 cubic overflow changing result | none (f32 internal) | x^3 in f32; even f32 overflow → ±inf feeds the stable-sigmoid correctly |
| Larger exact tile assumed safe from source count | none | design forbids it; only compile-gate evidence grows the tile |
| `one`/`zero` hoist breaks under unroll | low | if it does, move them inside the loop (UB impact negligible; they are read-only) |

## 11. Summary of design choices

- **Two kernels** (`_gelu_none_kernel` @512/uf2, `_gelu_tanh_kernel`
  @1024/uf2), dispatched by the `approximate` string on the host.
- **Exact**: cancellation-free `where(x>=0, x - 0.5*x*erfc(|x|/sqrt(2)),
  0.5*x*erfc(|x|/sqrt(2)))` with the NR 10-coefficient erfc Chebyshev fit.
- **Tanh**: stable-sigmoid `x * exp(min(s,0)) / (1 + exp(-|s|))` with
  `s = 2*sqrt(2/pi)*(x + 0.044715*x^3)`.
- **f32 internal compute** for f16/bf16; single cast on `copy_out`.
- **Grid-stride + `real_shape` tails**, `pad_value=0` (auto, operation-neutral
  because all divisors are `>= 1`).
- **72 cores** for every case; no wide/narrow host split (tiles are UB-capped).
- **No host special-casing** of Inf/NaN/zero — IEEE propagation matches golden
  positions.
- **Anti-cheat clean**: all math in `@asctile.jit`; torch only for
  allocation/metadata/contiguity; tensors passed directly to the JIT.

DESIGN_DONE
