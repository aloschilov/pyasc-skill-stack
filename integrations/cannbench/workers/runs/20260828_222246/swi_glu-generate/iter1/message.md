You are implementing ONE operator for the CANN Bench benchmark (https://gitcode.com/cann/cann-bench) as a pyasc asc2 JIT kernel module targeting an Ascend 950PR NPU. You have NO NPU access — the official harness evaluates your file on real hardware after you finish. Score per operator = 0.2 compile + 0.3 accuracy (all 20 cases must pass a relative-error check against golden.py) + 0.5 performance (profiler-measured kernel time vs an aclnn baseline).

# Task

Operator: **SwiGlu**. Write the file `candidate.py` in the current working directory: a complete Python module whose public callable `swi_glu` implements the schema below. Nothing else — no tests, no other files.

# Operator specification

## desc.md

# SwiGlu 算子 API 描述

## 1. 算子简介

标准 SwiGLU 激活函数(Shazeer 2020 "GLU Variants Improve Transformer")。输入在指定 dim 上拆分为 x0 / x1 两等份,x0 经 SiLU 激活(`Swish_1(x) = x · sigmoid(x)`)后与 x1 做门控乘法。

**主要应用场景**：
- LLM FFN 中的 SwiGLU 门控(Llama / PaLM / Gemma 等)
- 替代传统 GLU 与 ReLU FFN,提供更平滑的梯度与更强表达力

**算子特征**：
- 难度等级：L1（Elementwise）
- P2 op：torch 无同名接口;reference 实现取自 `torch_npu.npu_swiglu` / ACLNN `aclnnSwiGlu`,两者都固定 Swish 的 β = 1(即 SiLU)
- 单输入单输出,沿指定 dim 拆分 + 元素级运算

## 2. 算子定义

### 数学公式

$$
\text{output} = \text{SiLU}(x_0) \odot x_1 = \left( x_0 \cdot \sigma(x_0) \right) \odot x_1
\quad \text{其中} \quad (x_0, x_1) = \text{chunk}(input, 2, \text{dim})
$$

`σ(·)` 为 sigmoid;`⊙` 为逐元素乘法。

> **注**:历史上 Swish 有可调参数 β(`Swish_β(x) = x · sigmoid(β·x)`),但 SwiGLU 在文献与主流实现(Llama / PaLM / torch_npu / aclnnSwiGlu)中**统一固定 β = 1**(等价 SiLU)。本 spec 不暴露 β,与 reference 实现严格对齐。

## 3. 接口规范

### 算子原型

```python
cann_bench.swi_glu(Tensor input, int dim=-1) -> Tensor output
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input | Tensor | 必选 | 输入张量,dim 维 size 必须是偶数 |
| dim | int64 | -1 | 拆分维度 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| output | 与 input 同 rank,dim 维 size 折半 | 与 input 相同 | SwiGLU 激活输出 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16  | float16  |
| float32  | float32  |
| bfloat16 | bfloat16 |

### 规则与约束

- input 在 dim 维上 size 必须是偶数(否则无法等分 x0 / x1)
- `dim` 支持负数索引(如 -1 表示最后一维)
- output dtype 与 input 一致
- FP16 / BF16 输入内部计算时升精度到 FP32,再 cast 回原 dtype(与 ACLNN 一致)

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `rank(input)` | 2 ~ 8 | cases.csv 实测 2 ~ 5 维 |
| 每个维度大小 `dim_i` | ≥ 2 (split 维需偶) | cases.csv 实测最大 1,000,003 |
| 张量总元素数 | 1 ~ 2^30 | cases.csv 实测最大约 134M (8192×16384) |
| `dim` | -rank(input) ~ rank(input)-1 | cases.csv 实测 dim=-1 / 0 / 1 / 2 |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**:

1. 平均相对误差(MERE):采样点中相对误差平均值
2. 最大相对误差(MARE):采样点中相对误差最大值

**通过标准**:

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---------|----------|---------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 |

当 MERE < Threshold,MARE < 10 × Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
import torch
import torch.nn.functional as F


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """标准 SwiGLU 激活的 Torch Golden 参考实现 (P2 op).

    公式: output = silu(x0) * x1 = (x0 * sigmoid(x0)) * x1
    其中 x0, x1 = input.chunk(2, dim=dim).
    """
    out_dtype = input.dtype
    x = input.to(torch.float)
    x0, x1 = x.chunk(2, dim=dim)
    output = F.silu(x0) * x1
    return output.to(out_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 2048, dtype=torch.float32, device="npu")
y = cann_bench.swi_glu(x, dim=-1)         # 输出 shape [1024, 1024]

x = torch.randn(8192, 16384, dtype=torch.float16, device="npu")
y = cann_bench.swi_glu(x, dim=0)          # 输出 shape [4096, 16384]
```

## proto.yaml

```yaml
operator:
  name: SwiGlu
  category: Elementwise
  difficulty: L1
  formula: output = silu(x0) * x1 = (x0 * sigmoid(x0)) * x1
  description: 标准 SwiGLU 激活——输入在指定 dim 上等分成 x0 / x1,对 x0 做 SiLU (Swish_1),再与 x1 逐元素相乘
  shape_support: 单输入,在指定 dim 上拆分成两等份;该 dim 的 size 必须是偶数
  attrs:
  - name: dim
    type: int64
    description: 拆分维度,默认为最后一维
    required: false
    default: -1
  inputs:
  - name: input
    description: 输入张量,会在 dim 上拆分成 x0 / x1 两等份
    dtype:
    - float16
    - float32
    - bfloat16
  outputs:
  - name: output
    description: 输出张量,shape 同 input 但 dim 维大小折半
    dtype:
    - float16
    - float32
    - bfloat16
  schema: swi_glu(Tensor input, int dim=-1) -> Tensor output
```

## golden.py — reference semantics ONLY (you must NOT compute with torch like this; reimplement the math in asc2 kernels)

```python
#!/usr/bin/python3
# coding=utf-8


import torch
import torch.nn.functional as F


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """标准 SwiGLU 激活的 Torch Golden 参考实现 (P2 op).

    公式: output = silu(x0) * x1 = (x0 * sigmoid(x0)) * x1
    其中 x0, x1 = input.chunk(2, dim=dim).

    标准 SwiGLU (Shazeer 2020 / Llama / PaLM) 固定 Swish 的 beta = 1
    (即等价于 SiLU),没有可调 beta 参数。aclnnSwiGlu / torch_npu.npu_swiglu
    也是同样定义,本 spec 与之对齐。

    Args:
        input: 输入张量,dim 维度上 size 必须是偶数
        dim: 拆分维度,默认 -1

    Returns:
        output: 与 input 同 dtype/shape 但 dim 维度大小减半
    """
    # FP16/BF16 升精度计算,与 ACLNN 内部一致.
    out_dtype = input.dtype
    x = input.to(torch.float)
    x0, x1 = x.chunk(2, dim=dim)
    output = F.silu(x0) * x1
    return output.to(out_dtype)
```

## Evaluation cases your module must handle (shapes, dtypes, value ranges, attrs)

| case | shapes | dtype | value_range | attrs |
|---|---|---|---|---|
| 1 | [[1024, 2048]] | ['float16'] | [-1, 1] | {'dim': -1} |
| 2 | [[2048, 4096]] | ['float32'] | [-2, 2] | {'dim': -1} |
| 3 | [[4096, 8192]] | ['bfloat16'] | [-3, 3] | {'dim': -1} |
| 4 | [[8192, 16384]] | ['float16'] | [-10, 10] | {'dim': 0} |
| 5 | [[2039, 65520]] | ['float32'] | [-100, 100] | {'dim': -1} |
| 6 | [[1022, 2047]] | ['bfloat16'] | [-0.1, 0.1] | {'dim': 0} |
| 7 | [[1009, 2016]] | ['float16'] | [-1, 2] | {'dim': -1} |
| 8 | [[1538, 1537]] | ['float32'] | [-5, 10] | {'dim': 0} |
| 9 | [[363, 367, 14]] | ['bfloat16'] | [-50, 100] | {'dim': -1} |
| 10 | [[2049, 1024]] | ['float16'] | [-65504, 65504] | {'dim': 1} |
| 11 | [[3, 7, 13, 1018]] | ['float32'] | [-88, 88] | {'dim': -1} |
| 12 | [[1000003, 2]] | ['bfloat16'] | [-inf, inf] | {'dim': 1} |
| 13 | [[11, 13, 16, 67]] | ['float32'] | [nan, nan] | {'dim': 2} |
| 14 | [[3, 7, 11, 13, 1012]] | ['float16'] | [0, 0] | {'dim': -1} |
| 15 | [[512, 4096]] | ['float32'] | [-0.5, 0.5] | {'dim': -1} |
| 16 | [[254, 16383]] | ['bfloat16'] | [-1, 3] | {'dim': 0} |
| 17 | [[4096, 1023]] | ['float16'] | [-1000, 1000] | {'dim': 0} |
| 18 | [[2, 1023, 4096]] | ['float32'] | [-0.2, 0.2] | {'dim': -1} |
| 19 | [[4, 510, 4097]] | ['bfloat16'] | [-3, 6] | {'dim': 1} |
| 20 | [[2, 3, 17, 512, 100]] | ['float32'] | [-20, 40] | {'dim': -1} |

# Reference module — sigmoid.py from this submission (structure to copy; it scores 100% accuracy on this harness)

```python
"""CANN Bench Sigmoid interface implemented as a pyasc asc2 kernel.

Kernel design (pyasc-api-patterns Pattern A, grid-stride variant):
  - 1-D flatten; each block strides over tiles so any element count works.
  - Tail tiles handled with ``real_shape`` loads/stores (no host padding).
  - f16/bf16 inputs are promoted to f32 inside the kernel for precision
    (the spec's precision standard expects f32 internal compute).
  - Host selects between two compiled tile sizes: a wide tile (3072) when
    the element count fills all 72 cores, otherwise a narrow tile (1024) to
    maximize core utilization on small shapes.
  - sigmoid(x) = 1 / (1 + e^(-x)); IEEE saturation gives the correct
    limits at extreme inputs (e^inf -> inf -> y=0, e^-inf -> 0 -> y=1).
"""

import torch

import asc
import asc2

from ._pyasc_runtime import ensure_npu_platform

_WIDE_TILE = 3072
_NARROW_TILE = 1024
_MAX_CORES = 72


@asc2.jit
def _sigmoid_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                    size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.global_tensor(x_ptr, [size])
    out_gm = asc2.global_tensor(out_ptr, [size])
    for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        y = asc2.div(1.0, asc2.exp(-xf) + 1.0)
        asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Element-wise sigmoid of an NPU tensor via a pyasc asc2 kernel."""
    ensure_npu_platform()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    size = x.numel()
    if size == 0:
        return out
    if size >= _MAX_CORES * _WIDE_TILE:
        num_tiles = asc.ceildiv(size, _WIDE_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _sigmoid_kernel[cores](x, out, size, num_tiles, _WIDE_TILE)
    else:
        num_tiles = asc.ceildiv(size, _NARROW_TILE)
        cores = min(_MAX_CORES, num_tiles)
        _sigmoid_kernel[cores](x, out, size, num_tiles, _NARROW_TILE)
    return out

```

# pyasc asc2 kernel contract (follow EXACTLY — every rule below was learned from real failures on this hardware)

## Module shape

Your file becomes `cann_bench/<module>.py` inside the submission wheel. It must contain:

- imports at module top: `import torch`, `import asc`, `import asc2`,
  `from ._pyasc_runtime import ensure_npu_platform` (and `import math` if needed)
- one or more `@asc2.jit` kernel functions
- ONE public callable matching the operator schema exactly (name and signature)
- wrapper body: call `ensure_npu_platform()` first; make inputs contiguous if
  needed (`x = x.contiguous()` is allowed); allocate outputs with
  `torch.empty_like(x)` or `torch.empty(shape, dtype=..., device=x.device)`;
  launch `kernel[cores](tensor_args..., int_args..., float_args..., constexpr_args...)`;
  return contiguous NPU tensor(s)

## Kernel authoring rules

- Global memory views: `asc2.global_tensor(ptr, [size])` (1-D) or
  `asc2.global_tensor(ptr, [rows, cols])` (2-D). Ranks of global_tensor /
  copy_in / copy_out / offsets must ALL match — never mix 1-D and 2-D.
- Kernel params: pointers typed `asc.GlobalAddress`; sizes as plain `int`
  (runtime); tile sizes as `asc.ConstExpr[int]` (compile-time; REQUIRED for any
  value used inside a copy_in tile shape); scalars as `float`.
- Grid-stride tile loop (the proven pattern):

```python
for t in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(), unroll_factor=2):
    off = t * tile_size
    n = tile_size if off + tile_size <= size else size - off   # tail handling
    x = asc2.copy_in(x_gm, [off], [tile_size], real_shape=[n])
    ...compute on tiles...
    asc2.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])
```

- Launch: `kernel[cores](...)` with `cores = min(72, num_tiles)` (72 AIV cores
  on this 950PR box). No stream argument.
- Available tile ops: `+ - * /` (tile-tile and tile-scalar), `asc2.abs`,
  `asc2.exp`, `asc2.exp2`, `asc2.log`, `asc2.log2`, `asc2.sqrt`,
  `asc2.rsqrt`, `asc2.tanh`, `asc2.erf`, `asc2.sin`, `asc2.cos`,
  `asc2.floor`, `asc2.ceil`, `asc2.relu`, `asc2.maximum`, `asc2.minimum`,
  comparisons (`x >= 0.0`, `asc2.less(a, b)`, ... — NO int64 operands),
  `asc2.where(cond, a, b)`, `asc2.reduce_sum(x)`, `asc2.reduce_max(x)`,
  `asc2.reduce_min(x)`, `asc2.full([shape], scalar, dtype=...)`,
  `asc2.cast(tile, dtype)` / `tile.to(dtype)` casts, integer
  `asc2.left_shift`/`asc2.right_shift`, tile-shape ops `asc2.reshape`,
  `asc2.transpose`, `asc2.ravel`, `asc2.expand_dims`, `asc2.squeeze`,
  `asc2.broadcast_to`, `asc2.concat`, unary `-x`.
- int8 tiles: loading (copy_in) is fine but NO vector op accepts int8 input
  (not even `.to`); convert with `asc2.cast(t, asc.float16)` first. There is
  no uint8 tile dtype at all.
- Scalars go on the RIGHT of tile arithmetic (Tile has no `__rmul__`):
  write `x * 0.5`, NEVER `0.5 * x`. Same for `+ - /`.
- f16/bf16 inputs: promote to f32 in-kernel (`xf = x.to(asc.float32)`),
  compute in f32, cast back on copy_out (`y.to(x.dtype)`).
- UB (unified buffer) budget: ~253952 bytes total under static allocation.
  Every distinct f32 tile value costs `4 * TILE` bytes, `unroll_factor=2`
  doubles the total, and the compiler adds hidden temporaries — MEASURED
  calibration: the sigmoid chain (f16 load, f32 cast, `-x`, `exp`, `+1`,
  `div`, f16 store ≈ 6 visible values) uses 155648 bytes at TILE=2048 and
  311296 (OVERFLOW) at TILE=4096, i.e. real usage ≈ 1.6x the naive
  `visible_values * 4 * TILE * 2` estimate. Budget with that 1.6x factor.
  Rule of thumb: TILE=2048 for short chains (< 8 values), 1024 for medium,
  512 for long (> 16). A launch failing with `RuntimeError: UB overflow: X
  bytes are available, Y bytes are used` means: halve TILE (do NOT drop
  cases).
- `asc2.where` / comparison destination tiles must be a multiple of 256 bytes
  (`TILE * 4 % 256 == 0` for f32 — any TILE >= 64 is safe).
- Loop-carried scalar accumulators (VERIFIED on this build): seed with
  `acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))`
  (a bare `acc = 0.0` fails codegen with "re-assigned to an object with
  different type"), then `acc = acc + asc2.reduce_sum(x)` inside a plain
  `asc2.range(...)` loop. `asc2.range` accepts ONLY `unroll_factor` and
  `parallel` — there is NO `gm_barrier` kwarg on this build (it raises
  TypeError).
- Cross-core reductions (VERIFIED): `asc2.atomic_add(src_tile, dst_gm,
  [offset])` atomically accumulates a tile into global memory (dtypes int16/
  int32/f16/bf16/f32; also `asc2.atomic_max`). Host must zero the
  destination first (`torch.zeros(...)` — tensor creation is allowed).
  Pattern: each core reduce_sums its tiles into a scalar, widens it with
  `asc2.full([8], s, dtype=...)`, and atomic_adds slot [0]; a second tiny
  kernel (or the same one on one core) applies any final transform.
- Scalar reduction results must be widened before store:
  `asc2.copy_out(asc2.full([8], s, dtype=...), out_gm, [0], real_shape=[8])`
  style (min 32 bytes).
- Inside `@asc2.jit`: NO `print`, NO imports, NO `break`/`continue`/early
  `return`, NO exceptions, NO Python `range()` over runtime values (use
  `asc2.range`), NO `math.*` calls (precompute module-level constants).

## Numerical stability (MANDATORY — f32 cases use ranges like [-88, 88] and [-100, 100] under a ~1.2e-4 relative-error threshold)

- Never let `exp()` see a positive argument that can overflow; never subtract
  nearly-equal quantities (catastrophic cancellation); never rely on
  `log(1 + tiny)` (flushes to 0 below tiny < 6e-8).
- Proven cancellation-free building blocks (all verified on this harness):
  - `sigmoid(s) = exp(min(s, 0)) / (1 + exp(-|s|))`
  - `1 + tanh(u) = 2 * sigmoid(2u)`
  - `tanh(softplus(x))`: with `w = exp(-|x|)`, equals
    `(1 + 2w) / (1 + 2w + 2w^2)` for `x >= 0`, `(w^2 + 2w) / (w^2 + 2w + 2)`
    for `x < 0` (exact identities; blend with `asc2.where(xf >= 0.0, ...)`)
  - `erfc(z)` for `z >= 0`: Numerical Recipes fit `t * exp(-z*z + P(t))`,
    `t = 1/(1 + z/2)`, rel. err < 1.2e-7 (see the gelu reference module for
    the 9-coefficient Horner chain)
- IEEE special values (inf/nan scalars or extreme inputs) propagate correctly
  through the hardware ops — do NOT special-case them with host branches
  unless the golden does.

## Anti-cheat (violations zero the submission)

- ALL numerical work happens inside `@asc2.jit` kernels launched on the NPU.
- torch usage is allowed ONLY for: output allocation (`torch.empty`,
  `torch.empty_like`), metadata (`.shape`, `.numel()`, `.stride()`, `.dtype`,
  `.is_contiguous()`), contiguity (`.contiguous()`), and views (`.view`,
  `.reshape`, `.narrow`, indexing that returns a view).
- FORBIDDEN anywhere in the module: torch math/compute ops (`torch.mul`,
  `torch.norm`, `torch.nn.functional.*`, tensor arithmetic like `a + b`,
  `x.sigmoid()`, `.to(dtype)` casts of device data, `torch.cat`,
  `torch.clone`, `torch.sum`, ...). The harness hooks torch dispatch and
  rotates input data pointers between calls — caching outputs by `data_ptr`
  is detected and scored as cheating.
- Outputs must be contiguous NPU tensors with exactly the golden's
  shape/dtype. Do not return views of inputs.


# Operator-specific guidance

- input splits into x0 / x1 halves along attr dim (cases use dim 0, 1, 2 and
  -1; that dim is always even). output = silu(x0) * x1 where
  silu(v) = v * sigmoid(v); use the stable sigmoid form.
- Zero-copy strided access (do NOT call .contiguous()/.narrow() to
  materialize x0/x1 — extra device copies destroy the perf score): on the
  host compute outer = prod(shape[:dim]), C = shape[dim],
  inner = prod(shape[dim+1:]), half_cols = (C // 2) * inner. The contiguous
  input viewed as 2-D [outer, C * inner] has x0 in columns [0, half_cols)
  and x1 in columns [half_cols, 2 * half_cols) of every row. Kernel: 2-D
  global tensors (runtime row/col counts are fine), loop over (row,
  col-chunk) with [1, TILE] tiles and real_shape tails; output is
  [outer, half_cols].
- Distribute work over rows AND column-chunks so small-outer cases still use
  many cores (e.g. grid-stride over row * num_col_tiles + col_tile).
- MEASURED FAILURE to avoid: 2-D copy tiles require the LAST dimension to be
  32-byte aligned. Case 12 is [1000003, 2] bf16 with dim=1 -> half_cols = 1
  element = 2 bytes -> "RuntimeError: Last dimension of tensor must be
  aligned by 32 bytes, got 1 x 2 bytes". Add a host-side fallback for
  degenerate layouts: when `half_cols * input.element_size() < 32`, split
  with metadata ops (which the anti-cheat explicitly allows):
  `x0 = input.narrow(dim, 0, C // 2).contiguous()` and
  `x1 = input.narrow(dim, C // 2, C // 2).contiguous()`, then run a plain
  1-D elementwise silu-mul kernel over x0/x1. The two extra copy kernels
  cost perf only on that one case; all aligned cases must keep the
  zero-copy 2-D path.

# Deliverable

Write ONLY `candidate.py` (complete, self-contained module). Public callable named exactly `swi_glu` with the exact schema signature including attr defaults. Think hard about numerical stability across the full value ranges listed above, and about the UB budget for your op chain before picking TILE.

IMPORTANT — no local execution: pyasc/asc/asc2/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
