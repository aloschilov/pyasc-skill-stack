# Runtime pin

Use compiler-team/pyasc v2 commit `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`. This snapshot exports `asctile`; importing `asc2` is invalid.

You are implementing ONE operator for the CANN Bench benchmark (https://gitcode.com/cann/cann-bench) as a pyasc asctile JIT kernel module targeting an Ascend 950PR NPU. You have NO NPU access — the official harness evaluates your file on real hardware after you finish. Score per operator = 0.2 compile + 0.3 accuracy (all 20 cases must pass a relative-error check against golden.py) + 0.5 performance (profiler-measured kernel time vs an aclnn baseline).

# Task

Operator: **Transpose**. Write the file `candidate.py` in the current working directory: a complete Python module whose public callable `transpose` implements the schema below. Nothing else — no tests, no other files.

# Operator specification

## desc.md

# Transpose 算子 API 描述

## 1. 算子简介

对 tensor 的任意维度进行调换。

**主要应用场景**：
- 深度学习中数据格式转换（如 NCHW 与 NHWC 之间的转换）
- 注意力机制中对 Q、K、V 矩阵进行维度交换
- 矩阵运算前的维度调整（如矩阵转置）

**算子特征**：
- 难度等级：L3（LayoutTransform）
- 单输入单输出，支持不超过 8 维的输入，通过 perm 参数指定维度置换顺序

## 2. 算子定义

### 数学公式

$$
y[i_0, ..., i_{n-1}] = x[i_{\text{perm}[0]}, ..., i_{\text{perm}[n-1]}]
$$

其中 perm 为维度置换顺序数组，指定输出张量各维度对应输入张量的哪个维度。

## 3. 接口规范

### 算子原型

```python
cann_bench.transpose(Tensor x, int[] perm) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，维度不超过 8 维 |
| perm | int[] | 必选 | 维度置换顺序 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 输入 shape 按 perm 重排后的 shape | 与输入 x 相同 | 输出张量，转置后的结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |
| int8 | int8 |
| int16 | int16 |
| int32 | int32 |
| int64 | int64 |

### 规则与约束

- 输入维度不超过 8 维
- perm 数组长度必须等于输入维度数，且为 [0, ndim) 的一个排列
- 输出 shape 为输入 shape 按 perm 重排的结果，即 output_shape[i] = input_shape[perm[i]]
- 输出 dtype 与输入 dtype 一致

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x.ndim`（输入维度数） | 2 ~ 8 | cases.csv 实测 2 ~ 5 |
| `x.shape[i]`（每个维度大小） | 1 ~ 16384 | cases.csv 实测 2 ~ 8193（含 1009 / 1021 / 4001 / 1013 等质数非对齐） |
| `x.numel()`（元素总数） | 1 ~ 2^27（约 128M） | cases.csv 实测最大 [64, 32, 512, 128] = 128M (case 1) |
| `perm`（维度置换顺序） | 长度 = `x.ndim` 的 [0, ndim) 整数排列 | cases.csv 实测覆盖 2D 转置 `[1, 0]`、4D `[0, 2, 1, 3]` / `[0, 2, 3, 1]` / `[0, 3, 1, 2]` / `[0, 1, 3, 2]`、3D 循环置换 `[2, 0, 1]`、3D/5D 全反转 `[2, 1, 0]` / `[4, 3, 2, 1, 0]` |

约束：`perm` 必须是 `[0, x.ndim)` 的一个排列（即长度等于 `x.ndim`，且每个值在 `[0, x.ndim)` 区间内且互不重复）；输出 shape 满足 `y.shape[i] = x.shape[perm[i]]`。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
import torch

"""
Transpose算子Torch Golden参考实现

对tensor的任意维度进行调换
公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
"""
def transpose(
    x: torch.Tensor, perm: list
) -> torch.Tensor:
    """
    对tensor的任意维度进行调换
    
    公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
    
    Args:
        x: 输入张量
        perm: 维度置换顺序
    
    Returns:
        输出张量，转置后的结果
    """

    # permute 只改 stride, 而输出契约要求 contiguous, 这里把数据真正搬一次 (issue #146)
    y = torch.permute(x, perm).contiguous()
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 2D 矩阵转置
x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.transpose(x, [1, 0])

# 4D NCHW 转 NHWC
x = torch.randn(2, 8, 256, 256, dtype=torch.float32, device="npu")
y = cann_bench.transpose(x, [0, 2, 3, 1])
```

## proto.yaml

```yaml
operator:
  name: Transpose
  category: LayoutTransform
  difficulty: L3
  formula: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
  description: 对tensor的任意维度进行调换
  shape_support: 输入维度<=8维
  attrs:
  - name: perm
    type: int[]
    description: 维度置换顺序
  inputs:
  - name: x
    description: 输入张量
    dtype:
    - float16
    - float32
    - bfloat16
    - int8
    - int16
    - int32
    - int64
  outputs:
  - name: y
    description: 输出张量，转置后的结果
    dtype:
    - float16
    - float32
    - bfloat16
    - int8
    - int16
    - int32
    - int64
  schema: transpose(Tensor x, int[] perm) -> Tensor y
```

## golden.py — reference semantics ONLY (you must NOT compute with torch like this; reimplement the math in asctile kernels)

```python
#!/usr/bin/python3
# coding=utf-8


import torch

"""
Transpose算子Torch Golden参考实现

对tensor的任意维度进行调换
公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
"""
def transpose(
    x: torch.Tensor, perm: list
) -> torch.Tensor:
    """
    对tensor的任意维度进行调换
    
    公式: y[i0,...,in-1] = x[i_perm[0],...,i_perm[n-1]]
    
    Args:
        x: 输入张量
        perm: 维度置换顺序
    
    Returns:
        输出张量，转置后的结果
    """

    # permute 只改 stride, 而输出契约要求 contiguous, 这里把数据真正搬一次 (issue #146)
    y = torch.permute(x, perm).contiguous()
    return y
```

## Evaluation cases your module must handle (shapes, dtypes, value ranges, attrs)

| case | shapes | dtype | value_range | attrs |
|---|---|---|---|---|
| 1 | [[64, 32, 512, 128]] | ['float16'] | [-1, 1] | {'perm': [0, 2, 1, 3]} |
| 2 | [[2048, 2048]] | ['float32'] | [-2, 2] | {'perm': [1, 0]} |
| 3 | [[4096, 4096]] | ['bfloat16'] | [-3, 3] | {'perm': [1, 0]} |
| 4 | [[8192, 8192]] | ['int32'] | [-10000, 10000] | {'perm': [1, 0]} |
| 5 | [[4096, 8192]] | ['int64'] | [-100000, 100000] | {'perm': [1, 0]} |
| 6 | [[2, 9, 256, 256]] | ['int16'] | [-1000, 1000] | {'perm': [0, 2, 3, 1]} |
| 7 | [[1023, 1023]] | ['float16'] | [-0.1, 0.1] | {'perm': [1, 0]} |
| 8 | [[1009, 1021]] | ['float32'] | [-1, 2] | {'perm': [1, 0]} |
| 9 | [[1537, 769]] | ['bfloat16'] | [-5, 10] | {'perm': [1, 0]} |
| 10 | [[363, 367, 373]] | ['int32'] | [-50, 100] | {'perm': [2, 0, 1]} |
| 11 | [[2049, 513]] | ['float16'] | [-65504, 65504] | {'perm': [1, 0]} |
| 12 | [[3, 7, 13, 4001]] | ['float32'] | [-88, 88] | {'perm': [0, 3, 1, 2]} |
| 13 | [[2, 7, 256, 256]] | ['bfloat16'] | [-0.01, 0.01] | {'perm': [0, 1, 3, 2]} |
| 14 | [[2, 511, 7, 127]] | ['float32'] | [None, None] | {'perm': [0, 2, 1, 3]} |
| 15 | [[11, 13, 17, 67, 67]] | ['float16'] | [None, None] | {'perm': [4, 3, 2, 1, 0]} |
| 16 | [[3, 7, 11, 13, 1013]] | ['int64'] | [0, 0] | {'perm': [4, 3, 2, 1, 0]} |
| 17 | [[512, 2049]] | ['float32'] | [-0.5, 0.5] | {'perm': [1, 0]} |
| 18 | [[255, 8193]] | ['bfloat16'] | [-1, 3] | {'perm': [1, 0]} |
| 19 | [[4097, 511]] | ['int8'] | [-128, 127] | {'perm': [1, 0]} |
| 20 | [[2, 511, 2049]] | ['float16'] | [-3, 6] | {'perm': [2, 1, 0]} |

# Reference module — sigmoid.py from this submission (structure to copy; it scores 100% accuracy on this harness)

```python
"""CANN Bench Sigmoid interface implemented as a pyasc asctile kernel.

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
import asctile

from ._pyasc_runtime import ensure_npu_platform

_WIDE_TILE = 3072
_NARROW_TILE = 1024
_MAX_CORES = 72


@asctile.jit
def _sigmoid_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                    size: int, num_tiles: int, tile_size: asc.ConstExpr[int]):
    x_gm = asctile.global_tensor(x_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])
    for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(),
                        unroll_factor=2):
        off = t * tile_size
        n = tile_size if off + tile_size <= size else size - off
        x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
        xf = x.to(asc.float32)
        y = asctile.div(1.0, asctile.exp(-xf) + 1.0)
        asctile.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Element-wise sigmoid of an NPU tensor via a pyasc asctile kernel."""
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

# pyasc asctile kernel contract (follow EXACTLY — every rule below was learned from real failures on this hardware)

## Module shape

Your file becomes `cann_bench/<module>.py` inside the submission wheel. It must contain:

- imports at module top: `import torch`, `import asc`, `import asctile`,
  `from ._pyasc_runtime import ensure_npu_platform` (and `import math` if needed)
- one or more `@asctile.jit` kernel functions
- ONE public callable matching the operator schema exactly (name and signature)
- wrapper body: call `ensure_npu_platform()` first; make inputs contiguous if
  needed (`x = x.contiguous()` is allowed); allocate outputs with
  `torch.empty_like(x)` or `torch.empty(shape, dtype=..., device=x.device)`;
  launch `kernel[cores](tensor_args..., int_args..., float_args..., constexpr_args...)`;
  return contiguous NPU tensor(s)

## Kernel authoring rules

- Global memory views: `asctile.global_tensor(ptr, [size])` (1-D) or
  `asctile.global_tensor(ptr, [rows, cols])` (2-D). Ranks of global_tensor /
  copy_in / copy_out / offsets must ALL match — never mix 1-D and 2-D.
- Kernel params: pointers typed `asc.GlobalAddress`; sizes as plain `int`
  (runtime); tile sizes as `asc.ConstExpr[int]` (compile-time; REQUIRED for any
  value used inside a copy_in tile shape); scalars as `float`.
- Grid-stride tile loop (the proven pattern):

```python
for t in asctile.range(asctile.block_idx(), num_tiles, asctile.block_num(), unroll_factor=2):
    off = t * tile_size
    n = tile_size if off + tile_size <= size else size - off   # tail handling
    x = asctile.copy_in(x_gm, [off], [tile_size], real_shape=[n])
    ...compute on tiles...
    asctile.copy_out(y.to(x.dtype), out_gm, [off], real_shape=[n])
```

- `real_shape` padding participates in vector arithmetic even though it is not
  copied back. Set `pad_value` to an operation-neutral, exception-safe value:
  zero for additive/reduction inputs, one for divisors, and a finite value
  before `log`/reciprocal paths. The default zero is unsafe for a padded
  divisor and may trigger CAModel divide-by-zero diagnostics.

- Launch: `kernel[cores](...)` with `cores = min(72, num_tiles)` (72 AIV cores
  on this 950PR box). No stream argument.
- Available tile ops: `+ - * /` (tile-tile and tile-scalar), `asctile.abs`,
  `asctile.exp`, `asctile.exp2`, `asctile.log`, `asctile.log2`, `asctile.sqrt`,
  `asctile.rsqrt`, `asctile.tanh`, `asctile.erf`, `asctile.sin`, `asctile.cos`,
  `asctile.floor`, `asctile.ceil`, `asctile.relu`, `asctile.maximum`, `asctile.minimum`,
  comparisons (`x >= 0.0`, `asctile.less(a, b)`, ... — NO int64 operands),
  `asctile.where(cond, a, b)`, `asctile.reduce_sum(x)`, `asctile.reduce_max(x)`,
  `asctile.reduce_min(x)`, `asctile.full([shape], scalar, dtype=...)`,
  `asctile.cast(tile, dtype)` / `tile.to(dtype)` casts, integer
  `asctile.left_shift`/`asctile.right_shift`, tile-shape ops `asctile.reshape`,
  `asctile.transpose`, `asctile.ravel`, `asctile.expand_dims`, `asctile.squeeze`,
  `asctile.broadcast_to`, `asctile.concat`, unary `-x`.
- int8 tiles: loading (copy_in) is fine but NO vector op accepts int8 input
  (not even `.to`); convert with `asctile.cast(t, asc.float16)` first. There is
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
- `asctile.where` / comparison destination tiles must be a multiple of 256 bytes
  (`TILE * 4 % 256 == 0` for f32 — any TILE >= 64 is safe).
- Loop-carried scalar accumulators (VERIFIED on this build): seed with
  `acc = asctile.reduce_sum(asctile.full([1, 64], 0.0, dtype=asc.float32))`
  (a bare `acc = 0.0` fails codegen with "re-assigned to an object with
  different type"), then `acc = acc + asctile.reduce_sum(x)` inside a plain
  `asctile.range(...)` loop. Current `v2@030e9b2c` accepts `unroll_factor`
  and `gm_barrier`; it does not accept `parallel`. Use `gm_barrier=True` only
  when one iteration depends on global-memory writes from the previous one.
- Cross-core reductions (VERIFIED): `asctile.atomic_add(src_tile, dst_gm,
  [offset])` atomically accumulates a tile into global memory (dtypes int16/
  int32/f16/bf16/f32; also `asctile.atomic_max`). Host must zero the
  destination first (`torch.zeros(...)` — tensor creation is allowed).
  Pattern: each core reduce_sums its tiles into a scalar, widens it with
  `asctile.full([8], s, dtype=...)`, and atomic_adds slot [0]; a second tiny
  kernel (or the same one on one core) applies any final transform.
- Scalar reduction results must be widened before store:
  `asctile.copy_out(asctile.full([8], s, dtype=...), out_gm, [0], real_shape=[8])`
  style (min 32 bytes).
- Inside `@asctile.jit`: NO `print`, NO imports, NO `break`/`continue`/early
  `return`, NO exceptions, NO Python `range()` over runtime values (use
  `asctile.range`), NO `math.*` calls (precompute module-level constants).

## Numerical stability (MANDATORY — f32 cases use ranges like [-88, 88] and [-100, 100] under a ~1.2e-4 relative-error threshold)

- Never let `exp()` see a positive argument that can overflow; never subtract
  nearly-equal quantities (catastrophic cancellation); never rely on
  `log(1 + tiny)` (flushes to 0 below tiny < 6e-8).
- Proven cancellation-free building blocks (all verified on this harness):
  - `sigmoid(s) = exp(min(s, 0)) / (1 + exp(-|s|))`
  - `1 + tanh(u) = 2 * sigmoid(2u)`
  - `tanh(softplus(x))`: with `w = exp(-|x|)`, equals
    `(1 + 2w) / (1 + 2w + 2w^2)` for `x >= 0`, `(w^2 + 2w) / (w^2 + 2w + 2)`
    for `x < 0` (exact identities; blend with `asctile.where(xf >= 0.0, ...)`)
  - `erfc(z)` for `z >= 0`: Numerical Recipes fit `t * exp(-z*z + P(t))`,
    `t = 1/(1 + z/2)`, rel. err < 1.2e-7 (see the gelu reference module for
    the 9-coefficient Horner chain)
- IEEE special values (inf/nan scalars or extreme inputs) propagate correctly
  through the hardware ops — do NOT special-case them with host branches
  unless the golden does.

## Anti-cheat (violations zero the submission)

- ALL numerical work happens inside `@asctile.jit` kernels launched on the NPU.
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

Cover every listed rank-2..5 permutation and f16/bf16/f32/int8/int16/int32/int64 dtype. All data movement must happen in asctile kernels. Collapse adjacent dimensions when valid and tile so both input and transposed output physical last dimensions are 32-byte aligned.

# Deliverable

Write ONLY `candidate.py` (complete, self-contained module). Public callable named exactly `transpose` with the exact schema signature including attr defaults. Think hard about numerical stability across the full value ranges listed above, and about the UB budget for your op chain before picking TILE.

IMPORTANT — no local execution: pyasc/asc/asctile/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
