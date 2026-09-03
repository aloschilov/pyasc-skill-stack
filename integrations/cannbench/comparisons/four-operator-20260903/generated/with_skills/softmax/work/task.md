# Runtime pin

Use compiler-team/pyasc v2 commit `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`. This snapshot exports `asctile`; importing `asc2` is invalid.

You are implementing ONE operator for the CANN Bench benchmark (https://gitcode.com/cann/cann-bench) as a pyasc asctile JIT kernel module targeting an Ascend 950PR NPU. You have NO NPU access — the official harness evaluates your file on real hardware after you finish. Score per operator = 0.2 compile + 0.3 accuracy (all 20 cases must pass a relative-error check against golden.py) + 0.5 performance (profiler-measured kernel time vs an aclnn baseline).

# Task

Operator: **Softmax**. Write the file `candidate.py` in the current working directory: a complete Python module whose public callable `softmax` implements the schema below. Nothing else — no tests, no other files.

# Operator specification

## desc.md

# Softmax 算子 API 描述

## 1. 算子简介

沿指定维度计算 Softmax 归一化。

**主要应用场景**：
- 分类模型的输出层，将 logits 转换为概率分布
- 注意力机制中计算注意力权重
- 强化学习中的策略概率输出

**算子特征**：
- 难度等级：L2（Normalization）
- 单输入单输出，涉及指数运算、求和、除法等多步计算
- 输出元素值在 [0, 1] 范围内，沿指定维度求和为 1

## 2. 算子定义

### 数学公式

**基本公式**：

$$
y_i = \frac{\exp(x_i)}{\sum_{j}\exp(x_j)}
$$

数值稳定版本（内部实现）：

$$
y_i = \frac{\exp(x_i - \max(x))}{\sum_{j}\exp(x_j - \max(x))}
$$

其中：
- `x_i` 为输入张量沿指定 dim 维度上的第 i 个元素
- 输出满足 `0 <= y_i <= 1` 且 `sum(y) = 1`（沿 dim 维度）
- 数值稳定版本减去最大值以避免指数溢出

## 3. 接口规范

### 算子原型

```python
cann_bench.softmax(Tensor x, int dim=-1) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| dim | int | -1 | 计算 Softmax 的维度 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | Softmax 归一化后的张量 |

### 数据类型

| x dtype | 输出 dtype |
|---------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x 可以为任意维度的张量
- dim 指定计算 Softmax 的维度，支持负数索引
- 输出 shape 和 dtype 与输入完全一致
- 需注意数值稳定性：内部实现应使用减最大值技巧避免指数溢出
- 特殊值行为（须与 `torch.nn.functional.softmax` 一致，均源自 `x - max(x)` 重整）：
  - 某切片含任意 `+inf`（无论是否同时含其它有限/`-inf` 元素）：整切片输出 `NaN`（`inf - inf = NaN` 沿切片传播）
  - 某切片全部为 `-inf`：整切片输出 `NaN`（`max = -inf`，`-inf - (-inf) = NaN`）
  - 某切片含 `-inf` 与有限元素：`-inf` 位置输出 `0`，有限元素按正常 softmax 归一化
  - 输入含 `NaN`：整切片输出 `NaN`

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 | 1 ~ 8 | cases.csv 实测 2 ~ 5 维 |
| `x` 各维度大小 | 1 ~ 2097152 | cases.csv 实测 2 ~ 1000003 |
| `dim` | `[-rank, rank-1]` | cases.csv 实测 -1 / 0 / 1 / 2；支持负数索引 |

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
Softmax 算子 Torch Golden 参考实现

沿指定维度计算 Softmax 归一化

公式:
    y_i = exp(x_i) / sum(exp(x_j))

参考 PyTorch API: torch.nn.functional.softmax
    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html

Parameters:
    - x: 任意维度输入张量
    - dim: int, 默认 -1 - 计算 Softmax 的维度
"""


def softmax(
    x: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    沿指定维度计算 Softmax 归一化

    Args:
        x: 输入张量，任意 shape
        dim: 计算 Softmax 的维度，默认为 -1（最后一维）

    Returns:
        Softmax 归一化后的张量，shape 与输入相同
        输出元素值在 [0, 1] 范围内，且沿 dim 维度求和为 1

    Examples:
        >>> x = torch.randn(1024, 2048)
        >>> y = softmax(x, dim=-1)
    """
    y = torch.nn.functional.softmax(x, dim=dim)

    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 2048, dtype=torch.float32, device="npu")

y = cann_bench.softmax(x, dim=-1)
y = cann_bench.softmax(x, dim=0)
y = cann_bench.softmax(x, dim=1)
```

## proto.yaml

```yaml
operator:
  name: Softmax
  category: Normalization
  difficulty: L2
  formula: 'y_i = exp(x_i) / sum(exp(x_j))

    '
  description: 沿指定维度计算 Softmax 归一化
  shape_support: 'x (input): 任意维度张量，沿 dim 指定的维度计算 Softmax

    '
  attrs:
  - name: dim
    type: Int
    default: -1
    description: 计算 Softmax 的维度
    required: false
  inputs:
  - name: x
    description: 输入张量
    dtype:
    - float16
    - float32
    - bfloat16
  outputs:
  - name: y
    description: Softmax 归一化后的张量，与输入 shape 和 dtype 相同
    dtype:
    - float16
    - float32
    - bfloat16
  schema: softmax(Tensor x, int dim=-1) -> Tensor y
```

## golden.py — reference semantics ONLY (you must NOT compute with torch like this; reimplement the math in asctile kernels)

```python
#!/usr/bin/python3
# coding=utf-8


import torch

"""
Softmax 算子 Torch Golden 参考实现

沿指定维度计算 Softmax 归一化

公式:
    y_i = exp(x_i) / sum(exp(x_j))

参考 PyTorch API: torch.nn.functional.softmax
    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html

Parameters:
    - x: 任意维度输入张量
    - dim: int, 默认 -1 - 计算 Softmax 的维度
"""


def softmax(
    x: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    沿指定维度计算 Softmax 归一化

    Args:
        x: 输入张量，任意 shape
        dim: 计算 Softmax 的维度，默认为 -1（最后一维）

    Returns:
        Softmax 归一化后的张量，shape 与输入相同
        输出元素值在 [0, 1] 范围内，且沿 dim 维度求和为 1

    Examples:
        >>> x = torch.randn(1024, 2048)
        >>> y = softmax(x, dim=-1)
    """
    y = torch.nn.functional.softmax(x, dim=dim)

    return y
```

## Evaluation cases your module must handle (shapes, dtypes, value ranges, attrs)

| case | shapes | dtype | value_range | attrs |
|---|---|---|---|---|
| 1 | [[1024, 1024]] | ['float16'] | [-1, 1] | {'dim': -1} |
| 2 | [[2048, 2048]] | ['float32'] | [-2, 2] | {'dim': -1} |
| 3 | [[4096, 4096]] | ['bfloat16'] | [-3, 3] | {'dim': -1} |
| 4 | [[8192, 8192]] | ['float16'] | [-10, 10] | {'dim': 0} |
| 5 | [[8192, 8192]] | ['float32'] | [-100, 100] | {'dim': 1} |
| 6 | [[31, 67, 127, 257]] | ['bfloat16'] | [-5, 5] | {'dim': 2} |
| 7 | [[1023, 2047]] | ['float16'] | [-0.1, 0.1] | {'dim': -1} |
| 8 | [[2049, 4097]] | ['float32'] | [-1, 1] | {'dim': -1} |
| 9 | [[127, 257, 1023]] | ['bfloat16'] | [-0.5, 0.5] | {'dim': -2} |
| 10 | [[1009, 1021]] | ['float16'] | [-1, 2] | {'dim': -1} |
| 11 | [[367, 373, 379]] | ['float32'] | [-50, 100] | {'dim': 1} |
| 12 | [[11, 13, 17, 4001]] | ['bfloat16'] | [-3, 6] | {'dim': -1} |
| 13 | [[1000003, 2]] | ['float16'] | [None, None] | {'dim': -1} |
| 14 | [[11, 13, 17, 67, 67]] | ['float32'] | [None, None] | {'dim': -1} |
| 15 | [[3, 7, 11, 13, 1013]] | ['bfloat16'] | [0, 0] | {'dim': -1} |
| 16 | [[512, 2049]] | ['float16'] | [-0.5, 0.5] | {'dim': 0} |
| 17 | [[255, 8193]] | ['float32'] | [-1000, 1000] | {'dim': 1} |
| 18 | [[2, 511, 2049]] | ['bfloat16'] | [-0.2, 0.2] | {'dim': -1} |
| 19 | [[4, 255, 2049]] | ['float16'] | [-65504, 65504] | {'dim': 1} |
| 20 | [[2, 3, 17, 1024, 101]] | ['float32'] | [-20, 40] | {'dim': 3} |

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

Normalize dim, including negative values. Treat x as [outer, axis_size, inner]. Do not use torch.permute/softmax. Use a full-row path when inner==1 and an asctile local-transpose path otherwise. Cover axis_size through 8193 and preserve special values.

# Deliverable

Write ONLY `candidate.py` (complete, self-contained module). Public callable named exactly `softmax` with the exact schema signature including attr defaults. Think hard about numerical stability across the full value ranges listed above, and about the UB budget for your op chain before picking TILE.

IMPORTANT — no local execution: pyasc/asc/asctile/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
