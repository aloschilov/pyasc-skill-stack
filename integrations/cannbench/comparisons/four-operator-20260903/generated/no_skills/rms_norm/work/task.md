# Runtime pin

Use compiler-team/pyasc v2 commit `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d`. This snapshot exports `asctile`; importing `asc2` is invalid.

You are implementing ONE operator for the CANN Bench benchmark (https://gitcode.com/cann/cann-bench) as a pyasc asctile JIT kernel module targeting an Ascend 950PR NPU. You have NO NPU access — the official harness evaluates your file on real hardware after you finish. Score per operator = 0.2 compile + 0.3 accuracy (all 20 cases must pass a relative-error check against golden.py) + 0.5 performance (profiler-measured kernel time vs an aclnn baseline).

# Task

Operator: **RmsNorm**. Write the file `candidate.py` in the current working directory: a complete Python module whose public callable `rms_norm` implements the schema below. Nothing else — no tests, no other files.

# Operator specification

## desc.md

# RmsNorm 算子 API 描述

## 1. 算子简介

计算 RMS (均方根) 归一化。

**主要应用场景**：
- 大语言模型中的归一化层（LLaMA、Gemma 等使用 RMSNorm 替代 LayerNorm）
- Transformer 架构中的预归一化（Pre-Norm）
- 相比 LayerNorm 省去均值计算，推理效率更高

**算子特征**：
- 难度等级：L2（Normalization）
- 双输入（x 和 gamma）单输出，涉及平方、均值、开方、除法、乘法等多步计算
- 沿最后一维进行归一化，gamma 为可学习的缩放参数

## 2. 算子定义

### 数学公式

**基本公式**：

$$
y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
$$

展开为：

$$
y_i = \frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2 + \epsilon}} \cdot \gamma_i
$$

其中：
- `D` 为最后一维的大小（归一化维度）
- `epsilon` 为数值稳定性参数，防止除零
- `gamma` 为逐元素的缩放参数，shape 为 (D,)
- 与 LayerNorm 不同，RMSNorm 不计算均值，也没有偏置（beta）参数

## 3. 接口规范

### 算子原型

```python
cann_bench.rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| gamma | Tensor | 必选 | 缩放参数，shape 为输入最后一维大小 |
| epsilon | float | 1e-6 | 数值稳定性参数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | RMS 归一化后的张量 |

### 数据类型

| x dtype | gamma dtype | 输出 dtype |
|---------|------------|-----------|
| float16 | float16 | float16 |
| float32 | float32 | float32 |
| bfloat16 | bfloat16 | bfloat16 |

### 规则与约束

- x 的 shape 为 (..., D)，gamma 的 shape 为 (D,)，其中 D 为最后一维大小
- gamma 的 dtype 需与 x 一致
- epsilon 为正数，通常取 1e-6 或 1e-5
- 需注意数值稳定性：当输入值极小时，mean(x^2) 可能下溢；当输入值极大时，x^2 可能溢出

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 | 2 ~ 8 | cases.csv 实测 2 ~ 5 维 |
| `D`（最后一维/归一化维度） | 1 ~ 16384 | cases.csv 实测 2 ~ 8192；`gamma` 的 shape 必须为 `(D,)` |
| 前导维度乘积 `S = N0*N1*...` | 1 ~ 2097152 | cases.csv 实测 231 ~ 1000003 |
| `gamma` 维度数 | 1 | 固定为 1 维 |
| `epsilon` | 1e-12 ~ 1 | cases.csv 实测 1e-12 ~ 1e-3；须为正数，常用 1e-6 / 1e-5 |

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
RmsNorm 算子 Torch Golden 参考实现

计算 RMS (均方根) 归一化

公式:
    y = x / sqrt(mean(x^2) + eps) * gamma

参考 PyTorch API: torch.nn.functional.rms_norm
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html

参考论文: Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467

Parameters:
    - x: (..., D) 输入张量，最后一维为归一化维度
    - gamma: (D,) 缩放参数
    - epsilon: float, 默认 1e-6 - 数值稳定性参数
"""


def rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算 RMS (均方根) 归一化

    Args:
        x: 输入张量，shape (..., D)
           最后一维 D 为归一化维度
        gamma: 缩放参数，shape (D,)
               与输入最后一维大小相同
        epsilon: 数值稳定性参数，防止除零
                 默认值 1e-6

    Returns:
        RMS 归一化后的张量，shape 与输入相同

    Examples:
        >>> x = torch.randn(32, 128, 4096)
        >>> gamma = torch.ones(4096)
        >>> y = rms_norm(x, gamma, epsilon=1e-6)
    """
    # 直接调用 PyTorch 原生 RMSNorm 实现；fp16/bf16 输入由 F.rms_norm 内部
    # 自动以 fp32 累加，避免 |x|>256 时 x^2 上溢。
    return torch.nn.functional.rms_norm(
        x, normalized_shape=gamma.shape, weight=gamma, eps=epsilon
    )
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(32, 128, 4096, dtype=torch.float32, device="npu")
gamma = torch.ones(4096, dtype=torch.float32, device="npu")

y = cann_bench.rms_norm(x, gamma, epsilon=1e-6)
y = cann_bench.rms_norm(x, gamma, epsilon=1e-5)
```

## proto.yaml

```yaml
operator:
  name: RmsNorm
  category: Normalization
  difficulty: L2
  formula: 'y = x / sqrt(mean(x^2) + eps) * gamma

    '
  description: 计算 RMS (均方根) 归一化
  shape_support: 'x (input): (..., D) - 任意前导维度，最后一维为归一化维度 D

    gamma (weight): (D,) - 缩放参数，shape 为输入最后一维大小

    '
  attrs:
  - name: epsilon
    type: float
    default: 1e-6
    description: 数值稳定性参数
    required: false
  inputs:
  - name: x
    description: 输入张量
    dtype:
    - float16
    - float32
    - bfloat16
  - name: gamma
    description: 缩放参数，shape 为输入最后一维大小
    dtype:
    - float16
    - float32
    - bfloat16
  outputs:
  - name: y
    description: RMS 归一化后的张量，与输入 shape 和 dtype 相同
    dtype:
    - float16
    - float32
    - bfloat16
  schema: rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> Tensor y
```

## golden.py — reference semantics ONLY (you must NOT compute with torch like this; reimplement the math in asctile kernels)

```python
#!/usr/bin/python3
# coding=utf-8


import torch

"""
RmsNorm 算子 Torch Golden 参考实现

计算 RMS (均方根) 归一化

公式:
    y = x / sqrt(mean(x^2) + eps) * gamma

参考 PyTorch API: torch.nn.functional.rms_norm
    https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html

参考论文: Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467

Parameters:
    - x: (..., D) 输入张量，最后一维为归一化维度
    - gamma: (D,) 缩放参数
    - epsilon: float, 默认 1e-6 - 数值稳定性参数
"""


def rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算 RMS (均方根) 归一化

    Args:
        x: 输入张量，shape (..., D)
           最后一维 D 为归一化维度
        gamma: 缩放参数，shape (D,)
               与输入最后一维大小相同
        epsilon: 数值稳定性参数，防止除零
                 默认值 1e-6

    Returns:
        RMS 归一化后的张量，shape 与输入相同

    Examples:
        >>> x = torch.randn(32, 128, 4096)
        >>> gamma = torch.ones(4096)
        >>> y = rms_norm(x, gamma, epsilon=1e-6)
    """
    # 直接调用 PyTorch 原生 RMSNorm 实现；fp16/bf16 输入由 F.rms_norm 内部
    # 自动以 fp32 累加，避免 |x|>256 时 x^2 上溢。
    return torch.nn.functional.rms_norm(
        x, normalized_shape=gamma.shape, weight=gamma, eps=epsilon
    )
```

## Evaluation cases your module must handle (shapes, dtypes, value ranges, attrs)

| case | shapes | dtype | value_range | attrs |
|---|---|---|---|---|
| 1 | [[32, 128, 768], [768]] | ['float16', 'float16'] | [-1, 1] | {'epsilon': 1e-06} |
| 2 | [[32, 128, 1024], [1024]] | ['float32', 'float32'] | [-2, 2] | {'epsilon': 1e-06} |
| 3 | [[32, 128, 2048], [2048]] | ['bfloat16', 'bfloat16'] | [-3, 3] | {'epsilon': 1e-06} |
| 4 | [[16, 256, 4096], [4096]] | ['float16', 'float16'] | [-10, 10] | {'epsilon': 1e-06} |
| 5 | [[8, 512, 8192], [8192]] | ['float32', 'float32'] | [-100, 100] | {'epsilon': 1e-06} |
| 6 | [[4, 1023, 4097], [4097]] | ['bfloat16', 'bfloat16'] | [-5, 5] | {'epsilon': 1e-05} |
| 7 | [[63, 67, 1023], [1023]] | ['float16', 'float16'] | [-0.1, 0.1] | {'epsilon': 1e-08} |
| 8 | [[16, 511, 2049], [2049]] | ['float32', 'float32'] | [-1, 1] | {'epsilon': 0.0001} |
| 9 | [[8, 1021, 4099], [4099]] | ['bfloat16', 'bfloat16'] | [-0.5, 0.5] | {'epsilon': 1e-12} |
| 10 | [[33, 127, 769], [769]] | ['float16', 'float16'] | [-1, 2] | {'epsilon': 1e-06} |
| 11 | [[31, 129, 2049], [2049]] | ['float32', 'float32'] | [-50, 100] | {'epsilon': 1e-06} |
| 12 | [[17, 255, 4097], [4097]] | ['bfloat16', 'bfloat16'] | [-3, 6] | {'epsilon': 1e-06} |
| 13 | [[7, 1009, 1021], [1021]] | ['float16', 'float16'] | [-1, 1] | {'epsilon': 1e-07} |
| 14 | [[11, 367, 373], [373]] | ['float32', 'float32'] | [-10, 10] | {'epsilon': 1e-05} |
| 15 | [[1000003, 2], [2]] | ['bfloat16', 'bfloat16'] | [None, None] | {'epsilon': 1e-06} |
| 16 | [[11, 13, 17, 67], [67]] | ['float16', 'float16'] | [None, None] | {'epsilon': 1e-08} |
| 17 | [[3, 7, 11, 4096], [4096]] | ['float32', 'float32'] | [0, 0] | {'epsilon': 0.0001} |
| 18 | [[2, 511, 8192], [8192]] | ['bfloat16', 'bfloat16'] | [-0.2, 0.2] | {'epsilon': 1e-06} |
| 19 | [[4, 255, 4096], [4096]] | ['float16', 'float16'] | [-65504, 65504] | {'epsilon': 0.001} |
| 20 | [[2, 3, 17, 1024, 128], [128]] | ['float32', 'float32'] | [-20, 40] | {'epsilon': 1e-06} |

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

Normalize independently over the last dimension. Cover D=2..8192 and all listed f16/bf16/f32 routes. Accumulate squares in f32 and do not emit the optional rstd output from the upstream target; CANNBench returns y only.

# Deliverable

Write ONLY `candidate.py` (complete, self-contained module). Public callable named exactly `rms_norm` with the exact schema signature including attr defaults. Think hard about numerical stability across the full value ranges listed above, and about the UB budget for your op chain before picking TILE.

IMPORTANT — no local execution: pyasc/asc/asctile/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
