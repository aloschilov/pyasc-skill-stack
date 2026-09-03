You are implementing ONE operator for the CANN Bench benchmark (https://gitcode.com/cann/cann-bench) as a pyasc asc2 JIT kernel module targeting an Ascend 950PR NPU. You have NO NPU access — the official harness evaluates your file on real hardware after you finish. Score per operator = 0.2 compile + 0.3 accuracy (all 20 cases must pass a relative-error check against golden.py) + 0.5 performance (profiler-measured kernel time vs an aclnn baseline).

# Task

Operator: **Exp**. Write the file `candidate.py` in the current working directory: a complete Python module whose public callable `exp` implements the schema below. Nothing else — no tests, no other files.

# Operator specification

## desc.md

# Exp 算子 API 描述

## 1. 算子简介

Exp 算子用于计算输入张量的广义指数函数，支持自定义底数（base）、缩放因子（scale）和偏移量（shift）三个参数，涵盖自然指数、任意底数指数等多种变体。

**主要应用场景**：
- Softmax 中的自然指数计算
- 注意力机制中的指数缩放
- 概率分布与对数域间的转换
- 学习率调度与指数衰减

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致

## 2. 算子定义

### 数学公式

**通用公式**：

$$
y = e^{(x \cdot scale + shift) \cdot \ln(base)}, \quad base > 0
$$

**自然指数**（当 $base \leq 0$ 时，使用自然底数 $e$）：

$$
y = e^{x \cdot scale + shift}
$$

### 特殊情况

| 条件 | 简化公式 |
|------|---------|
| base ≤ 0, scale=1, shift=0 | $y = e^x$ |
| base > 0, scale=1, shift=0 | $y = base^x$ |
| base=1（任意 scale, shift） | $y = 1$（因 $\ln 1 = 0$） |

## 3. 接口规范

### 算子原型

```python
cann_bench.exp(Tensor x, float base=-1.0, float scale=1.0, float shift=0.0) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持任意维度 |
| base | float | -1.0 | 指数底数；≤ 0 表示使用自然底数 $e$，> 0 表示自定义底数 |
| scale | float | 1.0 | 输入缩放因子 |
| shift | float | 0.0 | 输入偏移量 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | 指数计算结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- `base` 参数：≤ 0 时一律视为自然底数 $e$；> 0 时使用该值作为底数
- `x` 支持任意维度（1D ~ 5D 及更高维），不限制具体 shape
- 需注意数值溢出：float16 的有效范围约 [-65504, 65504]，float32 下 $e^x$ 在 $|x| > 88$ 左右可能溢出为 inf

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 | 1 ~ 8 | cases.csv 实测 1D ~ 5D；逐元素算子，不限维度数 |
| `x` 各维大小 | 1 ~ 1048576 | cases.csv 各维实测 2 ~ 8192（含 1D 张量长度 1000007） |
| `x` 元素总数 | 1 ~ 64M | cases.csv 实测 ~1M ~ 64M |
| `base` | -1.0 ~ 1024.0 | cases.csv 实测 -1.0 / 1.0 / 2.0 / 10.0；≤ 0 时一律视为自然底数 e |
| `scale` | -1024.0 ~ 1024.0 | cases.csv 实测 0.5 ~ 2.0 |
| `shift` | -1024.0 ~ 1024.0 | cases.csv 实测 0.0 ~ 2.0 |

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

def exp(
    x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0
) -> torch.Tensor:
    """
    计算输入张量的指数函数

    - base <= 0: y = exp(scale * x + shift)
    - base > 0: y = exp((shift + scale * x) * ln(base))

    Args:
        x: 输入张量
        base: 指数底数，base <= 0 表示使用自然底数 e
        scale: 输入缩放因子
        shift: 输入偏移量

    Returns:
        指数计算结果
    """
    # FP16/BF16 输入为保证精度会先提升到 FP32 计算
    input_dtype = x.dtype
    compute_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
    x_compute = x.to(compute_dtype)

    temp = scale * x_compute + shift
    if base > 0:
        temp = temp * torch.log(torch.tensor(base, dtype=temp.dtype, device=temp.device))
    y = torch.exp(temp)

    # 计算完成后恢复到输入 dtype
    return y.to(input_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.exp(x, base=-1.0, scale=1.0, shift=0.0)  # 自然指数 e^x
y = cann_bench.exp(x, base=2.0, scale=1.0, shift=0.0)   # 2^x
y = cann_bench.exp(x, base=-1.0, scale=2.0, shift=1.0)  # e^(2x+1)
```

## proto.yaml

```yaml
operator:
  name: Exp
  category: Elementwise
  difficulty: L1
  formula: y = e^((x * scale + shift) * ln(base))
  description: 计算输入张量的指数函数，支持自定义底数、缩放和偏移
  shape_support: 输入任意维度，输出与输入相同shape
  attrs:
  - name: base
    type: float
    default: -1.0
    description: 指数底数，-1.0表示使用自然底数e，正值表示自定义底数
  - name: scale
    type: float
    default: 1.0
    description: 输入缩放因子
  - name: shift
    type: float
    default: 0.0
    description: 输入偏移量
  note: 当base=-1时，公式简化为 y = e^(x * scale + shift)
  inputs:
  - name: x
    description: 输入张量
    dtype:
    - float16
    - float32
    - bfloat16
  outputs:
  - name: y
    description: 指数计算结果，输出数据类型与输入一致
    dtype:
    - float16
    - float32
    - bfloat16
  schema: exp(Tensor x, float base=-1.0, float scale=1.0, float shift=0.0) -> Tensor y
```

## golden.py — reference semantics ONLY (you must NOT compute with torch like this; reimplement the math in asc2 kernels)

```python
#!/usr/bin/python3
# coding=utf-8


import torch

"""
Exp算子Torch Golden参考实现

计算输入张量的指数函数
- base <= 0: y = exp(scale * x + shift)
- base > 0: y = exp((shift + scale * x) * ln(base))
"""
def exp(
    x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0
) -> torch.Tensor:
    """
    计算输入张量的指数函数

    - base <= 0: y = exp(scale * x + shift)
    - base > 0: y = exp((shift + scale * x) * ln(base))

    Args:
        x: 输入张量
        base: 指数底数，base <= 0 表示使用自然底数 e
        scale: 输入缩放因子
        shift: 输入偏移量

    Returns:
        指数计算结果
    """
    # 检测输入 dtype
    input_dtype = x.dtype

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    x_compute = x.to(compute_dtype)

    temp = scale * x_compute + shift
    if base > 0:
        temp = temp * torch.log(torch.tensor(base, dtype=temp.dtype, device=temp.device))
    y = torch.exp(temp)

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return y.to(input_dtype)
    return y
```

## Evaluation cases your module must handle (shapes, dtypes, value ranges, attrs)

| case | shapes | dtype | value_range | attrs |
|---|---|---|---|---|
| 1 | [[1024, 1024]] | ['float16'] | [-1, 1] | {'base': -1.0, 'scale': 1.0, 'shift': 0.0} |
| 2 | [[2048, 2048]] | ['float32'] | [-2, 2] | {'base': -1.0, 'scale': 1.5, 'shift': 0.0} |
| 3 | [[4096, 4096]] | ['bfloat16'] | [-3, 3] | {'base': -1.0, 'scale': 2.0, 'shift': 0.0} |
| 4 | [[8192, 8192]] | ['float16'] | [-10, 10] | {'base': -1.0, 'scale': 0.5, 'shift': 0.0} |
| 5 | [[8192, 8192]] | ['float32'] | [-100, 100] | {'base': -1.0, 'scale': 1.0, 'shift': 1.0} |
| 6 | [[1023, 1023]] | ['bfloat16'] | [-0.1, 0.1] | {'base': 2.0, 'scale': 1.0, 'shift': 0.0} |
| 7 | [[1009, 1021]] | ['float16'] | [-1, 2] | {'base': 2.0, 'scale': 1.5, 'shift': 0.0} |
| 8 | [[1537, 769]] | ['float32'] | [-5, 10] | {'base': 10.0, 'scale': 1.0, 'shift': 0.0} |
| 9 | [[363, 367, 373]] | ['bfloat16'] | [-50, 100] | {'base': -1.0, 'scale': 2.0, 'shift': 0.5} |
| 10 | [[2049, 513]] | ['float16'] | [-65504, 65504] | {'base': -1.0, 'scale': 1.0, 'shift': 2.0} |
| 11 | [[3, 7, 13, 4003]] | ['float32'] | [-88, 88] | {'base': 1.0, 'scale': 2.0, 'shift': 0.0} |
| 12 | [[1000007]] | ['bfloat16'] | [-inf, inf] | {'base': -1.0, 'scale': 0.5, 'shift': 0.5} |
| 13 | [[11, 13, 17, 67, 67]] | ['float32'] | [nan, nan] | {'base': 2.0, 'scale': 1.0, 'shift': 1.0} |
| 14 | [[3, 7, 11, 13, 1013]] | ['float16'] | [0, 0] | {'base': -1.0, 'scale': 2.0, 'shift': 1.0} |
| 15 | [[512, 2049]] | ['float32'] | [-0.5, 0.5] | {'base': -1.0, 'scale': 1.0, 'shift': 0.5} |
| 16 | [[255, 8193]] | ['bfloat16'] | [-1, 3] | {'base': -1.0, 'scale': 1.2, 'shift': 0.0} |
| 17 | [[4097, 511]] | ['float16'] | [-1000, 1000] | {'base': 1.0, 'scale': 0.5, 'shift': 0.0} |
| 18 | [[2, 511, 2049]] | ['float32'] | [-0.2, 0.2] | {'base': 2.0, 'scale': 0.5, 'shift': 0.0} |
| 19 | [[4, 255, 2049]] | ['bfloat16'] | [-3, 6] | {'base': 10.0, 'scale': 0.5, 'shift': 0.5} |
| 20 | [[2, 3, 17, 1024, 101]] | ['float16'] | [-20, 40] | {'base': 1.0, 'scale': 1.5, 'shift': 1.0} |

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

-

# Deliverable

Write ONLY `candidate.py` (complete, self-contained module). Public callable named exactly `exp` with the exact schema signature including attr defaults. Think hard about numerical stability across the full value ranges listed above, and about the UB budget for your op chain before picking TILE.

IMPORTANT — no local execution: pyasc/asc/asc2/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
