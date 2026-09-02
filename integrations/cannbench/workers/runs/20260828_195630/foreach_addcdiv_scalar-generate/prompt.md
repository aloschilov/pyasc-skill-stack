You are implementing ONE operator for the CANN Bench benchmark (https://gitcode.com/cann/cann-bench) as a pyasc asc2 JIT kernel module targeting an Ascend 950PR NPU. You have NO NPU access — the official harness evaluates your file on real hardware after you finish. Score per operator = 0.2 compile + 0.3 accuracy (all 20 cases must pass a relative-error check against golden.py) + 0.5 performance (profiler-measured kernel time vs an aclnn baseline).

# Task

Operator: **ForeachAddcdivScalar**. Write the file `candidate.py` in the current working directory: a complete Python module whose public callable `foreach_addcdiv_scalar` implements the schema below. Nothing else — no tests, no other files.

# Operator specification

## desc.md

# ForeachAddcdivScalar 算子 API 描述

## 1. 算子简介

ForeachAddcdivScalar 算子对多个张量列表进行逐元素的加、除、乘复合操作，是优化器（如 Adam）中常用的基础运算。

**主要应用场景**：
- Adam / AdamW 优化器的参数更新步骤
- 需要对多组参数同时执行 addcdiv 运算的场景
- 分布式训练中的批量参数更新

**算子特征**：
- 难度等级：L1（FusedComposite）
- 三组 TensorList 输入，逐元素复合运算，输出 TensorList 与输入 shape 一致

## 2. 算子定义

### 数学公式

对列表中第 $i$ 个张量：

$$
y_i = x1_i + \frac{x2_i}{x3_i} \cdot scalar
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.foreach_addcdiv_scalar(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[] y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x1 | Tensor[] | 必选 | 第 1 个输入张量列表（TensorList），被加数 |
| x2 | Tensor[] | 必选 | 第 2 个输入张量列表（TensorList），被除数的分子 |
| x3 | Tensor[] | 必选 | 第 3 个输入张量列表（TensorList），被除数的分母 |
| scalar | float | 必选 | 缩放因子 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 TensorList 各元素 shape 相同 | 与输入 dtype 相同 | 逐元素复合运算结果列表 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- x1、x2、x3 三个 TensorList 长度必须相同
- 对应位置的张量 shape 必须一致
- 列表中各张量的 dtype 须一致
- x3 中的元素不应为零（除以零会产生 inf/nan）

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| TensorList 长度（`L`） | 1 ~ 64 | cases.csv 实测 1 ~ 4；x1/x2/x3 三个列表长度必须相同 |
| 每个张量维度数 | 1 ~ 8 | cases.csv 实测 1D ~ 5D |
| 每个张量各维大小 | 1 ~ 1048576 | cases.csv 各维实测 2 ~ 8193（含 1D 张量长度 1000003） |
| 每个张量元素总数 | 1 ~ 64M | cases.csv 实测 ~1M ~ 64M |
| `scalar` | -1024.0 ~ 1024.0 | cases.csv 实测 -1.0 ~ 2.0（含 inf / nan 特殊值） |

约束：x1[i]、x2[i]、x3[i] 三者 shape 与 dtype 必须一致；x3 中元素应非零。

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
from typing import List

def foreach_addcdiv_scalar(
    x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对多个张量进行逐元素加、乘、除操作

    公式：y_i = x1_i + (x2_i / x3_i) * scalar

    Args:
        x1: 第 1 个输入张量列表 (TensorList)
        x2: 第 2 个输入张量列表 (TensorList)
        x3: 第 3 个输入张量列表 (TensorList)
        scalar: 缩放因子

    Returns:
        输出张量列表
    """

    # FP16/BF16 输入为保证精度会先提升到 FP32 计算
    input_dtype = x1[0].dtype if x1 else torch.float32
    compute_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype

    x1_compute = [t.to(compute_dtype) for t in x1]
    x2_compute = [t.to(compute_dtype) for t in x2]
    x3_compute = [t.to(compute_dtype) for t in x3]

    y = [x1_i + (x2_i / x3_i) * scalar for x1_i, x2_i, x3_i in zip(x1_compute, x2_compute, x3_compute)]

    # 计算完成后恢复到输入 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return [t.to(input_dtype) for t in y]
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x1 = [torch.randn(1024, 1024, dtype=torch.float32, device="npu")]
x2 = [torch.randn(1024, 1024, dtype=torch.float32, device="npu")]
x3 = [torch.rand(1024, 1024, dtype=torch.float32, device="npu") + 0.1]  # 避免除零
y = cann_bench.foreach_addcdiv_scalar(x1, x2, x3, scalar=1.0)
```

## proto.yaml

```yaml
operator:
  name: ForeachAddcdivScalar
  category: FusedComposite
  difficulty: L1
  formula: y_i = x1_i + (x2_i / x3_i) * scalar
  description: 对多个张量进行逐元素加、乘、除操作
  shape_support: 输入 shape 必须一致
  attrs:
  - name: scalar
    type: float
    description: 缩放因子
    required: true
  inputs:
  - name: x1
    description: 第 1 个输入张量列表 (TensorList)
    dtype:
    - float32
    - float16
    - bfloat16
    is_list: true
  - name: x2
    description: 第 2 个输入张量列表 (TensorList)
    dtype:
    - float32
    - float16
    - bfloat16
    is_list: true
  - name: x3
    description: 第 3 个输入张量列表 (TensorList)
    dtype:
    - float32
    - float16
    - bfloat16
    is_list: true
  outputs:
  - name: y
    description: 输出张量列表
    dtype:
    - float32
    - float16
    - bfloat16
  schema: foreach_addcdiv_scalar(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[] y
  # 除法类算子精度敏感，使用更宽松的阈值
  precision_thresholds:
    float32: 0.005  # 除法类算子精度敏感，需要一定宽松度
    float16: 0.01   # 除法类算子精度敏感，需要一定宽松度
    bfloat16: 0.01  # 除法类算子精度敏感，需要一定宽松度
```

## golden.py — reference semantics ONLY (you must NOT compute with torch like this; reimplement the math in asc2 kernels)

```python
#!/usr/bin/python3
# coding=utf-8


import torch
from typing import List

"""
ForeachAddcdivScalar 算子 Torch Golden 参考实现

对多个张量进行逐元素加、乘、除操作
公式：y_i = x1_i + (x2_i / x3_i) * scalar
"""
def foreach_addcdiv_scalar(
    x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalar: float
) -> List[torch.Tensor]:
    """
    对多个张量进行逐元素加、乘、除操作

    公式：y_i = x1_i + (x2_i / x3_i) * scalar

    Args:
        x1: 第 1 个输入张量列表 (TensorList)
        x2: 第 2 个输入张量列表 (TensorList)
        x3: 第 3 个输入张量列表 (TensorList)
        scalar: 缩放因子

    Returns:
        输出张量列表
    """
    # 检测输入 dtype
    input_dtype = x1[0].dtype if x1 else torch.float32

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    x1_compute = [t.to(compute_dtype) for t in x1]
    x2_compute = [t.to(compute_dtype) for t in x2]
    x3_compute = [t.to(compute_dtype) for t in x3]

    # 计算
    y = [x1_i + (x2_i / x3_i) * scalar for x1_i, x2_i, x3_i in zip(x1_compute, x2_compute, x3_compute)]

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return [t.to(input_dtype) for t in y]
    return y
```

## Evaluation cases your module must handle (shapes, dtypes, value ranges, attrs)

| case | shapes | dtype | value_range | attrs |
|---|---|---|---|---|
| 1 | [[[1024, 1024], [1024, 1024]], [[1024, 1024], [1024, 1024]], [[1024, 1024], [1024, 1024]]] | ['float32'] | [[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.0]] | {'scalar': 1.0} |
| 2 | [[[2048, 2048], [2048, 2048], [2048, 2048]], [[2048, 2048], [2048, 2048], [2048, 2048]], [[2048, 2048], [2048, 2048], [2048, 2048]]] | ['float16'] | [[-2.0, 2.0], [-2.0, 2.0], [1.0, 2.0]] | {'scalar': 1.0} |
| 3 | [[[4096, 4096]], [[4096, 4096]], [[4096, 4096]]] | ['bfloat16'] | [[-3.0, 3.0], [-3.0, 3.0], [0.3, 3.0]] | {'scalar': 1.0} |
| 4 | [[[8192, 8192]], [[8192, 8192]], [[8192, 8192]]] | ['float32'] | [[1.0, 10.0], [-10.0, 10.0], [5.0, 10.0]] | {'scalar': 0.5} |
| 5 | [[[8192, 4096], [8192, 4096]], [[8192, 4096], [8192, 4096]], [[8192, 4096], [8192, 4096]]] | ['float16'] | [[-100.0, 100.0], [-100.0, 100.0], [50.0, 100.0]] | {'scalar': 2.0} |
| 6 | [[[1023, 1023]], [[1023, 1023]], [[1023, 1023]]] | ['bfloat16'] | [[-0.1, 0.1], [-0.1, 0.1], [0.1, 0.1]] | {'scalar': -1.0} |
| 7 | [[[1009, 1021]], [[1009, 1021]], [[1009, 1021]]] | ['float32'] | [[-1.0, 2.0], [-1.0, 2.0], [1.0, 2.0]] | {'scalar': 1.5} |
| 8 | [[[1537, 769]], [[1537, 769]], [[1537, 769]]] | ['float16'] | [[-5.0, 10.0], [-5.0, 10.0], [0.1, 10.0]] | {'scalar': 1.0} |
| 9 | [[[363, 367, 373], [363, 367, 373]], [[363, 367, 373], [363, 367, 373]], [[363, 367, 373], [363, 367, 373]]] | ['bfloat16'] | [[-50.0, 100.0], [-50.0, 100.0], [0.1, 100.0]] | {'scalar': 1.0} |
| 10 | [[[2049, 513]], [[2049, 513]], [[2049, 513]]] | ['float32'] | [[-65504.0, 65504.0], [-65504.0, 65504.0], [1.0, 65504.0]] | {'scalar': 1.0} |
| 11 | [[[3, 7, 13, 4001], [3, 7, 13, 4001]], [[3, 7, 13, 4001], [3, 7, 13, 4001]], [[3, 7, 13, 4001], [3, 7, 13, 4001]]] | ['float16'] | [[-88.0, 88.0], [-88.0, 88.0], [0.1, 88.0]] | {'scalar': 1.0} |
| 12 | [[[1000003]], [[1000003]], [[1000003]]] | ['float32'] | [[-inf, inf], [-inf, inf], [0.1, inf]] | {'scalar': inf} |
| 13 | [[[11, 13, 17, 67, 67]], [[11, 13, 17, 67, 67]], [[11, 13, 17, 67, 67]]] | ['bfloat16'] | [[nan, nan], [nan, nan], [nan, nan]] | {'scalar': nan} |
| 14 | [[[3, 7, 11, 13, 1013]], [[3, 7, 11, 13, 1013]], [[3, 7, 11, 13, 1013]]] | ['float32'] | [[0, 0], [0, 0], [0.001, 0.001]] | {'scalar': 1.0} |
| 15 | [[[512, 2049], [512, 2049]], [[512, 2049], [512, 2049]], [[512, 2049], [512, 2049]]] | ['float32'] | [[-0.5, 0.5], [-0.5, 0.5], [0.25, 0.5]] | {'scalar': 1.0} |
| 16 | [[[255, 8193], [255, 8193], [255, 8193], [255, 8193]], [[255, 8193], [255, 8193], [255, 8193], [255, 8193]], [[255, 8193], [255, 8193], [255, 8193], [255, 8193]]] | ['bfloat16'] | [[-1.0, 3.0], [-1.0, 3.0], [0.1, 3.0]] | {'scalar': 1.0} |
| 17 | [[[4097, 511]], [[4097, 511]], [[4097, 511]]] | ['float16'] | [[-1000.0, 1000.0], [-1000.0, 1000.0], [0.1, 1000.0]] | {'scalar': 0.0} |
| 18 | [[[2, 511, 2049], [2, 511, 2049]], [[2, 511, 2049], [2, 511, 2049]], [[2, 511, 2049], [2, 511, 2049]]] | ['float32'] | [[-0.2, 0.2], [-0.2, 0.2], [0.1, 0.2]] | {'scalar': 2.0} |
| 19 | [[[4, 255, 2049]], [[4, 255, 2049]], [[4, 255, 2049]]] | ['bfloat16'] | [[-3.0, 6.0], [-3.0, 6.0], [0.1, 6.0]] | {'scalar': -0.5} |
| 20 | [[[2, 3, 17, 1024, 101]], [[2, 3, 17, 1024, 101]], [[2, 3, 17, 1024, 101]]] | ['float32'] | [[-20.0, 40.0], [-20.0, 40.0], [20.0, 40.0]] | {'scalar': 1.5} |

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
  `asc2.exp`, `asc2.log`, `asc2.sqrt`, `asc2.tanh`, `asc2.erf`, `asc2.sin`,
  `asc2.cos`, `asc2.maximum`, `asc2.minimum`, comparisons
  (`x >= 0.0`, `asc2.less(a, b)`, ... — NO int64 operands),
  `asc2.where(cond, a, b)`, `asc2.reduce_sum(x)`, `asc2.reduce_max(x)`,
  `asc2.full([shape], scalar, dtype=...)`, `tile.to(asc.float32)` casts,
  unary `-x`.
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
- Loop-carried scalar accumulators: seed with
  `acc = asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32))`
  (a bare `acc = 0.0` fails codegen with "re-assigned to an object with
  different type"), and the accumulating loop must pass `gm_barrier=True`.
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

- Signature: foreach_addcdiv_scalar(x1: List[Tensor], x2, x3, scalar: float)
  -> List[Tensor]. Elementwise per list entry: y_i = x1_i + (x2_i / x3_i) *
  scalar; shapes match within each triple, lists are short.
- Host loops over the list, one kernel launch per tensor triple,
  empty_like output each.
- scalar values include 0.0, +-0.5, +-1.0, 1.5, 2.0, inf and nan — pass it
  as a runtime float kernel argument; IEEE propagation through the hardware
  ops matches the golden. NO host special-casing.
- Compute in f32 internally even for f16/bf16 (division is
  precision-sensitive).

# Deliverable

Write ONLY `candidate.py` (complete, self-contained module). Public callable named exactly `foreach_addcdiv_scalar` with the exact schema signature including attr defaults. Think hard about numerical stability across the full value ranges listed above, and about the UB budget for your op chain before picking TILE.

IMPORTANT — no local execution: pyasc/asc/asc2/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
