You are implementing ONE operator for the CANN Bench benchmark (https://gitcode.com/cann/cann-bench) as a pyasc asc2 JIT kernel module targeting an Ascend 950PR NPU. You have NO NPU access — the official harness evaluates your file on real hardware after you finish. Score per operator = 0.2 compile + 0.3 accuracy (all 20 cases must pass a relative-error check against golden.py) + 0.5 performance (profiler-measured kernel time vs an aclnn baseline).

# Task

Operator: **Gelu**. Write the file `candidate.py` in the current working directory: a complete Python module whose public callable `gelu` implements the schema below. Nothing else — no tests, no other files.

# Operator specification

## desc.md

# Gelu 算子 API 描述

## 1. 算子简介

Gelu（高斯误差线性单元）是一种广泛应用于 Transformer 架构的激活函数，支持精确计算和 tanh 近似两种模式。

**主要应用场景**：
- BERT、GPT 等 Transformer 模型的前馈网络激活层
- Vision Transformer (ViT) 中的 MLP 模块
- 各类预训练语言模型的中间激活

**算子特征**：
- 难度等级：L1（Elementwise）
- 单输入单输出，逐元素运算，输出 shape 与输入完全一致
- 支持 0~8 维输入

## 2. 算子定义

### 数学公式

**精确模式**（approximate="none"）：

$$
y = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

**tanh 近似模式**（approximate="tanh"）：

$$
y = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
$$

### 特殊情况

| 输入 | 输出 |
|------|------|
| x = 0 | y = 0 |
| x → +∞ | y → x |
| x → -∞ | y → 0 |

## 3. 接口规范

### 算子原型

```python
cann_bench.gelu(Tensor x, str approximate="none") -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量，支持 0~8 维 |
| approximate | str | "none" | GELU 近似计算算法，可选值：'none'（精确计算）或 'tanh'（tanh 近似） |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | 与输入 x 相同 | GELU 激活结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| float32 | float32 |
| bfloat16 | bfloat16 |

### 规则与约束

- 输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 一致
- `approximate` 参数仅支持 "none" 和 "tanh" 两种取值
- 输入支持 0~8 维

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `x` 维度数 | 0 ~ 8 | cases.csv 实测 1D ~ 5D；接口规范支持 0 ~ 8 维 |
| `x` 各维大小 | 1 ~ 1048576 | cases.csv 各维实测 2 ~ 8192（含 1D 张量长度 1000003） |
| `x` 元素总数 | 1 ~ 64M | cases.csv 实测 ~1M ~ 64M |
| `approximate` | {"none", "tanh"} | cases.csv 两种取值均覆盖；仅支持这两种字符串 |

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

def gelu(
    x: torch.Tensor,
    approximate: str = "none"
) -> torch.Tensor:
    """
    高斯误差线性单元激活函数

    公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: 输入张量
        approximate: GELU 近似计算算法，可选值：'none'(精确计算) 或 'tanh'(tanh 近似)

    Returns:
        输出张量，GELU 激活结果
    """

    y = torch.nn.functional.gelu(x, approximate=approximate)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = cann_bench.gelu(x)                          # 精确模式
y = cann_bench.gelu(x, approximate="tanh")       # tanh 近似模式
```

## proto.yaml

```yaml
operator:
  name: Gelu
  category: Elementwise
  difficulty: L1
  formula: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  description: 高斯误差线性单元激活函数
  shape_support: 输入 0-8 维
  attrs:
  - name: approximate
    type: str
    default: "none"
    description: GELU 近似计算算法，可选值：'none'(精确计算) 或 'tanh'(tanh 近似)
  inputs:
  - name: x
    description: 输入张量
    dtype:
    - float32
    - float16
    - bfloat16
  outputs:
  - name: y
    description: 输出张量，GELU 激活结果
    dtype:
    - float32
    - float16
    - bfloat16
  schema: gelu(Tensor x, str approximate="none") -> Tensor y
```

## golden.py — reference semantics ONLY (you must NOT compute with torch like this; reimplement the math in asc2 kernels)

```python
#!/usr/bin/python3
# coding=utf-8


import torch

"""
Gelu 算子 Torch Golden 参考实现

高斯误差线性单元激活函数
公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""
def gelu(
    x: torch.Tensor,
    approximate: str = "none"
) -> torch.Tensor:
    """
    高斯误差线性单元激活函数

    公式：y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: 输入张量
        approximate: GELU 近似计算算法，可选值：'none'(精确计算) 或 'tanh'(tanh 近似)

    Returns:
        输出张量，GELU 激活结果
    """

    y = torch.nn.functional.gelu(x, approximate=approximate)
    return y
```

## Evaluation cases your module must handle (shapes, dtypes, value ranges, attrs)

| case | shapes | dtype | value_range | attrs |
|---|---|---|---|---|
| 1 | [[1024, 1024]] | ['float16'] | [-1, 1] | {'approximate': 'none'} |
| 2 | [[2048, 2048]] | ['float32'] | [-2, 2] | {'approximate': 'none'} |
| 3 | [[4096, 4096]] | ['bfloat16'] | [-3, 3] | {'approximate': 'none'} |
| 4 | [[8192, 8192]] | ['float16'] | [-10, 10] | {'approximate': 'tanh'} |
| 5 | [[8192, 8192]] | ['float32'] | [-100, 100] | {'approximate': 'tanh'} |
| 6 | [[1023, 1023]] | ['bfloat16'] | [-0.1, 0.1] | {'approximate': 'tanh'} |
| 7 | [[1009, 1021]] | ['float16'] | [-1, 2] | {'approximate': 'none'} |
| 8 | [[1537, 769]] | ['float32'] | [-5, 10] | {'approximate': 'tanh'} |
| 9 | [[363, 367, 373]] | ['bfloat16'] | [-50, 100] | {'approximate': 'none'} |
| 10 | [[2049, 513]] | ['float16'] | [-65504, 65504] | {'approximate': 'tanh'} |
| 11 | [[3, 7, 13, 4001]] | ['float32'] | [-88, 88] | {'approximate': 'none'} |
| 12 | [[1000003]] | ['bfloat16'] | [-inf, inf] | {'approximate': 'tanh'} |
| 13 | [[11, 13, 17, 67, 67]] | ['float32'] | [nan, nan] | {'approximate': 'none'} |
| 14 | [[3, 7, 11, 13, 1009]] | ['float16'] | [0, 0] | {'approximate': 'tanh'} |
| 15 | [[512, 2049]] | ['float32'] | [-0.5, 0.5] | {'approximate': 'none'} |
| 16 | [[255, 8193]] | ['bfloat16'] | [-1, 3] | {'approximate': 'none'} |
| 17 | [[4097, 511]] | ['float16'] | [-1000, 1000] | {'approximate': 'tanh'} |
| 18 | [[2, 511, 2049]] | ['float32'] | [-0.2, 0.2] | {'approximate': 'none'} |
| 19 | [[4, 255, 2049]] | ['bfloat16'] | [-3, 6] | {'approximate': 'tanh'} |
| 20 | [[2, 3, 17, 1024, 101]] | ['float32'] | [-20, 40] | {'approximate': 'none'} |

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

Write ONLY `candidate.py` (complete, self-contained module). Public callable named exactly `gelu` with the exact schema signature including attr defaults. Think hard about numerical stability across the full value ranges listed above, and about the UB budget for your op chain before picking TILE.

IMPORTANT — no local execution: pyasc/asc/asc2/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.


# Evaluator feedback from the previous iteration

The previous three-phase workflow did not pass its mandatory skill provenance gates. Start over from the full original operator task and produce a fresh candidate.