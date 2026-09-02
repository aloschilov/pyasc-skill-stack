# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

__all__ = []

from asc.language.core.constexpr import ConstExpr
from asc.language.core.dtype import (
    DataType,
    bfloat16,
    bool_,
    float16,
    float32,
    float64,
    float_,
    int8,
    int16,
    int32,
    int64,
    int_,
)
from asc.language.core.ir_value import GlobalAddress
from asc.lib import profiling, runtime
from asc.runtime.config import Backend, KernelType, Platform, set_platform

__all__ += [
    "Backend",
    "ConstExpr",
    "DataType",
    "GlobalAddress",
    "KernelType",
    "Platform",
    "bfloat16",
    "bool_",
    "float16",
    "float32",
    "float64",
    "float_",
    "int8",
    "int16",
    "int32",
    "int64",
    "int_",
    "profiling",
    "runtime",
    "set_platform",
]

from .language.global_tensor import GlobalTensor, global_tensor
from .language.local_tensor import LocalTensor, RoundMode, TensorLocation
from .language.range import range, static_range
from .language.utils import ceildiv

# Tile operations
from .language.atomic_ops import (
    atomic_add,
    atomic_max,
    atomic_min,
)
from .language.binary_ops import (
    add,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    div,
    equal,
    greater,
    greater_equal,
    left_shift,
    less,
    less_equal,
    matmul,
    matmul_acc,
    maximum,
    minimum,
    mul,
    not_equal,
    right_shift,
    sub,
)
from .language.creation_ops import (
    cast,
    concat,
    full,
    full_like,
    zeros,
    zeros_acc,
    zeros_like,
)
from .language.debug_ops import (
    inline,
    inline_vf,
)
from .language.memory_ops import (
    copy,
    copy_in,
    copy_out,
    gather,
    scatter,
)
from .language.prog_model_ops import (
    block_idx,
    block_num,
    sub_block_idx,
    sub_block_num,
)
from .language.shape_ops import (
    broadcast_shapes,
    broadcast_tensors,
    broadcast_to,
    expand_dims,
    ravel,
    reshape,
    squeeze,
    transpose,
)
from .language.unary_ops import (
    abs,
    bitwise_not,
    ceil,
    cos,
    cosh,
    erf,
    exp,
    exp2,
    floor,
    layer_norm,
    log,
    log2,
    negative,
    relu,
    rms_norm,
    rsqrt,
    sin,
    sinh,
    softmax,
    sqrt,
    tan,
    tanh,
)
from .language.indexing_ops import (
    mask,
    where,
)
from .language.reduction_ops import (
    reduce_max,
    reduce_min,
    reduce_sum,
    reduce_prod,
)

from .runtime.compiler import CompileOptions
from .runtime.jit import jit

__all__ += [
    # .language.global_tensor
    "GlobalTensor",
    "global_tensor",
    # .language.local_tensor
    "LocalTensor",
    "RoundMode",
    "TensorLocation",
    # .language.range
    "range",
    "static_range",
    # .language.utils
    "ceildiv",
    # .language.atomic_ops
    "atomic_add",
    "atomic_max",
    "atomic_min",
    # .language.binary_ops
    "add",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "div",
    "equal",
    "greater",
    "greater_equal",
    "left_shift",
    "less",
    "less_equal",
    "matmul",
    "matmul_acc",
    "maximum",
    "minimum",
    "mul",
    "not_equal",
    "right_shift",
    "sub",
    # .language.creation_ops
    "cast",
    "concat",
    "full",
    "full_like",
    "zeros",
    "zeros_acc",
    "zeros_like",
    # .language.debug_ops
    "inline",
    "inline_vf",
    # .language.memory_ops
    "copy",
    "copy_in",
    "copy_out",
    "gather",
    "scatter",
    # .language.prog_model_ops
    "block_idx",
    "block_num",
    "sub_block_idx",
    "sub_block_num",
    # .language.shape_ops
    "broadcast_shapes",
    "broadcast_tensors",
    "broadcast_to",
    "expand_dims",
    "ravel",
    "reshape",
    "squeeze",
    "transpose",
    # .language.unary_ops
    "abs",
    "bitwise_not",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "exp",
    "exp2",
    "floor",
    "layer_norm",
    "log",
    "log2",
    "negative",
    "relu",
    "rms_norm",
    "rsqrt",
    "sin",
    "sinh",
    "softmax",
    "sqrt",
    "tan",
    "tanh",
    # .language.indexing_ops
    "mask",
    "where",
    # .language.reduction_ops
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    # .runtime.compiler
    "CompileOptions",
    # .runtime.jit
    "jit",
]
