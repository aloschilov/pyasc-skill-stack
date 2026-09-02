# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from .activation import softmax, swiglu

from .math import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    ceil,
    cos,
    cosh,
    digamma,
    erf,
    erfc,
    exp,
    floor,
    frac,
    lgamma,
    log,
    round,
    sign,
    sin,
    sinh,
    tan,
    tanh,
    trunc,
    power,
    xor,
    axpy,
)
from .matmul import (
    Matmul,
    MatmulType,
    get_matmul_api_tiling,
    get_basic_config,
    get_ib_share_norm_config,
    get_mdl_config,
    get_mm_config,
    get_normal_config,
    get_special_basic_config,
    get_special_mdl_config,
    register_matmul,
)
from .normalization import rmsnorm
from .quantization import quant
from .tiling import (
    MatmulApiStaticTiling,
    RmsNormTiling,
    SoftmaxTiling,
    TCubeTiling,
)
from .sort import (
    concat,
    extract,
)
from .types import (
    MatmulBatchParams,
    MatmulConfig,
    MatmulFuncParams,
    MatmulQuantParams,
    MatmulShapeParams,
    QuantConfig,
)

__all__ = [
    # .activation
    "softmax",
    "swiglu",
    # .math
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "ceil",
    "cos",
    "cosh",
    "digamma",
    "erf",
    "erfc",
    "exp",
    "floor",
    "frac",
    "lgamma",
    "log",
    "round",
    "sign",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "trunc",
    "power",
    "xor",
    "axpy",
    # .matmul
    "Matmul",
    "MatmulType",
    "get_basic_config",
    "get_ib_share_norm_config",
    "get_matmul_api_tiling",
    "get_mdl_config",
    "get_mm_config",
    "get_normal_config",
    "get_special_basic_config",
    "get_special_mdl_config",
    "register_matmul",
    # .normalization
    "rmsnorm",
    # .quantization
    "quant",
    # .sort
    "concat",
    "extract",
    # .tiling
    "MatmulApiStaticTiling",
    "RmsNormTiling",
    "SoftmaxTiling",
    "TCubeTiling",
    # .types
    "MatmulBatchParams",
    "MatmulConfig",
    "MatmulFuncParams",
    "MatmulQuantParams",
    "MatmulShapeParams",
    "QuantConfig",
]
