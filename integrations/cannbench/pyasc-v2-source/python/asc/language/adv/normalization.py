# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, Union, overload

from ..core.ir_value import RuntimeNumeric, \
                            materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import require_jit, global_builder
from .tiling import RmsNormTiling


@overload
def rmsnorm(dst: LocalTensor, src: LocalTensor, gamma: LocalTensor, epsilon: Union[float, int], tiling: RmsNormTiling,
            temp_buffer: Optional[LocalTensor] = None, basic_block: bool = False) -> None:
    ...


@require_jit
def rmsnorm(dst: LocalTensor, src: LocalTensor, gamma: LocalTensor, epsilon: RuntimeNumeric, tiling: RmsNormTiling,
            temp_buffer: Optional[LocalTensor] = None, basic_block: bool = False) -> None:
    temp_buffer = temp_buffer.to_ir() if temp_buffer is not None else None
    epsilon = _mat(epsilon, src.dtype)
    global_builder.get_ir_builder().create_asc_RmsNormOp(basicBlock=basic_block, dst=dst.to_ir(), src=src.to_ir(),
                                                         gamma=gamma.to_ir(), epsilon=epsilon.to_ir(),
                                                         tiling=tiling.to_ir(), sharedTmpBuffer=temp_buffer)
