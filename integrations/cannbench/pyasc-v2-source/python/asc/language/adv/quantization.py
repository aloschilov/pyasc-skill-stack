# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, overload

from ..core.ir_value import RuntimeFloat, RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import require_jit, global_builder
from .types import QuantConfig


@overload
def quant(dst: LocalTensor, src: LocalTensor, scale: float, offset: float, count: Optional[int] = None,
          temp_buffer: Optional[LocalTensor] = None, reuse_source: bool = False,
          config: Optional[QuantConfig] = None) -> None:
    ...


@require_jit
def quant(dst: LocalTensor, src: LocalTensor, scale: RuntimeFloat, offset: RuntimeFloat,
          count: Optional[RuntimeInt] = None, temp_buffer: Optional[LocalTensor] = None, reuse_source: bool = False,
          config: Optional[QuantConfig] = None) -> None:
    scale = _mat(scale, src.dtype).to_ir()
    offset = _mat(offset, src.dtype).to_ir()
    count = _mat(count).to_ir() if count is not None else None
    temp_buffer = temp_buffer.to_ir() if temp_buffer is not None else None
    config = config.to_ir() if config is not None else None
    global_builder.get_ir_builder().create_asc_QuantOp(isReuseSource=reuse_source, dst=dst.to_ir(),
                                                       srcTensor=src.to_ir(), scale=scale, offset=offset,
                                                       calCount=count, sharedTmpBuffer=temp_buffer, config=config)
