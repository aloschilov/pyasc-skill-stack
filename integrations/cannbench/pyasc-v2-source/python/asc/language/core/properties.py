# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass

from .dtype import DataType
from .dtype import KnownTypes as KT
from .ir_value import PlainValue
from .utils import require_jit, global_builder


@dataclass(frozen=True)
class Property:
    name: str
    dtype: DataType


@require_jit
def property(prop: Property) -> PlainValue:
    builder = global_builder.get_ir_builder()
    handle = builder.create_emitc_ConstantOp(builder.get_opaque_attr(prop.name), prop.dtype.to_ir())
    return PlainValue(handle, prop.dtype)


DEFAULT_C0_SIZE = Property("::AscendC::DEFAULT_C0_SIZE", KT.int32)
ONE_BLK_SIZE = Property("::AscendC::ONE_BLK_SIZE", KT.int16)
TOTAL_L1_SIZE = Property("::AscendC::TOTAL_L1_SIZE", KT.int32)
TOTAL_L0C_SIZE = Property("::AscendC::TOTAL_L0C_SIZE", KT.int32)