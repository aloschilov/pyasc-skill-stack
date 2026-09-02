# Copyright (c) 2025 ISE Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ..core.dtype import KnownTypes as KT
from ..core.ir_value import materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.types import BrcbRepeatParams
from ..core.utils import require_jit, global_builder
from .utils import set_common_docstring


@require_jit
@set_common_docstring(api_name="brcb")
def brcb(dst: LocalTensor, src0: LocalTensor, repeat_times: int, repeat_params: BrcbRepeatParams) -> None:
    builder = global_builder.get_ir_builder()

    builder.create_asc_BrcbL0Op(dst.to_ir(), src0.to_ir(), _mat(repeat_times, KT.uint8).to_ir(), repeat_params.to_ir())
