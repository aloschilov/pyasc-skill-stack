# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ..core.utils import require_jit, global_builder
from .utils import set_common_docstring
from ..core.tensor import GlobalTensor, LocalTensor


@require_jit
@set_common_docstring(api_name="notify_next_block")
def notify_next_block(gm_workspace: GlobalTensor, ub_workspace: LocalTensor) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_NotifyNextBlockOp(gm_workspace.to_ir(), ub_workspace.to_ir())


@require_jit
@set_common_docstring(api_name="wait_pre_block")
def wait_pre_block(gm_workspace: GlobalTensor, ub_workspace: LocalTensor) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_WaitPreBlockOp(gm_workspace.to_ir(), ub_workspace.to_ir())
