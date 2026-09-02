# Copyright (c) 2025 ISE Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload

from ..core.dtype import KnownTypes as KT
from ..core.ir_value import materialize_ir_value as _mat, RuntimeInt
from ..core.tensor import LocalTensor, GlobalTensor
from ..core.types import FixpipeConfig, FixpipeParamsV220
from ..core.utils import require_jit, global_builder, OverloadDispatcher
from .utils import set_common_docstring


@overload
def fixpipe(dst: GlobalTensor, src: LocalTensor, params: FixpipeParamsV220,
            config: FixpipeConfig = FixpipeConfig.cfg_row_major) -> None:
    ...


@overload
def fixpipe(dst: GlobalTensor, src: LocalTensor, cbuf_workspace: LocalTensor, params: FixpipeParamsV220,
            config: FixpipeConfig = FixpipeConfig.cfg_row_major) -> None:
    ...


@require_jit
@set_common_docstring("fixpipe")
def fixpipe(dst: GlobalTensor, src: LocalTensor, *args, **kwargs) -> None:
    builder = global_builder.get_ir_builder()
    dispatcher = OverloadDispatcher("fixpipe")

    @dispatcher.register_auto
    def _(params: FixpipeParamsV220, config: FixpipeConfig = FixpipeConfig.cfg_row_major):
        if callable(config):
            config = config()
        builder.create_asc_FixpipeOp(dst.to_ir(), src.to_ir(), params.to_ir(), config.to_ir())

    @dispatcher.register_auto
    def _(cbuf_workspace: LocalTensor, params: FixpipeParamsV220, config: FixpipeConfig = FixpipeConfig.cfg_row_major):
        if callable(config):
            config = config()
        builder.create_asc_FixpipeWithWorkspaceOp(dst.to_ir(), src.to_ir(), cbuf_workspace.to_ir(), params.to_ir(),
                                                  config.to_ir())

    dispatcher(*args, **kwargs)


@overload
def set_fix_pipe_pre_quant_flag(config: int) -> None:
    ...


@require_jit
@set_common_docstring("set_fix_pipe_pre_quant_flag")
def set_fix_pipe_pre_quant_flag(config: RuntimeInt) -> None:
    builder = global_builder.get_ir_builder()
    builder.create_asc_SetFixpipePreQuantFlagOp(_mat(config, KT.uint64).to_ir())
