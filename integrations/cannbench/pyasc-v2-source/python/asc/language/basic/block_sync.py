# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload, Optional

from ..core.enums import HardEvent, PipeID, MemDsbT
from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.utils import require_jit, global_builder
from .utils import set_common_docstring
from ..core.tensor import GlobalTensor, LocalTensor


@require_jit
@set_common_docstring(api_name="data_sync_barrier")
def data_sync_barrier(arg0: MemDsbT) -> None:
    global_builder.get_ir_builder().create_asc_DataSyncBarrierOp(arg0.value)


@require_jit
@set_common_docstring(api_name="pipe_barrier")
def pipe_barrier(pipe: PipeID) -> None:
    global_builder.get_ir_builder().create_asc_PipeBarrierOp(pipe)


@overload
def set_flag(event: HardEvent, event_id: int = 0) -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_flag")
def set_flag(event: HardEvent, event_id: RuntimeInt = 0) -> None:
    event_id = _mat(event_id).to_ir()
    global_builder.get_ir_builder().create_asc_SetFlagOp(event, event_id)


@overload
def wait_flag(event: HardEvent, event_id: int = 0) -> None:
    ...


@require_jit
@set_common_docstring(api_name="wait_flag")
def wait_flag(event: HardEvent, event_id: RuntimeInt = 0) -> None:
    event_id = _mat(event_id).to_ir()
    global_builder.get_ir_builder().create_asc_WaitFlagOp(event, event_id)


@overload
def ib_set(gm_workspace: GlobalTensor, ub_workspace: LocalTensor, block_idx: int, event_id: int,
           is_aiv_only: bool = True) -> None:
    ...


@require_jit
@set_common_docstring(api_name="ib_set")
def ib_set(gm_workspace: GlobalTensor, ub_workspace: LocalTensor, block_idx: RuntimeInt, event_id: RuntimeInt,
           is_aiv_only: bool = True) -> None:
    builder = global_builder.get_ir_builder()
    block_idx_ir = _mat(block_idx).to_ir()
    event_id_ir = _mat(event_id).to_ir()
    if is_aiv_only:
        builder.create_asc_IBSetOp(gm_workspace.to_ir(), ub_workspace.to_ir(), block_idx_ir, event_id_ir)
    else:
        builder.create_asc_IBSetOp(gm_workspace.to_ir(), ub_workspace.to_ir(), block_idx_ir, event_id_ir,
                                   is_aiv_only=True)


@overload
def ib_wait(gm_workspace: GlobalTensor, ub_workspace: LocalTensor, block_idx: int, event_id: int,
            is_aiv_only: bool = True) -> None:
    ...


@require_jit
@set_common_docstring(api_name="ib_wait")
def ib_wait(gm_workspace: GlobalTensor, ub_workspace: LocalTensor, block_idx: RuntimeInt, event_id: RuntimeInt,
            is_aiv_only: bool = True) -> None:
    builder = global_builder.get_ir_builder()
    block_idx_ir = _mat(block_idx).to_ir()
    event_id_ir = _mat(event_id).to_ir()
    if is_aiv_only:
        builder.create_asc_IBWaitOp(gm_workspace.to_ir(), ub_workspace.to_ir(), block_idx_ir, event_id_ir)
    else:
        builder.create_asc_IBWaitOp(gm_workspace.to_ir(), ub_workspace.to_ir(), block_idx_ir, event_id_ir,
                                    is_aiv_only=True)


@overload
def sync_all(is_aiv_only: bool = True) -> None:
    ...


@overload
def sync_all(gm_workspace: GlobalTensor, ub_workspace: LocalTensor, used_cores: int = 0,
             is_aiv_only: bool = True) -> None:
    ...


@require_jit
@set_common_docstring(api_name="sync_all")
def sync_all(gm_workspace: Optional[GlobalTensor] = None, ub_workspace: Optional[LocalTensor] = None,
             used_cores: RuntimeInt = 0, is_aiv_only: bool = True) -> None:
    builder = global_builder.get_ir_builder()
    if gm_workspace is None or ub_workspace is None:
        if is_aiv_only:
            builder.create_asc_SyncAllHardOp()
        else:
            builder.create_asc_SyncAllHardOp(is_aiv_only=True)
    else:
        used_cores_ir = _mat(used_cores).to_ir()
        if is_aiv_only:
            builder.create_asc_SyncAllSoftOp(gm_workspace.to_ir(), ub_workspace.to_ir(), used_cores_ir)
        else:
            builder.create_asc_SyncAllSoftOp(gm_workspace.to_ir(), ub_workspace.to_ir(), used_cores_ir,
                                             is_aiv_only=True)


@overload
def cross_core_set_flag(flag_id: int, mode_id: int, pipe: PipeID) -> None:
    ...


@require_jit
@set_common_docstring(api_name="cross_core_set_flag")
def cross_core_set_flag(flag_id: RuntimeInt, mode_id: int, pipe: PipeID) -> None:
    builder = global_builder.get_ir_builder()
    flag_id_ir = _mat(flag_id).to_ir()
    builder.create_asc_CrossCoreSetFlagOp(flag_id_ir, mode_id, pipe)


@overload
def cross_core_wait_flag(flag_id: int, mode_id: int, pipe: PipeID) -> None:
    ...


@require_jit
@set_common_docstring(api_name="cross_core_wait_flag")
def cross_core_wait_flag(flag_id: RuntimeInt, mode_id: int, pipe: PipeID) -> None:
    builder = global_builder.get_ir_builder()
    flag_id_ir = _mat(flag_id).to_ir()
    builder.create_asc_CrossCoreWaitFlagOp(flag_id_ir, mode_id, pipe)
