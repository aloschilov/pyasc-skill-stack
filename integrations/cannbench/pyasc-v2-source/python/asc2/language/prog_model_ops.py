# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.language.basic.sys_var import get_block_idx, get_block_num
from asc.language.core.dtype import KnownTypes
from asc.language.core.ir_value import PlainValue
from asc.language.core.utils import global_builder, require_jit


@require_jit
def block_idx() -> PlainValue:
    """
    Returns the current block (NPU core) index.

    In the PyAsc2 programming model, kernels are executed across multiple NPU blocks (cores). This function returns the
    index of the current block, which can be used to determine which portion of the data to process.

    Returns:
        PlainValue: The current block index (0-based)

    Examples:
        Get the current block index to compute the data offset: ::

            idx = asc2.block_idx()
            offset = idx * TILE_SIZE
            tile = asc2.copy_in(x_gm, [offset], [TILE_SIZE])
    """
    return get_block_idx()


@require_jit
def block_num() -> PlainValue:
    """
    Returns the total number of blocks (NPU cores) allocated for the kernel.

    This function returns the total number of NPU blocks that are executing the kernel, which was specified when
    launching the kernel.

    Returns:
        PlainValue: The total number of blocks

    Examples:
        Use block count to compute a stride across blocks: ::

            idx = asc2.block_idx()
            n_blocks = asc2.block_num()
            stride = n_blocks * TILE_SIZE
    """
    return get_block_num()


@require_jit
def sub_block_idx() -> PlainValue:
    """
    Returns the current sub-block index of vector or cube unit on the AI core.

    This function is useful to distinguish between vector sub-cores if the platform has multiple of them, especially
    when the kernel employs both vector and cube units. For example, there are 2 vector sub-cores and 1 cube sub-core
    on Ascend950PR_9599 chip.

    Returns:
        PlainValue: The current sub-block index

    Examples:
        Use with block_idx to compute starting iteration in a loop: ::

            idx = asc2.block_idx()
            num = asc2.sub_block_num()
            start = idx / num
            for i in asc2.range(start, total, step=asc2.block_num()):
                ...
    """
    return PlainValue(global_builder.get_ir_builder().create_asc_GetSubBlockIdxOp(KnownTypes.int_.to_ir()))


@require_jit
def sub_block_num() -> PlainValue:
    """
    Returns the number of vector or cube sub-cores on the AI core.

    This function returns the count of sub-cores available on the current AI core. For example, on Ascend950PR_9599
    chip there are 2 vector sub-cores and 1 cube sub-core, so this function returns 2 when called from vector code
    and 1 when called from cube code.

    Returns:
        PlainValue: The number of sub-cores on the AI core

    Examples:
        Use with block_idx to compute starting iteration in a loop: ::

            idx = asc2.cast(asc2.block_idx(), asc2.int32)
            num = asc2.cast(asc2.sub_block_num(), asc2.int32)
            start = idx / num
            for i in asc2.range(start, total, step=asc2.block_num()):
                ...
    """
    return PlainValue(global_builder.get_ir_builder().create_asc_GetSubBlockNumOp(KnownTypes.int_.to_ir()))
