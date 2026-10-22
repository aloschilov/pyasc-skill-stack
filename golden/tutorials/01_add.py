# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Vector Addition
===============

This tutorial demonstrates vector addition operator and its launch on Ascend simulator.

"""

import asctile
import torch


# The functions which are executed on Ascend NPU must be marked with `@asctile.jit` decorator.
# Available parameters for @asctile.jit decorator can be seen in the documentation:
# https://compiler-team-ru.github.io/pyasc/python-api/rst/runtime/index.html
@asctile.jit
def vector_add(
        # Pointers to input and output tensors should have `asctile.GlobalAddress` type.
        x_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress, out_ptr: asctile.GlobalAddress,
        # Scalar parameters are passed as Python types (e.g. `int`, `float`).
        # For optimization purposes it is recommended to pass scalar parameter as constants (e.g. `asctile.ConstExpr[int]`).
        size: int,
        # tile_size is used in asctile.copy_in must be asctile.ConstExpr
        tile_size: asctile.ConstExpr):

    # Tensor descriptor is created from `asctile.GlobalAddress` to represent entire tensor.
    x_gm = asctile.global_tensor(x_ptr, [size])
    y_gm = asctile.global_tensor(y_ptr, [size])
    out_gm = asctile.global_tensor(out_ptr, [size])

    # Python expressions are used to calculate offset and define the loop iterating over tiles:
    # `asctile.block_num()` function provides number of AICOREs launched.
    block_loop_num = asctile.ceildiv(asctile.ceildiv(size, asctile.block_num()), tile_size)
    # `asctile.block_idx()` function is used to get current AICORE index.
    block_offset = asctile.block_idx() * tile_size * block_loop_num

    # `unroll_factor` parameter of `asctile.range` in `for` loop can be used to manage software pipelining. Set it to `2` to enable double buffering.
    # `parallel` parameter of `asctile.range` in `for` loop enable overlapping of store operation of `i`-th iteration and load of `i+1`-th iteration.
    # It is user responsibility to ensure that there are no data dependencies between overlapped iterations.
    for i in asctile.range(block_loop_num, unroll_factor=2):
        tile_offset = block_offset + i * tile_size
        # `asctile.copy_in` is used to create a local tensor object which is used for further calculations. Data movement from GM to UB happens in this operation.
        x = asctile.copy_in(x_gm, [tile_offset], [tile_size])
        y = asctile.copy_in(y_gm, [tile_offset], [tile_size])
        # One or more operations can be applied for tensors. It is compiler responsibility to allocate required number of memory blocks in UB.
        out = x + y
        # `asctile.copy_out` is used to move data from UB back to GM.
        asctile.copy_out(out, out_gm, [tile_offset])


if __name__ == "__main__":
    backend = asctile.Backend.Model  # can be "Model" for simulator or "NPU" for device
    soc_version = asctile.Platform.Ascend950PR_9599  # Device version
    device_id = 0  # might be necessary to provide if more than one NPU device is present in the system
    asctile.set_platform(backend, soc_version, device_id)

    size = 8192
    block_num = 16
    tile_size = 128
    dtype = torch.float32
    # Allocate tensors
    x = torch.randn(size, dtype=dtype)
    y = torch.randn(size, dtype=dtype)
    out = torch.empty_like(x)

    # For the kernel invocation, number of AICOREs should be provided in brackets:
    vector_add[block_num](x, y, out, size, tile_size)
    assert torch.allclose(out, x + y)
