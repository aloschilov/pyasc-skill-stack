# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

import asctile
import pytest
import torch


# Reads (h,w) sub tile
@asctile.jit(reuse_alloc=1)
def transpose_block(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, width: asctile.ConstExpr[int],
                    height: asctile.ConstExpr[int], block_width: asctile.ConstExpr[int],
                    block_height: asctile.ConstExpr[int], tile_width: asctile.ConstExpr[int],
                    tile_height: asctile.ConstExpr[int], repeat: asctile.ConstExpr[int],
                    unroll_factor: asctile.ConstExpr[int]):
    total_tiles_w = asctile.ceildiv(width, block_width)

    global_tensor = asctile.global_tensor(input_ptr, [height, width])
    result_tensor = asctile.global_tensor(output_ptr, [width, height])
    for i in asctile.range(asctile.block_idx(), repeat, asctile.block_num(), unroll_factor=unroll_factor):
        offset_x = (i % total_tiles_w) * block_width
        offset_y = (i // total_tiles_w) * block_height
        load_width = block_width if block_width < width - offset_x else width - offset_x
        load_height = block_height if block_height < height - offset_y else height - offset_y
        input = asctile.copy_in(global_tensor, [offset_y, offset_x], [tile_height, tile_width],
                                real_shape=[load_height, load_width])
        transposed = input.transpose()
        asctile.copy_out(transposed, result_tensor, [offset_x, offset_y], real_shape=[load_width, load_height])


# Reads (height,n) at once
@asctile.jit(reuse_alloc=1)
def transpose_column(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, width: asctile.ConstExpr[int],
                     height: asctile.ConstExpr[int], block_size: asctile.ConstExpr[int],
                     tile_width: asctile.ConstExpr[int], tile_height: asctile.ConstExpr[int],
                     total_count: asctile.ConstExpr[int], unroll_factor: asctile.ConstExpr[int]):

    global_tensor = asctile.global_tensor(input_ptr, [height, width])
    result_tensor = asctile.global_tensor(output_ptr, [width, height])
    for i in asctile.range(asctile.block_idx(), total_count, asctile.block_num(), unroll_factor=unroll_factor):
        offset = i * block_size
        load_width = block_size if block_size < width - offset else width - offset
        input = asctile.copy_in(global_tensor, [0, offset], [tile_height, tile_width], real_shape=[height, load_width])
        transposed = input.transpose()
        asctile.copy_out(transposed, result_tensor, [offset, 0], real_shape=[load_width, height])


# Reads (n,width) at once
@asctile.jit(reuse_alloc=1)
def transpose_line(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, width: asctile.ConstExpr[int],
                   height: asctile.ConstExpr[int], block_size: asctile.ConstExpr[int],
                   tile_width: asctile.ConstExpr[int], tile_height: asctile.ConstExpr[int],
                   total_count: asctile.ConstExpr[int], unroll_factor: asctile.ConstExpr[int]):
    global_tensor = asctile.global_tensor(input_ptr, [height, width])
    result_tensor = asctile.global_tensor(output_ptr, [width, height])
    for i in asctile.range(asctile.block_idx(), total_count, asctile.block_num(), unroll_factor=unroll_factor):
        offset = i * block_size
        load_height = block_size if block_size < height - offset else height - offset
        input = asctile.copy_in(global_tensor, [offset, 0], [tile_height, tile_width], real_shape=[load_height, width])
        transposed = input.transpose()
        asctile.copy_out(transposed, result_tensor, [0, offset], real_shape=[width, load_height])


# For permutation like [0,2,1,3] when last dimension untouch
# Iterate over first dimension
@asctile.jit(reuse_alloc=1)
def transpose_nlast_axis(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                         axis_step: asctile.ConstExpr, repeats: asctile.ConstExpr, permute: asctile.ConstExpr,
                         gm_read_shape: asctile.ConstExpr, gm_write_shape: asctile.ConstExpr,
                         ub_shape: asctile.ConstExpr, read_shape: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    input_tensor = asctile.global_tensor(input_ptr, gm_read_shape)
    output_tensor = asctile.global_tensor(output_ptr, gm_write_shape)
    for i in asctile.range(asctile.block_idx(), repeats, asctile.block_num(), unroll_factor=1):

        store_offsets = [0] * len(permute)
        store_offsets[permute[0]] = i * axis_step

        tile = asctile.copy_in(input_tensor, [i * axis_step, 0], read_shape).reshape(*ub_shape)
        tile = tile.transpose(*permute)
        asctile.copy_out(tile, output_tensor, store_offsets)


@asctile.jit(reuse_alloc=1)
def transpose_nlast_axis_fat(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                             axis_step: asctile.ConstExpr, repeats: asctile.ConstExpr, inner_steps: asctile.ConstExpr,
                             permute: asctile.ConstExpr, gm_read_shape: asctile.ConstExpr,
                             gm_write_shape: asctile.ConstExpr, ub_shape: asctile.ConstExpr,
                             read_shape: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    input_tensor = asctile.global_tensor(input_ptr, gm_read_shape)
    output_tensor = asctile.global_tensor(output_ptr, gm_write_shape)

    for i in asctile.range(asctile.block_idx(), repeats, asctile.block_num(), unroll_factor=unroll_factor):
        store_offsets = [0] * len(permute)
        outer_id = i // inner_steps
        axis_id = i % inner_steps
        store_offsets[permute[0]] = outer_id
        store_offsets[permute[1]] = axis_id * axis_step
        tile = asctile.copy_in(input_tensor, [axis_id * axis_step + outer_id * gm_write_shape[permute[1]], 0],
                               read_shape).reshape(*ub_shape)
        tile = tile.transpose(*permute)
        asctile.copy_out(tile, output_tensor, store_offsets)


# For cases when last dimension is unchanged.
def launch_nlast_axis(input, permute, unroll_factor, block_num, input_dtype, ub_size, runs, profiler):
    input_shape = list(input.shape)
    output_shape = [input_shape[permute[i]] for i in range(0, len(input_shape))]
    output = torch.zeros(output_shape, dtype=input.dtype)
    items_in_block = 32 // input.element_size()
    inner_dimensions = math.prod(input_shape[1:])
    axis_step = ub_size // (inner_dimensions * unroll_factor * input.element_size()) // 2
    ub_shape = [i for i in input_shape]
    ub_shape[-1] = asctile.ceildiv(ub_shape[-1], items_in_block) * items_in_block
    gm_write_shape = output_shape
    if axis_step > 0:
        ub_shape[0] = axis_step
        read_shape = [axis_step, inner_dimensions]
        gm_read_shape = [input_shape[0], inner_dimensions]
        repeats = asctile.ceildiv(input_shape[0], axis_step)
        with profiler.profile():
            for _ in range(runs):
                transpose_nlast_axis[block_num](input, output, axis_step, repeats, permute, gm_read_shape,
                                                gm_write_shape, ub_shape, read_shape, unroll_factor)
    else:
        gm_read_shape = [input_shape[0] * input_shape[1], math.prod(input_shape[2:])]
        axis_step = ub_size // (math.prod(input_shape[2:]) * unroll_factor * input.element_size())
        assert (axis_step > 0)
        ub_shape[0] = 1
        ub_shape[1] = axis_step
        read_shape = [axis_step, math.prod(input_shape[2:])]
        axis_steps_count = asctile.ceildiv(input_shape[1], axis_step)
        repeats = axis_steps_count * input_shape[0] * axis_steps_count
        with profiler.profile():
            for _ in range(runs):
                transpose_nlast_axis_fat[block_num](input, output, axis_step, repeats, axis_steps_count, permute,
                                                    gm_read_shape, gm_write_shape, ub_shape, read_shape, unroll_factor)
    return output


# Split input data for one axis loads at once slice of size axis_step on that dimension
@asctile.jit(reuse_alloc=1)
def transpose_one_axis(
    input_ptr: asctile.GlobalAddress,
    output_ptr: asctile.GlobalAddress,
    input_shape: asctile.ConstExpr,  # Read tensor shape
    axis_step: asctile.ConstExpr,  # How many compute per step
    load_shape_axis: asctile.ConstExpr,  # Dim num in input_shape
    store_shape_axis: asctile.ConstExpr,  # Dim num in transposed shape
    ub_load_shape: asctile.ConstExpr,  # Buffer size in ub
    load_shape: asctile.ConstExpr,  # Used to fill real_shape on load
    permute: asctile.ConstExpr,  # Dimensions to permute
    block_count: asctile.ConstExpr,  # How many steps do total
    unroll_factor: asctile.ConstExpr,
):

    output_shape = []
    ub_store_shape = []
    store_shape = []
    for dim in asctile.static_range(0, len(input_shape)):
        output_shape += [input_shape[permute[dim]]]
        ub_store_shape += [ub_load_shape[permute[dim]]]
        store_shape += [load_shape[permute[dim]]]
    input_tensor = asctile.global_tensor(input_ptr, input_shape)
    output_tensor = asctile.global_tensor(output_ptr, output_shape)
    for i in asctile.range(asctile.block_idx(), block_count, asctile.block_num(), unroll_factor=unroll_factor):
        offset = i * axis_step
        read_offsets = [0] * (load_shape_axis) + [offset] + [0] * (len(load_shape) - 1 - load_shape_axis)
        load_tensor = asctile.copy_in(input_tensor, read_offsets, ub_load_shape)
        transposed_tensor = load_tensor.transpose(*permute)
        write_offsets = [0] * (store_shape_axis) + [offset] + [0] * (len(store_shape) - 1 - store_shape_axis)
        asctile.copy_out(transposed_tensor, output_tensor, write_offsets)


def launch_one_axis(input, permute, axis, axis_step, unroll_factor, cores, dtype, runs, profiler):
    input_shape = list(input.shape)
    output_shape = [input_shape[permute[i]] for i in range(0, len(input_shape))]
    items_in_block = 32 // input.element_size()
    output = torch.zeros(output_shape, dtype=dtype)

    load_shape_axis = axis
    store_shape_axis = 0  # Where load_shape_axis goes in target shape
    for i in range(0, len(input_shape)):
        if permute[i] == load_shape_axis:
            store_shape_axis = i

    block_count = asctile.ceildiv(input_shape[load_shape_axis], axis_step)
    load_shape = [input_shape[i] if i != load_shape_axis else axis_step for i in range(0, len(input_shape))]

    ub_shape = [i for i in load_shape]
    ub_shape[load_shape_axis] = axis_step
    ub_shape[-1] = asctile.ceildiv(ub_shape[-1], items_in_block) * items_in_block
    ub_shape[permute[-1]] = asctile.ceildiv(ub_shape[permute[-1]], items_in_block) * items_in_block
    ub_load_shape = ub_shape

    with profiler.profile():
        for _ in range(runs):
            transpose_one_axis[cores](input, output, input_shape, axis_step, load_shape_axis, store_shape_axis,
                                      ub_load_shape, load_shape, permute, block_count, unroll_factor)
    return output


@asctile.jit(reuse_alloc=1)
def simple_copy(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, input_lenth,
                tile_shape: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_lenth])
    out_gm = asctile.global_tensor(output_ptr, [input_lenth])
    total_repeats = asctile.ceildiv(input_lenth, tile_shape)

    for i in asctile.range(asctile.block_idx(), total_repeats, asctile.block_num(), unroll_factor=unroll_factor):
        data = asctile.copy_in(in_gm, [i * tile_shape], [tile_shape])
        asctile.copy_out(data, out_gm, [i * tile_shape])


# Split tensor by 2 dimensions (store_axis), loads subpart
# Example: output_shape = [1024,2048,512], store_axis=[0,1] iterate by [STEP1,STEP2,512]
@asctile.jit(reuse_alloc=1)
def transpose_2_axis(
    input_ptr: asctile.GlobalAddress,
    output_ptr: asctile.GlobalAddress,
    input_shape: asctile.ConstExpr,  # Read tensor shape
    axis_step: asctile.ConstExpr,  # How many compute per step x2
    store_axis: asctile.ConstExpr,  # Dim num in *output_shape*
    ub_load_shape: asctile.ConstExpr,  # Buffer size in ub
    load_shape: asctile.ConstExpr,  # Used to fill real_shape on load
    permute: asctile.ConstExpr,  # Dimensions to permute
    block_count: asctile.ConstExpr,  # How many steps do total
    block_width: asctile.ConstExpr,
    unroll_factor: asctile.ConstExpr,
):

    output_shape = []
    ub_store_shape = []
    store_shape = []
    for dim in asctile.static_range(0, len(input_shape)):
        output_shape += [input_shape[permute[dim]]]
        ub_store_shape += [ub_load_shape[permute[dim]]]
        store_shape += [load_shape[permute[dim]]]
    load_shape_axis0 = permute[store_axis[0]]
    load_shape_axis1 = permute[store_axis[1]]

    input_tensor = asctile.global_tensor(input_ptr, input_shape)
    output_tensor = asctile.global_tensor(output_ptr, output_shape)
    for i in asctile.range(asctile.block_idx(), block_count, asctile.block_num(), unroll_factor=unroll_factor):
        offset0 = i % block_width * axis_step[0]
        offset1 = i // block_width * axis_step[1]
        count0 = axis_step[0] if offset0 + axis_step[0] < input_shape[
            load_shape_axis0] else input_shape[load_shape_axis0] - offset0
        count1 = axis_step[1] if offset1 + axis_step[1] < input_shape[
            load_shape_axis1] else input_shape[load_shape_axis1] - offset1

        read_offsets = [0] * len(load_shape)
        read_offsets[load_shape_axis0] = offset0
        read_offsets[load_shape_axis1] = offset1
        read_count = load_shape
        read_count[load_shape_axis0] = count0
        read_count[load_shape_axis1] = count1

        load_tensor = asctile.copy_in(input_tensor, read_offsets, ub_load_shape)
        transposed_tensor = load_tensor.transpose(*permute)

        write_offsets = [0] * len(load_shape)
        write_offsets[store_axis[0]] = offset0
        write_offsets[store_axis[1]] = offset1
        write_count = store_shape
        write_count[store_axis[0]] = count0
        write_count[store_axis[1]] = count1

        asctile.copy_out(transposed_tensor, output_tensor, write_offsets)


def launch_2axis(input, permute, axis, step, dtype, cores, unroll_factor, runs, profiler):
    output_size = [input.shape[i] for i in permute]
    output = torch.zeros(output_size, dtype=dtype)
    items_in_block = 32 // input.element_size()
    axis0 = permute[axis[0]]
    axis1 = permute[axis[1]]

    ub_size = list(input.shape)
    ub_size[axis0] = step[0]
    ub_size[axis1] = step[1]
    ub_size[-1] = asctile.ceildiv(ub_size[-1], items_in_block) * items_in_block
    ub_size[permute[-1]] = asctile.ceildiv(ub_size[permute[-1]], items_in_block) * items_in_block

    load_size = list(input.shape)
    load_size[axis0] = step[0]
    load_size[axis1] = step[1]

    blocks_axis0 = asctile.ceildiv(input.shape[axis0], step[0])
    blocks_axis1 = asctile.ceildiv(input.shape[axis1], step[1])
    total_blocks = blocks_axis0 * blocks_axis1

    with profiler.profile():
        for _ in range(runs):
            transpose_2_axis[cores](input, output, list(input.shape), step, axis, ub_size,\
                                    load_size, permute, total_blocks, blocks_axis0, unroll_factor)
    return output


def simplify_shape(input, permute):
    # Remove empty dimensions (of one element)
    dim_dec = [0] * len(permute)
    counter = 0
    for i in range(0, len(permute)):
        dim_dec[i] = counter
        if input[i] == 1:
            counter = counter + 1
            dim_dec[i] = -1
    new_permute = []
    new_shape = []
    for i in range(0, len(input)):
        if dim_dec[i] != -1:
            new_shape = new_shape + [input[i]]
    for i in range(0, len(permute)):
        if dim_dec[permute[i]] != -1:
            new_permute = new_permute + [permute[i] - dim_dec[permute[i]]]
    permute = new_permute
    input = new_shape
    # Merge dimensions together if they keep order in permute: 2,3,0,1 -> 1,0
    assert (len(input) == len(permute))
    result_shape = []
    dims = []
    # Check dimensions we can merge
    for i in range(0, len(input)):
        if i > 0 and permute[i - 1] + 1 == permute[i]:
            result_shape[-1] = result_shape[-1] * input[i]
            dims.append(dims[-1])
        else:
            result_shape.append(input[i])
            dims.append(permute[i])
    # fix dim order
    dim_id = 0
    for i in range(0, len(input)):
        count = dims.count(i)
        if count > 0:
            dims = [dim_id if j == i else j for j in dims]
            dim_id = dim_id + 1
    # remove duplicates
    result_permute = []
    for i in dims:
        if len(result_permute) == 0 or result_permute[-1] != i:
            result_permute.append(i)
    return result_shape, result_permute


# yapf: disable
@pytest.mark.parametrize("test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes, compile_params, runtime_params, tiling_key, tiling_params", [
# PYASC_TESTS_BEGIN
    ("transpose_test_1", 72, ([92, 256, 80], ), (torch.float32, ), ([256, 92, 80], ), (torch.float32, ), None, (2, [1, 0, 2]), 10004, (3, 0, 0, 1, 0, 0, 0, 72, 1, 20, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [92, 256, 80, 0, 0, 0, 0, 0], [256, 92, 80, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 1, 92, 256, 80], [1, 1, 256, 92, 80], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])),
    ("transpose_test_2", 72, ([1024, 4, 15, 64], ), (torch.float32, ), ([1024, 15, 4, 64], ), (torch.float32, ), None, (2, [0, 2, 1, 3]), 10004, (4, 0, 0, 8, 0, 0, 0, 72, 1, 56, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [1024, 4, 15, 64, 0, 0, 0, 0], [1024, 15, 4, 64, 0, 0, 0, 0], [0, 2, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 1024, 4, 15, 64], [1, 1024, 15, 4, 64], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])),
    ("transpose_test_3", 72, ([128, 32, 12, 64], ), (torch.float16, ), ([128, 12, 32, 64], ), (torch.float16, ), None, (2, [0, 2, 1, 3]), 10004, (4, 0, 0, 1, 0, 0, 0, 72, 1, 56, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [128, 32, 12, 64, 0, 0, 0, 0], [128, 12, 32, 64, 0, 0, 0, 0], [0, 2, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 128, 32, 12, 64], [1, 128, 12, 32, 64], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])),
    ("transpose_test_4", 72, ([7000, 8, 4, 32], ), (torch.float32, ), ([7000, 4, 32, 8], ), (torch.float32, ), None, (1, [0, 2, 3, 1]), 10002, (3, 1, 0, 1, 49, 0, 42, 72, 1, 71, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [7000, 8, 128, 0, 0, 0, 0, 0], [7000, 128, 8, 0, 0, 0, 0, 0], [0, 2, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 2, 4, 3], [1, 1, 7000, 8, 128], [1, 1, 7000, 128, 8], [1, 1, 49, 8, 128], [1, 1, 49, 128, 8], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 42, 8, 128], [1, 1, 42, 128, 8])),
    ("transpose_test_5", 72, ([7000, 100, 4, 32], ), (torch.float32, ), ([7000, 4, 32, 100], ), (torch.float32, ), None, (1, [0, 2, 3, 1]), 10002, (3, 1, 0, 1, 4, 0, 0, 72, 24, 22, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [7000, 100, 128, 0, 0, 0, 0, 0], [7000, 128, 100, 0, 0, 0, 0, 0], [0, 2, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 2, 4, 3], [1, 1, 7000, 100, 128], [1, 1, 7000, 128, 100], [1, 1, 4, 100, 128], [1, 1, 4, 128, 100], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 100, 128], [1, 1, 0, 128, 100])),
    ("transpose_test_6", 72, ([3072, 6, 8, 64], ), (torch.bfloat16, ), ([3072, 8, 6, 64], ), (torch.bfloat16, ), None, (2, [0, 2, 1, 3]), 10004, (4, 0, 0, 15, 0, 12, 0, 72, 2, 61, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [3072, 6, 8, 64, 0, 0, 0, 0], [3072, 8, 6, 64, 0, 0, 0, 0], [0, 2, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 3072, 6, 8, 64], [1, 3072, 8, 6, 64], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])),
    ("transpose_test_7", 72, ([3072, 8, 384], ), (torch.bfloat16, ), ([8, 3072, 384], ), (torch.bfloat16, ), None, (2, [1, 0, 2]), 10004, (3, 0, 0, 15, 0, 12, 0, 72, 2, 61, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [3072, 8, 384, 0, 0, 0, 0, 0], [8, 3072, 384, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 1, 3072, 8, 384], [1, 1, 8, 3072, 384], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])),
    ("transpose_test_8", 72, ([8, 3072, 384], ), (torch.bfloat16, ), ([3072, 8, 384], ), (torch.bfloat16, ), None, (2, [1, 0, 2]), 10004, (3, 1, 0, 119, 0, 97, 0, 72, 2, 64, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [8, 3072, 384, 0, 0, 0, 0, 0], [3072, 8, 384, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 1, 8, 3072, 384], [1, 1, 3072, 8, 384], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])),
    ("transpose_test_9", 72, ([256, 12, 256, 64], ), (torch.float32, ), ([256, 256, 12, 64], ), (torch.float32, ), None, (2, [0, 2, 1, 3]), 10004, (4, 1, 0, 1, 0, 0, 0, 72, 42, 48, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [256, 12, 256, 64, 0, 0, 0, 0], [256, 256, 12, 64, 0, 0, 0, 0], [0, 2, 1, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 256, 12, 256, 64], [1, 256, 256, 12, 64], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])),
    ("transpose_test_10", 66, ([50, 4096, 16], ), (torch.float32, ), ([4096, 50, 16], ), (torch.float32, ), None, (1, [1, 0, 2]), 10002, (3, 1, 0, 15, 63, 1, 1, 66, 1, 0, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [50, 4096, 16, 0, 0, 0, 0, 0], [4096, 50, 16, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 3, 2, 4], [1, 1, 50, 4096, 16], [1, 1, 4096, 50, 16], [1, 1, 50, 63, 16], [1, 1, 63, 50, 16], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 50, 1, 16], [1, 1, 1, 50, 16])),
    ("transpose_test_11", 65, ([50, 4096, 16], ), (torch.float32, ), ([4096, 16, 50], ), (torch.float32, ), None, (1, [1, 2, 0]), 10002, (2, 1, 0, 251, 1023, 25, 64, 65, 1, 0, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [50, 65536, 0, 0, 0, 0, 0, 0], [65536, 50, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 2, 4, 3], [1, 1, 1, 50, 65536], [1, 1, 1, 65536, 50], [1, 1, 1, 50, 1023], [1, 1, 1, 1023, 50], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 50, 64], [1, 1, 1, 64, 50])),
    ("transpose_test_12", 72, ([4096, 39, 1500], ), (torch.float32, ), ([4096, 1500, 39], ), (torch.float32, ), None, (1, [0, 2, 1]), 10002, (3, 2, 0, 251, 1, 245, 0, 72, 56, 64, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [4096, 39, 1500, 0, 0, 0, 0, 0], [4096, 1500, 39, 0, 0, 0, 0, 0], [0, 2, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 2, 4, 3], [1, 1, 4096, 39, 1500], [1, 1, 4096, 1500, 39], [1, 1, 1, 39, 1500], [1, 1, 1, 1500, 39], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 39, 1500], [1, 1, 0, 1500, 39])),
    ("transpose_test_13", 72, ([4096, 1500, 39], ), (torch.float32, ), ([4096, 39, 1500], ), (torch.float32, ), None, (1, [0, 2, 1]), 10002, (3, 1, 0, 6, 1, 0, 0, 72, 56, 64, 253952, 1, 0, 0, 0, 0, 0, 0, 0, [4096, 1500, 39, 0, 0, 0, 0, 0], [4096, 39, 1500, 0, 0, 0, 0, 0], [0, 2, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 2, 4, 3], [1, 1, 4096, 1500, 39], [1, 1, 4096, 39, 1500], [1, 1, 1, 1500, 39], [1, 1, 1, 39, 1500], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 1500, 39], [1, 1, 0, 39, 1500])),
# PYASC_TESTS_END
])
# yapf: enable
def test_transpose(profiler, runs, test_name, block_num, input_shapes, input_dtypes, output_shapes, output_dtypes,
                   compile_params, runtime_params, tiling_key, tiling_params):
    raw_input_shape = input_shapes[0]
    input_dtype = input_dtypes[0]
    in_cut_index = tiling_params[1]
    out_cut_index = tiling_params[2]
    in_ub_factor = tiling_params[3]
    out_ub_factor = tiling_params[4]
    ub_size = tiling_params[10]
    unroll_factor = runtime_params[0]
    raw_permute = runtime_params[1]

    input_shape, permute = simplify_shape(raw_input_shape, raw_permute)

    input = torch.randn(input_shape).to(input_dtype)
    out_shape = [input_shape[permute[i]] for i in range(0, len(input_shape))]
    out = torch.zeros(out_shape, dtype=input_dtype)
    items_in_block = 32 // input_dtype.itemsize
    if len(input_shape) == 1:
        tile_size = min(ub_size // unroll_factor // input.element_size(), asctile.ceildiv(input_shape[0], block_num))
        tile_size = asctile.ceildiv(tile_size, items_in_block) * items_in_block
        with profiler.profile():
            for _ in range(runs):
                simple_copy[block_num](input, out, input_shape[0], tile_size, unroll_factor)
    elif tiling_key == 10001 and len(permute) == 2:
        width = input_shape[1]
        height = input_shape[0]
        load_sizes = tiling_params[29]
        ub_size = [load_sizes[4], load_sizes[3]]

        if width > height:
            # load n columns
            block_size = tiling_params[4]
            if block_size is None or block_size == 0:
                block_size = asctile.ceildiv(width, block_num)
            tile_size = [
                asctile.ceildiv(height, items_in_block) * items_in_block,
                asctile.ceildiv(block_size, items_in_block) * items_in_block,
            ]
            total_tiles = asctile.ceildiv(width, block_size)
            with profiler.profile():
                for _ in range(runs):
                    transpose_column[block_num](input, out, width, height, block_size, tile_size[1], tile_size[0],
                                                total_tiles, unroll_factor)
        else:
            # load n rows
            block_size = tiling_params[4]
            if block_size is None or block_size == 0:
                block_size = asctile.ceildiv(height, block_num)
            tile_size = [
                asctile.ceildiv(block_size, items_in_block) * items_in_block,
                asctile.ceildiv(width, items_in_block) * items_in_block,
            ]
            total_tiles = asctile.ceildiv(height, block_size)
            with profiler.profile():
                for _ in range(runs):
                    transpose_line[block_num](input, out, width, height, block_size, tile_size[1], tile_size[0],
                                              total_tiles, unroll_factor)
    elif tiling_key == 10001:
        input_shape = list(input.shape)
        longest_axis = max(input.shape)
        axis = list(input.shape).index(longest_axis)

        axis_step = asctile.ceildiv(input.shape[axis], block_num)
        out = launch_one_axis(input, permute, axis, axis_step, unroll_factor, block_num, input_dtype, runs, profiler)
    elif tiling_key == 10004:  #Split on N-last dims
        out = launch_nlast_axis(input, permute, unroll_factor, block_num, input_dtype, ub_size, runs, profiler)
    elif tiling_key == 10002:  #Split on single axis
        longest_axis = max(input.shape)
        axis = list(input.shape).index(longest_axis)

        axis_step = asctile.ceildiv(input.shape[axis], block_num)
        ub_fit = ub_size * input.shape[axis] // math.prod(input.shape) // input.element_size() // unroll_factor
        assert (ub_fit > 0)
        axis_step = min(axis_step, ub_fit)

        out = launch_one_axis(input, permute, axis, axis_step, unroll_factor, block_num, input_dtype, runs, profiler)
    elif tiling_key == 10003:  #Split on two axes
        input_shape = list(input.shape)
        step = [in_ub_factor, out_ub_factor]
        axis = [0, out_cut_index]  # number in out_shape!
        for i in range(0, len(permute)):
            if permute[i] == in_cut_index:
                axis[0] = i
        assert (out_ub_factor != 0 and in_ub_factor != 0 and out_shape[axis[0]] > in_ub_factor
                and out_shape[axis[1]] > out_ub_factor)
        out = launch_2axis(input, permute, axis, step, input_dtype, block_num, unroll_factor, runs, profiler)
    else:
        raise RuntimeError(f"Wrong tilekey: {tiling_key}")

    golden = input.permute(*permute)
    torch.testing.assert_close(out, golden)
