# Source-derived from compiler-team/pyasc v2; see PROVENANCE.json.
import math
import torch
import asctile

from ._pyasc_runtime import ensure_npu_platform

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

@asctile.jit(reuse_alloc=1)
def simple_copy(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, input_lenth,
                tile_shape: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_lenth])
    out_gm = asctile.global_tensor(output_ptr, [input_lenth])
    total_repeats = asctile.ceildiv(input_lenth, tile_shape)

    for i in asctile.range(asctile.block_idx(), total_repeats, asctile.block_num(), unroll_factor=unroll_factor):
        data = asctile.copy_in(in_gm, [i * tile_shape], [tile_shape])
        asctile.copy_out(data, out_gm, [i * tile_shape])

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

def _ceildiv_host(value, divisor):
    return (value + divisor - 1) // divisor


def _align_host(value, alignment):
    return _ceildiv_host(value, alignment) * alignment


def transpose(x, perm):
    ensure_npu_platform()
    raw_shape = list(x.shape)
    raw_perm = [int(i) for i in perm]
    output_shape = [raw_shape[i] for i in raw_perm]
    if not x.is_contiguous():
        x = x.contiguous()
    input_shape, reduced_perm = simplify_shape(raw_shape, raw_perm)
    x_view = x.view(input_shape)
    reduced_output_shape = [input_shape[i] for i in reduced_perm]
    out = torch.empty(reduced_output_shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return out.view(output_shape)

    items_in_block = 32 // x.element_size()
    if len(input_shape) == 1 or reduced_perm == list(range(len(input_shape))):
        count = x.numel()
        tile = min(_align_host(_ceildiv_host(count, 72), items_in_block), 16384)
        repeats = _ceildiv_host(count, tile)
        simple_copy[min(72, repeats)](x_view, out, count, tile, 1)
        return out.view(output_shape)

    if len(input_shape) == 2 and reduced_perm == [1, 0]:
        height, width = input_shape
        block_width = 64 if x.element_size() <= 4 else 32
        block_height = block_width
        tile_width = _align_host(block_width, items_in_block)
        tile_height = _align_host(block_height, items_in_block)
        repeat = _ceildiv_host(width, block_width) * _ceildiv_host(height, block_height)
        transpose_block[min(72, repeat)](
            x_view, out, width, height, block_width, block_height,
            tile_width, tile_height, repeat, 1,
        )
        return out.view(output_shape)

    # The upstream target's general path splits two output axes.  Select the
    # two largest corresponding input dimensions so the unsplit physical tile
    # remains bounded.  This is transport/tiling only; transpose_2_axis is kept
    # byte-for-byte from the v2 target source.
    input_axes = sorted(range(len(input_shape)), key=lambda axis: input_shape[axis], reverse=True)[:2]
    store_axes = [reduced_perm.index(axis) for axis in input_axes]
    steps = []
    for axis in input_axes:
        step = items_in_block if axis in {len(input_shape) - 1, reduced_perm[-1]} else 1
        steps.append(min(input_shape[axis], step))
    ub_shape = list(input_shape)
    load_shape = list(input_shape)
    for axis, step in zip(input_axes, steps):
        ub_shape[axis] = step
        load_shape[axis] = step
    ub_shape[-1] = _align_host(ub_shape[-1], items_in_block)
    ub_shape[reduced_perm[-1]] = _align_host(ub_shape[reduced_perm[-1]], items_in_block)
    blocks0 = _ceildiv_host(input_shape[input_axes[0]], steps[0])
    blocks1 = _ceildiv_host(input_shape[input_axes[1]], steps[1])
    total_blocks = blocks0 * blocks1
    transpose_2_axis[min(72, total_blocks)](
        x_view, out, input_shape, steps, store_axes, ub_shape, load_shape,
        reduced_perm, total_blocks, blocks0, 1,
    )
    return out.view(output_shape)
