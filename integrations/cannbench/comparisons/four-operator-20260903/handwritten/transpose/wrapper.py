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
