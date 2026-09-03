def rms_norm(x, gamma, epsilon=1e-6):
    ensure_npu_platform()
    original_shape = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    num_col = x.shape[-1]
    num_row = x.numel() // num_col
    x_2d = x.view(num_row, num_col)
    y = torch.empty_like(x_2d)
    if num_row == 0 or num_col == 0:
        return y.view(original_shape)
    rstd = torch.zeros(num_row, dtype=torch.float32, device=x.device)
    items_in_block = 32 // x.element_size()
    num_col_align = ((num_col + items_in_block - 1) // items_in_block) * items_in_block
    target_cores = min(72, num_row)
    block_factor = (num_row + target_cores - 1) // target_cores
    cores = (num_row + block_factor - 1) // block_factor
    last_block_factor = num_row - block_factor * (cores - 1)
    # The kernel already divides the square sum by num_col.  CANNBench's
    # standard RMS contract therefore requires no second scaling here.
    avg_factor = 1.0
    rms_norm_kernel[cores](
        x_2d, gamma, y, rstd,
        num_row, num_col, num_col_align,
        block_factor, 1, 1,
        epsilon, avg_factor, last_block_factor,
    )
    return y.view(original_shape)

